"""
Training script for Transformer Diffusion Ablation Study
Usage:
    python train.py --config depth_2
    python train.py --config depth_4
    python train.py --config film
    ... (see ABLATION_CONFIGS below)
"""

import os
import math
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from model import TransformerDenoiser
from diffusion import GaussianDiffusion


# ─────────────────────────────────────────────
# Ablation Configurations
# ─────────────────────────────────────────────
ABLATION_CONFIGS = {
    # ── Architectural ──────────────────────────────────────────
    "depth_2":      dict(depth=2, d_model=256, use_pe=True,  cond_type="add",  use_flash_attn=True),
    "depth_4":      dict(depth=4, d_model=256, use_pe=True,  cond_type="add",  use_flash_attn=True),  # baseline
    "depth_6":      dict(depth=6, d_model=256, use_pe=True,  cond_type="add",  use_flash_attn=True),
    "width_128":    dict(depth=4, d_model=128, use_pe=True,  cond_type="add",  use_flash_attn=True),
    "width_256":    dict(depth=4, d_model=256, use_pe=True,  cond_type="add",  use_flash_attn=True),  # = baseline
    "no_pe":        dict(depth=4, d_model=256, use_pe=False, cond_type="add",  use_flash_attn=True),
    "film":         dict(depth=4, d_model=256, use_pe=True,  cond_type="film", use_flash_attn=True),
    # ── Flash Attention ablation ────────────────────────────────
    "no_flash":     dict(depth=4, d_model=256, use_pe=True,  cond_type="add",  use_flash_attn=False),
    # ── Sampling (all use depth_4 baseline model) ──────────────
    # trained with cfg_dropout so we can sweep guidance at inference
    "cfg_train":    dict(depth=4, d_model=256, use_pe=True,  cond_type="add",  use_flash_attn=True,
                         cfg_dropout_prob=0.15),
}

DEFAULT_TRAIN_CFG = dict(
    epochs=30,
    batch_size=256,
    lr=3e-4,
    T=1000,
    patch_size=4,
    img_size=28,
    in_channels=1,
    num_classes=10,
    cfg_dropout_prob=0.0,
    grad_clip=1.0,
    log_every=100,
    save_every=5,
    dataset="fashion_mnist",   # "mnist" or "fashion_mnist"
    seed=42,
)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataset(name, root="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),   # to [-1, 1]
    ])
    if name == "mnist":
        train = datasets.MNIST(root, train=True,  download=True, transform=tf)
        test  = datasets.MNIST(root, train=False, download=True, transform=tf)
    elif name == "fashion_mnist":
        train = datasets.FashionMNIST(root, train=True,  download=True, transform=tf)
        test  = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    elif name == "cifar10":
        tf_c = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        train = datasets.CIFAR10(root, train=True,  download=True, transform=tf_c)
        test  = datasets.CIFAR10(root, train=False, download=True, transform=tf_c)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test


class Trainer:
    def __init__(self, config_name, train_cfg=None, output_dir="./runs"):
        # Merge configs
        ablation = ABLATION_CONFIGS[config_name].copy()
        self.tcfg = {**DEFAULT_TRAIN_CFG, **(train_cfg or {})}
        self.tcfg["cfg_dropout_prob"] = ablation.pop("cfg_dropout_prob",
                                                       self.tcfg["cfg_dropout_prob"])
        self.model_cfg = ablation

        self.config_name = config_name
        self.out_dir = Path(output_dir) / config_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

        set_seed(self.tcfg["seed"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps"  if torch.backends.mps.is_available() else
                                   "cpu")
        print(f"[{config_name}] Device: {self.device}")

        # Dataset
        # Adjust for CIFAR-10
        dataset_name = self.tcfg["dataset"]
        if dataset_name == "cifar10":
            self.tcfg["img_size"] = 32
            self.tcfg["in_channels"] = 3
            self.tcfg["patch_size"] = 4   # 8x8=64 patches
        train_ds, _ = get_dataset(dataset_name)
        self.loader = DataLoader(train_ds, batch_size=self.tcfg["batch_size"],
                                 shuffle=True, num_workers=4, pin_memory=True)

        # Model
        self.model = TransformerDenoiser(
            img_size=self.tcfg["img_size"],
            patch_size=self.tcfg["patch_size"],
            in_channels=self.tcfg["in_channels"],
            d_model=self.model_cfg["d_model"],
            n_heads=max(1, self.model_cfg["d_model"] // 64),
            depth=self.model_cfg["depth"],
            num_classes=self.tcfg["num_classes"],
            use_pe=self.model_cfg["use_pe"],
            cond_type=self.model_cfg["cond_type"],
            use_flash_attn=self.model_cfg.get("use_flash_attn", True),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"[{config_name}] Model params: {n_params:.2f}M")

        self.diffusion = GaussianDiffusion(T=self.tcfg["T"], device=self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.tcfg["lr"])
        self.scaler    = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        # Logs
        self.log = {"loss": [], "step": [], "grad_norm": [], "epoch_loss": []}

    def save_checkpoint(self, epoch):
        path = self.out_dir / f"ckpt_ep{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config_name": self.config_name,
            "model_cfg": self.model_cfg,
            "train_cfg": self.tcfg,
        }, path)
        # Also save "latest"
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "config_name": self.config_name,
            "model_cfg": self.model_cfg,
            "train_cfg": self.tcfg,
        }, self.out_dir / "latest.pt")

    def train(self):
        import torch.nn.functional as F

        global_step = 0
        for epoch in range(1, self.tcfg["epochs"] + 1):
            self.model.train()
            epoch_losses = []
            t0 = time.time()

            for x0, labels in self.loader:
                x0     = x0.to(self.device)
                labels = labels.to(self.device)

                # Sample t, forward diffuse
                B = x0.shape[0]
                t = torch.randint(0, self.tcfg["T"], (B,), device=self.device)
                x_t, noise = self.diffusion.q_sample(x0, t)

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    pred = self.model(x_t, t, labels,
                                      cfg_dropout_prob=self.tcfg["cfg_dropout_prob"])
                    loss = F.mse_loss(pred, noise)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     self.tcfg["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_losses.append(loss.item())
                self.log["loss"].append(loss.item())
                self.log["step"].append(global_step)
                self.log["grad_norm"].append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                global_step += 1

                if global_step % self.tcfg["log_every"] == 0:
                    recent = np.mean(self.log["loss"][-self.tcfg["log_every"]:])
                    print(f"  step {global_step:6d} | loss {recent:.4f} | gnorm {grad_norm:.3f}")

            epoch_mean = np.mean(epoch_losses)
            epoch_std  = np.std(epoch_losses)
            self.log["epoch_loss"].append({"epoch": epoch, "mean": epoch_mean, "std": epoch_std})
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{self.tcfg['epochs']} | loss {epoch_mean:.4f} ± {epoch_std:.4f} | {elapsed:.1f}s")

            if epoch % self.tcfg["save_every"] == 0 or epoch == self.tcfg["epochs"]:
                self.save_checkpoint(epoch)

        # Save logs
        with open(self.out_dir / "log.json", "w") as f:
            json.dump(self.log, f)
        print(f"[{self.config_name}] Training complete. Logs saved to {self.out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="depth_4", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dataset", default=None, choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--output_dir", default="./runs")
    args = parser.parse_args()

    overrides = {}
    if args.epochs:     overrides["epochs"]     = args.epochs
    if args.batch_size: overrides["batch_size"] = args.batch_size
    if args.lr:         overrides["lr"]         = args.lr
    if args.dataset:    overrides["dataset"]    = args.dataset

    trainer = Trainer(args.config, train_cfg=overrides, output_dir=args.output_dir)
    trainer.train()


if __name__ == "__main__":
    main()
