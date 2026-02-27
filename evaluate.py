"""
Sampling & Evaluation Script
Generates samples, runs ablation sweeps, produces paper-ready figures.

Usage:
    # Generate 8×10 grid (8 samples per class) for a given checkpoint:
    python evaluate.py --ckpt runs/depth_4/latest.pt --sampler ddim --steps 50

    # Sweep DDIM steps for a CFG-trained model:
    python evaluate.py --ckpt runs/cfg_train/latest.pt --sweep ddim_steps

    # Sweep CFG guidance scale:
    python evaluate.py --ckpt runs/cfg_train/latest.pt --sweep cfg_scale

    # Plot training curves comparing all ablations:
    python evaluate.py --plot_curves --runs_dir runs/
"""

import os
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import make_grid

from model import TransformerDenoiser
from diffusion import GaussianDiffusion


FASHION_MNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Boot"
]

MNIST_CLASSES = [str(i) for i in range(10)]


# ─────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = ckpt["model_cfg"]
    tcfg = ckpt["train_cfg"]

    model = TransformerDenoiser(
        img_size=tcfg.get("img_size", 28),
        patch_size=tcfg.get("patch_size", 4),
        in_channels=tcfg.get("in_channels", 1),
        d_model=mcfg["d_model"],
        n_heads=max(1, mcfg["d_model"] // 64),
        depth=mcfg["depth"],
        num_classes=tcfg.get("num_classes", 10),
        use_pe=mcfg["use_pe"],
        cond_type=mcfg["cond_type"],
        use_flash_attn=mcfg.get("use_flash_attn", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(T=tcfg.get("T", 1000), device=device)
    return model, diffusion, tcfg


# ─────────────────────────────────────────────
# Sample grid (8 per class)
# ─────────────────────────────────────────────
@torch.no_grad()
def sample_grid(model, diffusion, tcfg, sampler="ddim", steps=50,
                guidance_scale=3.0, n_per_class=8, seed=42, device="cpu"):
    torch.manual_seed(seed)
    num_classes = tcfg.get("num_classes", 10)
    img_size    = tcfg.get("img_size", 28)
    in_channels = tcfg.get("in_channels", 1)

    all_samples = []
    for cls in range(num_classes):
        labels = torch.full((n_per_class,), cls, device=device, dtype=torch.long)
        shape  = (n_per_class, in_channels, img_size, img_size)

        if sampler == "ddpm":
            samples = diffusion.ddpm_sample(model, shape, labels,
                                             guidance_scale=guidance_scale)
        else:
            samples = diffusion.ddim_sample(model, shape, labels, steps=steps,
                                             guidance_scale=guidance_scale)
        all_samples.append(samples.cpu())

    # (num_classes * n_per_class, C, H, W)
    return torch.cat(all_samples, dim=0)


def save_sample_grid(samples, out_path, num_classes=10, n_per_class=8,
                     class_names=None, title=""):
    """Save (num_classes × n_per_class) image grid with class labels."""
    # Normalize from [-1,1] to [0,1]
    imgs = (samples * 0.5 + 0.5).clamp(0, 1)
    in_channels = imgs.shape[1]

    fig, axes = plt.subplots(num_classes, n_per_class,
                             figsize=(n_per_class * 1.2, num_classes * 1.2))
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    for c in range(num_classes):
        for i in range(n_per_class):
            ax = axes[c, i]
            img = imgs[c * n_per_class + i]
            if in_channels == 1:
                ax.imshow(img[0].numpy(), cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis("off")
            if i == 0 and class_names:
                ax.set_ylabel(class_names[c], fontsize=7, rotation=0,
                              labelpad=35, va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid: {out_path}")


# ─────────────────────────────────────────────
# DDIM steps sweep
# ─────────────────────────────────────────────
def sweep_ddim_steps(model, diffusion, tcfg, out_dir, device,
                     steps_list=(10, 25, 50, 100, 200, 1000), seed=42):
    """Compare quality at different DDIM step counts (same noise seed)."""
    num_classes = tcfg.get("num_classes", 10)
    img_size    = tcfg.get("img_size", 28)
    in_channels = tcfg.get("in_channels", 1)
    n_show = 5  # one sample per class, 5 classes for display

    torch.manual_seed(seed)
    noise_ref = torch.randn(n_show, in_channels, img_size, img_size, device=device)
    labels    = torch.arange(n_show, device=device, dtype=torch.long)

    fig, axes = plt.subplots(len(steps_list), n_show,
                             figsize=(n_show * 1.5, len(steps_list) * 1.5))
    fig.suptitle("DDIM: Steps sweep (same noise init)", fontsize=11)

    for row, steps in enumerate(steps_list):
        sampler = "ddpm" if steps == 1000 else "ddim"
        if sampler == "ddim":
            samples = diffusion.ddim_sample(
                model,
                (n_show, in_channels, img_size, img_size),
                labels,
                steps=steps,
                x_T=noise_ref,
            )
        else:
            samples = diffusion.ddpm_sample(
                model,
                (n_show, in_channels, img_size, img_size),
                labels,
                x_T=noise_ref,
            )
        imgs = (samples.cpu() * 0.5 + 0.5).clamp(0, 1)

        for col in range(n_show):
            ax = axes[row, col]
            if in_channels == 1:
                ax.imshow(imgs[col, 0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(imgs[col].permute(1, 2, 0))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"T={steps}", fontsize=8, rotation=0,
                              labelpad=30, va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "sweep_ddim_steps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved DDIM steps sweep.")


# ─────────────────────────────────────────────
# CFG guidance scale sweep
# ─────────────────────────────────────────────
def sweep_cfg(model, diffusion, tcfg, out_dir, device,
              scales=(0.0, 1.0, 2.0, 3.0, 5.0, 7.0), seed=42):
    num_classes = tcfg.get("num_classes", 10)
    img_size    = tcfg.get("img_size", 28)
    in_channels = tcfg.get("in_channels", 1)
    n_show = 5

    labels = torch.arange(n_show, device=device, dtype=torch.long)
    fig, axes = plt.subplots(len(scales), n_show,
                             figsize=(n_show * 1.5, len(scales) * 1.5))
    fig.suptitle("CFG Guidance Scale Sweep (DDIM 50 steps)", fontsize=11)

    for row, w in enumerate(scales):
        torch.manual_seed(seed)
        samples = diffusion.ddim_sample(model, (n_show, in_channels, img_size, img_size),
                                         labels, steps=50, guidance_scale=w)
        imgs = (samples.cpu() * 0.5 + 0.5).clamp(0, 1)

        for col in range(n_show):
            ax = axes[row, col]
            if in_channels == 1:
                ax.imshow(imgs[col, 0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(imgs[col].permute(1, 2, 0))
            ax.axis("off")
            if col == 0:
                lbl = "Uncond." if w == 0 else f"w={w}"
                ax.set_ylabel(lbl, fontsize=8, rotation=0, labelpad=35, va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "sweep_cfg_scale.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved CFG scale sweep.")


# ─────────────────────────────────────────────
# Plot training curves from log.json files
# ─────────────────────────────────────────────
def moving_average(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training_curves(runs_dir, out_path, configs=None):
    """Overlay training loss curves for all (or specified) configs."""
    runs_dir = Path(runs_dir)
    if configs is None:
        configs = [p.name for p in runs_dir.iterdir() if p.is_dir()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for i, cfg_name in enumerate(configs):
        log_path = runs_dir / cfg_name / "log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            log = json.load(f)

        steps  = np.array(log["step"])
        losses = np.array(log["loss"])
        grad_norms = np.array(log["grad_norm"])

        # Smooth
        if len(losses) > 100:
            smooth_loss = moving_average(losses, 100)
            smooth_steps = steps[99:]
        else:
            smooth_loss = losses
            smooth_steps = steps

        axes[0].plot(smooth_steps, smooth_loss, label=cfg_name, color=colors[i], linewidth=1.5)

        # Epoch-level std (stability proxy)
        epoch_data = log.get("epoch_loss", [])
        if epoch_data:
            epochs = [e["epoch"] for e in epoch_data]
            stds   = [e["std"] for e in epoch_data]
            axes[1].plot(epochs, stds, label=cfg_name, color=colors[i], linewidth=1.5, marker="o", ms=3)

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("MSE Loss (smoothed)")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Intra-epoch Loss Std")
    axes[1].set_title("Training Stability (Loss Variance)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves: {out_path}")


# ─────────────────────────────────────────────
# Summary table of final losses
# ─────────────────────────────────────────────
def print_summary_table(runs_dir):
    runs_dir = Path(runs_dir)
    rows = []
    for cfg_dir in sorted(runs_dir.iterdir()):
        log_path = cfg_dir / "log.json"
        if not log_path.exists():
            continue
        with open(log_path) as f:
            log = json.load(f)
        epoch_data = log.get("epoch_loss", [])
        if not epoch_data:
            continue
        last = epoch_data[-1]
        rows.append({
            "config": cfg_dir.name,
            "final_loss": f"{last['mean']:.4f}",
            "final_std":  f"{last['std']:.4f}",
            "epochs":     last["epoch"],
        })

    if not rows:
        print("No completed runs found.")
        return

    print(f"\n{'Config':<20} {'Final Loss':>12} {'Loss Std':>10} {'Epochs':>8}")
    print("─" * 55)
    for r in rows:
        print(f"{r['config']:<20} {r['final_loss']:>12} {r['final_std']:>10} {r['epochs']:>8}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path for sampling")
    parser.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG guidance scale")
    parser.add_argument("--n_per_class", type=int, default=8)
    parser.add_argument("--sweep", default=None, choices=["ddim_steps", "cfg_scale"])
    parser.add_argument("--plot_curves", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--runs_dir", default="./runs")
    parser.add_argument("--out_dir", default="./figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_curves:
        plot_training_curves(args.runs_dir,
                              out_dir / "training_curves.png")

    if args.summary:
        print_summary_table(args.runs_dir)

    if args.ckpt:
        model, diffusion, tcfg = load_model(args.ckpt, device)
        dataset = tcfg.get("dataset", "fashion_mnist")
        class_names = FASHION_MNIST_CLASSES if "fashion" in dataset else MNIST_CLASSES

        if args.sweep == "ddim_steps":
            sweep_ddim_steps(model, diffusion, tcfg, out_dir, device, seed=args.seed)
        elif args.sweep == "cfg_scale":
            sweep_cfg(model, diffusion, tcfg, out_dir, device, seed=args.seed)
        else:
            # Default: full sample grid
            cfg_name = Path(args.ckpt).parent.name
            title = f"{cfg_name} | {args.sampler.upper()} steps={args.steps} w={args.guidance}"
            samples = sample_grid(model, diffusion, tcfg,
                                  sampler=args.sampler, steps=args.steps,
                                  guidance_scale=args.guidance,
                                  n_per_class=args.n_per_class,
                                  seed=args.seed, device=device)
            out_path = out_dir / f"samples_{cfg_name}_{args.sampler}_{args.steps}steps.png"
            save_sample_grid(samples, out_path,
                             num_classes=tcfg.get("num_classes", 10),
                             n_per_class=args.n_per_class,
                             class_names=class_names,
                             title=title)


if __name__ == "__main__":
    main()
