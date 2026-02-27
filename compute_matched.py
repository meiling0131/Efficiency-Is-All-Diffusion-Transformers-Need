"""
Compute-Matched Experiment
─────────────────────────────────────────────────────────────────
Key insight for the efficiency paper:
  Flash Attention trains faster → more iterations in same wall-clock time
  → better FID under the same compute budget

This script:
1. Measures how many iters/sec Flash vs Standard can do (at their max batch size)
2. Trains both for the same wall-clock time budget
3. Evaluates FID at regular checkpoints
4. Plots: FID vs Wall-Clock Time (the main paper figure)

Usage:
    python compute_matched.py --budget_minutes 30 --dataset fashion_mnist
    python compute_matched.py --budget_minutes 60 --dataset cifar10
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import TransformerDenoiser
from diffusion import GaussianDiffusion
from fid import FIDEvaluator


def get_dataset(name, root="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    if name == "fashion_mnist":
        train = datasets.FashionMNIST(root, train=True,  download=True, transform=tf)
        test  = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    elif name == "mnist":
        train = datasets.MNIST(root, train=True,  download=True, transform=tf)
        test  = datasets.MNIST(root, train=False, download=True, transform=tf)
    elif name == "cifar10":
        tf_c = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        train = datasets.CIFAR10(root, train=True,  download=True, transform=tf_c)
        test  = datasets.CIFAR10(root, train=False, download=True, transform=tf_c)
    return train, test


def make_model(use_flash, d_model, depth, img_size, in_channels, device):
    return TransformerDenoiser(
        img_size=img_size, patch_size=4, in_channels=in_channels,
        d_model=d_model, n_heads=max(1, d_model // 64),
        depth=depth, num_classes=10,
        use_pe=True, cond_type="add",
        use_flash_attn=use_flash,
    ).to(device)


# ─────────────────────────────────────────────
# Auto-find max batch size without OOM
# ─────────────────────────────────────────────
def find_max_batch_size(use_flash, d_model, depth, img_size, in_channels,
                         device, candidates=(64, 128, 256, 512, 1024)):
    diffusion = GaussianDiffusion(T=1000, device=device)
    max_bs = None
    for bs in candidates:
        try:
            model = make_model(use_flash, d_model, depth, img_size, in_channels, device)
            opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)
            x0    = torch.randn(bs, in_channels, img_size, img_size, device=device)
            t     = torch.randint(0, 1000, (bs,), device=device)
            labs  = torch.randint(0, 10, (bs,), device=device)
            x_t, noise = diffusion.q_sample(x0, t)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(x_t, t, labs)
                loss = nn.functional.mse_loss(pred, noise)
            loss.backward()
            max_bs = bs
            del model, opt, x0, x_t, pred, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            raise
    if max_bs is None:
        kind = "Flash" if use_flash else "Standard"
        raise RuntimeError(
            f"{kind} attention OOM for all tested batch sizes {tuple(candidates)}. "
            "Reduce model size or provide smaller batch-size candidates."
        )
    return max_bs


# ─────────────────────────────────────────────
# Train for a fixed wall-clock budget
# ─────────────────────────────────────────────
def train_budget(use_flash, batch_size, d_model, depth, img_size, in_channels,
                 train_ds, dataset_name, device, budget_seconds,
                 fid_eval, fid_interval_secs=120,
                 fid_n_samples=2000, out_dir=None, seed=42):
    """
    Train for `budget_seconds` wall-clock seconds.
    Evaluate FID every `fid_interval_secs` seconds.
    Returns: list of (wall_time_secs, fid_score, n_iters)
    """
    torch.manual_seed(seed)
    label = "Flash" if use_flash else "Standard"
    print(f"\n{'='*55}")
    print(f"Training: {label} | batch={batch_size} | budget={budget_seconds//60}min")
    print(f"{'='*55}")

    model    = make_model(use_flash, d_model, depth, img_size, in_channels, device)
    diffusion = GaussianDiffusion(T=1000, device=device)
    opt      = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler   = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    loader   = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)

    tcfg = {
        "img_size": img_size, "in_channels": in_channels,
        "num_classes": 10, "dataset": dataset_name,
        "T": 1000, "patch_size": 4,
    }

    history = []   # (wall_secs, fid, n_iters)
    n_iters = 0
    t_start = time.time()
    t_last_fid = t_start
    loader_iter = iter(loader)

    while True:
        wall = time.time() - t_start
        if wall >= budget_seconds:
            break

        # FID checkpoint
        if (time.time() - t_last_fid) >= fid_interval_secs or n_iters == 0:
            if n_iters > 0:   # skip FID at step 0
                model.eval()
                fid = fid_eval.compute_fid(model, diffusion, tcfg,
                                            n_samples=fid_n_samples,
                                            sampler="ddim", steps=50,
                                            guidance_scale=3.0)
                wall_now = time.time() - t_start
                history.append({"wall_secs": wall_now, "fid": fid, "n_iters": n_iters})
                print(f"  t={wall_now/60:.1f}min | iter={n_iters:6d} | FID={fid:.2f}")
                t_last_fid = time.time()
                model.train()

                if out_dir:
                    torch.save({"model": model.state_dict(), "n_iters": n_iters,
                                "use_flash": use_flash},
                               Path(out_dir) / f"{'flash' if use_flash else 'std'}_iter{n_iters}.pt")

        # Training step
        try:
            x0, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x0, labels = next(loader_iter)

        x0, labels = x0.to(device), labels.to(device)
        t_step = torch.randint(0, 1000, (x0.shape[0],), device=device)
        x_t, noise = diffusion.q_sample(x0, t_step)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            pred = model(x_t, t_step, labels)
            loss = nn.functional.mse_loss(pred, noise)

        opt.zero_grad()
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        n_iters += 1

    # Final FID
    model.eval()
    fid = fid_eval.compute_fid(model, diffusion, tcfg,
                                n_samples=fid_n_samples,
                                sampler="ddim", steps=50, guidance_scale=3.0)
    wall_now = time.time() - t_start
    history.append({"wall_secs": wall_now, "fid": fid, "n_iters": n_iters})
    print(f"  FINAL: t={wall_now/60:.1f}min | iter={n_iters} | FID={fid:.2f}")

    iters_per_sec = n_iters / wall_now
    print(f"  Throughput: {iters_per_sec:.1f} iters/sec ({iters_per_sec * batch_size:.0f} img/sec)")

    return history, model


# ─────────────────────────────────────────────
# Plot: FID vs Wall-Clock Time
# ─────────────────────────────────────────────
def plot_fid_vs_time(flash_history, std_history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = {"flash": "#2196F3", "standard": "#FF5722"}

    for ax_idx, metric in enumerate(["wall_mins", "n_iters"]):
        ax = axes[ax_idx]
        for key, history, label in [
            ("flash",    flash_history, "Flash Attention"),
            ("standard", std_history,   "Standard Attention"),
        ]:
            if metric == "wall_mins":
                xs = [h["wall_secs"] / 60 for h in history]
                ax.set_xlabel("Wall-Clock Time (minutes)", fontsize=11)
            else:
                xs = [h["n_iters"] for h in history]
                ax.set_xlabel("Training Iterations", fontsize=11)

            ys = [h["fid"] for h in history]
            ax.plot(xs, ys, marker="o", label=label,
                    color=colors[key], linewidth=2, markersize=5)

        ax.set_ylabel("FID ↓", fontsize=11)
        ax.set_title(
            "FID vs Wall-Clock Time\n(same compute budget)" if metric == "wall_mins"
            else "FID vs Training Iterations",
            fontsize=11
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_minutes", type=int, default=30)
    parser.add_argument("--fid_interval_minutes", type=float, default=5.0)
    parser.add_argument("--fid_n_samples", type=int, default=2000)
    parser.add_argument("--dataset", default="fashion_mnist",
                        choices=["mnist", "fashion_mnist", "cifar10"])
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--depth",   type=int, default=4)
    parser.add_argument("--out_dir", default="./runs/compute_matched")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("./figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    img_size    = 32 if args.dataset == "cifar10" else 28
    in_channels = 3  if args.dataset == "cifar10" else 1
    train_ds, test_ds = get_dataset(args.dataset)

    # Real FID stats (computed once)
    real_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    fid_eval    = FIDEvaluator(device=device, batch_size=64)
    fid_eval.compute_real_stats(real_loader)

    budget_secs    = args.budget_minutes * 60
    fid_interval   = args.fid_interval_minutes * 60

    # Find max batch size for each
    print("\nFinding max batch sizes...")
    bs_flash = find_max_batch_size(True,  args.d_model, args.depth,
                                    img_size, in_channels, device)
    bs_std   = find_max_batch_size(False, args.d_model, args.depth,
                                    img_size, in_channels, device)
    print(f"  Flash max batch: {bs_flash}")
    print(f"  Standard max batch: {bs_std}")

    # Train both
    flash_history, flash_model = train_budget(
        use_flash=True, batch_size=bs_flash,
        d_model=args.d_model, depth=args.depth,
        img_size=img_size, in_channels=in_channels,
        train_ds=train_ds, dataset_name=args.dataset,
        device=device, budget_seconds=budget_secs,
        fid_eval=fid_eval, fid_interval_secs=fid_interval,
        fid_n_samples=args.fid_n_samples,
        out_dir=out_dir, seed=args.seed,
    )

    std_history, std_model = train_budget(
        use_flash=False, batch_size=bs_std,
        d_model=args.d_model, depth=args.depth,
        img_size=img_size, in_channels=in_channels,
        train_ds=train_ds, dataset_name=args.dataset,
        device=device, budget_seconds=budget_secs,
        fid_eval=fid_eval, fid_interval_secs=fid_interval,
        fid_n_samples=args.fid_n_samples,
        out_dir=out_dir, seed=args.seed,
    )

    # Save results
    results = {
        "flash":    {"batch_size": bs_flash, "history": flash_history},
        "standard": {"batch_size": bs_std,   "history": std_history},
        "config":   vars(args),
    }
    with open(out_dir / "compute_matched_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_fid_vs_time(flash_history, std_history,
                     fig_dir / "fid_vs_walltime.png")

    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Flash   : batch={bs_flash}, final FID={flash_history[-1]['fid']:.2f}, "
          f"iters={flash_history[-1]['n_iters']}")
    print(f"  Standard: batch={bs_std},   final FID={std_history[-1]['fid']:.2f}, "
          f"iters={std_history[-1]['n_iters']}")
    iters_ratio = flash_history[-1]["n_iters"] / max(std_history[-1]["n_iters"], 1)
    print(f"  Flash ran {iters_ratio:.2f}x more iterations in same wall-clock time")


if __name__ == "__main__":
    main()
