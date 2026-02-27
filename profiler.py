"""
Efficiency Profiler: Throughput, Memory, Latency
Measures training throughput and inference latency for Flash vs Standard attention.

Usage:
    python profiler.py --mode all
    python profiler.py --mode throughput
    python profiler.py --mode memory
    python profiler.py --mode inference
"""

import argparse
import time
import json
import gc
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from model import TransformerDenoiser
from diffusion import GaussianDiffusion


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def make_model(d_model=256, depth=4, use_flash=True, img_size=28,
               in_channels=1, device="cuda"):
    return TransformerDenoiser(
        img_size=img_size,
        patch_size=4,
        in_channels=in_channels,
        d_model=d_model,
        n_heads=max(1, d_model // 64),
        depth=depth,
        num_classes=10,
        use_pe=True,
        cond_type="add",
        use_flash_attn=use_flash,
    ).to(device)


def peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


# ─────────────────────────────────────────────
# 1. Training Throughput (samples/sec)
# ─────────────────────────────────────────────
def benchmark_throughput(device, batch_sizes=(32, 64, 128, 256, 512),
                          d_model=256, depth=4, img_size=28, in_channels=1,
                          n_iters=50, warmup=10):
    """
    Measure forward+backward throughput (samples/sec) for Flash vs Standard.
    """
    print("\n" + "=" * 60)
    print("Throughput Benchmark (training fwd+bwd)")
    print("=" * 60)
    diffusion = GaussianDiffusion(T=1000, device=device)
    results = {"flash": {}, "standard": {}}

    for use_flash in [True, False]:
        key = "flash" if use_flash else "standard"
        label = "Flash Attn" if use_flash else "Standard  "
        print(f"\n{label}")
        print(f"  {'batch':>8} | {'throughput (img/s)':>20} | {'mem (MB)':>10}")
        print(f"  {'-'*8}-+-{'-'*20}-+-{'-'*10}")

        for bs in batch_sizes:
            clear_cache()
            try:
                model = make_model(d_model, depth, use_flash, img_size, in_channels, device)
                opt   = torch.optim.AdamW(model.parameters(), lr=3e-4)
                scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

                x0     = torch.randn(bs, in_channels, img_size, img_size, device=device)
                labels = torch.randint(0, 10, (bs,), device=device)
                t      = torch.randint(0, 1000, (bs,), device=device)

                # Warm-up
                for _ in range(warmup):
                    x_t, noise = diffusion.q_sample(x0, t)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        pred = model(x_t, t, labels)
                        loss = nn.functional.mse_loss(pred, noise)
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                for _ in range(n_iters):
                    x_t, noise = diffusion.q_sample(x0, t)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        pred = model(x_t, t, labels)
                        loss = nn.functional.mse_loss(pred, noise)
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed   = time.time() - t0
                throughput = bs * n_iters / elapsed
                mem_mb     = peak_memory_mb()

                results[key][bs] = {"throughput": throughput, "mem_mb": mem_mb}
                print(f"  {bs:>8} | {throughput:>20.1f} | {mem_mb:>10.1f}")

                del model, opt
                clear_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  {bs:>8} | {'OOM':>20} | {'OOM':>10}")
                    results[key][bs] = {"throughput": None, "mem_mb": None}
                    clear_cache()
                else:
                    raise

    return results


# ─────────────────────────────────────────────
# 2. Memory Scaling (batch × d_model)
# ─────────────────────────────────────────────
def benchmark_memory(device, d_models=(128, 256, 512, 768),
                     batch_size=64, img_size=28, in_channels=1):
    """
    Peak memory (MB) as function of d_model for Flash vs Standard.
    Shows Flash Attention's O(N) memory advantage.
    """
    print("\n" + "=" * 60)
    print("Memory Scaling (forward pass, fixed batch=64)")
    print("=" * 60)
    diffusion = GaussianDiffusion(T=1000, device=device)
    results = {"flash": {}, "standard": {}}

    for use_flash in [True, False]:
        key = "flash" if use_flash else "standard"
        label = "Flash Attn" if use_flash else "Standard  "
        print(f"\n{label}")
        print(f"  {'d_model':>8} | {'n_params (M)':>14} | {'peak mem (MB)':>14}")
        print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*14}")

        for d in d_models:
            clear_cache()
            try:
                model = make_model(d, depth=4, use_flash=use_flash,
                                   img_size=img_size, in_channels=in_channels,
                                   device=device)
                n_params = sum(p.numel() for p in model.parameters()) / 1e6

                x0     = torch.randn(batch_size, in_channels, img_size, img_size, device=device)
                t      = torch.randint(0, 1000, (batch_size,), device=device)
                labels = torch.randint(0, 10, (batch_size,), device=device)
                x_t, noise = diffusion.q_sample(x0, t)

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pred = model(x_t, t, labels)
                    loss = nn.functional.mse_loss(pred, noise)
                loss.backward()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                mem = peak_memory_mb()

                results[key][d] = {"n_params_M": n_params, "peak_mem_mb": mem}
                print(f"  {d:>8} | {n_params:>14.2f} | {mem:>14.1f}")

                del model
                clear_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  {d:>8} | {'—':>14} | {'OOM':>14}")
                    results[key][d] = {"n_params_M": None, "peak_mem_mb": None}
                    clear_cache()
                else:
                    raise

    return results


# ─────────────────────────────────────────────
# 3. Inference Latency (ms per batch)
# ─────────────────────────────────────────────
def benchmark_inference(device, steps_list=(10, 25, 50, 100, 200),
                        batch_size=32, d_model=256, depth=4,
                        img_size=28, in_channels=1, n_iters=20):
    """
    DDIM inference latency (ms/batch) for Flash vs Standard × different step counts.
    """
    print("\n" + "=" * 60)
    print("Inference Latency (DDIM, ms per batch)")
    print("=" * 60)
    diffusion = GaussianDiffusion(T=1000, device=device)
    results = {"flash": {}, "standard": {}}

    for use_flash in [True, False]:
        key = "flash" if use_flash else "standard"
        label = "Flash Attn" if use_flash else "Standard  "
        print(f"\n{label} (batch={batch_size})")
        print(f"  {'steps':>8} | {'ms/batch':>12} | {'img/s':>10}")
        print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}")

        model = make_model(d_model, depth, use_flash, img_size, in_channels, device)
        model.eval()

        for steps in steps_list:
            clear_cache()
            labels = torch.randint(0, 10, (batch_size,), device=device)
            shape  = (batch_size, in_channels, img_size, img_size)

            # Warm-up
            for _ in range(3):
                diffusion.ddim_sample(model, shape, labels, steps=steps)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n_iters):
                diffusion.ddim_sample(model, shape, labels, steps=steps)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            ms_per_batch = elapsed / n_iters * 1000
            imgs_per_sec  = batch_size * n_iters / elapsed
            results[key][steps] = {"ms_per_batch": ms_per_batch, "imgs_per_sec": imgs_per_sec}
            print(f"  {steps:>8} | {ms_per_batch:>12.1f} | {imgs_per_sec:>10.1f}")

        del model
        clear_cache()

    return results


# ─────────────────────────────────────────────
# 4. Plot Efficiency Figures
# ─────────────────────────────────────────────
def plot_efficiency_figures(throughput_results, memory_results, inference_results,
                             out_dir="./figures"):
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = {"flash": "#2196F3", "standard": "#FF5722"}
    labels = {"flash": "Flash Attention", "standard": "Standard Attention"}

    # ── Figure A: Throughput vs Batch Size ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for key in ["flash", "standard"]:
        data = throughput_results[key]
        bs_list = sorted(data.keys())
        tp = [data[bs]["throughput"] for bs in bs_list]
        ax.plot(bs_list, tp, marker="o", label=labels[key],
                color=colors[key], linewidth=2)
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Throughput (images/sec)", fontsize=11)
    ax.set_title("Training Throughput vs Batch Size", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)

    # ── Figure B: Peak Memory vs Batch Size ─────────────────────
    ax = axes[1]
    for key in ["flash", "standard"]:
        data = throughput_results[key]
        bs_list = sorted(data.keys())
        mem = [data[bs]["mem_mb"] for bs in bs_list
               if data[bs]["mem_mb"] is not None]
        bs_valid = [bs for bs in bs_list if data[bs]["mem_mb"] is not None]
        ax.plot(bs_valid, mem, marker="s", label=labels[key],
                color=colors[key], linewidth=2)
    ax.set_xlabel("Batch Size", fontsize=11)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=11)
    ax.set_title("Memory Usage vs Batch Size", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "efficiency_throughput_memory.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: efficiency_throughput_memory.png")

    # ── Figure C: Memory Scaling vs d_model ─────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key in ["flash", "standard"]:
        data = memory_results[key]
        dims = sorted(data.keys())
        mem  = [data[d]["peak_mem_mb"] for d in dims if data[d]["peak_mem_mb"] is not None]
        dims_valid = [d for d in dims if data[d]["peak_mem_mb"] is not None]
        ax.plot(dims_valid, mem, marker="o", label=labels[key],
                color=colors[key], linewidth=2)
    ax.set_xlabel("d_model", fontsize=11)
    ax.set_ylabel("Peak Memory (MB)", fontsize=11)
    ax.set_title("Memory Scaling with Model Width", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "efficiency_memory_scaling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: efficiency_memory_scaling.png")

    # ── Figure D: Inference Latency vs DDIM Steps ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for key in ["flash", "standard"]:
        data = inference_results[key]
        steps_list = sorted(data.keys())
        ms = [data[s]["ms_per_batch"] for s in steps_list]
        ax.plot(steps_list, ms, marker="o", label=labels[key],
                color=colors[key], linewidth=2)
    ax.set_xlabel("DDIM Steps", fontsize=11)
    ax.set_ylabel("Latency (ms/batch)", fontsize=11)
    ax.set_title("Inference Latency vs Sampling Steps", fontsize=12)
    ax.legend(); ax.grid(alpha=0.3)

    # ── Speedup ratio ────────────────────────────────────────────
    ax = axes[1]
    steps_common = sorted(set(inference_results["flash"].keys()) &
                          set(inference_results["standard"].keys()))
    speedup = [inference_results["standard"][s]["ms_per_batch"] /
               inference_results["flash"][s]["ms_per_batch"]
               for s in steps_common]
    ax.bar(range(len(steps_common)), speedup, color=colors["flash"], alpha=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(steps_common)))
    ax.set_xticklabels([str(s) for s in steps_common])
    ax.set_xlabel("DDIM Steps", fontsize=11)
    ax.set_ylabel("Speedup (Standard / Flash)", fontsize=11)
    ax.set_title("Flash Attention Speedup", fontsize=12)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "efficiency_inference.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: efficiency_inference.png")


# ─────────────────────────────────────────────
# 5. Pareto Frontier: FID vs Throughput
# ─────────────────────────────────────────────
def plot_pareto(results_list, out_dir="./figures"):
    """
    results_list: list of dicts with keys:
        name, fid, throughput, use_flash, batch_size, d_model
    """
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"flash": "#2196F3", "standard": "#FF5722"}

    for r in results_list:
        color = colors["flash"] if r["use_flash"] else colors["standard"]
        ax.scatter(r["throughput"], r["fid"], color=color, s=80, zorder=3)
        ax.annotate(r["name"], (r["throughput"], r["fid"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    # Draw Pareto frontier for flash configs
    flash_pts = [(r["throughput"], r["fid"]) for r in results_list if r["use_flash"]]
    flash_pts.sort(key=lambda x: x[0])
    # Pareto: for increasing throughput, FID should decrease
    pareto = []
    min_fid = float("inf")
    for tp, fid in flash_pts:
        if fid < min_fid:
            pareto.append((tp, fid))
            min_fid = fid
    if pareto:
        px, py = zip(*pareto)
        ax.plot(px, py, "--", color=colors["flash"], linewidth=1.5, alpha=0.7,
                label="Flash Pareto frontier")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["flash"],
               markersize=9, label="Flash Attention"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["standard"],
               markersize=9, label="Standard Attention"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("Training Throughput (images/sec)", fontsize=11)
    ax.set_ylabel("FID ↓", fontsize=11)
    ax.set_title("Pareto Frontier: Sample Quality vs Efficiency", fontsize=12)
    ax.invert_yaxis()   # lower FID is better → up
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_fid_throughput.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: pareto_fid_throughput.png")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "throughput", "memory", "inference"])
    parser.add_argument("--img_size",    type=int, default=28)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--d_model",     type=int, default=256)
    parser.add_argument("--depth",       type=int, default=4)
    parser.add_argument("--out_dir",     default="./figures")
    parser.add_argument("--save_json",   default="./figures/efficiency_results.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cpu":
        print("WARNING: No GPU detected. Results will be slow and memory stats unavailable.")

    all_results = {}

    if args.mode in ("all", "throughput"):
        r = benchmark_throughput(device, img_size=args.img_size,
                                 in_channels=args.in_channels,
                                 d_model=args.d_model, depth=args.depth)
        all_results["throughput"] = r

    if args.mode in ("all", "memory"):
        r = benchmark_memory(device, img_size=args.img_size,
                              in_channels=args.in_channels)
        all_results["memory"] = r

    if args.mode in ("all", "inference"):
        r = benchmark_inference(device, img_size=args.img_size,
                                 in_channels=args.in_channels,
                                 d_model=args.d_model, depth=args.depth)
        all_results["inference"] = r

    # Save JSON
    Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.save_json}")

    # Plot
    if "throughput" in all_results and "memory" in all_results and "inference" in all_results:
        plot_efficiency_figures(
            all_results["throughput"],
            all_results["memory"],
            all_results["inference"],
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
