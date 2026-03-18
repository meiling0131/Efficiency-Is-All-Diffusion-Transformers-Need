# Efficiency Is All Diffusion Transformers Need

A compute-first study of Transformer-based diffusion models.

This repository tests a simple claim:

> **Flash Attention enables better quality-efficiency Pareto frontiers in diffusion Transformers under matched compute budgets.**

Instead of only comparing training loss, we compare models under the same wall-clock budget (same GPU time) and ask: **who gets lower FID per unit time?**

## Abstract-Style Summary

Transformer denoisers for diffusion are often bottlenecked by attention memory and throughput. Using PyTorch SDPA (`scaled_dot_product_attention`) as Flash Attention backend, we evaluate training throughput, memory scaling, and inference latency, then run compute-matched training where Flash and Standard attention are given the same training time. We show that Flash shifts the Pareto frontier by enabling larger feasible batch sizes and more optimization steps within fixed compute.

## Main Contributions

1. Compute-matched protocol for diffusion Transformers (`FID vs wall-clock`).
2. Unified efficiency benchmark suite (`throughput`, `peak memory`, `latency`).
3. Practical evidence that Flash Attention changes reachable operating points, not just implementation speed.

## Repository Layout

```text
.
├── model.py                  # TransformerDenoiser + SDPA/standard attention toggle
├── diffusion.py              # DDPM/DDIM + classifier-free guidance
├── train.py                  # Ablation training configs
├── evaluate.py               # Sampling, sweeps, curve plotting
├── fid.py                    # InceptionV3-based FID computation
├── profiler.py               # Throughput/memory/latency benchmark
├── compute_matched.py        # Core experiment: FID vs wall-clock
├── run_efficiency_suite.sh   # One-command efficiency pipeline
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Reproduce Core Efficiency Results

### One-command pipeline

```bash
bash run_efficiency_suite.sh \
  --budget_minutes 60 \
  --dataset fashion_mnist
```

### Manual steps

```bash
# 1) Efficiency profiling (no training checkpoint needed)
python profiler.py --mode all --out_dir ./figures

# 2) Compute-matched training (main figure)
python compute_matched.py \
  --budget_minutes 60 \
  --fid_interval_minutes 5 \
  --dataset fashion_mnist
```

## Key Outputs

- `figures/fid_vs_walltime.png`:
  main claim figure (same compute budget, different FID trajectory)
- `figures/efficiency_throughput_memory.png`:
  throughput + memory vs batch size
- `figures/efficiency_memory_scaling.png`:
  memory vs model width (`d_model`)
- `figures/efficiency_inference.png`:
  DDIM inference latency and speedup
- `runs/efficiency_suite/compute_matched/compute_matched_results.json`:
  raw experiment records

## Method Notes

- Attention backend:
  `use_flash_attn=True/False` in `TransformerDenoiser`
- Flash implementation:
  `torch.nn.functional.scaled_dot_product_attention`
- Sampling:
  DDPM and DDIM with classifier-free guidance
- Compute-matching:
  both variants are trained for identical wall-clock budgets

## Typical Baseline Commands

```bash
# Train one ablation config
python train.py --config depth_4 --dataset fashion_mnist --epochs 30

# Generate samples from a checkpoint
python evaluate.py --ckpt runs/depth_4/latest.pt --sampler ddim --steps 50 --guidance 3.0
```

