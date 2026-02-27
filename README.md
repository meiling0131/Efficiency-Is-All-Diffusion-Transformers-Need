# Diffusion Models as Sequence Models: Architectural & Sampling Trade-offs
## (Efficiency-Focused, AAAI Target)

Patch-based Transformer denoiser for class-conditional diffusion.  
Core contribution: **Flash Attention enables Pareto-optimal trade-offs between sample quality and compute efficiency.**

---

## Project Structure

```
diffusion_transformer/
├── model.py              # TransformerDenoiser + FlashSelfAttention (SDPA)
├── diffusion.py          # DDPM + DDIM samplers with CFG
├── train.py              # Training loop + ablation configs
├── evaluate.py           # Sample grids, DDIM/CFG sweeps, training curves
├── fid.py                # FID evaluation (InceptionV3 features)
├── profiler.py           # Throughput / memory / latency benchmarks
├── compute_matched.py    # KEY: FID vs wall-clock (same compute budget)
├── run_all_ablations.sh
├── generate_figures.sh
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Run efficiency benchmarks first (no training needed)
python profiler.py --mode all

# 2. Compute-matched experiment (the main paper figure)
python compute_matched.py --budget_minutes 60 --dataset fashion_mnist

# 3. Full ablation suite
bash run_all_ablations.sh && bash generate_figures.sh
```
---

## Experiment Pipeline

### Step 1: Efficiency Benchmarks
```bash
python profiler.py --mode all --out_dir ./figures
```
Produces:
- `efficiency_throughput_memory.png` — throughput & memory vs batch size
- `efficiency_memory_scaling.png`    — memory vs d_model (Flash O(N) advantage)
- `efficiency_inference.png`         — DDIM latency + speedup bar chart

### Step 2: Compute-Matched Training (Core Figure)
```bash
python compute_matched.py \
    --budget_minutes 60 \
    --fid_interval_minutes 5 \
    --dataset fashion_mnist
```
Produces:
- `fid_vs_walltime.png` — **FID vs Wall-Clock Time** (Flash trains more iters → better FID)
- `pareto_fid_throughput.png` (via `profiler.plot_pareto()`)

### Step 3: Ablation Study
```bash
bash run_all_ablations.sh && bash generate_figures.sh
```

---

## Key Figures for Paper

| Figure | File | What it shows |
|--------|------|---------------|
| **Fig 1** | `fid_vs_walltime.png` | Flash reaches better FID in same wall time |
| **Fig 2** | `pareto_fid_throughput.png` | Pareto frontier across configs |
| **Fig 3** | `efficiency_throughput_memory.png` | Throughput & memory scaling |
| **Fig 4** | `training_curves.png` | Architectural ablations |
| **Fig 5** | `sweep_cfg_scale.png` + `sweep_ddim_steps.png` | Sampling trade-offs |
| **Table 1** | (from `evaluate.py --summary`) | FID + throughput summary |

---

## Flash Attention Implementation

Uses `torch.nn.functional.scaled_dot_product_attention` (PyTorch ≥ 2.0):
- CUDA: dispatches to Flash Attention 2 automatically (O(N) memory)
- CPU/MPS: falls back to math attention (correct, slower)
- No extra packages needed

```python
# In FlashSelfAttention.forward():
attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=...)
```

Toggle via `use_flash_attn=True/False` in `TransformerDenoiser`.

---

## Expected Results (Fashion-MNIST, 60-min budget, A100)

| Config | Batch | Iters | Final FID |
|--------|-------|-------|-----------|
| Flash (d=256) | 512 | ~180k | ~18 |
| Standard (d=256) | 128 | ~45k | ~32 |
| Flash (d=512) | 256 | ~90k | ~15 |
| Standard (d=512) | 64 | ~22k | ~40 |

Flash runs **~4× more iterations** in the same wall-clock time → significantly better FID.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
tqdm
```


---

## Project Structure

```
diffusion_transformer/
├── model.py          # TransformerDenoiser (patch embed + blocks + FiLM/AdaLN)
├── diffusion.py      # GaussianDiffusion: DDPM forward, DDPM/DDIM reverse
├── train.py          # Training loop + all ablation configs
├── evaluate.py       # Sampling, sweeps, figure generation
├── run_all_ablations.sh   # Train all 7 configs in sequence
├── generate_figures.sh    # Produce all paper figures after training
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Train baseline (depth=4, d_model=256, PE, additive cond)
python train.py --config depth_4 --dataset fashion_mnist --epochs 30

# Sample from it
python evaluate.py --ckpt runs/depth_4/latest.pt --sampler ddim --steps 50 --guidance 3.0

# Run ALL ablations (~2-4h on 1 GPU)
bash run_all_ablations.sh

# Generate all figures
bash generate_figures.sh
```

---

## Ablation Configurations

| Config       | Depth | d_model | Pos. Enc. | Cond. Type | Notes                  |
|-------------|-------|---------|-----------|------------|------------------------|
| `depth_2`   | 2     | 256     | ✓         | Add        | Shallow                |
| `depth_4`   | 4     | 256     | ✓         | Add        | **Baseline**           |
| `depth_6`   | 6     | 256     | ✓         | Add        | Deep                   |
| `width_128` | 4     | 128     | ✓         | Add        | Narrow                 |
| `no_pe`     | 4     | 256     | ✗         | Add        | No positional encoding |
| `film`      | 4     | 256     | ✓         | FiLM/AdaLN | Strong conditioning    |
| `cfg_train` | 4     | 256     | ✓         | Add        | CFG dropout=15%        |

---

## Architecture

```
x_t (B,C,H,W)
   │
   ├─ Patchify → (B, N, patch_dim)          # e.g. 7×7=49 patches for 28×28/4
   ├─ Linear patch embed → (B, N, d_model)
   ├─ + 2D Learned PE (optional)
   │
   ├─ Timestep: sinusoidal → MLP → (B, d_model)
   ├─ Class: Embedding → (B, d_model)
   ├─ Combined cond: concat → MLP → (B, d_model)
   │
   ├─ Add cond to tokens
   │
   ├─ L × TransformerBlock(MHA + FFN)
   │     Additive: LayerNorm + residuals (standard)
   │     FiLM:     AdaLN scale/shift from cond per block
   │
   ├─ LayerNorm → Linear → (B, N, patch_dim)
   └─ Unpatchify → ε̂ (B, C, H, W)
```

Training objective: **ε-prediction** MSE loss.  
CFG: 15% class-dropout during training; `class_label = num_classes` = unconditional token.

---

## Sampling

### DDPM (1000 steps)
Standard ancestral sampling. Slow but accurate upper-bound on quality.

### DDIM (10–200 steps)
Deterministic (η=0) or stochastic (η=1). Use `--steps` to control step count.

### Classifier-Free Guidance
```
ε̂ = ε_uncond + w × (ε_cond − ε_uncond)
```
Sweep `w ∈ {0, 1, 2, 3, 5, 7}` with `evaluate.py --sweep cfg_scale`.

---

## Expected Results (Fashion-MNIST, 30 epochs)

**Architectural trade-offs:**

| Config       | Final Loss ↓ | Stability | Visual Quality |
|-------------|-------------|-----------|----------------|
| depth_2     | ~0.045      | Stable    | Blurry details |
| depth_4     | ~0.035      | Stable    | Good baseline  |
| depth_6     | ~0.033      | Stable    | Slightly better|
| width_128   | ~0.042      | Stable    | Slightly worse |
| no_pe       | ~0.038      | Stable    | Spatial artifacts |
| film        | ~0.033      | Stable    | Best class consistency |

**Sampling trade-offs (cfg_train, w=3.0):**

| Sampler     | Steps | Quality  | Time   |
|-------------|-------|----------|--------|
| DDPM        | 1000  | Best     | Slow   |
| DDIM        | 100   | ≈ DDPM   | 10×    |
| DDIM        | 50    | Good     | 20×    |
| DDIM        | 25    | OK       | 40×    |
| DDIM        | 10    | Blurry   | 100×   |

**CFG guidance (DDIM 50 steps):**

| w   | Quality | Diversity |
|-----|---------|-----------|
| 0   | Low     | High      |
| 1   | Medium  | High      |
| 3   | Good    | Medium    |
| 5   | High    | Low       |
| 7   | Oversat.| Very Low  |

Sweet spot: **w ≈ 3.0**

---

## Extending to CIFAR-10

```bash
python train.py --config depth_6 --dataset cifar10 --epochs 50
```
Also increase `patch_size=4` (auto-detected for 32×32 images → 8×8=64 patches).

---

## Paper Proposal

> We implement a patch-based Transformer denoiser for class-conditional diffusion and
> systematically study architectural choices (depth, positional encoding, conditioning
> injection) and sampling strategies (DDPM vs DDIM, classifier-free guidance),
> characterizing trade-offs among training stability, sample fidelity, and sampling efficiency.
