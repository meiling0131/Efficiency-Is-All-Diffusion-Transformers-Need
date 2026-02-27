#!/usr/bin/env bash
set -euo pipefail

# One-shot efficiency pipeline for the paper's core figures:
# 1) Throughput/Memory/Latency benchmarks
# 2) Compute-matched training (same wall-clock budget)
#
# Usage:
#   bash run_efficiency_suite.sh
#   bash run_efficiency_suite.sh --budget_minutes 60 --dataset fashion_mnist

BUDGET_MINUTES=60
FID_INTERVAL_MINUTES=5
FID_N_SAMPLES=2000
DATASET="fashion_mnist"
DMODEL=256
DEPTH=4
RUN_ROOT="./runs/efficiency_suite"
FIG_DIR="./figures"
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget_minutes) BUDGET_MINUTES="$2"; shift 2 ;;
    --fid_interval_minutes) FID_INTERVAL_MINUTES="$2"; shift 2 ;;
    --fid_n_samples) FID_N_SAMPLES="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --d_model) DMODEL="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;
    --run_root) RUN_ROOT="$2"; shift 2 ;;
    --fig_dir) FIG_DIR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_efficiency_suite.sh [options]"
      echo "  --budget_minutes INT"
      echo "  --fid_interval_minutes FLOAT"
      echo "  --fid_n_samples INT"
      echo "  --dataset {mnist,fashion_mnist,cifar10}"
      echo "  --d_model INT"
      echo "  --depth INT"
      echo "  --run_root PATH"
      echo "  --fig_dir PATH"
      echo "  --seed INT"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$RUN_ROOT" "$FIG_DIR"

echo "[1/2] Running efficiency profiler..."
python profiler.py \
  --mode all \
  --d_model "$DMODEL" \
  --depth "$DEPTH" \
  --out_dir "$FIG_DIR" \
  --save_json "$RUN_ROOT/efficiency_results.json"

echo "[2/2] Running compute-matched experiment..."
python compute_matched.py \
  --budget_minutes "$BUDGET_MINUTES" \
  --fid_interval_minutes "$FID_INTERVAL_MINUTES" \
  --fid_n_samples "$FID_N_SAMPLES" \
  --dataset "$DATASET" \
  --d_model "$DMODEL" \
  --depth "$DEPTH" \
  --out_dir "$RUN_ROOT/compute_matched" \
  --seed "$SEED"

echo
echo "Done."
echo "Key outputs:"
echo "  - ${FIG_DIR}/efficiency_throughput_memory.png"
echo "  - ${FIG_DIR}/efficiency_memory_scaling.png"
echo "  - ${FIG_DIR}/efficiency_inference.png"
echo "  - ${FIG_DIR}/fid_vs_walltime.png"
echo "  - ${RUN_ROOT}/compute_matched/compute_matched_results.json"
