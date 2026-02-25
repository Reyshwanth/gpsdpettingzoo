#!/bin/bash
# Sequential sweep: Transformer (3 runs) then MLP (9 runs)
# Total: 12 runs

set -e  # Stop on first error

echo "=============================================="
echo "  Starting sweep: $(date)"
echo "  Transformer: 3 runs | MLP: 9 runs | Total: 12"
echo "=============================================="

# ── Transformer runs (clip_coef sweep) ──
for clip in 0.05 0.1 0.15; do
    echo ""
    echo ">>> [Transformer] clip_coef=$clip"
    python3 train_gpsd_mappo_transformer.py --clip-coef $clip
done

# ── MLP runs (clip_coef × local_ratio sweep) ──
for clip in 0.05 0.1 0.15; do
    for lr in 0.990 0.995 0.999; do
        echo ""
        echo ">>> [MLP] clip_coef=$clip  local_ratio=$lr"
        python3 train_gpsd_mappo.py --clip-coef $clip --local-ratio $lr
    done
done

echo ""
echo "=============================================="
echo "  All 12 runs complete: $(date)"
echo "=============================================="
