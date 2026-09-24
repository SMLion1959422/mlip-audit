#!/usr/bin/env bash
set -uo pipefail
cd /c/Users/srika/Documents/mlip-audit
source .venv/Scripts/activate

OUT_DIR=results/test3_dimer_sp1_robustness
mkdir -p "$OUT_DIR"

for model in mace-off23-small ani2x; do
  for fmax in 0.01 0.005; do
    for optimizer in lbfgs fire; do
      for k in 1 10 100; do
        tag="${model}_fmax${fmax}_${optimizer}_k${k}"
        out_csv="$OUT_DIR/${tag}.csv"
        if [ -f "$out_csv" ]; then
          echo "=== SKIP (exists): $tag ==="
          continue
        fi
        echo "=== RUN: $tag ==="
        timeout -k 10 900 python scripts/check5_optimizer_robustness.py \
          --model "$model" --fmax "$fmax" --optimizer "$optimizer" --k-gj "$k" \
          --max-steps 1000 --out "$out_csv" 2>&1 | grep -v "Using\|Cached\|Environment variable\|cuequivariance\|_Jd\|torch.load\|TORCHANI_NO_WARN\|warnings.warn"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
          echo "=== TIMED OUT (900s): $tag -- marking and moving on ==="
          echo "TIMED_OUT" > "$OUT_DIR/${tag}.TIMEDOUT"
        fi
      done
    done
  done
done
echo "=== ALL 24 RUNS COMPLETE ==="
