#!/usr/bin/env bash
# Setup for mlip-audit, intended to work unmodified on a fresh Colab runtime
# or a fresh local machine with a POSIX shell (Linux/macOS/Git-Bash-on-Windows).
#
# Usage:
#   bash setup.sh ani-mace     # installs ANI-2x + MACE-OFF23-small stack (Test 3 only)
#   bash setup.sh uma          # installs UMA-S stack (Test 3 only)
#   bash setup.sh md           # installs UMA-S + eSEN + ANI-2x + RDKit (Tests 2 and 4)
#
# Why "ani-mace"/"uma" are separate (Test 3 only): mace-torch hard-pins
# e3nn==0.4.4; fairchem-core (needed for UMA-S) requires e3nn>=0.5. These
# conflict for real -- not just a slow pip resolution -- across all current
# releases of both packages. You cannot pip-install mace-torch and
# fairchem-core into the same environment. See requirements.txt.
#
# Why "md" is its own (single, unified) stack: Tests 2 and 4 use UMA-S,
# eSEN-conserving, eSEN-direct, and ANI-2x -- NOT MACE-OFF23, so the
# mace-torch/fairchem-core conflict above never arises. torchani (ANI-2x)
# and fairchem-core (UMA/eSEN) do NOT conflict with each other (verified by
# installing both together) -- so unlike Test 3, Tests 2/4 need only ONE
# environment for all four of their models.
#
# What this does NOT do: it does not touch HuggingFace credentials. Run
# `hf auth login` yourself (interactively, so the token isn't captured in
# any log) before running anything that touches UMA-S or eSEN. See README.md.

set -euo pipefail

STACK="${1:-}"
if [[ "$STACK" != "ani-mace" && "$STACK" != "uma" && "$STACK" != "md" ]]; then
    echo "Usage: bash setup.sh {ani-mace|uma|md}"
    echo ""
    echo "  ani-mace  -- ANI-2x + MACE-OFF23-small, Test 3 only (requirements-ani-mace.txt)"
    echo "  uma       -- UMA-S, Test 3 only (requirements-uma.txt)"
    echo "  md        -- UMA-S + eSEN-conserving + eSEN-direct + ANI-2x, Tests 2 and 4 (requirements-md.txt)"
    echo ""
    echo "ani-mace/uma are separate because mace-torch and fairchem-core have a"
    echo "genuine, unresolvable e3nn version conflict. md doesn't use MACE, so it"
    echo "doesn't hit that conflict and is a single unified stack. See requirements.txt."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "== mlip-audit setup (stack: $STACK) =="

# --- torch -------------------------------------------------------------
# On Colab, torch + CUDA are already installed and matched to the runtime.
# Do not reinstall it there. Locally (no CUDA), install a CPU build if
# torch is missing.
if python -c "import torch" 2>/dev/null; then
    echo "torch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "torch not found -- installing CPU build (local dev machine assumed)."
    echo "If this is Colab and you see this message, something is wrong with the runtime."
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# --- stack-specific requirements ----------------------------------------
pip install -r "requirements-${STACK}.txt"

# --- editable install of this package so `import mlip_audit` works -----
pip install -e .

echo ""
if [[ "$STACK" == "uma" || "$STACK" == "md" ]]; then
    echo "== Checking HuggingFace login (required for UMA-S / eSEN) =="
    if python -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)" 2>/dev/null; then
        echo "HuggingFace token found via huggingface_hub cache. OK."
    else
        echo "No cached HuggingFace token found."
        echo "Run:  hf auth login"
        echo "(then accept the gated UMA/eSEN model licenses at huggingface.co)"
    fi
fi

echo ""
echo "== Checking CUDA visibility =="
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "Setup complete ($STACK stack). Try:"
if [[ "$STACK" == "uma" ]]; then
    echo "  python -m pytest tests/test_charge_spin.py -v"
    echo "  python -m mlip_audit.test3_dimer --model uma-s-1p1"
elif [[ "$STACK" == "md" ]]; then
    echo "  python -m mlip_audit.test2_md_stability --molecule drug_like_benchmark --model ani2x --seed 0"
    echo "  python -m mlip_audit.test4_condensed_water --model ani2x --phase nvt"
else
    echo "  python -m mlip_audit.test3_dimer --model mace-off23-small"
    echo "  python -m mlip_audit.test3_dimer --model ani2x"
fi
