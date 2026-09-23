"""Central configuration: model registry metadata, reference DFT levels, and paths.

Nothing in this file is hardware- or Colab-specific. All paths are resolved
relative to the repository root (the parent of this package) unless
overridden by environment variables, so the same config works locally and
on Colab.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Repo root = parent of the mlip_audit/ package directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = Path(os.environ.get("MLIP_AUDIT_RESULTS_DIR", REPO_ROOT / "results"))
CHECKPOINT_DIR = Path(
    os.environ.get("MLIP_AUDIT_CHECKPOINT_DIR", RESULTS_DIR / "checkpoints")
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
# Canonical model keys used everywhere in this codebase (CLI args, CSV file
# names, plot legends). Do not add models beyond these three without
# explicit instruction -- the audit protocol for this session is scoped to
# exactly these.

MODEL_KEYS = ("ani2x", "mace-off23-small", "uma-s-1p1")

# Level of theory each model was trained/fitted against. Test 3 does not use
# this yet, but later tests (which compare each model's predictions to its
# OWN training-level reference, not a single universal reference) will.
REFERENCE_LEVELS: dict[str, str] = {
    "ani2x": "wB97X/6-31G(d)",
    "mace-off23-small": "wB97M-D3(BJ)/def2-TZVPPD",
    "uma-s-1p1": "wB97M-V/def2-TZVPD",
}

# UMA's `omol` task requires charge and spin multiplicity on every structure
# (set via atoms.info). ANI-2x and MACE-OFF23 are closed-shell neutral-only
# models with no such input -- they silently ignore atoms.info.
MODEL_REQUIRES_CHARGE_SPIN: dict[str, bool] = {
    "ani2x": False,
    "mace-off23-small": False,
    "uma-s-1p1": True,
}

# --------------------------------------------------------------------------
# Test 3 (water dimer O-O scan) parameters
# --------------------------------------------------------------------------

DIMER_SCAN_START_ANG = 7.0   # start of scan (well-separated, physical region)
DIMER_SCAN_END_ANG = 0.2     # end of scan (deep atomic clash)
DIMER_SCAN_STEP_ANG = 0.1
DIMER_SCAN_DECIMALS = 4      # rounding used for CSV keys / resume matching

LBFGS_FMAX = 0.05            # eV / Angstrom
LBFGS_MAX_STEPS = 200

# Multi-start restarts per scan point: the warm-started continuation from
# the previous point, plus (N-1) randomly-reoriented starting geometries
# (see geometry.random_perturbed_water_dimer). This exists because the
# short-O-O-distance region of some models' restrained-relaxation landscape
# is multi-basin/rugged (empirically confirmed for MACE-OFF23-small during
# this project -- a single warm-started LBFGS chain can land in a basin
# several thousand eV shallower than the deepest one reachable from a
# different starting orientation). A single local optimizer chain is not a
# reliable way to test "does a spurious minimum exist here"; N>=2 restarts
# are needed for that question to be answered honestly. Default N=5 trades
# off runtime against coverage of that ruggedness.
DIMER_N_RESTARTS = 5
DIMER_RANDOM_SEED = 0

# Neutral singlet water dimer.
DIMER_CHARGE = 0
DIMER_SPIN = 1  # multiplicity (2S+1), i.e. singlet
