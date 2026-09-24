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
# names, plot legends). The original session scope was exactly
# {ani2x, mace-off23-small, uma-s-1p1}; eSEN-conserving/eSEN-direct were
# added later, explicitly requested for Tests 2 and 4 (MD stability /
# condensed-phase water), not used in Test 3.

MODEL_KEYS = (
    "ani2x",
    "mace-off23-small",
    "uma-s-1p1",
    "esen-conserving",
    "esen-direct",
)

# fairchem pretrained_mlip checkpoint names for the eSEN variants. Both are
# the "sm" (small) size class, matching uma-s-1p1's size for consistency
# and CPU/Colab-T4 compute budget -- fairchem also publishes "md" (medium)
# variants (e.g. esen-md-direct-all-omol) but no esen-md-conserving-all-omol
# exists, so "sm" is used for both to keep conserving/direct comparable.
# Checked available via fairchem.core.pretrained_mlip.available_models on
# fairchem-core==2.22.0; re-verify if upgrading.
ESEN_CHECKPOINT_NAMES: dict[str, str] = {
    "esen-conserving": "esen-sm-conserving-all-omol",
    "esen-direct": "esen-sm-direct-all-omol",
}

# Level of theory each model was trained/fitted against. Test 3 does not use
# this yet, but later tests (which compare each model's predictions to its
# OWN training-level reference, not a single universal reference) will.
# eSEN levels: assumed identical to UMA (both are OMol25-generation fairchem
# models trained on the OMol25 dataset at the same reference level) -- not
# independently re-verified against fairchem's own documentation this
# session; flag if this assumption turns out wrong.
REFERENCE_LEVELS: dict[str, str] = {
    "ani2x": "wB97X/6-31G(d)",
    "mace-off23-small": "wB97M-D3(BJ)/def2-TZVPPD",
    "uma-s-1p1": "wB97M-V/def2-TZVPD",
    "esen-conserving": "wB97M-V/def2-TZVPD",
    "esen-direct": "wB97M-V/def2-TZVPD",
}

# UMA's and eSEN's `omol` task requires charge and spin multiplicity on
# every structure (set via atoms.info). ANI-2x and MACE-OFF23 are
# closed-shell neutral-only models with no such input -- they silently
# ignore atoms.info.
MODEL_REQUIRES_CHARGE_SPIN: dict[str, bool] = {
    "ani2x": False,
    "mace-off23-small": False,
    "uma-s-1p1": True,
    "esen-conserving": True,
    "esen-direct": True,
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

# --------------------------------------------------------------------------
# MD root directory -- Colab/Google-Drive-aware
# --------------------------------------------------------------------------
# On Colab, set MLIP_AUDIT_MD_ROOT to a path under a mounted Google Drive
# (e.g. "/content/drive/MyDrive/mlip-audit-md") BEFORE importing this
# module, so Test 2/Test 4 checkpoints and results survive a runtime
# disconnect. Locally, defaults under RESULTS_DIR like everything else.
MD_ROOT = Path(os.environ.get("MLIP_AUDIT_MD_ROOT", RESULTS_DIR / "md"))
MD_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Test 2 (short-MD molecular stability) parameters
# --------------------------------------------------------------------------
# Reference protocol, Ranasinghe et al. 2025 Sec. 2.2.2: the actual
# 349-atom artificial drug-like benchmark molecule (Fig. 2 -- a composite
# of clarithromycin, dexamethasone, diazepam, morphine, penicillin,
# sildenafil, and tryptophan dipeptide fragments), 400 K, Langevin
# friction 1 ps^-1, timestep 1.0 fs, length 0.25 ns (250 ps), trajectory
# saved every 0.5 ps, LBFGS geometry optimization before each ML MD run.
#
# REVISION NOTE: an earlier round of this session substituted 4 small
# molecules (ethanol/THF/phenol/ala-dipeptide, drawn from a DIFFERENT
# paper test -- the SI's "14 simple benchmark systems" quasi-harmonic
# set) for this 349-atom molecule, as a compute-budget simplification.
# That was superseded: the paper deposits the benchmark molecule's EXACT
# SMILES in its SI (Appendix D), extracted and validated this session
# (RDKit parses it to exactly 349 atoms / C130H169ClFN15O31S2, matching
# the paper's stated count and elements) -- see
# scripts/build_drug_like_benchmark_geometry.py for the extraction,
# validation, and 3D-embedding process, and its cached output,
# geometries/drug_like_benchmark_349atoms.xyz. Using the real molecule is
# strictly more defensible than a substitute when the real one is
# available, so it replaces the small-molecule set entirely. See
# RESULTS.md's deviations table for the full account, including this
# supersession.
TEST2_BENCHMARK_GEOMETRY_PATH = REPO_ROOT / "geometries" / "drug_like_benchmark_349atoms.xyz"

TEST2_MOLECULES: dict[str, str] = {
    "drug_like_benchmark": None,  # built from TEST2_BENCHMARK_GEOMETRY_PATH, not from SMILES on the fly
}

TEST2_MODELS = ("uma-s-1p1", "esen-conserving", "esen-direct", "ani2x")
# ANI-2x element-support check (explicitly requested): the paper states
# ANI-1ccx could NOT be used on this molecule (S, F, Cl present) -- but
# ANI-1ccx is a different, older model, not ANI-2x. Verified directly this
# session: torchani.models.ANI2x().symbols == ('H','C','N','O','S','F','Cl'),
# an exact match to this molecule's element set (no missing element), and
# an actual ANI-2x single-point energy/force evaluation on the real
# 349-atom structure succeeded (E=-260224 eV, max|F|=2.2 eV/A, 0.26s on
# CPU). UMA-S was also directly verified the same way (E=-260290 eV,
# 8s on CPU/batch-mode). eSEN-conserving/eSEN-direct were NOT
# independently tested this session (same FAIRChemCalculator
# infrastructure as UMA-S, presumed to behave the same way for element
# support, but this is an inference, not a direct check -- verify on
# first real use).

TEST2_SEEDS = (0,)  # ONE seed per model this round (was 2 for the small-molecule substitute)

TEST2_TEMPERATURE_K = 400.0
TEST2_LANGEVIN_FRICTION_PER_FS = 1.0 / 1000.0  # 1 ps^-1 -> fs^-1 (ASE Langevin takes 1/fs)
TEST2_TIMESTEP_FS = 1.0
# paper: 250 ps. Session: 100 ps -- disclosed compute-budget deviation,
# justified by the paper's own reported instability timings: of the three
# models the paper found unstable on this exact test (MACE RXRX XXXS:
# unstable after 10 ps; MACE RXRX XXS: unstable within 1 ps; published
# MACE-OFF23 S: unstable after 44 ps), ALL manifested within 44 ps -- the
# longest of the three. 100 ps gives 2.3x margin past that single
# empirical precedent. Caveat, disclosed: that precedent is from
# MACE-family reduced-parameter models, not from UMA/eSEN/ANI-2x (which
# postdate the paper and were never run on this molecule before) -- it is
# the best available evidence for how early instabilities in this class of
# test tend to appear, not a guarantee about these specific four models.
TEST2_TOTAL_TIME_PS = 100.0
TEST2_CHECKPOINT_EVERY_PS = 5.0
TEST2_TRAJ_SAVE_EVERY_PS = 0.5  # matches paper
TEST2_CHARGE = 0
TEST2_SPIN = 1  # all 4 molecules are neutral closed-shell

# --------------------------------------------------------------------------
# Test 4 (condensed-phase water) parameters
# --------------------------------------------------------------------------
# Reference protocol, Ranasinghe et al. 2025 Sec. 2.3: 168 TIP3P water
# molecules (box equilibrated classically to 1.706 nm side length), PME
# electrostatics (8 A cutoff), SETTLE-constrained rigid water, a SOLUTE
# optimized before each ML run, 125 ps NVT equilibration, 125 ps (0.125 ns)
# NPT production, 300 K (Nose-Hoover thermostat), 1 bar (Monte Carlo
# barostat), 0.5 fs timestep, trajectory saved every 0.5 ps. See
# RESULTS.md's deviations table -- notably, this session's Test 4 has NO
# solute (pure water box, ML potential describing every atom, testing bulk
# water structure/stability directly) and uses a 1 fs timestep (vs. 0.5 fs)
# and 50 ps NPT production (vs. 125 ps), both compute-budget-driven.
TEST4_N_WATERS = 168
TEST4_MODELS = TEST2_MODELS  # same four models
TEST4_TARGET_DENSITY_G_CM3 = 1.0  # initial packing guess; NPT relaxes it
TEST4_TEMPERATURE_K = 300.0  # matches paper (not specified by user this session; defaulted to paper's value)
TEST4_PRESSURE_BAR = 1.0
TEST4_TIMESTEP_FS = 1.0  # paper: 0.5 fs; session compute budget: 1 fs (disclosed limitation)
TEST4_NVT_TIME_PS = 125.0  # matches paper
TEST4_NPT_TIME_PS = 50.0   # paper: 125 ps; session compute budget: 50 ps (disclosed deviation)
TEST4_CHECKPOINT_EVERY_PS = 5.0
TEST4_TRAJ_SAVE_EVERY_PS = 0.5  # matches paper
TEST4_CHARGE = 0
TEST4_SPIN = 1  # neutral closed-shell water box
# Nose-Hoover-chain damping times (ASE convention, fs); standard literature
# defaults, not specified by the paper (which uses OpenMM's own internal
# Nose-Hoover/MC-barostat implementation, not directly comparable 1:1).
TEST4_TDAMP_FS = 100.0
TEST4_PDAMP_FS = 1000.0
