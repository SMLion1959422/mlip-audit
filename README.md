# mlip-audit

Physical-reliability audit of OMol25-generation machine-learning interatomic
potentials (Meta's UMA and eSEN family, plus ANI-2x and MACE-OFF23 as
comparison points), following the four-test protocol of Ranasinghe et al.
2025 (JCIM 65(17), 8980-8999; preprint arXiv:2503.11537).

**Scope:** repo scaffold, model-loading harness, **Test 3** (water dimer
O-O potential energy scan -- complete for all 5 models, see `RESULTS.md`),
and **Tests 2 & 4** (short-MD molecular stability; condensed-phase water --
infrastructure built and locally smoke-tested, full-scale runs pending on
Colab, see `RESULTS.md`'s "Test 2 and Test 4" section and
`notebooks/02_md_tests.ipynb`). Test 1 is not implemented.

## Models

| Key | Loaded via | Trained/reference level of theory | Used in |
|---|---|---|---|
| `ani2x` | `torchani.models.ANI2x().ase()` | wB97X/6-31G(d) | Tests 2, 3, 4 |
| `mace-off23-small` | `mace.calculators.mace_off(model="small")` | wB97M-D3(BJ)/def2-TZVPPD | Test 3 |
| `uma-s-1p1` | `fairchem.core.pretrained_mlip` + `FAIRChemCalculator` | wB97M-V/def2-TZVPD | Tests 2, 3, 4 |
| `esen-conserving` | fairchem, checkpoint `esen-sm-conserving-all-omol` | wB97M-V/def2-TZVPD (assumed, same as UMA) | Tests 2, 4 |
| `esen-direct` | fairchem, checkpoint `esen-sm-direct-all-omol` | wB97M-V/def2-TZVPD (assumed, same as UMA) | Tests 2, 4 |

All five are wrapped behind one factory, `mlip_audit.models.get_calc(name)`,
so test code never branches on model identity. Reference levels are recorded
in `mlip_audit/config.py` for later tests that compare each model against
its *own* training level (Test 3 doesn't use this yet).

## Important: environments (three stacks, not one)

`mace-torch` (every release up to and including the current 0.3.10) hard-pins
`e3nn==0.4.4`. `fairchem-core` (required for UMA-S/eSEN) requires `e3nn>=0.5`.
These conflict for real -- `pip install mace-torch fairchem-core` together
is `ResolutionImpossible`, confirmed against current PyPI releases of both,
not a range we guessed at. There is no single environment that can run
MACE-OFF23-small alongside UMA-S/eSEN.

**Consequence for Test 3** (which uses MACE-OFF23-small): run ANI-2x +
MACE-OFF23-small in one environment/Colab session, and UMA-S in a separate
one. `setup.sh` takes a stack argument to make this explicit:

```bash
bash setup.sh ani-mace   # installs requirements-ani-mace.txt (Test 3: ANI-2x, MACE-OFF23-small)
bash setup.sh uma        # installs requirements-uma.txt (Test 3: UMA-S, separate env/runtime)
```

**Tests 2 and 4** don't use MACE-OFF23-small at all (their 4 models are
UMA-S, eSEN-conserving, eSEN-direct, ANI-2x), so the conflict above never
arises for them -- `torchani` and `fairchem-core` install together into
ONE environment with no issue (verified). Use the third stack:

```bash
bash setup.sh md         # installs requirements-md.txt (Tests 2 & 4: all 4 models, one env)
```

## Setup

### Colab (recommended for actually running models)

```
!git clone <this-repo-url> /content/mlip-audit
%cd /content/mlip-audit
!bash setup.sh ani-mace   # or: !bash setup.sh uma
```

Run ANI-2x/MACE-OFF23 scans in one Colab runtime, then **Runtime > Restart
session** (or open a second Colab session) before running `bash setup.sh
uma` and the UMA-S scan -- installing the `uma` stack on top of an
`ani-mace` environment (or vice versa) will break one of them via the e3nn
conflict above.

`setup.sh` leaves torch alone on Colab (it's preinstalled with a
CUDA-matched build) and installs everything else from the chosen stack's
requirements file, plus an editable install of this package.

**HuggingFace login (required for UMA-S only):** UMA's checkpoint is a
gated model. Log in *interactively* so your token is never captured in a
notebook cell, log file, or shell history:

```
hf auth login
```

(or, inside a notebook cell: `from huggingface_hub import login; login()`,
which opens a token-entry widget). Then accept the gated model's license at
https://huggingface.co on the account whose token you just cached.

This codebase reads the token **only** via `huggingface_hub.get_token()`
(the cache `hf auth login` writes to, normally `~/.cache/huggingface/token`).
It never hardcodes, prints, logs, or otherwise touches the token value
itself -- see `mlip_audit/models.py::_check_hf_login`.

`notebooks/01_dimer_scan.ipynb` is a ready-to-run Colab driver that does all
of the above plus the scan, plotting, and a results download step (Colab
sessions are ephemeral -- download your CSVs before the runtime recycles).

### Local development (no GPU required for code/scaffold work)

```bash
python -m venv .venv-ani-mace
source .venv-ani-mace/bin/activate   # or .venv-ani-mace\Scripts\activate on Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
bash setup.sh ani-mace
```

For UMA-S, repeat in a **separate** venv (`.venv-uma`) with `bash setup.sh
uma` -- see "Important: two separate environments" above.

CPU is fine for ANI-2x and MACE-OFF23-small (both are small models). UMA-S
will run on CPU too but is slow; a T4 or better is strongly recommended for
a full 69-point scan.

## Running Test 3

```bash
python -m mlip_audit.test3_dimer --model mace-off23-small
python -m mlip_audit.test3_dimer --model ani2x
python -m mlip_audit.test3_dimer --model uma-s-1p1
```

Each run is **resumable**: results are flushed to
`results/test3_dimer/<model>.csv` after every point, and the relaxed
geometry is checkpointed to `results/checkpoints/<model>_dimer_scan.extxyz`.
Re-running the same command after a crash/disconnect skips already-completed
distances and warm-starts from the last checkpointed geometry. Use
`--no-resume` to force a clean restart (this deletes that model's existing
CSV/checkpoint first).

### Scan direction (a deliberate implementation choice)

The spec describes the scan as covering 0.2-7.0 A. We **compute points in
descending order (7.0 -> 0.2 A)**, using each step's relaxed geometry as the
warm start for the next (closer) step, rather than building an independent
starting guess at every distance. Two reasons:

1. An independent starting geometry at, say, 0.3 A is physically
   meaningless -- there's no principled way to place two overlapping water
   molecules from scratch at that separation.
2. This is the standard way to run a 1-D adiabatic PES scan, and it is the
   procedure that actually tests for the failure mode Test 3 is looking
   for: whether, as the dimer is pushed together from the physical region,
   the optimizer relaxes into a spurious low-energy clash instead of the
   energy rising monotonically.

The CSV rows are written in this descending order; `mlip_audit.plotting`
sorts by distance before plotting/analysis, so this is transparent
downstream.

## Plotting and the acceptance check

```bash
python -m mlip_audit.plotting --csv results/test3_dimer/*.csv
```

Saves two plots -- `results/test3_dimer/dimer_scan.png` (log-scale
energy vs. distance) and `results/test3_dimer/force_convergence.png`
(LBFGS convergence) -- and prints, per model, a **quantitative** depth
metric (eV and kcal/mol) for how much deeper the global minimum is than
the physical ~2.9 A one.

**Read this before trusting any depth number**: a deep energy at short
O-O distance only means something if the underlying relaxed structure is
still a chemically intact water dimer. This project's own diagnostic work
found that a naive "just take the minimum energy" reading gets fooled by
relaxations that collapse or dissociate an O-H bond -- so every depth is
reported TWO ways, "raw" (unfiltered) and "valid" (restricted to
`converged=True AND geometry_valid=True` points, per
`mlip_audit.geometry.check_dimer_geometry`). **Always check both, and
prefer the "valid" number.** See `RESULTS.md` for this session's actual
numbers and the full diagnostic trail -- the acceptance criterion below
was NOT met once this filtering was applied, which is a real, current
finding, not a placeholder.

**Acceptance criterion for this session:** `mace-off23-small` must show
exactly this failure -- a spurious minimum at short O-O separation deeper
than the physical one near 2.9 A, in a structure that is still a genuine
water dimer. This is a known, published result. If our pipeline does not
reproduce it, that is treated as a bug (in the pipeline, the methodology,
or our understanding) to be debugged, not a finding to report at face
value -- see `RESULTS.md` for where that debugging currently stands.

## Charge/spin sanity check (UMA)

UMA's `omol` task silently ignores charge/spin if the config doesn't reach
it -- it does not error out. `tests/test_charge_spin.py` catches this by
scoring acetate (CH3COO-) at a fixed geometry with `charge=-1` (correct) vs
`charge=0` (wrong) and asserting the energies differ:

```bash
python -m pytest tests/test_charge_spin.py -v
```

This test is skipped (not failed) if UMA can't be loaded in the current
environment (no GPU, no HF login, no network) -- that's an environment
gap, not a code failure.

## Running Tests 2 & 4

**Read `RESULTS.md`'s "Test 2 and Test 4" section first** -- it has the
full deviations table (shortened production lengths, no solute in Test 4,
`inference_settings="turbo"`'s untested-outside-Colab status, etc.) and
current status (infrastructure built + locally smoke-tested; no full-scale
run yet). Use the `md` environment stack (see above), then:

```bash
# Test 2: short-MD stability, 4 molecules x 4 models x 2 seeds x 100 ps.
python -m mlip_audit.test2_md_stability --verbose
# subset: --molecule ethanol --model uma-s-1p1 --seed 0

# Test 4: condensed-phase water, 4 models x (125 ps NVT + 50 ps NPT).
python -m mlip_audit.test4_condensed_water --verbose
# subset: --model uma-s-1p1 --phase nvt
```

Both are resumable the same way Test 3 is (re-running the same command
picks up from the last checkpoint) -- see `notebooks/02_md_tests.ipynb`
for the full Colab driver, including mounting Google Drive for
`MLIP_AUDIT_MD_ROOT` so checkpoints survive a runtime disconnect.

## Repo layout

```
mlip_audit/
  config.py             reference DFT levels, paths, all tests' constants
  models.py             get_calc() factory (5 models) + charge/spin wiring
  geometry.py            water dimer builder + O-O distance utilities (Test 3)
  restraints.py           harmonic distance restraint (Test 3)
  test3_dimer.py          the resumable O-O scan -> CSV
  plotting.py             Test 3 log-scale plot + spurious-minimum check
  molecules.py             Test 2/4 molecule + water-box builders
  md_common.py             resumable MD driver (checkpointing, Test 2/4)
  md_analysis.py           bond-length stability (Test 2) + O-O RDF (Test 4)
  test2_md_stability.py    Test 2 driver
  test4_condensed_water.py Test 4 driver
tests/
  test_charge_spin.py
notebooks/
  01_dimer_scan.ipynb   Test 3 Colab driver
  02_md_tests.ipynb     Tests 2 & 4 Colab driver
scripts/                 one-off diagnostic/investigation scripts (see RESULTS.md)
results/                  gitignored: CSVs, checkpoints, plots land here
```

## Known limitations of this session's build

- No CI/automated run of Test 3 with real GPU models is included in this
  repo -- results depend on actually executing the notebook or CLI in an
  environment with the model weights and (for UMA) HF access.
- The water dimer starting geometry (`geometry.build_water_dimer`) is an
  idealized, non-optimized guess for the *first* point of the scan only;
  every subsequent point warm-starts from the previous relaxation (see
  "Scan direction" above), so the initial guess's exact orientation is not
  expected to matter for the converged results.
