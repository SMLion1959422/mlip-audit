# Test 3 results (water dimer O-O potential energy scan)

Session date: 2026-09-22/23. Scan performed on CPU (local dev machine, no
GPU available this session); Colab GPU not used yet.

**This document supersedes an earlier version of itself that declared the
acceptance criterion "MET" for MACE-OFF23-small.** That conclusion was
premature -- it was built on energies from relaxed structures that turned
out, on inspection, to have collapsed or dissociated O-H bonds (not an
intact water dimer). Three requested diagnostic checks (geometry, LBFGS
convergence, constraint-mechanism sensitivity) were run before writing this
version; see "Diagnostic checks" below for the full trail. The corrected,
current status is in Section 1.

## 1. Acceptance criterion (as set for this session)

> MACE-OFF23-small must reproduce its published failure: a spurious energy
> minimum at short O-O separation that is DEEPER than the physical minimum
> near 2.9 A. This is a known, published result (Ranasinghe et al. 2025,
> JCIM 65(17), 8980-8999; arXiv:2503.11537). If we cannot reproduce it, the
> pipeline is wrong and we stop and debug rather than proceeding.

**Status: NOT MET, under rigorous (geometry- and convergence-validated)
analysis.** Across four independent methodological variants tried in this
session (multi-start + restraint, single-chain + restraint, single-chain +
hard constraint, all starting from independently-verified clean
geometries), MACE-OFF23-small's energy rises smoothly and monotonically as
O-O shrinks for as long as the relaxed structure remains a chemically
intact water dimer. Below a threshold separation, EVERY relaxation
attempt -- regardless of method -- converges to a structure with either a
collapsed or a dissociated O-H bond. No run, under any method tried,
produced a **geometry-valid** structure with energy below the physical
~2.9 A minimum. This is stated per the session's own explicit instruction
("if we cannot reproduce it... we stop and debug rather than proceeding")
-- this is that stop. See "What would still need to happen" at the end of
this document for what remains untried.

ANI-2x, run in parallel as a comparison point, shows the same qualitative
picture (see Section 3): smooth monotonic rise while geometry-valid, then
total breakdown of intact-dimer geometry below a threshold -- no
geometry-valid point deeper than physical.

## 2. Quantitative depth metric (replaces the old boolean)

For both models, `mlip_audit.plotting.check_spurious_minimum` now reports
depth in eV and kcal/mol two ways:
- **raw**: minimum energy over ALL scanned points, no filtering (this is
  what a naive "just read off the minimum" analysis would report, and is
  probably close to what the original session's premature "MET" claim was
  built on).
- **valid**: minimum energy restricted to points that are BOTH
  `converged=True` AND `geometry_valid=True` (no collapsed or dissociated
  O-H -- see `mlip_audit.geometry.check_dimer_geometry`).

| Model | Physical min (2.9ish A) | RAW global min | RAW depth | VALID global min | VALID depth |
|---|---|---|---|---|---|
| mace-off23-small | -4162.4494 eV @ 2.9 A | -4165.8032 eV @ 0.6 A | 3.35 eV = **77.3 kcal/mol** | -4162.4494 eV @ 2.9 A (i.e. no deeper valid point) | **0.00 eV = 0.00 kcal/mol** |
| ani2x | -4157.6191 eV @ 2.7 A | -4258.9194 eV @ 0.6 A | 101.30 eV = **2336.0 kcal/mol** | -4157.6191 eV @ 2.7 A (i.e. no deeper valid point) | **0.00 eV = 0.00 kcal/mol** |

These are NOT the same finding, as requested -- and now that both are
computed the same rigorous way, neither is actually a confirmed spurious
minimum. The RAW numbers are large and clearly different in magnitude
between the two models (77 vs. 2336 kcal/mol), but both RAW numbers come
from geometry-invalid points, so neither should be cited as "the depth of
the spurious minimum" without the caveat that it is not evidence about the
intact-dimer PES.

64/69 points are `geometry_valid=True` for BOTH models (see per-point data
in `results/test3_dimer/{model}.csv`, columns `min_oh_ang`, `max_oh_ang`,
`n_proton_transfer_flags`, `geometry_valid`).

- MACE-OFF23-small invalid points (contiguous): O-O target = 0.2, 0.3, 0.4,
  0.5, 0.6 A. Valid everywhere from 0.7 A up to 7.0 A.
- ANI-2x invalid points (NOT contiguous): O-O target = 0.3, 0.6, 0.7, 0.8,
  0.9 A -- but 0.2, 0.4, 0.5 A are valid. This non-contiguous pattern is
  itself a signal of a genuinely rugged/multi-basin breakdown region for
  ANI-2x, not a clean single threshold.

## 3. Diagnostic checks (run before writing any conclusion, as requested)

### Check 1: geometry dump at 0.5/0.6/0.7/0.8/1.0/1.5/2.9 A

Script: `scripts/check1_geometry.py` (dumps all O-H distances and H-O-H
angles from the checkpoint trajectories; its own pass/fail threshold
predates the canonical dual-threshold check and should not be trusted over
the CSV's `geometry_valid` column, but its raw distance dumps are the
underlying evidence). Representative findings from the FIRST (now
retracted) multi-start run, which is what motivated building the canonical
check in the first place:
- MACE at target=0.7 A: O_A-H(bridge) = 0.62 A (collapsed; real O-H is
  ~0.96-0.98 A), O_B-H2 = 0.62 A (collapsed). This was the geometry behind
  the originally-reported -8792 eV point.
- ANI-2x at target=0.8/1.0 A: O_A-H(bridge) = 6.65/7.20 A -- a hydrogen
  displaced several Angstrom from its own oxygen. Not "a dimer with a
  short O-O distance" by any reading.
- At target=1.5 A and 2.9 A, BOTH models show fully normal geometry (O-H
  ~0.96-0.98 A, H-O-H ~97-117 deg, no proton-transfer flags) -- confirming
  the normal/physical region of the scan is trustworthy as-is.

### Check 2: LBFGS convergence (max force vs. distance)

Plot: `results/test3_dimer/force_convergence.png`. Non-converged points
are marked with an X, not silently included.

This check found a real bug, now fixed: the originally-reported MACE
-8792 eV point (O-O target 0.7 A) was labeled `converged=True` by the
pipeline, but recomputing forces at the exact same saved geometry gave a
max force of 2.22 eV/A -- nowhere near the fmax=0.05 target. ASE's LBFGS
had accepted one more position update after its internal convergence
check, landing in a region of this rugged PES with a completely different
(large) force -- a real discrepancy between "the state LBFGS checked" and
"the state that got saved," not a reporting error on our part after the
fact. Fixed in `mlip_audit/test3_dimer.py::_relax_candidate`: the max
force is now read immediately after `opt.run()` returns, before the
calculator is swapped to compute the reported energy, so this class of bug
cannot recur silently. After the fix, only ONE point in the entire
re-run (MACE, O-O target 0.5 A) is genuinely non-converged, and it is
correctly labeled as such.

### Check 3: method sensitivity (restraint vs. hard FixBondLength)

Script: `scripts/check3_method_sensitivity.py`. Both models, starting from
an independently-verified clean 1.5 A geometry, single warm-started chain
(NO multi-start, to isolate the constraint-mechanism variable), swept
1.4 A down to 0.2 A under (a) a hard `FixBondLengths` constraint
(tolerance loosened to 1e-6 to avoid the numerical failure documented in
the original debugging trail) and (b) the harmonic restraint. Full data:
`results/test3_dimer/{model}_method_sensitivity.csv`.

Result: the two methods agree closely everywhere both produce a
geometry-valid point (energies typically match to within ~0.1-0.3 eV at
the same target distance), and BOTH methods show the same qualitative
picture as the main scan: smooth monotonic rise while valid, then total
geometry breakdown. One specific test of the original hypothesis ("if the
ANI-2x artifact disappears under restraints but MACE's persists, that
explains the discrepancy") is informative here: under the hard constraint,
MACE showed an energy of -4322.47 eV at O-O=0.6 A -- deeper than physical,
and it would have been a clean confirmation of the acceptance criterion,
EXCEPT its geometry has max_OH = 2.64 A (a dissociated O-H), so it fails
the same validity gate. The hypothesis as stated is not what explains the
discrepancy: the discrepancy dissolved once geometry validity was checked
at all, for both models, under both methods.

## 4. Pipeline changes made in response to this diagnostic

- `mlip_audit/geometry.py`: added `check_dimer_geometry()` /
  `DimerGeometryReport`, the canonical structural-sanity check. Flags a
  point invalid if ANY of the four O-H distances is either collapsed
  (`< OH_COLLAPSE_THRESHOLD_ANG = 0.85`) or dissociated
  (`> OH_STRETCH_WARN_ANG = 1.3`). Both thresholds are judgment calls
  informed by the real O-H equilibrium bond length (~0.96-0.98 A); they
  are not derived from anything more rigorous than "clearly not a bonded
  O-H distance in either direction," and a different reasonable person
  could draw them slightly differently. The dissociation half of this
  check was added only after finding that the collapse-only version
  missed points where EVERY O-H distance was 3-11 A (fully fragmented,
  not merely "stretched").
- `mlip_audit/test3_dimer.py`: the multi-start candidate-selection logic
  now prefers a converged, geometry-valid candidate over one that is
  merely lower-energy -- previously, an invalid-geometry candidate's
  (meaningless) energy could win the multi-start comparison outright. New
  CSV columns: `final_max_force_eV_per_ang` (captured live, see Check 2),
  `min_oh_ang`, `max_oh_ang`, `n_proton_transfer_flags`, `geometry_valid`.
- `mlip_audit/plotting.py`: `check_spurious_minimum` now returns
  quantitative `raw_depth_eV` / `raw_depth_kcal_mol` /
  `valid_depth_eV` / `valid_depth_kcal_mol` instead of a boolean (see
  Section 2). Added `plot_force_convergence()`. `plot_dimer_scan()` now
  marks geometry-invalid points with hollow markers.
- Both official scans (MACE-OFF23-small, ANI-2x) were re-run end to end
  with this corrected pipeline; the numbers in this document are from
  those re-runs, not the original (retracted) ones.

## 5. Exact package versions (pip freeze, key packages)

Two separate environments were used (see `requirements.txt` for why:
`mace-torch` hard-pins `e3nn==0.4.4`, incompatible with `fairchem-core`'s
`e3nn>=0.5` requirement).

`.venv` (ani-mace stack; used for both results above):
```
ase==3.29.0
torch==2.13.0
torchani==2.9.0
mace-torch==0.3.16
e3nn==0.4.4
numpy==2.4.6
pandas==3.0.6
matplotlib==3.11.2
scipy==1.17.1
pytest==9.1.1
huggingface_hub==1.32.0
```

`.venv-uma` (UMA stack; installed this session, not yet used for a scan --
see "What is NOT yet built"):
```
ase==3.29.0
torch==2.13.0
fairchem-core==2.22.0
e3nn==0.6.0
huggingface_hub==1.32.0
numpy==2.4.6
```

Both venvs were created fresh with Python 3.11.9
(`C:\Users\srika\AppData\Local\Programs\Python\Python311\python.exe`) on
this Windows machine, torch installed from the CPU-only wheel index
(`https://download.pytorch.org/whl/cpu`) since there is no local GPU.

## 6. What is NOT yet built / run

- **UMA-S: not run.** `.venv-uma` is created and `requirements-uma.txt`
  installed successfully (confirmed: `fairchem-core==2.22.0` imports). A
  cached HuggingFace token exists. Neither the real
  `test_charge_reaches_model` test nor a UMA-S Test 3 scan have been
  executed. When this happens, it must go through the SAME
  geometry-validity + convergence gating as MACE/ANI-2x, not the original
  (retracted) raw-minimum approach.
- **Tests 1, 2, 4: not built at all**, per this session's explicit scope.
- **Colab: not exercised.** Everything above ran on local CPU.
- **`results/checkpoints/*.extxyz`** exist locally but are gitignored
  (regenerable) -- not part of this commit.

## What would still need to happen to give this a final verdict

The acceptance criterion is NOT confirmed, but it is also not cleanly
falsified -- "we could not find a valid example with the methods we tried"
is not the same as "no valid example exists." Concretely untried:
1. **The literal starting geometry.** Ranasinghe et al. start from "the
   Smith stationary point 1," then a real wB97X/6-31G(d) optimization.
   This session used a hand-built idealized Cs-symmetric guess instead
   (`geometry.build_water_dimer`), reasoning that the optimizer should
   wash out the difference -- but the whole finding of this diagnostic
   round is that the short-range landscape is rugged/history-dependent
   enough for the exact path to matter a great deal. Building the actual
   literature starting geometry (or at least a DFT-optimized one) and
   re-running is the single most direct way to close this gap.
2. **Finer sampling right at the breakdown boundary.** This scan steps in
   0.1 A increments (matching the paper's stated 0.01 nm). A valid,
   deeper minimum occupying a window narrower than 0.1 A between two
   sampled points would be invisible here. Given how sharply behavior
   changed between adjacent 0.1 A points in this data, this is plausible.
3. **The paper's own geometries.** Unavailable to this session. If
   Ranasinghe et al.'s SI includes structures (not just energies) for
   their MACE-OFF23 "spurious minima," checking those against the same
   `check_dimer_geometry` gate used here would directly settle whether
   their own published finding involves an intact dimer or not -- which
   would also settle whether this session's inability to reproduce it
   reflects a real difference in conclusion, or just a difference in how
   carefully the geometry was checked.
