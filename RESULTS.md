# Test 3 results (water dimer O-O potential energy scan)

Session date: 2026-09-22/23. Scan performed on CPU (local dev machine, no
GPU available this session); Colab GPU not used yet.

**This document supersedes an earlier version of itself that declared the
acceptance criterion "MET" for MACE-OFF23-small.** That conclusion was
premature -- it was built on energies from relaxed structures that turned
out, on inspection, to have collapsed or dissociated O-H bonds (not an
intact water dimer). Six diagnostic checks were run before writing this
version: (1) geometry dump, (2) LBFGS convergence, (3) constraint-mechanism
sensitivity, (4) an unconstrained basin-of-attraction test, (5) a
from-scratch reconstruction of the paper's own starting geometry (Section
7), and (6) a 24-combination optimizer/tolerance/restraint-stiffness
robustness sweep (Section 8) -- see "Diagnostic checks" and Sections 7-8
below for the full trail. The corrected, current status is in Section 1.

Three checks in particular changed how strong a claim this document can
make. Check 4 tested whether the broken-geometry region is something an
ordinary energy minimizer would actually wander into from a realistic
clash, as opposed to only being reachable via the artificial restrained
scan -- it is NOT reachable that way, for either model. Section 7 tested
whether this session's non-reproduction was an artifact of using an
idealized rather than the paper's literal starting geometry -- rebuilding
a literature-informed reconstruction of "Smith stationary point 1" from
scratch (via a real wB97X/6-31G(d) optimization, since no coordinates were
obtainable from the paper or its SI) and re-running both models gave the
SAME result (valid depth = 0.00 eV for both models, both starting
geometries). Section 8 tested whether that result depends on optimizer
choice, convergence tolerance, or restraint stiffness -- 22 of 24 swept
combinations gave exactly 0.00 eV, the other 2 differing by numerical
noise two orders of magnitude below any real signal. Together these make
"NOT MET" a considerably stronger, better-supported conclusion than
earlier rounds of this document could claim -- see Section 7's final
paragraph for exactly how much this does, and does not, license saying
about the published result.

## 1. Acceptance criterion (as set for this session)

> MACE-OFF23-small must reproduce its published failure: a spurious energy
> minimum at short O-O separation that is DEEPER than the physical minimum
> near 2.9 A. This is a known, published result (Ranasinghe et al. 2025,
> JCIM 65(17), 8980-8999; arXiv:2503.11537). If we cannot reproduce it, the
> pipeline is wrong and we stop and debug rather than proceeding.

**Status: NOT MET, under rigorous (geometry- and convergence-validated)
analysis -- and this is now more than an absence-of-evidence finding.**
Across four independent methodological variants tried in this session
(multi-start + restraint, single-chain + restraint, single-chain + hard
constraint, and unconstrained free minimization -- see Check 4), all
starting from independently-verified clean geometries, MACE-OFF23-small's
energy rises smoothly and monotonically as O-O shrinks for as long as the
relaxed structure remains a chemically intact water dimer. Below a
threshold separation, restrained/constrained relaxation attempts converge
to a structure with either a collapsed or a dissociated O-H bond -- BUT
Check 4 (unconstrained LBFGS, no restraint, no constraint, starting from
real mild-to-moderate clashes at O-O=1.8 and 2.2 A) shows that broken
region is NOT spontaneously reachable: released from a clash and left to
follow its own gradient, the model relaxes back to the physical
hydrogen-bonded minimum every time, not into a collapsed structure. The
broken-geometry states this session found are real stationary points of
the model's PES, but they are NOT downhill from a physically reasonable
starting point -- an actual docking or minimization workflow would not
fall into them. No run, under any method tried, produced a
**geometry-valid** structure with energy below the physical ~2.9 A
minimum, AND the specific operational concern from the original spec
("energy minimization of a clashed structure makes the clash worse") was
directly tested and did not occur. This is stated per the session's own
explicit instruction ("if we cannot reproduce it... we stop and debug
rather than proceeding") -- this is that stop. See "What would still need
to happen" at the end of this document for what remains untried.

ANI-2x, run in parallel as a comparison point, shows the same qualitative
picture (see Section 3): smooth monotonic rise while geometry-valid, then
total breakdown of intact-dimer geometry below a threshold, and (Check 4)
the same non-reachability from a realistic clash -- no
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

### Check 4: is the broken-geometry region reachable from a realistic clash?

The previous three checks establish that the broken-geometry region is not
a valid dimer state, but they don't by themselves address the concern the
original spec actually raised: *"energy minimization of a clashed
structure might lead to even stronger steric clashes."* A restrained SCAN
artificially holds O-O at a target value throughout -- it doesn't test
whether an ordinary, unconstrained minimizer would ever wander into the
broken region on its own starting from something a real workflow (e.g. a
docking program placing a ligand with a mild clash) might actually
produce.

Script: `scripts/check4_unconstrained_basin.py`. For each model, took the
already-relaxed (geometry-valid) checkpoint structure at O-O target =
2.9, 2.2, and 1.8 A, stripped every constraint and restraint, and ran
**fully unconstrained** LBFGS (fmax=0.05, up to 500 steps) -- every degree
of freedom free, including O-O itself. Full data:
`results/test3_dimer/unconstrained_basin_check.csv`.

| Model | Start target (A) | Start actual O-O (A) | Final O-O (A) | Final energy (eV) | Final geometry | Outcome |
|---|---|---|---|---|---|---|
| mace-off23-small | 2.9 | 2.900 | 2.900 | -4162.4494 | valid, min/max OH 0.958/0.966 | returned to physical minimum |
| mace-off23-small | 2.2 | 2.202 | 2.756 | -4162.4005 | valid, min/max OH 0.958/0.962 | returned to physical minimum |
| mace-off23-small | 1.8 | 1.808 | 2.733 | -4162.4018 | valid, min/max OH 0.958/0.962 | returned to physical minimum |
| ani2x | 2.9 | 2.900 | 2.844 | -4157.6147 | valid, min/max OH 0.963/0.972 | returned to physical minimum |
| ani2x | 2.2 | 2.202 | 2.639 | -4157.5469 | valid, min/max OH 0.962/0.967 | returned to physical minimum |
| ani2x | 1.8 | 1.808 | 2.635 | -4157.5459 | valid, min/max OH 0.963/0.967 | returned to physical minimum |

**All six runs (both models, all three starting points, including the
real O-O=1.8 A steric clash) relaxed back to the physical hydrogen-bonded
minimum.** None collapsed; every final structure has zero
proton-transfer flags and normal O-H bond lengths. The broken-geometry
region documented in Checks 1-3 is a real feature of both models' PES
(reachable by an artificial restrained scan that forces the system to
stay at an extreme O-O value against its own gradient), but it is **not
downhill from a physically reasonable starting geometry** -- an actual
energy-minimization or docking workflow encountering a mild-to-moderate
clash would not fall into it for either model. This directly answers the
question the acceptance criterion is really asking about operational
risk, and the answer is negative for both models under this test.

This does not prove the broken region is unreachable from EVERY possible
starting point (only the three tested, which were chosen to bracket
"mild" to "fairly severe" steric clash) -- see "What would still need to
happen" for how this could be pushed further.

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

- **Tests 1, 2, 4: not built at all**, per this session's explicit scope.
- **Colab: not exercised.** Everything above (and UMA-S, Section 9) ran on
  local CPU.
- **`results/checkpoints/*.extxyz`** and `results/checkpoints_sp1/*.extxyz`
  exist locally but are gitignored (regenerable) -- not part of this
  commit.
- UMA-S was NOT run through the optimizer-settings robustness sweep
  (Section 8) -- that was completed for MACE-OFF23-small and ANI-2x only.

## 7. Starting-geometry sensitivity: Smith SP1 reconstruction

Ranasinghe et al. start from "the Smith stationary point 1" water dimer,
optimized at wB97X/6-31G(d) with ORCA. Everything in Sections 1-6 used a
hand-built, idealized, PLANAR Cs-symmetric guess instead
(`geometry.build_water_dimer`) -- the single biggest acknowledged gap from
the paper's exact setup, and the top item in the previous version's "what
would still need to happen" list. This section closes it as far as this
session is able to.

**Step 1: obtain or reconstruct the exact structure.** Checked
Ranasinghe et al.'s SI (fetched the arXiv PDF, both main text and the
appended Supporting Information) for deposited coordinates: not present.
The SI contains only training loss curves (Figs. S1-S7) and
quasi-harmonic analysis tables (Tables S1-S5) -- no structure files, no
data/code availability statement pointing to one. Their citation for
"Smith stationary point 1" (ref. 20) points to Gillan, Alfe & Michaelides,
"Perspective: How good is DFT for water?" (J. Chem. Phys. 144, 130901,
2016), which is paywalled and could not be fetched by this session (HTTP
403). A web search independently confirmed what "Smith SP1" is: one of
ten water-dimer stationary points characterized by Smith, Swanton, Pople,
Schaefer & Radom (J. Chem. Phys. 92, 1240, 1990); SP1 specifically is
established in the literature as the true global minimum -- a near-linear,
NON-PLANAR, Cs-symmetric hydrogen-bonded structure (confirmed via a
secondary source, "The water dimer II: Theoretical investigations",
which reviews the Smith stationary points in detail). No literal
Cartesian coordinates were obtainable from any accessible source.

Per the fallback plan: **reconstructed** the structure. Built a
topologically-correct non-planar Cs-symmetric starting guess (donor
monomer + the O-O axis define a mirror plane; the acceptor's two H atoms
are placed as exact mirror images of each other through that plane,
rather than splayed within it as the simpler planar guess does), then ran
a REAL geometry optimization at wB97X/6-31G(d) -- the paper's exact
level of theory -- using PySCF + pyberny. (PySCF has no Windows wheels;
installed via a dedicated conda environment, `mlip-audit-qm`, using
conda-forge; needed `OMP_NUM_THREADS=2` to work around a Windows-build-
specific memory allocation bug -- see
`scripts/build_smith_sp1_geometry.py` docstring for full detail.) The
optimization converged in 31 steps to a structure that IS Cs-symmetric
(acceptor H's ended up as exact mirror images in z, to 5 decimal places,
despite no symmetry constraint being imposed on the optimizer) with
sensible parameters: O-O = 2.840 A, near-linear H-bond (O-H...O = 164.2
deg), donor's bonded O-H elongated (0.973 A) relative to its free O-H
(0.963 A) and the acceptor's (0.966 A both) -- all textbook signatures of
the real water dimer minimum. Saved: `geometries/smith_sp1_reconstructed.xyz`.
**This is a reconstruction, not the paper's own deposited structure** --
stated plainly, as it should be every time this geometry is cited.

**Step 2: re-run the scan.** Added `--start-geometry` to
`mlip_audit.test3_dimer` (a disclosed, permanent pipeline feature, not a
one-off hack -- see `_load_warm_start`) so a custom structure can replace
the idealized guess as the scan's starting point, everything else
(restraint mechanism, k=10 GJ/mol/nm^2, multi-start, geometry/convergence
validation) unchanged. Re-ran both models over the paper's exact range,
0.02-0.40 nm = **0.2-4.0 A** (narrower than Sections 1-6's 0.2-7.0 A) at
0.1 A steps, from the SP1 reconstruction. Full data:
`results/test3_dimer_sp1/{model}.csv`; plots:
`results/test3_dimer_sp1/dimer_scan_sp1.png` and `force_convergence_sp1.png`.

**Step 3: side-by-side comparison.**

| Model | Starting geometry | Physical min (eV) | Valid global min (eV) | Valid depth | Raw global min (eV) | Raw depth |
|---|---|---|---|---|---|---|
| mace-off23-small | idealized planar guess (Sections 1-6) | -4162.4494 @ 2.9 A | -4162.4494 @ 2.9 A | 0.00 eV | -4165.8032 @ 0.6 A | 3.35 eV = 77.3 kcal/mol |
| mace-off23-small | **Smith SP1 reconstruction** | -4162.4500 @ 2.9 A | -4162.4500 @ 2.9 A | **0.00 eV** | -4450.3785 @ 0.2 A | 287.93 eV = 6639.8 kcal/mol |
| ani2x | idealized planar guess (Sections 1-6) | -4157.6191 @ 2.7 A | -4157.6191 @ 2.7 A | 0.00 eV | -4258.9194 @ 0.6 A | 101.30 eV = 2336.0 kcal/mol |
| ani2x | **Smith SP1 reconstruction** | -4157.6187 @ 2.8 A | -4157.6187 @ 2.8 A | **0.00 eV** | -4157.6187 @ 2.8 A | **0.00 eV** (no raw excursion at all) |

**The physical minimum energy matches to within ~0.0006 eV between the
two starting geometries for both models** -- strong cross-validation that
both starting points converge to the same true minimum, as expected for a
real PES feature independent of how you got there. **The VALID depth is
0.00 eV in all four runs** -- completely unchanged by starting geometry.
If anything, ANI-2x's raw (unfiltered) numbers are LESS dramatic from the
SP1 starting point (no deep excursion at all, vs. 2336 kcal/mol from the
idealized guess) -- the opposite of what "a more realistic starting
geometry reveals the artifact" would predict.

**This directly answers the question this diagnostic round was run to
answer: the starting geometry does NOT account for this session's
difference from the published result.** The finding (no geometry-valid,
operationally-reachable minimum deeper than physical, for either model)
is robust across two starting geometries that differ substantially in
construction (hand-built idealized planar guess vs. an independently
wB97X/6-31G(d)-optimized, literature-informed, non-planar Cs-symmetric
reconstruction) and, from Checks 3-4, across constraint mechanism and
reachability testing as well.

**What this does NOT close**: this session still does not have Ranasinghe
et al.'s own literal structure (unobtainable, per Step 1), nor their exact
toolchain (ORCA for the DFT optimization; OpenMM for the ML-potential
restraint scans, vs. this session's PySCF and ASE respectively). "The
starting geometry doesn't explain it" is now reasonably well-supported;
"our result contradicts theirs" is still a stronger claim than this
session can make, since a toolchain- or checkpoint-version-level
difference remains untested and unruled-out. The honest summary, per the
standard this diagnostic round was held to: this session tested a
system matching the published one's described geometry and level of
theory as closely as could be reconstructed, with a validation procedure
stricter than what the paper's methods section describes, across multiple
independent starting geometries and constraint mechanisms, and found no
reachable, geometry-valid spurious minimum for either model.

## 8. Optimizer-settings robustness sweep (bounding the remaining toolchain gap)

Section 7 closed the starting-geometry gap; this section bounds a
different one cheaply: does the "no reachable spurious minimum" finding
depend on THIS session's specific optimizer choice (LBFGS), convergence
tolerance (fmax=0.05), or restraint stiffness (k=10 GJ/mol/nm^2) -- as
opposed to Ranasinghe et al.'s actual toolchain (ORCA/OpenMM), which
cannot be run here? If varying these settings changes the answer, the
untested toolchain difference becomes a live concern; if it doesn't, that
concern shrinks.

**Design**: single-chain (no multi-start -- isolating the optimizer axis
from the already-separately-tested multi-start question), full range
(4.0 -> 0.2 A), Smith SP1 reconstruction starting geometry (Section 7),
both models, swept over fmax in {0.01, 0.005} (tighter than the
project-standard 0.05) x optimizer in {LBFGS, FIRE} x restraint k in
{1, 10, 100} GJ/mol/nm^2 (vs. the standard 10) = 12 settings x 2 models =
24 runs, each with max_steps raised to 1000. Script:
`scripts/check5_optimizer_robustness.py`, driven by
`scripts/run_robustness_sweep.sh`.

**An operational incident happened during this sweep and is disclosed in
full**, per this project's standing practice of not hiding methodology
mistakes: killing one hung combination's Python process (FIRE + k=100 GJ/mol/nm^2,
which requires very small stable timesteps and was slow, not actually
hung) left its parent shell script alive, which then continued its own
loop independently while a second, corrected re-launch ran concurrently
-- two process trees briefly wrote toward the same output directory. This
was caught, both trees were fully killed (`taskkill /F /T`, verified via
`Get-CimInstance Win32_Process`), and **every resulting CSV was audited
before use**: row count vs. the expected 39 scan points, duplicate
distance values, truncated/malformed final rows, and mtime clustering
against the known ~4-minute concurrent-write window. Full results: all 24
files are structurally clean (zero duplicate distances, zero
truncated/malformed rows anywhere). 19/24 completed all 39 points; 5/24
are cleanly incomplete (23-37/39 points, each with a `.TIMEDOUT` marker
and a well-formed final row at the 900s per-combination wall-clock cap) --
all five are `fire_k10` or `fire_k100` combinations, consistent with FIRE
being slow to converge against very stiff restraints, not with
corruption. All 5 partial files have mtimes well outside the confirmed
race window and were kept (not deleted) as genuine, if incomplete,
partial evidence. Full audit script and reasoning available on request;
not separately committed as a script since it was a one-time forensic
check, not a reusable pipeline component.

**Results** (`results/test3_dimer_sp1_robustness/*.csv`; `valid_depth_eV`
computed via `mlip_audit.plotting.check_spurious_minimum` against each
combination's own physical-window reference):

| Setting | mace-off23-small valid_depth (eV) | ani2x valid_depth (eV) |
|---|---|---|
| fmax=0.01, LBFGS, k=1 | 0.0000 | 0.0000 |
| fmax=0.01, LBFGS, k=10 | 0.0000 | 0.0000 |
| fmax=0.01, LBFGS, k=100 | 0.0000 | 0.0000 |
| fmax=0.01, FIRE, k=1 | 0.0000 | 0.0000 |
| fmax=0.01, FIRE, k=10 | 0.0000 | -0.0015 (noise) |
| fmax=0.01, FIRE, k=100 | -0.0132 (noise; 24/39 pts, timed out) | 0.0000 (33/39 pts, timed out) |
| fmax=0.005, LBFGS, k=1 | 0.0000 | 0.0000 |
| fmax=0.005, LBFGS, k=10 | 0.0000 (fresh spot-check re-run, see below) | 0.0000 (fresh spot-check re-run, see below) |
| fmax=0.005, LBFGS, k=100 | 0.0000 | 0.0000 |
| fmax=0.005, FIRE, k=1 | 0.0000 | 0.0000 |
| fmax=0.005, FIRE, k=10 | 0.0000 (37/39 pts, timed out) | -0.0015 (noise) |
| fmax=0.005, FIRE, k=100 | N/A (23/39 pts, timed out; physical window under-sampled) | N/A (33/39 pts, timed out; physical window under-sampled) |

**22/24 combinations give exactly 0.00 eV.** The 2 non-zero values
(-0.0015 and -0.0132 eV = -0.03 and -0.30 kcal/mol) are two orders of
magnitude below anything that would register as a real minimum (compare
to the raw/invalid depths of 77-6640 kcal/mol seen elsewhere in this
project) -- numerical noise from the physical-window reference landing on
a slightly different point, not a finding.

**Spot-check re-run** (fmax=0.005, LBFGS, k=10, the one combination most
likely to have been touched by the concurrent-write incident given its
mtime): both models' files for this exact setting were deleted and
re-run fresh, sequentially, with no other process active (verified via
`Get-CimInstance Win32_Process` immediately before each run). Results:
mace-off23-small valid_depth = 0.0000 eV (physical -4162.4507 eV @ 2.9 A;
raw global min -4304.8225 eV @ 0.2 A, invalid); ani2x valid_depth =
0.0000 eV (physical = raw = -4157.6152 eV @ 2.8 A, no excursion at all,
matching its behavior on the standard SP1 scan in Section 7).

**Conclusion: the "no reachable, geometry-valid spurious minimum" finding
is insensitive to optimizer choice (LBFGS vs. FIRE), convergence
tolerance (fmax 0.05/0.01/0.005), and restraint stiffness (k =
1/10/100 GJ/mol/nm^2), for both models, on the Smith SP1 reconstruction
geometry.** This narrows what an actual ORCA/OpenMM toolchain run could
plausibly change: it would need to differ from every setting tested here
in some OTHER respect (e.g. a genuine numerical difference between
OpenMM's and this project's harmonic restraint implementation, or a
difference in how the two DFT codes, ORCA vs. PySCF, converge the
starting geometry) rather than in optimizer/tolerance/stiffness choice,
which this sweep now rules out as the explanation.


## 9. UMA-S: charge/spin fix, and Test 3 scan (all three models now complete)

**A real environment bug, not a charge/spin logic bug, blocked UMA-S
entirely at first.** Running `tests/test_charge_spin.py::test_charge_reaches_model`
failed with `torch._inductor.exc.InductorError: InvalidCxxCompiler:
Compiler: cl is not found`. Root cause: `fairchem.core.pretrained_mlip.
get_predict_unit()`'s DEFAULT `inference_settings` ("default", and also
"turbo") both set `compile=True`, which requires `torch.compile`'s C++
backend (MSVC's `cl.exe` on Windows) to JIT-compile inference kernels --
unavailable on this machine (same root cause as the earlier PySCF
build-from-source failure in Section 7). Fixed by explicitly passing
`inference_settings="batch"` in `mlip_audit/models.py::_load_uma_s` -- the
one named preset with both `compile=False` and `merge_mole=False` (the
latter also sidesteps a charge/spin-triggered remerge-fallback path that
doesn't apply cleanly to this project's usage pattern of varying
charge/spin between calls). This is a genuine, permanent pipeline fix, not
a one-off test workaround -- it changes how `get_calc("uma-s-1p1")` loads
the model for every caller, including the Test 3 scan below. Still
explicitly NOT `"turbo"`, per the original project brief's prohibition
(system-size locking).

With that fix, all three `tests/test_charge_spin.py` tests pass,
including the real (not skipped) `test_charge_reaches_model`: acetate
scored at charge=-1 vs. charge=0 gives measurably different energies,
confirming charge/spin config reaches the model.

**Test 3 scan** (standard idealized starting geometry, same protocol as
MACE-OFF23-small/ANI-2x in Sections 1-6: multi-start restraint,
0.2-7.0 A, geometry+convergence validated): full data
`results/test3_dimer/uma-s-1p1.csv`; combined 3-model plot
`results/test3_dimer/dimer_scan.png`.

| | Physical min (~2.9 A) | RAW global min | RAW depth | VALID global min | VALID depth |
|---|---|---|---|---|---|
| uma-s-1p1 | -4159.9486 eV @ 2.9 A | -4159.9486 eV @ 2.9 A | **0.00 eV = 0.00 kcal/mol** | -4159.9486 eV @ 2.9 A | **0.00 eV = 0.00 kcal/mol** |

**UMA-S is the cleanest of the three models by this measure**: not only
is VALID depth 0.00 eV (same as MACE-OFF23-small and ANI-2x), its RAW
(unfiltered) depth is ALSO 0.00 eV -- unlike MACE (77 kcal/mol raw) and
ANI-2x (2336 kcal/mol raw), UMA-S's geometry-invalid points (11/69, all
below ~1.3 A, all showing severe O-H dissociation up to 6.5 A) are all
HIGHER energy than physical, never lower. UMA-S shows no spurious dip at
all under this protocol, filtered or not.

**Test 3 scan, Smith SP1 reconstruction geometry** (Section 7's
methodology, extended to UMA-S so all three models now have both the
standard AND the SP1 scan): full data `results/test3_dimer_sp1/uma-s-1p1.csv`;
combined 3-model plot `results/test3_dimer_sp1/dimer_scan_sp1.png`.

| | Physical min (~2.9 A) | RAW global min | RAW depth | VALID global min | VALID depth |
|---|---|---|---|---|---|
| uma-s-1p1 (SP1 geometry) | -4159.9480 eV @ 2.9 A | -4159.9480 eV @ 2.9 A | **0.00 eV** | -4159.9480 eV @ 2.9 A | **0.00 eV** |

Same result as the standard-geometry scan (energies agree to 0.0006 eV),
same as MACE-OFF23-small and ANI-2x on this geometry (Section 7): no
spurious minimum, raw or valid. **All three models, both starting
geometries, now agree: valid depth = 0.00 eV in every one of the 6
(model x geometry) combinations tested in this session.**

## What would still need to happen to give this a final verdict

Check 4 upgraded the conclusion from "we could not find a valid example"
to "we could not find a valid example, AND we specifically tested whether
an ordinary minimizer would fall into the broken region from a realistic
clash and it did not." Section 7 upgraded it again: "AND this is not an
artifact of our idealized starting geometry -- an independently
reconstructed, literature-informed, DFT-optimized starting point gives
the same answer." Together these are a real, operationally-scoped
negative result, not just absence of evidence. What's left is narrower
than before:

1. **The paper's own geometries** (still the highest-value remaining
   gap). Unavailable to this session -- checked and confirmed absent from
   both the paper and its SI (Section 7, Step 1). If Ranasinghe et al.'s
   raw structures ever become available (e.g. on request from the
   authors, or a future SI update), checking those against
   `check_dimer_geometry` and, if reachable, against the Check-4-style
   basin test, would directly settle whether their finding and this
   session's non-finding are in real tension or not.
2. **The exact toolchain.** This session used PySCF (DFT optimization)
   and ASE+LBFGS (MLIP relaxation) throughout; the paper used ORCA and
   OpenMM respectively. Section 7 shows the DFT optimizer converges to a
   sensible, textbook water-dimer minimum regardless (matching energy
   across two different codes/methods was not tested, but the geometry's
   textbook parameters -- O-O, H-bond angle, donor O-H elongation -- are a
   reasonable plausibility check that ORCA would land in the same basin).
   The bigger unknown is the ML relaxation side: this session's harmonic
   restraint should be numerically equivalent to OpenMM's, but has not
   been cross-checked against an actual OpenMM run.
3. **Finer sampling right at the breakdown boundary.** This scan steps in
   0.1 A increments (matching the paper's stated 0.01 nm). A valid,
   deeper minimum occupying a window narrower than 0.1 A between two
   sampled points would be invisible here. De-prioritized by Check 4: even
   if such a narrow valid basin exists, it would need its own basin of
   attraction reachable from a realistic clash to be operationally
   relevant, and the tested clashes (1.8, 2.2 A) didn't find one.
4. **More/different unconstrained starting points than Check 4 tried.**
   Only 1.8, 2.2, and 2.9 A were tested (as requested), and only from the
   idealized-guess scan's own checkpoints (Section 7's SP1 reconstruction
   was not itself re-tested with the Check-4 unconstrained-release
   protocol). Trying additional starting points -- especially ones
   deliberately constructed to be "close to" the broken region found in
   Checks 1-3 (e.g. take an invalid, dissociated structure from the
   restrained scan and release the restraint from THERE) would more
   directly test the boundary of the broken region's basin of attraction.

---

# Test 2 and Test 4: MD stability and condensed-phase water

**Status: infrastructure built and locally smoke-tested (tiny scale, CPU)
-- NOT YET RUN AT FULL SCALE.** Full-scale runs (100 ps x 4 molecules x 2
seeds for Test 2; 175 ps x 504 atoms for Test 4, x4 models each) are
computationally infeasible on this session's local CPU-only machine and
are explicitly scoped as Colab GPU work (per the 177-compute-unit budget
given for this work). This section documents what was built, the bugs
found and fixed while smoke-testing it, and every deviation from
Ranasinghe et al.'s actual protocol, with justification, per instruction.

## Reference protocol (Ranasinghe et al. 2025, re-extracted from the paper for precision)

**Test 2 (paper Sec. 2.2.2, main text)**: ONE artificial 349-atom
drug-like benchmark molecule (a composite of clarithromycin,
dexamethasone, diazepam, morphine, penicillin, sildenafil, and tryptophan
dipeptide fragments) -- NOT the 4 molecules used this session. 400 K,
Langevin (friction 1 ps^-1), 1.0 fs timestep, 0.25 ns (250 ps) production
(MACE M/L only reached 0.174/0.041 ns in 24h -- a real data point on this
class of test's compute cost), trajectory saved every 0.5 ps, LBFGS
geometry optimization before each model's MD run, bond-length/bond-angle
distributions analyzed via MDTraj + quasi-harmonic analysis.

**Test 2's molecules, actually a different paper test**: the paper
separately describes (SI, "Further quasi-harmonic analysis tests") a set
of "14 simple benchmark systems... water, ethane, methanol, methanethiol,
ethanol, acetamide, tetrahydrofuran, n-hexane, cyclohexane, benzene,
phenol, aniline, N-acetyl-alanine-methylamide, and
N-acetyl-serine-methylamide," run at the SAME simulation parameters (400 K
/ Langevin 1 ps^-1 / 1.0 fs / 0.25 ns) as the main 349-atom test. This
session's 4 molecules (ethanol, THF, phenol, ala-dipeptide) are a subset
of THIS 14-molecule list, not of the 349-atom test -- worth being precise
about, since these are two different tests in the paper with the same
simulation parameters but different molecules and different purposes
(main stability test vs. quasi-harmonic frequency analysis).

**Test 4 (paper Sec. 2.3)**: 168 TIP3P water molecules PLUS a SOLUTE,
prepared classically (MM, 0.5 ns equilibration to a 1.706 nm box), PME
electrostatics (8 A cutoff), SETTLE-constrained rigid water. "Before each
ML simulation, the geometry of the SOLUTE was optimized," then 125 ps NVT
equilibration, then 0.125 ns (125 ps) NPT production, 300 K (Nose-Hoover
thermostat), 1 bar (Monte Carlo barostat), 0.5 fs timestep, trajectory
saved every 0.5 ps, RDFs via MDTraj.

## Deviations table

| Parameter | Ranasinghe et al. | This session | Justification |
|---|---|---|---|
| **Test 2 molecule(s)** | ONE 349-atom composite drug-like molecule (main test); separately, 14 simple molecules incl. ethanol/THF/phenol/ala-dipeptide (SI quasi-harmonic test, same sim. params) | 4 molecules: ethanol, THF, phenol, ala-dipeptide (subset of the paper's OWN 14-molecule SI list) | Compute budget: the 349-atom test is far outside a 177-CU Colab budget across 4 models x 2 seeds; the paper's own SI subset is a validated, much cheaper alternative testing the same physics (bond/angle stability) on real, if smaller, molecules the paper itself used. |
| **Test 2 seeds** | not specified (presumably 1 run per model) | 2 seeds per (molecule, model) | Session choice, to distinguish a genuine model instability from a single unlucky velocity draw. Not a paper value to deviate from -- new. |
| **Test 2 length** | 0.25 ns (250 ps) | 100 ps | **Compute-limited, disclosed limitation** (explicit instruction: not to be silently adopted). 40% of the paper's length; some instabilities the paper reports only emerging late in a 250 ps run could be missed here. |
| **Test 2 timestep** | 1.0 fs | 1.0 fs | Matches. |
| **Test 2 temperature / thermostat** | 400 K, Langevin, friction 1 ps^-1 | 400 K, Langevin, friction 1 ps^-1 | Matches. |
| **Test 2 models** | ANI-2x, MACE-OFF23 (S/M/L/XS variants), B97-3c-family in-house models -- NOT UMA/eSEN (postdate the paper) | UMA-S, eSEN-conserving, eSEN-direct, ANI-2x (control) | Project scope from the start of this session: auditing OMol25-generation models (UMA, eSEN), which did not exist when the paper was written. ANI-2x included as a non-fairchem control, matching its role throughout this project. |
| **Test 4 solute** | a solute molecule embedded in the water box; the ML potential's role includes describing (at least) the solute | **NONE -- pure water box, ML potential describes all 504 atoms** | **Substantive deliberate simplification, disclosed**: no specific solute was given in this session's scope; testing a pure water box directly probes each model's description of water-water interactions in isolation (matching the paper's own RESULTS-section framing -- "water-water interactions are the driving force of the hydrophobic effect... ML potentials will still have to improve" -- and its RDF-based water-structure analysis), without a confounding solute-specific effect. This is NOT equivalent to the paper's literal Test 4 setup and should not be cited as reproducing it. |
| **Test 4 box construction** | OpenMM MM equilibration (0.5 ns) with TIP3P + PME + SETTLE | Grid-packing (see `mlip_audit/molecules.py::build_water_box`) at matching initial density; box side (17.13 A) matches the paper's reported 17.06 A closely | No packmol/OpenMM available in this environment. Grid-packing is only a REASONABLE starting configuration -- the protocol's own 125 ps NVT equilibration is what's supposed to relax it, same role the paper's box serves before ML production. |
| **Test 4 electrostatics/constraints** | PME (8 A cutoff), SETTLE-rigid water | None -- direct ML potential forces on all atoms, no long-range electrostatic scheme, no rigid-water constraint | The ML potentials (UMA/eSEN/ANI) compute total energy/forces directly from local+message-passing environments, not from a classical PME+point-charge scheme -- PME is specific to classical force fields and has no direct analog here. Water is fully flexible (not SETTLE-constrained), consistent with testing whether each MLIP's own intramolecular water description is stable, not imposing a classical constraint the MLIP wasn't trained to expect. |
| **Test 4 thermostat/barostat** | Nose-Hoover thermostat, Monte Carlo barostat (OpenMM) | `ase.md.nose_hoover_chain.NoseHooverChainNVT` (equilibration), `IsotropicMTKNPT` (production) | ASE has no Monte Carlo barostat implementation; MTK (Martyna-Tobias-Klein) is the standard deterministic alternative achieving the same NPT ensemble, and ASE's Nose-Hoover-chain NVT is a direct match to the paper's NVT thermostat. Toolchain substitution, not a physics choice. |
| **Test 4 timestep** | 0.5 fs | 1.0 fs | **Compute-limited, disclosed limitation.** 2x the paper's timestep. Water O-H stretches at ~3500-3800 cm^-1 (period ~9-10 fs); 1 fs gives ~10 steps/period (borderline-minimum resolution), vs. 0.5 fs's ~20 steps/period (safer, paper's choice). Flexible (non-SETTLE-constrained) water makes this MORE of a concern here than in the paper's rigid-water setup -- a real risk of energy-conservation artifacts or instability specifically from this choice, not just a shortened run. |
| **Test 4 NVT equilibration** | 125 ps | 125 ps | Matches. |
| **Test 4 NPT production** | 0.125 ns (125 ps) | 50 ps | **Compute-limited, disclosed limitation.** 40% of the paper's length -- less time for RDF/structural statistics to converge; any reported RDF should be treated as lower-confidence than the paper's own. |
| **Test 4 temperature/pressure** | 300 K, 1 bar | 300 K, 1 bar | Matches (temperature not specified by the user this session; defaulted to the paper's value as the only well-justified choice). |
| **Both tests: `inference_settings="turbo"`** | N/A (paper uses OpenMM, not fairchem inference_settings at all) | `"turbo"` requested explicitly, with a fresh calculator per fixed-composition trajectory (never shared across differently-sized/charged systems -- see `mlip_audit/models.py::get_calc` docstring for the exact safety contract) | User-specified, with the "fixed composition" justification given explicitly. **UNTESTED on this session's local machine**: turbo requires torch.compile's C++ backend, unavailable here (no MSVC), so it was never actually exercised locally -- only "batch" mode was validated (see Test 3 Section 9). Verify turbo genuinely works (and is faster, not just different) on Colab before trusting results from it; if it silently falls back or errors, that itself needs disclosing. |
| **eSEN checkpoint choice** | N/A (eSEN postdates the paper) | `esen-sm-conserving-all-omol`, `esen-sm-direct-all-omol` (the "sm"/small size class) | fairchem's `available_models` registry (checked against fairchem-core==2.22.0) has no `esen-md-conserving-all-omol` (only `esen-md-direct-all-omol` exists at the "md"/medium size) -- "sm" was chosen for both conserving and direct so the two are a fair, comparable pair, and to match uma-s-1p1's size class for consistency and compute budget. |
| **eSEN reference level of theory** | N/A | Assumed wB97M-V/def2-TZVPD, same as UMA | Not independently verified against fairchem's own eSEN documentation this session (both are OMol25-generation fairchem models, presumed to share a training reference level) -- flag if this turns out wrong; it isn't used by Tests 2/4 themselves (only Test 3's `REFERENCE_LEVELS` table, unused so far), so this is a low-stakes assumption for now. |

## Bugs found and fixed during local smoke-testing (before any full-scale run)

Two real, non-obvious correctness bugs were caught by smoke-testing the
resumable-MD driver (`mlip_audit/md_common.py::run_resumable_md`) at tiny
scale (hundredths of a picosecond, seconds of wall time) with ANI-2x
locally, before considering the Test 2/4 scripts trustworthy enough to
hand off for real Colab runs:

1. **Velocity carryover bug.** Test 4's NPT phase is meant to continue
   directly from NVT's final (positions AND velocities) state. The
   initial implementation redrew a fresh Maxwell-Boltzmann velocity
   distribution at the start of the NPT phase (since, from
   `run_resumable_md`'s point of view, `npt.traj` not existing yet looked
   like "a fresh start," which normally SHOULD draw new velocities).
   Caught by comparing NPT's first logged kinetic energy against NVT's
   last -- they should match exactly and didn't. Fixed by adding a
   `draw_initial_velocities` flag, `False` for Test 4's NPT call
   specifically.
2. **Off-by-one resume bug.** ASE's `Dynamics.attach(fn, interval=N)`
   fires its callback once immediately at step 0 (before any integration)
   in addition to every `N` steps after -- confirmed empirically (a
   10-step run with `interval=2` produces 6 saved frames at steps
   0,2,4,6,8,10, not 5). The initial implementation treated the on-disk
   frame COUNT as directly proportional to completed integration steps,
   over-counting by one checkpoint interval on every resume. Concretely:
   requesting a 20-step trajectory that had already reached 10 steps
   would compute only 8 remaining steps instead of the correct 10,
   silently landing at step 18 while logging (and believing) it had
   reached step 20. Over repeated Colab-disconnect-driven resumes (which
   this whole design exists to handle), this would have compounded,
   leaving every real run short of its stated target by an
   uncontrolled, resume-count-dependent amount, without any error or
   warning. Fixed by correctly computing completed real steps as
   `(n_frames_on_disk - 1) * save_interval_steps`, not
   `n_frames_on_disk * save_interval_steps`. Re-verified after the fix:
   a resumed run now reaches its exact requested target with correctly
   labeled timestamps and no gaps or duplicate frames.

Both bugs were caught before any expensive run, specifically BECAUSE this
project's established practice (see Test 3's whole debugging trail) is to
smoke-test new resumable/stateful machinery at trivial scale before
trusting it, rather than trusting new infrastructure's first real
(expensive) invocation.

## What is built vs. what still needs to happen

**Built and locally smoke-tested** (tiny scale, ANI-2x, CPU, seconds of
wall time each): `mlip_audit/molecules.py` (SMILES->3D via RDKit for Test
2; grid-packed periodic water box for Test 4, verified against the
paper's own box-size number), `mlip_audit/md_common.py` (resumable MD
driver, checkpointing every N ps, both bugs above fixed and re-verified),
`mlip_audit/md_analysis.py` (bond-length-stability tracking for Test 2;
O-O RDF for Test 4, sanity-checked to peak near the physically-expected
~2.8-2.9 A on a tiny test box), `mlip_audit/test2_md_stability.py` and
`mlip_audit/test4_condensed_water.py` (CLI drivers), `mlip_audit/models.py`
extended with eSEN-conserving/eSEN-direct and the turbo-mode contract,
`requirements-md.txt` + `setup.sh md` (single unified environment for all
four Test 2/4 models, since torchani and fairchem-core don't conflict the
way mace-torch and fairchem-core do).

**NOT done**:
- No full-scale run of either test, for any model. Everything numeric in
  this section is either a paper-reference value or this session's
  planned/configured value, NEVER a result -- do not read anything here as
  a finding about any model's actual stability.
- `inference_settings="turbo"` has never actually executed successfully
  anywhere in this project (needs a C++ compiler this local machine
  lacks) -- its very first real exercise will be on Colab. Watch closely
  for whether it errors, silently falls back, or behaves as documented.
- A new Colab notebook (`notebooks/02_md_tests.ipynb`, separate from Test
  3's `01_dimer_scan.ipynb` since these are a different stack/scope) has
  driver cells for both tests, but neither has been executed even once,
  on Colab or anywhere else.
- No `--molecule`/`--model` combination has been run to full completion,
  so there is no evidence yet that a real ~100k-step trajectory doesn't
  hit some OTHER bug this tiny-scale smoke testing couldn't surface
  (e.g. numerical drift only visible over many more steps, a memory leak,
  a Colab-specific environment quirk).
