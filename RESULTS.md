# Test 3 results (water dimer O-O potential energy scan)

Session date: 2026-09-22/23. Scan performed on CPU (local dev machine, no
GPU available this session); Colab GPU not used yet.

**This document supersedes an earlier version of itself that declared the
acceptance criterion "MET" for MACE-OFF23-small.** That conclusion was
premature -- it was built on energies from relaxed structures that turned
out, on inspection, to have collapsed or dissociated O-H bonds (not an
intact water dimer). Five diagnostic checks were run before writing this
version: (1) geometry dump, (2) LBFGS convergence, (3) constraint-mechanism
sensitivity, (4) an unconstrained basin-of-attraction test, and (5) a
from-scratch reconstruction of the paper's own starting geometry (Section
7) to test whether this session's idealized starting guess was itself the
source of the disagreement -- see "Diagnostic checks" and Section 7 below
for the full trail. The corrected, current status is in Section 1.

Two checks in particular changed how strong a claim this document can
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
geometries). Together these make "NOT MET" a considerably stronger,
better-supported conclusion than earlier rounds of this document could
claim -- see Section 7's final paragraph for exactly how much this does,
and does not, license saying about the published result.

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
