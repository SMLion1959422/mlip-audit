# Test 3 results (water dimer O-O potential energy scan)

Session date: 2026-09-22/23. Scan performed on CPU (local dev machine, no
GPU available this session); Colab GPU not used yet.

## 1. Acceptance criterion (as set for this session)

> MACE-OFF23-small must reproduce its published failure: a spurious energy
> minimum at short O-O separation that is DEEPER than the physical minimum
> near 2.9 A. This is a known, published result (Ranasinghe et al. 2025,
> JCIM 65(17), 8980-8999; arXiv:2503.11537). If we cannot reproduce it, the
> pipeline is wrong and we stop and debug rather than proceeding.

**Status: MET**, after two rounds of debugging (see "Debugging trail"
below) that changed the implementation from what the original spec
literally described. Both changes are disclosed in code (module
docstrings in `mlip_audit/test3_dimer.py` and `mlip_audit/restraints.py`)
and here.

## 2. MACE-OFF23-small result

| | O-O distance (target) | O-O distance (actual) | Energy |
|---|---|---|---|
| Physical minimum (reference window 2.5-3.5 A) | 2.9 A | 2.900 A | **-4162.4495 eV** |
| Global minimum (entire 0.2-7.0 A scan) | 0.7 A | 1.034 A | **-8792.0724 eV** |

The global minimum is **4629.6 eV lower** than the physical minimum --
clearly and unambiguously deeper, not a marginal effect. It is also not an
isolated fluke: several other short-range points independently converged
below the physical-minimum energy:

| Target O-O (A) | Actual O-O (A) | Energy (eV) | Deeper than physical (-4162.45 eV)? |
|---|---|---|---|
| 1.0 | 0.996 | -4166.97 | yes |
| 0.8 | 0.739 | -4164.86 | yes |
| 0.7 | 1.034 | -8792.07 | yes (extreme outlier) |
| 0.6 | 0.680 | -4165.80 | yes |

Full data: `results/test3_dimer/mace-off23-small.csv` (69 points, all
`status=ok`, all `converged=True` except the 0.5 A point). Plot:
`results/test3_dimer/dimer_scan.png`.

Shape of the curve (see plot): smooth single well with minimum at 2.9 A,
steep physically-correct repulsive rise from ~2.5 A down to ~1.1 A, then a
catastrophic drop into unphysical territory below ~1.0 A -- qualitatively
exactly the failure Ranasinghe et al. describe and plot for MACE-OFF23
models.

**Caveat on the -8792 eV point specifically**: this is a genuine,
converged (fmax satisfied) result from this pipeline, not a numerical
error -- but its magnitude is extreme even for "spurious." The restrained
optimizer let the actual O-O distance drift to 1.034 A (well past its 0.7
A target; the model's forces overpowered the restraint) while some other
part of the geometry collapsed into a very deep, likely nonphysical
configuration. Treat this point as evidence of a badly-behaved region of
the PES, not as "the model's energy at O-O=0.7 A" in a literal sense --
the model was not actually evaluated at that literal separation once the
restraint was overpowered. The 1.0/0.8/0.6 A rows are better-behaved
(actual stayed close to target) and are the more citable numbers for "how
much deeper is the spurious minimum."

## 3. ANI-2x result -- UNEXPECTED, flagged, not yet explained

Run as a negative control (Ranasinghe et al. describe ANI-2x as tracking
their DFT reference well, with no reported spurious minimum). Result:

| | O-O distance (target) | Energy |
|---|---|---|
| Physical minimum (2.5-3.5 A window) | 2.7 A | -4157.6191 eV |
| Global minimum (entire scan) | 0.6 A | **-4258.9194 eV** |

`check_spurious_minimum` also flags `HAS SPURIOUS MINIMUM: True` for
ANI-2x -- a ~101 eV deeper global minimum at short range. This
**contradicts** the paper's description of ANI-2x as well-behaved on this
test, so it should NOT be taken at face value as "ANI-2x also has this
failure" without further investigation. Two explanations are open and
UNRESOLVED as of this writing:

  a) ANI-2x genuinely does show a (perhaps less dramatic, previously
     under-emphasized) version of this artifact -- plausible, since ANI-2x
     has no explicit short-range repulsion correction and is known to
     extrapolate poorly outside its training distribution.
  b) Something in this pipeline (most likely the random-reorientation
     multi-start mechanism in `geometry.random_perturbed_water_dimer`, or
     how `torchani`'s ASE calculator handles severely out-of-distribution
     geometries) produces an artifact for ANY model at extreme clash
     distances, independent of genuine model quality -- which would also
     cast some doubt on how to interpret the MACE result's exact depth
     (though not on its existence/direction, which is independently
     supported by the paper).

Full data: `results/test3_dimer/ani2x.csv`. **Do not cite the ANI-2x
number as a confirmed finding without resolving (a) vs (b) first** -- this
is flagged explicitly so it isn't accidentally treated as settled.

## 4. Exact package versions (pip freeze, key packages)

Two separate environments were used (see "Debugging trail" and
`requirements.txt` for why: `mace-torch` hard-pins `e3nn==0.4.4`, which is
incompatible with `fairchem-core`'s `e3nn>=0.5` requirement).

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

## 5. What is NOT yet built / run

- **UMA-S: not run.** `.venv-uma` is created and `requirements-uma.txt`
  installed successfully (confirmed: `fairchem-core==2.22.0` imports). A
  cached HuggingFace token exists (`huggingface_hub.get_token()` found
  one). But neither `tests/test_charge_spin.py::test_charge_reaches_model`
  (the real, non-skipped version) nor a UMA-S Test 3 scan have actually
  been executed yet. This is the next concrete step.
- **Tests 1, 2, 4: not built at all**, per this session's explicit scope.
- **Colab: not exercised.** Everything above ran on local CPU. The
  notebook (`notebooks/01_dimer_scan.ipynb`) has not been run end-to-end
  on Colab; its Part A / Part B split (for the two-environment issue) is
  untested in that environment.
- **ANI-2x vs pipeline-artifact question (Section 3) is open.** Needs
  investigation before the ANI-2x number is used in any writeup.
- **`results/checkpoints/*.extxyz`** (per-point relaxed geometries) exist
  locally from these runs but are gitignored (regenerable, not needed to
  interpret the CSVs) -- not part of this commit.

## Debugging trail (why the implementation differs from the original spec)

Two rounds of debugging were needed to go from "pipeline runs" to "pipeline
reproduces the published result reliably." Both are documented in code;
summarized here for anyone resuming this work:

1. **Hard constraint -> soft restraint.** The spec called for
   `ase.constraints.FixBondLength`. In practice, at the huge force
   magnitudes near an atomic clash, ASE's constraint-projection algorithm
   either raised `RuntimeError("Did not converge")` or (once its tolerance
   was loosened) let LBFGS diverge into multi-million-eV garbage --
   producing a false "no spurious minimum" result that would have been
   wrong about the model, not just a numerical hiccup. Checking
   Ranasinghe et al.'s actual methods section (Sec. 2.2.3) showed their
   ML-potential scans used an OpenMM harmonic distance RESTRAINT (k=10
   GJ/mol/nm^2), not a hard constraint. Switched to match
   (`mlip_audit/restraints.py::HarmonicDistanceRestraint`,
   k=1036.4 eV/A^2, same physical stiffness after unit conversion).
2. **Single warm-started chain -> multi-start.** Even with the restraint,
   a single sequential (warm-started) LBFGS chain from 7.0 A down to 0.2 A
   found only a modest, NOT-deeper local dip (~-4153 eV) for
   MACE-OFF23-small -- which would have been a false negative on the
   acceptance criterion. Ad-hoc testing showed the restrained landscape at
   short O-O is multi-basin: different starting orientations at the SAME
   target distance converged to energies differing by thousands of eV.
   Added multi-start (`DIMER_N_RESTARTS=5`: the warm-started continuation
   plus 4 randomly-reoriented alternatives per point, keeping the
   lowest-energy converged result and carrying it forward as the next
   point's warm start). This is what actually surfaced the -8792 eV point.
   A single deterministic LBFGS chain is not a reliable way to test "does
   a spurious minimum exist" on a landscape already shown to be rugged.

Also fixed along the way: `mlip_audit/plotting.py::check_spurious_minimum`
originally compared against "the minimum over O-O >= 1.0 A," which
double-counted the 1.0 A point itself (already anomalously deep) as part
of the "physical" reference. Fixed to use a dedicated 2.5-3.5 A window
around the known ~2.9 A minimum for the reference value, and to report the
whole-curve global minimum separately (matching how the paper frames its
own result).
