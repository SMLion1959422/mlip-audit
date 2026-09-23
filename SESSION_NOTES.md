# Session notes -- resume from here

Read this + `RESULTS.md` before doing anything else in a new session. This
file is about STATE and NEXT STEPS; `RESULTS.md` is about the actual
numbers/findings. Don't duplicate content between them -- update both if
something changes.

## Where things stand right now

- Repo scaffold, model-loading harness (`mlip_audit/models.py`), and
  Test 3 (`mlip_audit/test3_dimer.py`) are built and working.
- MACE-OFF23-small's Test 3 scan is DONE and the acceptance criterion
  (spurious minimum deeper than physical) is MET -- see `RESULTS.md`
  section 2. Don't re-derive this; it's settled.
- ANI-2x's Test 3 scan is DONE but its result is FLAGGED, not settled --
  see `RESULTS.md` section 3. It shows the same qualitative
  "spurious-minimum" flag as MACE-OFF23, which contradicts the paper's
  description of ANI-2x as well-behaved. **If you pick this back up, this
  is probably the single highest-value thing to resolve** -- see "Next
  step: investigate the ANI-2x result" below.
- UMA-S: environment is ready (`.venv-uma`, see below) but NOTHING has
  been run yet -- no charge/spin test, no Test 3 scan. This is the most
  concrete unstarted piece of work.

## Two environments -- do not `pip install` both stacks into one venv

`mace-torch` hard-pins `e3nn==0.4.4`; `fairchem-core` (needed for UMA-S)
needs `e3nn>=0.5`. They cannot coexist in one Python environment. This
repo has two local venvs already set up from this session:

- `C:\Users\srika\Documents\mlip-audit\.venv` -- ANI-2x + MACE-OFF23-small
  stack (`requirements-ani-mace.txt`). Used for both results in
  `RESULTS.md`. Activate: `source .venv/Scripts/activate` (git-bash) or
  `.venv\Scripts\Activate.ps1` (PowerShell).
- `C:\Users\srika\Documents\mlip-audit\.venv-uma` -- UMA-S stack
  (`requirements-uma.txt`). Created and installed this session
  (`fairchem-core==2.22.0` confirmed importable), but no UMA code has
  actually been run in it yet. Activate the same way, substituting
  `.venv-uma`.

Both venvs are gitignored (`.venv*/` pattern) -- they will NOT exist in a
fresh clone. Recreate with:
```bash
python -m venv .venv          && source .venv/Scripts/activate      && bash setup.sh ani-mace
python -m venv .venv-uma      && source .venv-uma/Scripts/activate  && bash setup.sh uma
```
(On this machine, use
`/c/Users/srika/AppData/Local/Programs/Python/Python311/python.exe -m venv ...`
specifically -- the default `python`/`py` on PATH may resolve to a broken
or free-threaded build; Python 3.11.9 at that path is known-good. See
`RESULTS.md` section 4 for exactly what versions ended up installed.)

## Immediate next steps, in priority order

1. **Run the real UMA-S charge/spin test.**
   ```bash
   cd /c/Users/srika/Documents/mlip-audit && source .venv-uma/Scripts/activate
   python -m pytest tests/test_charge_spin.py -v
   ```
   A cached HuggingFace token already exists on this machine
   (`huggingface_hub.get_token()` found one during this session) -- if
   this session's login has expired or you're on a different machine, run
   `hf auth login` first and make sure the gated UMA model license has
   been accepted on huggingface.co for that account. If
   `test_charge_reaches_model` fails (not skips) -- i.e. it loads UMA but
   the two charge states give near-identical energies -- STOP and debug
   the `FAIRChemCalculator`/`atoms.info` wiring in `mlip_audit/models.py`
   before trusting anything else UMA-related; that was called out as
   critical in the original project brief.

2. **Run UMA-S's Test 3 scan** (same command pattern as the other two):
   ```bash
   python -m mlip_audit.test3_dimer --model uma-s-1p1 --device cpu --no-resume --verbose
   ```
   UMA-S is heavier than ANI-2x/MACE-OFF23-small; on CPU this may be slow
   -- consider `--device cuda` on Colab, or reducing `--n-restarts` (default
   5) for a faster first pass if it's impractically slow on CPU. Then:
   ```bash
   python -m mlip_audit.plotting --csv results/test3_dimer/*.csv
   ```
   to get the combined plot and per-model spurious-minimum check. Record
   the result in `RESULTS.md` the same way MACE-OFF23/ANI-2x are recorded
   (a new numbered section), whatever it shows -- don't just fold it in
   silently.

3. **Investigate the ANI-2x result** (`RESULTS.md` section 3). Concretely:
   - Re-run ANI-2x's short-range points (say 0.2-1.0 A) with
     `--n-restarts 1` (i.e. warm-start only, no random reorientation) and
     see if the deep minimum still appears. If it disappears, the
     multi-start mechanism itself is implicated (or at least, is required
     to reproduce it -- doesn't necessarily mean it's wrong, but narrows
     the question).
   - Look at the actual relaxed geometry at ANI-2x's 0.6 A point (it's in
     `results/checkpoints/ani2x_dimer_scan.extxyz` locally -- NOT committed,
     regenerate by re-running if needed) and sanity-check it's a real
     structure (no NaN positions, no atoms on top of each other in a way
     that's clearly a numerical degenerate case) rather than an artifact
     of the perturbed-restart geometry construction.
   - Consider whether `torchani`'s ASE calculator handles extreme
     close-contact geometries reliably at all, independent of this test.

4. **Tests 1, 2, 4** -- out of scope for this session by explicit
   instruction; don't start these without being asked.

5. **Try the notebook on actual Colab** at some point -- it's never been
   run there, only reasoned about. The Part A / Part B split (for the
   two-environment issue) in particular is untested.

## Things a fresh session might get wrong if it doesn't read this file

- Don't be surprised that `test3_dimer.py` uses a harmonic RESTRAINT
  (`mlip_audit/restraints.py`) instead of `ase.constraints.FixBondLength`
  as the original project brief literally specified -- this was a
  deliberate, debugged, and disclosed deviation. See `RESULTS.md`
  "Debugging trail" and the module docstring in `test3_dimer.py` before
  "fixing" it back.
- Don't be surprised by the multi-start restarts
  (`DIMER_N_RESTARTS=5` in `mlip_audit/config.py`) -- also deliberate, also
  in the debugging trail. A single LBFGS chain was empirically shown to
  produce false negatives on this specific landscape.
- The `oo_distance_actual_ang` CSV column can differ meaningfully from
  `oo_distance_target_ang` at short range (the restraint is soft, not a
  hard constraint) -- this is expected and itself diagnostic, not a bug.
  See the -8792 eV point's caveat in `RESULTS.md` section 2 for a concrete
  example of why this matters when citing a specific number.
- `mlip_audit/plotting.py::check_spurious_minimum` compares the
  whole-curve global minimum against a minimum computed over a NARROW
  2.5-3.5 A window (`PHYSICAL_WINDOW_ANG`), not "everything >= 1.0 A" --
  this was a real bug that was fixed mid-session (see `RESULTS.md`
  Debugging trail, last paragraph). If you see `physical_min_distance`
  landing outside ~2.7-3.1 A for some model, something is wrong -- don't
  just trust the number.
