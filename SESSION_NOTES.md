# Session notes -- resume from here

Read this + `RESULTS.md` before doing anything else in a new session. This
file is about STATE and NEXT STEPS; `RESULTS.md` is about the actual
numbers/findings. Don't duplicate content between them -- update both if
something changes.

## Where things stand right now

- Repo scaffold, model-loading harness (`mlip_audit/models.py`), and
  Test 3 (`mlip_audit/test3_dimer.py`) are built and working.
- MACE-OFF23-small's and ANI-2x's Test 3 scans are DONE, and both went
  through a rigorous geometry-validity + convergence diagnostic (three
  checks: geometry dump, force-convergence, constraint-mechanism
  sensitivity) requested mid-session. **The acceptance criterion (a valid,
  geometry-intact spurious minimum deeper than physical) is currently NOT
  MET for either model** -- see `RESULTS.md` Section 1. An EARLIER version
  of `RESULTS.md` claimed it WAS met for MACE-OFF23-small; that was wrong
  and has been explicitly retracted in the current version. If you find
  any other document, comment, or cached belief claiming "MACE acceptance
  criterion MET," it is stale -- `RESULTS.md`'s current text is the
  source of truth.
- UMA-S: environment is ready (`.venv-uma`, see below) but NOTHING has
  been run yet -- no charge/spin test, no Test 3 scan. This is the most
  concrete unstarted piece of work, AND it must go through the same
  geometry/convergence gating from the start (don't repeat the mistake of
  reading off a raw minimum) -- see "Immediate next steps" below.

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
`RESULTS.md` Section 5 for exactly what versions ended up installed.)

## Immediate next steps, in priority order

1. **Run the real UMA-S charge/spin test.**
   ```bash
   cd /c/Users/srika/Documents/mlip-audit && source .venv-uma/Scripts/activate
   python -m pytest tests/test_charge_spin.py -v
   ```
   A cached HuggingFace token already exists on this machine -- if this
   session's login has expired or you're on a different machine, run
   `hf auth login` first and make sure the gated UMA model license has
   been accepted on huggingface.co for that account. If
   `test_charge_reaches_model` fails (not skips) STOP and debug the
   `FAIRChemCalculator`/`atoms.info` wiring in `mlip_audit/models.py`
   before trusting anything else UMA-related.

2. **Run UMA-S's Test 3 scan**, then its plot/analysis:
   ```bash
   python -m mlip_audit.test3_dimer --model uma-s-1p1 --device cpu --no-resume --verbose
   python -m mlip_audit.plotting --csv results/test3_dimer/*.csv
   ```
   The pipeline already includes geometry-validity gating and correct
   live convergence capture (both were bugs fixed THIS session -- see
   `RESULTS.md` Section 4), so UMA-S's numbers should come out trustworthy
   without extra work. Still: look at the `geometry_valid` column and the
   raw-vs-valid depth numbers before writing anything about UMA-S's
   result. Record it in `RESULTS.md` as a new section, whatever it shows.

3. **Close the three open gaps listed in `RESULTS.md`'s "What would still
   need to happen" section** -- in priority order, these are: (a) build
   the actual literature starting geometry ("Smith stationary point 1"
   from Ranasinghe et al., not the hand-built idealized guess this session
   used) and re-run, since the diagnostic work showed the short-range
   landscape is history-dependent enough that this could matter a lot;
   (b) finer-than-0.1-A sampling right at the boundary where geometry
   validity breaks down, in case a valid deeper minimum lives in a window
   narrower than the current grid; (c) if at all obtainable, the paper's
   own SI structures, to check whether their published spurious minimum
   is itself geometry-valid by the same check used here.

4. **Tests 1, 2, 4** -- out of scope for this session by explicit
   instruction; don't start these without being asked.

5. **Try the notebook on actual Colab** at some point -- it's never been
   run there, only reasoned about. The Part A / Part B split (for the
   two-environment issue) in particular is untested.

## Things a fresh session might get wrong if it doesn't read this file

- Don't be surprised that `test3_dimer.py` uses a harmonic RESTRAINT
  (`mlip_audit/restraints.py`) instead of `ase.constraints.FixBondLength`
  as the original project brief literally specified -- this was a
  deliberate, debugged, and disclosed deviation. See `RESULTS.md` and the
  module docstring in `test3_dimer.py` before "fixing" it back. A hard
  constraint was re-tested later in the session anyway (Check 3) and
  didn't change the bottom-line conclusion.
- Don't be surprised by the multi-start restarts
  (`DIMER_N_RESTARTS=5` in `mlip_audit/config.py`) -- also deliberate.
  A single LBFGS chain was empirically shown to be sensitive to which
  basin it lands in on this landscape.
- **Do not report a "spurious minimum" number without checking
  `geometry_valid` first.** This is the single biggest lesson of this
  session. `mlip_audit.geometry.check_dimer_geometry` is the canonical
  check (catches both collapsed AND dissociated O-H -- an earlier,
  narrower version of this check that only caught collapse was itself a
  bug found mid-session). `mlip_audit.plotting.check_spurious_minimum`
  reports BOTH a "raw" and a "valid" depth for exactly this reason --
  always prefer "valid," and treat a large raw/valid gap as a red flag
  about the raw number, not as extra evidence.
- The `oo_distance_actual_ang` CSV column can differ meaningfully from
  `oo_distance_target_ang` at short range (the restraint is soft, not a
  hard constraint) -- expected and itself diagnostic, not a bug.
- If you see a point marked `converged=True` that looks physically
  implausible, don't assume the label is right -- re-derive `final_max_force_eV_per_ang`
  yourself and check it's actually <=0.05. This exact class of bug (a
  point silently mislabeled converged) was found and fixed this session;
  the fix (`_relax_candidate` in `test3_dimer.py`) is believed correct,
  but treat any single surprising point with residual suspicion.
- `mlip_audit/plotting.py::check_spurious_minimum` compares the
  whole-curve global minimum against a minimum computed over a NARROW
  2.5-3.5 A window (`PHYSICAL_WINDOW_ANG`), not "everything >= 1.0 A" --
  a wider window double-counts already-anomalous points as "physical."
  If you see `physical_min_distance` landing outside ~2.7-3.1 A for some
  model, something is wrong -- don't just trust the number.
