"""Test 3: water dimer O-O potential energy scan.

Protocol: scan the O-O distance of a water dimer over 0.2-7.0 Angstrom in
0.1 A steps. At each step, restrain the O-O distance to the target value
and relax every other degree of freedom with LBFGS (fmax=0.05 eV/A, max
200 steps). Record the model's (restraint-free) energy vs. distance.

Physical expectation: a single minimum near 2.9 A, with energy rising
monotonically and steeply below it. A second minimum below ~1.0 A is a
failure mode: it means minimizing a clashed structure makes the clash
worse instead of the optimizer pushing the atoms apart.

CONSTRAINT MECHANISM -- deviation from the original spec, disclosed:
the project spec called for a hard constraint, ase.constraints.FixBondLength.
We use a soft harmonic distance RESTRAINT instead (mlip_audit.restraints.
HarmonicDistanceRestraint), added to the model's energy only during
optimization. Two things forced this change, in order:

  1. FixBondLength (really ase.constraints.FixBondLengths under the hood)
     projects forces onto the constraint manifold with a SHAKE/RATTLE-style
     fixed-point iteration. At the huge force magnitudes present near an
     atomic clash (~1e4-1e5 eV/A), that iteration either raises
     RuntimeError("Did not converge"), or -- once its tolerance is loosened
     enough to stop raising -- lets LBFGS diverge into multi-million-eV
     nonphysical energies instead of finding a real minimum. Neither is
     usable, and the second failure mode looks like "no spurious minimum
     found," which would have been a WRONG conclusion about the model
     caused by an optimizer artifact, not a real finding.
  2. Checking the actual methodology in Ranasinghe et al. 2025
     (arXiv:2503.11537, Section 2.2.3) confirms their ML-potential water
     dimer scans were not done with a hard constraint either: "distance
     restraints with a strong force constant of 10 GJ/mol/nm^2" via OpenMM,
     re-optimizing every 0.01 nm -- i.e. exactly the soft-restraint,
     sequential/warm-started scan this module implements.
Switching to the restraint resolved both numerical failure modes.

MULTI-START -- a second thing this test needed to be honest, also found by
debugging a non-reproduction: at short O-O distance, MACE-OFF23-small's
restrained-relaxation landscape turned out to be multi-basin/rugged. A
single warm-started LBFGS chain reproducibly converges to SOME basin, but
which one (and how deep) depends sensitively on the exact path taken --
different starting orientations at the same target distance were observed
to converge to energies differing by THOUSANDS of eV (e.g. -4153 eV from
one warm-started chain vs. below -13,000 eV from a randomly reoriented
start at the same target distance). A single local-optimizer chain
therefore cannot reliably answer "does a spurious minimum exist here" --
it can just as easily land in a shallow basin and produce a false
negative. So at every scan point we run LBFGS from N_RESTARTS starting
geometries (the warm-started continuation, plus N_RESTARTS-1 randomly
reoriented alternatives via geometry.random_perturbed_water_dimer, which
preserves the O-O distance exactly while randomizing each monomer's
orientation about its own oxygen) and keep the lowest-energy CONVERGED
result (falling back to the lowest energy found at all if none converged).
That winning geometry is also what gets carried forward as the next
point's warm start. This is standard basin-hopping-style practice for a
landscape already shown to be rugged, not an attempt to cherry-pick a
result -- see README.md for the debugging trail that motivated it.

Because the restraint is soft (not infinitely stiff), the ACTUAL converged
O-O distance can deviate from the target at the most extreme clash
distances, where the model's own repulsive force exceeds the restraint's
restoring force. Both are recorded (oo_distance_target_ang and
oo_distance_actual_ang) -- this deviation is itself diagnostic of how hard
a model's short-range repulsive wall is.

Scan DIRECTION: points are computed in DESCENDING distance order
(7.0 -> 0.2), warm-starting each step's initial geometry from the previous
(larger-distance) step's relaxed structure via geometry.set_oo_distance().
This matches the paper's "re-optimize every 0.01 nm" sequential scan and is
what actually exposes the failure mode: it tests whether, as the system is
pushed together from the physical minimum, the optimizer finds a spurious
low-energy clash instead of the energy rising monotonically. The CSV rows
are written in this order; sort by oo_distance_target_ang for analysis
(plotting.py does this already).

Resumable: each completed point is flushed to CSV immediately, and the
relaxed geometry is appended to a checkpoint trajectory file. Re-running
this script skips distances already present in the CSV and warm-starts
from the last checkpointed geometry, so a killed Colab session loses at
most the point that was in flight.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.mixing import SumCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.optimize import LBFGS

from mlip_audit.config import (
    CHECKPOINT_DIR,
    DIMER_CHARGE,
    DIMER_N_RESTARTS,
    DIMER_RANDOM_SEED,
    DIMER_SCAN_DECIMALS,
    DIMER_SCAN_END_ANG,
    DIMER_SCAN_START_ANG,
    DIMER_SCAN_STEP_ANG,
    DIMER_SPIN,
    LBFGS_FMAX,
    LBFGS_MAX_STEPS,
    RESULTS_DIR,
)
from mlip_audit.geometry import (
    O_INDEX_A,
    O_INDEX_B,
    build_water_dimer,
    check_dimer_geometry,
    get_oo_distance,
    random_perturbed_water_dimer,
    set_oo_distance,
)
from mlip_audit.models import get_calc, prepare_atoms_for_model
from mlip_audit.restraints import RESTRAINT_K_EV_PER_ANG2, HarmonicDistanceRestraint

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "oo_distance_target_ang",
    "oo_distance_actual_ang",
    "energy_eV",
    "converged",
    "final_max_force_eV_per_ang",
    "n_lbfgs_steps",
    "wall_time_s",
    "status",
    "winning_start",
    "n_converged_of_n_restarts",
    "min_oh_ang",
    "max_oh_ang",
    "n_proton_transfer_flags",
    "geometry_valid",
]


def _target_distances(
    start: float, end: float, step: float, decimals: int = DIMER_SCAN_DECIMALS
) -> list[float]:
    """List of O-O distances to visit, in the order they will be computed
    (descending if end < start, ascending otherwise)."""
    n = int(round(abs(start - end) / step)) + 1
    sign = -1.0 if end < start else 1.0
    return [round(start + sign * i * step, decimals) for i in range(n)]


def _csv_path(model: str, out_dir: Path) -> Path:
    return out_dir / f"{model}.csv"


def _checkpoint_path(model: str, checkpoint_dir: Path) -> Path:
    return checkpoint_dir / f"{model}_dimer_scan.extxyz"


def _load_completed_distances(csv_path: Path, decimals: int) -> set[float]:
    if not csv_path.exists():
        return set()
    completed = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok":
                completed.add(round(float(row["oo_distance_target_ang"]), decimals))
    return completed


def _append_csv_row(csv_path: Path, row: dict) -> None:
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _load_warm_start(
    checkpoint_path: Path, first_target: float, charge: int, spin: int
) -> Atoms:
    """Return the starting Atoms for the scan: the last checkpointed
    geometry (rigidly slid to `first_target`) if one exists, else a fresh
    idealized dimer built at `first_target`."""
    if checkpoint_path.exists():
        atoms = ase_read(checkpoint_path, index=-1)
        atoms.constraints = []
        set_oo_distance(atoms, first_target)
        logger.info(
            "Resuming from checkpoint %s (last saved O-O=%.3f A), warm-started to %.3f A",
            checkpoint_path,
            get_oo_distance(atoms),
            first_target,
        )
        return atoms
    atoms = build_water_dimer(first_target)
    atoms.info.update({"charge": charge, "spin": spin})
    return atoms


def _relax_candidate(
    atoms: Atoms,
    calc,
    model: str,
    target: float,
    charge: int,
    spin: int,
    fmax: float,
    max_steps: int,
) -> tuple[float, bool, int, float, Atoms]:
    """Restrained-relax one candidate starting geometry (mutated in place).

    Returns (model_only_energy_eV, converged, n_lbfgs_steps,
    final_max_force_eV_per_ang, atoms).

    final_max_force is read from the RESTRAINED system immediately after
    opt.run() returns, before switching to the model-only calculator --
    recomputing it later, after reloading/re-attaching calculators, was
    empirically found (during this project's ANI-2x/MACE diagnostic) to
    sometimes disagree with what LBFGS actually converged against on a
    rugged part of the PES, which had silently mislabeled at least one
    MACE-OFF23-small point as converged when it was not. Reading it here,
    in-place, avoids that class of bug by construction.
    """
    prepare_atoms_for_model(atoms, model, charge=charge, spin=spin)
    restraint = HarmonicDistanceRestraint(O_INDEX_A, O_INDEX_B, RESTRAINT_K_EV_PER_ANG2, target)
    atoms.calc = SumCalculator([calc, restraint])

    opt = LBFGS(atoms, logfile=None)
    converged = opt.run(fmax=fmax, steps=max_steps)
    final_max_force = float(np.abs(atoms.get_forces()).max())

    # Report the MODEL's own energy at the restrained-converged geometry
    # (excludes the restraint bias term), matching what Ranasinghe et al.
    # plot as the potential energy curve.
    atoms.calc = calc
    energy_eV = float(atoms.get_potential_energy())
    return energy_eV, converged, opt.nsteps, final_max_force, atoms


def run_scan(
    model: str,
    device: str | None = None,
    start: float = DIMER_SCAN_START_ANG,
    end: float = DIMER_SCAN_END_ANG,
    step: float = DIMER_SCAN_STEP_ANG,
    fmax: float = LBFGS_FMAX,
    max_steps: int = LBFGS_MAX_STEPS,
    charge: int = DIMER_CHARGE,
    spin: int = DIMER_SPIN,
    n_restarts: int = DIMER_N_RESTARTS,
    seed: int = DIMER_RANDOM_SEED,
    out_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
) -> Path:
    """Run (or resume) the Test 3 O-O scan for one model. Returns the CSV path."""
    out_dir = Path(out_dir) if out_dir else RESULTS_DIR / "test3_dimer"
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _csv_path(model, out_dir)
    checkpoint_path = _checkpoint_path(model, checkpoint_dir)

    if not resume:
        # --no-resume means start over: wipe any prior CSV/checkpoint for
        # this model rather than silently mixing old and new points/frames.
        for p in (csv_path, checkpoint_path):
            if p.exists():
                logger.info("--no-resume: removing existing %s", p)
                p.unlink()

    all_targets = _target_distances(start, end, step)
    completed = _load_completed_distances(csv_path, DIMER_SCAN_DECIMALS) if resume else set()
    remaining = [d for d in all_targets if d not in completed]

    if not remaining:
        logger.info("All %d points already completed for %s -> %s", len(all_targets), model, csv_path)
        return csv_path

    logger.info(
        "%s: %d/%d points remaining (resume=%s, n_restarts=%d). Loading calculator on device=%s...",
        model,
        len(remaining),
        len(all_targets),
        resume,
        n_restarts,
        device or "(auto)",
    )
    calc = get_calc(model, device=device)
    rng = np.random.default_rng(seed)

    atoms = _load_warm_start(checkpoint_path, remaining[0], charge, spin)

    for target in remaining:
        t0 = time.time()
        try:
            set_oo_distance(atoms, target)
            candidates: list[tuple[str, Atoms]] = [("warm_start", atoms.copy())]
            for i in range(n_restarts - 1):
                candidates.append(
                    (f"restart_{i}", random_perturbed_water_dimer(target, rng))
                )

            best = None  # (energy, converged, n_steps, max_force, atoms, label, geom_report)
            n_converged = 0
            for label, cand_atoms in candidates:
                e, conv, n_steps, max_force, relaxed = _relax_candidate(
                    cand_atoms, calc, model, target, charge, spin, fmax, max_steps
                )
                geom = check_dimer_geometry(relaxed)
                n_converged += int(conv)
                if best is None:
                    best = (e, conv, n_steps, max_force, relaxed, label, geom)
                else:
                    _, best_conv, _, _, _, _, best_geom = best
                    # Selection priority: (1) converged beats non-converged,
                    # (2) geometry-VALID beats invalid (an invalid-geometry
                    # candidate's energy is not a meaningful "dimer energy"
                    # and must not be allowed to win on energy alone --
                    # see RESULTS.md's diagnostic trail for why this matters),
                    # (3) only then, lower energy wins.
                    if conv != best_conv:
                        better = conv and not best_conv
                    elif geom.is_valid != best_geom.is_valid:
                        better = geom.is_valid and not best_geom.is_valid
                    else:
                        better = e < best[0]
                    if better:
                        best = (e, conv, n_steps, max_force, relaxed, label, geom)

            energy_eV, converged, n_steps, final_max_force, atoms, winning_start, geom_report = best
            actual_distance = get_oo_distance(atoms)
            status = "ok"
        except Exception as exc:
            logger.exception("Point O-O=%.3f A FAILED for model=%s", target, model)
            energy_eV = float("nan")
            actual_distance = float("nan")
            converged = False
            n_steps = -1
            final_max_force = float("nan")
            status = "error"
            winning_start = ""
            n_converged = 0
            geom_report = None
            failure = exc

        wall_time_s = time.time() - t0
        row = {
            "oo_distance_target_ang": target,
            "oo_distance_actual_ang": actual_distance,
            "energy_eV": energy_eV,
            "converged": converged,
            "final_max_force_eV_per_ang": round(final_max_force, 6) if final_max_force == final_max_force else final_max_force,
            "n_lbfgs_steps": n_steps,
            "wall_time_s": round(wall_time_s, 3),
            "status": status,
            "winning_start": winning_start,
            "n_converged_of_n_restarts": f"{n_converged}/{n_restarts}",
            "min_oh_ang": round(geom_report.min_oh_ang, 4) if geom_report else float("nan"),
            "max_oh_ang": round(geom_report.max_oh_ang, 4) if geom_report else float("nan"),
            "n_proton_transfer_flags": geom_report.n_proton_transfer_flags if geom_report else -1,
            "geometry_valid": geom_report.is_valid if geom_report else False,
        }
        _append_csv_row(csv_path, row)

        if status == "ok":
            # Persist charge/spin in the checkpoint too (extxyz round-trips
            # scalar atoms.info entries), so a resumed run doesn't need to
            # re-derive them.
            ase_write(checkpoint_path, atoms, append=True)
            logger.info(
                "O-O target=%6.3f A actual=%6.3f A  E=%.6f eV  converged=%s (%s, %d/%d)  "
                "steps=%d  (%.1fs)",
                target,
                actual_distance,
                energy_eV,
                converged,
                winning_start,
                n_converged,
                n_restarts,
                n_steps,
                wall_time_s,
            )
        else:
            # Do not silently continue past a failure with a fabricated
            # geometry checkpoint -- stop the scan so the failure is
            # investigated rather than propagated to every later point.
            logger.error(
                "Stopping scan for %s at O-O=%.3f A due to the error above. "
                "Re-run to resume once fixed.",
                model,
                target,
            )
            raise failure

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["ani2x", "mace-off23-small", "uma-s-1p1"])
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect)")
    parser.add_argument("--start", type=float, default=DIMER_SCAN_START_ANG)
    parser.add_argument("--end", type=float, default=DIMER_SCAN_END_ANG)
    parser.add_argument("--step", type=float, default=DIMER_SCAN_STEP_ANG)
    parser.add_argument("--fmax", type=float, default=LBFGS_FMAX)
    parser.add_argument("--max-steps", type=int, default=LBFGS_MAX_STEPS)
    parser.add_argument("--charge", type=int, default=DIMER_CHARGE)
    parser.add_argument("--spin", type=int, default=DIMER_SPIN)
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=DIMER_N_RESTARTS,
        help="Multi-start restarts per scan point (warm start + N-1 random reorientations); see module docstring.",
    )
    parser.add_argument("--seed", type=int, default=DIMER_RANDOM_SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    csv_path = run_scan(
        model=args.model,
        device=args.device,
        start=args.start,
        end=args.end,
        step=args.step,
        fmax=args.fmax,
        max_steps=args.max_steps,
        charge=args.charge,
        spin=args.spin,
        n_restarts=args.n_restarts,
        seed=args.seed,
        out_dir=args.out_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
    )
    logger.info("Done. Results: %s", csv_path)


if __name__ == "__main__":
    main()
