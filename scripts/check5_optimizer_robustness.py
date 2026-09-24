"""Diagnostic check 5: robustness of the "no reachable spurious minimum"
finding to optimizer settings, on the Smith SP1 reconstruction geometry.

Sweeps fmax x optimizer x restraint stiffness (2 x 2 x 3 = 12 settings),
each a full single-chain (NO multi-start -- isolating the optimizer/
convergence axis from the already-separately-tested multi-start question,
per the request to bound this cheaply) scan over the paper's range
(4.0 -> 0.2 A, 0.1 A steps), for both models. Reports valid_depth_eV per
(model, setting) using the same geometry-validity + convergence gate as
the rest of this project.

Usage: run for one (model, setting) combination at a time via CLI args,
so combinations can be run/monitored independently.
"""
import argparse
import csv as csv_mod
import time
from pathlib import Path

import numpy as np
from ase.calculators.mixing import SumCalculator
from ase.optimize import LBFGS, FIRE

import sys
sys.path.insert(0, "/c/Users/srika/Documents/mlip-audit")
from mlip_audit.test3_dimer import _target_distances
from mlip_audit.config import DIMER_CHARGE, DIMER_SPIN
from mlip_audit.geometry import O_INDEX_A, O_INDEX_B, set_oo_distance, get_oo_distance, check_dimer_geometry
from mlip_audit.models import get_calc, prepare_atoms_for_model
from mlip_audit.restraints import HarmonicDistanceRestraint
from ase.io import read as ase_read

OPTIMIZERS = {"lbfgs": LBFGS, "fire": FIRE}

# RESTRAINT_K_EV_PER_ANG2 (1036.4) corresponds to k=10 GJ/mol/nm^2; scale
# linearly for other k values (see mlip_audit/restraints.py derivation).
def k_ev_per_ang2(k_gj_mol_nm2: float) -> float:
    return 1036.4 * (k_gj_mol_nm2 / 10.0)

CSV_FIELDS = ["oo_distance_target_ang", "oo_distance_actual_ang", "energy_eV",
              "converged", "final_max_force_eV_per_ang", "n_steps", "status",
              "min_oh_ang", "max_oh_ang", "n_proton_transfer_flags", "geometry_valid"]


def run(model: str, fmax: float, optimizer_name: str, k_gj: float, max_steps: int, out_csv: Path):
    calc = get_calc(model, device="cpu")
    k = k_ev_per_ang2(k_gj)
    OptClass = OPTIMIZERS[optimizer_name]

    start_atoms = ase_read("geometries/smith_sp1_reconstructed.xyz")
    targets = _target_distances(4.0, 0.2, 0.1)

    atoms = start_atoms.copy()
    atoms.info.update({"charge": DIMER_CHARGE, "spin": DIMER_SPIN})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_csv, "w", newline="")
    w = csv_mod.DictWriter(f, fieldnames=CSV_FIELDS)
    w.writeheader()

    for target in targets:
        t0 = time.time()
        try:
            set_oo_distance(atoms, target)
            prepare_atoms_for_model(atoms, model, charge=DIMER_CHARGE, spin=DIMER_SPIN)
            restraint = HarmonicDistanceRestraint(O_INDEX_A, O_INDEX_B, k, target)
            atoms.calc = SumCalculator([calc, restraint])

            opt = OptClass(atoms, logfile=None)
            converged = opt.run(fmax=fmax, steps=max_steps)
            final_max_force = float(np.abs(atoms.get_forces()).max())

            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            actual_oo = get_oo_distance(atoms)
            geom = check_dimer_geometry(atoms)
            status = "ok"
        except Exception as exc:
            energy, actual_oo, geom, converged, final_max_force = float("nan"), float("nan"), None, False, float("nan")
            status = f"error: {exc}"

        row = {
            "oo_distance_target_ang": target, "oo_distance_actual_ang": actual_oo,
            "energy_eV": energy, "converged": converged,
            "final_max_force_eV_per_ang": final_max_force, "n_steps": opt.nsteps if status == "ok" else -1,
            "status": status,
            "min_oh_ang": geom.min_oh_ang if geom else float("nan"),
            "max_oh_ang": geom.max_oh_ang if geom else float("nan"),
            "n_proton_transfer_flags": geom.n_proton_transfer_flags if geom else -1,
            "geometry_valid": geom.is_valid if geom else False,
        }
        w.writerow(row)
        f.flush()
        print(f"  target={target:.2f}  actual={actual_oo:.4f}  E={energy:.4f}  conv={converged}  "
              f"maxf={final_max_force:.5f}  valid={row['geometry_valid']}  ({time.time()-t0:.1f}s)")
        if status != "ok":
            print(f"  STOPPING due to error: {status}")
            break

    f.close()
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mace-off23-small", "ani2x"])
    p.add_argument("--fmax", type=float, required=True)
    p.add_argument("--optimizer", required=True, choices=["lbfgs", "fire"])
    p.add_argument("--k-gj", type=float, required=True)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args.model, args.fmax, args.optimizer, args.k_gj, args.max_steps, args.out)
