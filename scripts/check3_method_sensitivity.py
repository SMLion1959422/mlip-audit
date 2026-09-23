"""Diagnostic check 3 (v2, using the canonical check_dimer_geometry).
Starting from the verified-clean 1.5A geometry, run a SINGLE warm-started
chain (no multi-start) down to 0.2A using (a) a hard FixBondLengths
constraint and (b) the harmonic restraint, for both models."""
import sys
sys.path.insert(0, "/c/Users/srika/Documents/mlip-audit")
import csv as csv_mod
import numpy as np
from ase.io import read
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixBondLengths
from ase.optimize import LBFGS

from mlip_audit.test3_dimer import _target_distances
from mlip_audit.config import DIMER_SCAN_START_ANG, DIMER_SCAN_END_ANG, DIMER_SCAN_STEP_ANG
from mlip_audit.geometry import O_INDEX_A, O_INDEX_B, set_oo_distance, get_oo_distance, check_dimer_geometry
from mlip_audit.models import get_calc, prepare_atoms_for_model
from mlip_audit.restraints import RESTRAINT_K_EV_PER_ANG2, HarmonicDistanceRestraint

FIX_BOND_LENGTH_TOLERANCE = 1e-6

all_targets = _target_distances(DIMER_SCAN_START_ANG, DIMER_SCAN_END_ANG, DIMER_SCAN_STEP_ANG)
target_to_index = {round(t, 4): i for i, t in enumerate(all_targets)}
sub15_targets = [t for t in all_targets if t <= 1.4 + 1e-9]

CHECKPOINTS = {
    "mace-off23-small": "results/checkpoints/mace-off23-small_dimer_scan.extxyz",
    "ani2x": "results/checkpoints/ani2x_dimer_scan.extxyz",
}

def run_chain(model, calc, start_atoms, method, fmax=0.05, max_steps=200):
    atoms = start_atoms.copy()
    atoms.info.update({"charge": 0, "spin": 1})
    rows = []
    for target in sub15_targets:
        set_oo_distance(atoms, target)
        prepare_atoms_for_model(atoms, model, charge=0, spin=1)
        try:
            if method == "hard":
                atoms.constraints = [FixBondLengths([(O_INDEX_A, O_INDEX_B)], tolerance=FIX_BOND_LENGTH_TOLERANCE)]
                atoms.calc = calc
            else:
                atoms.constraints = []
                restraint = HarmonicDistanceRestraint(O_INDEX_A, O_INDEX_B, RESTRAINT_K_EV_PER_ANG2, target)
                atoms.calc = SumCalculator([calc, restraint])

            opt = LBFGS(atoms, logfile=None)
            converged = opt.run(fmax=fmax, steps=max_steps)
            max_force = float(np.abs(atoms.get_forces()).max())

            atoms.constraints = []
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            actual_oo = get_oo_distance(atoms)
            geom = check_dimer_geometry(atoms)
            status = "ok"
        except Exception as exc:
            energy, actual_oo, geom, converged, max_force = float("nan"), float("nan"), None, False, float("nan")
            status = f"error: {exc}"

        row = {
            "method": method, "target": target,
            "actual_oo": round(actual_oo,4) if actual_oo==actual_oo else actual_oo,
            "energy_eV": energy, "converged": converged, "max_force": round(max_force,5) if max_force==max_force else max_force,
            "min_OH": round(geom.min_oh_ang,4) if geom else float("nan"),
            "max_OH": round(geom.max_oh_ang,4) if geom else float("nan"),
            "n_proton_transfer": geom.n_proton_transfer_flags if geom else -1,
            "geometry_valid": geom.is_valid if geom else False,
            "status": status,
        }
        rows.append(row)
        print(f"  [{method:9s}] target={target:.2f}  actual_OO={row['actual_oo']}  E={energy:.4f}  conv={converged}  max_f={row['max_force']}  min_OH={row['min_OH']}  max_OH={row['max_OH']}  valid={row['geometry_valid']}  status={status}")
        if status != "ok":
            break
    return rows

for model, path in CHECKPOINTS.items():
    print(f"\n{'='*70}\n{model}\n{'='*70}")
    calc = get_calc(model, device="cpu")
    frames = read(path, index=":")
    start_atoms = frames[target_to_index[1.5]]
    geom0 = check_dimer_geometry(start_atoms)
    print(f"Starting geometry (1.5A checkpoint): min_OH={geom0.min_oh_ang:.4f}, max_OH={geom0.max_oh_ang:.4f}, valid={geom0.is_valid}")

    all_rows = []
    print("\n-- HARD CONSTRAINT --")
    all_rows += run_chain(model, calc, start_atoms, "hard")
    print("\n-- RESTRAINT --")
    all_rows += run_chain(model, calc, start_atoms, "restraint")

    out_path = f"results/test3_dimer/{model}_method_sensitivity.csv"
    with open(out_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=["method","target","actual_oo","energy_eV","converged","max_force","min_OH","max_OH","n_proton_transfer","geometry_valid","status"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {out_path}")
