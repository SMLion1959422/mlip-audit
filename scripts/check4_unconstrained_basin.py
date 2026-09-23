"""Diagnostic check 4: is the broken-geometry region REACHABLE by plain,
UNCONSTRAINED energy minimization from a mildly clashed but structurally
intact starting point? This tests operational relevance directly: a real
docking/minimization workflow has no O-O restraint holding it in place --
it just follows the gradient. If starting from a valid geometry at O-O=1.8
or 2.2 A and minimizing with NO constraint/restraint at all leads to
collapse, the spurious-minimum failure mode is real and dangerous in
practice, regardless of whether our restrained SCAN could locate a valid
example of it on-lattice.
"""
import sys
sys.path.insert(0, "/c/Users/srika/Documents/mlip-audit")
import csv as csv_mod
from ase.io import read
from ase.optimize import LBFGS

from mlip_audit.test3_dimer import _target_distances
from mlip_audit.config import DIMER_SCAN_START_ANG, DIMER_SCAN_END_ANG, DIMER_SCAN_STEP_ANG
from mlip_audit.geometry import get_oo_distance, check_dimer_geometry
from mlip_audit.models import get_calc, prepare_atoms_for_model

STARTING_TARGETS = [2.9, 2.2, 1.8]
FMAX = 0.05
MAX_STEPS = 500

all_targets = _target_distances(DIMER_SCAN_START_ANG, DIMER_SCAN_END_ANG, DIMER_SCAN_STEP_ANG)
target_to_index = {round(t, 4): i for i, t in enumerate(all_targets)}

CHECKPOINTS = {
    "mace-off23-small": "results/checkpoints/mace-off23-small_dimer_scan.extxyz",
    "ani2x": "results/checkpoints/ani2x_dimer_scan.extxyz",
}

PHYSICAL_MIN_OO = 2.9  # for classifying "returned to physical" vs "collapsed"

all_rows = []

for model, path in CHECKPOINTS.items():
    print(f"\n{'='*70}\n{model}\n{'='*70}")
    calc = get_calc(model, device="cpu")
    frames = read(path, index=":")

    for start_target in STARTING_TARGETS:
        idx = target_to_index[start_target]
        atoms = frames[idx].copy()
        start_geom = check_dimer_geometry(atoms)
        start_oo = get_oo_distance(atoms)
        print(f"\n-- start target={start_target} A (checkpoint actual O-O={start_oo:.4f} A, "
              f"geometry_valid={start_geom.is_valid}) --")

        # Fully unconstrained: no FixBondLength, no restraint. Just the
        # bare model, every DOF free including O-O itself.
        atoms.constraints = []
        prepare_atoms_for_model(atoms, model, charge=0, spin=1)
        atoms.calc = calc

        opt = LBFGS(atoms, logfile=None)
        converged = opt.run(fmax=FMAX, steps=MAX_STEPS)
        n_steps = opt.nsteps

        final_oo = get_oo_distance(atoms)
        final_energy = float(atoms.get_potential_energy())
        final_geom = check_dimer_geometry(atoms)

        # Classification: did it return to (near) the physical minimum, or
        # end up somewhere else (short O-O and/or broken geometry)?
        returned_to_physical = abs(final_oo - PHYSICAL_MIN_OO) < 0.3 and final_geom.is_valid
        outcome = "RETURNED TO PHYSICAL MINIMUM" if returned_to_physical else (
            "COLLAPSED (geometry invalid)" if not final_geom.is_valid else
            "SETTLED SHORT BUT GEOMETRY-VALID (unexpected third case)"
        )

        print(f"  final O-O = {final_oo:.4f} A   final E = {final_energy:.4f} eV   "
              f"converged={converged}  steps={n_steps}")
        print(f"  O-H distances: {[f'{k}={v:.4f}' for k, v in final_geom.oh_distances.items()]}")
        print(f"  min_OH={final_geom.min_oh_ang:.4f}  max_OH={final_geom.max_oh_ang:.4f}  "
              f"n_proton_transfer={final_geom.n_proton_transfer_flags}  geometry_valid={final_geom.is_valid}")
        print(f"  ==> OUTCOME: {outcome}")

        all_rows.append({
            "model": model, "start_target_ang": start_target, "start_actual_oo_ang": round(start_oo, 4),
            "final_oo_ang": round(final_oo, 4), "final_energy_eV": final_energy,
            "converged": converged, "n_steps": n_steps,
            "min_OH": round(final_geom.min_oh_ang, 4), "max_OH": round(final_geom.max_oh_ang, 4),
            "n_proton_transfer": final_geom.n_proton_transfer_flags,
            "geometry_valid": final_geom.is_valid, "outcome": outcome,
        })

out_path = "results/test3_dimer/unconstrained_basin_check.csv"
with open(out_path, "w", newline="") as f:
    fieldnames = list(all_rows[0].keys())
    w = csv_mod.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_rows)
print(f"\n\nSaved: {out_path}")
