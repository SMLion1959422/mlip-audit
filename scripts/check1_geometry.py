"""Diagnostic check 1: dump relaxed geometries at specific O-O distances
and report O-H distances / H-O-H angles for both monomers, for both
MACE-OFF23-small and ANI-2x. Flags collapsed O-H bonds or possible proton
transfer (H closer to the "wrong" oxygen).

NOTE: this script's own OH_COLLAPSE_THRESHOLD-only flagging predates the
canonical, stricter check that also catches dissociated (too-long) O-H --
see mlip_audit.geometry.check_dimer_geometry / DimerGeometryReport, which
is what the actual pipeline (test3_dimer.py) and RESULTS.md's numbers use.
This script's raw distance/angle DUMPS are still accurate and were the
first evidence that something was wrong with the raw energies; just don't
trust its "OK: no collapsed O-H" verdicts as the final word -- cross-check
against geometry_valid in the CSVs instead.
"""
import sys
sys.path.insert(0, "/c/Users/srika/Documents/mlip-audit")
import numpy as np
from ase.io import read

from mlip_audit.test3_dimer import _target_distances
from mlip_audit.config import DIMER_SCAN_START_ANG, DIMER_SCAN_END_ANG, DIMER_SCAN_STEP_ANG

TARGETS_OF_INTEREST = [0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.9]
OH_COLLAPSE_THRESHOLD = 0.85  # Angstrom

all_targets = _target_distances(DIMER_SCAN_START_ANG, DIMER_SCAN_END_ANG, DIMER_SCAN_STEP_ANG)
target_to_index = {round(t, 4): i for i, t in enumerate(all_targets)}

CHECKPOINTS = {
    "mace-off23-small": "results/checkpoints/mace-off23-small_dimer_scan.extxyz",
    "ani2x": "results/checkpoints/ani2x_dimer_scan.extxyz",
}

def angle_deg(atoms, i_center, i1, i2):
    v1 = atoms.positions[i1] - atoms.positions[i_center]
    v2 = atoms.positions[i2] - atoms.positions[i_center]
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

for model, path in CHECKPOINTS.items():
    print(f"\n{'='*70}\n{model}\n{'='*70}")
    frames = read(path, index=":")
    for target in TARGETS_OF_INTEREST:
        idx = target_to_index.get(round(target, 4))
        if idx is None or idx >= len(frames):
            print(f"  target={target}: NO FRAME (index {idx}, have {len(frames)} frames)")
            continue
        atoms = frames[idx]
        oo = atoms.get_distance(0, 3)

        oh_a1 = atoms.get_distance(0, 1)
        oh_a2 = atoms.get_distance(0, 2)
        oh_b1 = atoms.get_distance(3, 4)
        oh_b2 = atoms.get_distance(3, 5)

        hoh_a = angle_deg(atoms, 0, 1, 2)
        hoh_b = angle_deg(atoms, 3, 4, 5)

        # cross distances: each H to the OTHER monomer's O (proton-transfer check)
        h1_to_ob = atoms.get_distance(1, 3)
        h2_to_ob = atoms.get_distance(2, 3)
        h4_to_oa = atoms.get_distance(4, 0)
        h5_to_oa = atoms.get_distance(5, 0)

        flags = []
        for label, d in [("O_A-H_bridge", oh_a1), ("O_A-H_free", oh_a2),
                          ("O_B-H1", oh_b1), ("O_B-H2", oh_b2)]:
            if d < OH_COLLAPSE_THRESHOLD:
                flags.append(f"COLLAPSED {label}={d:.4f}A")
        if h1_to_ob < oh_a1:
            flags.append(f"H_bridge(A) closer to O_B ({h1_to_ob:.3f}) than its own O_A ({oh_a1:.3f}) -- possible proton transfer")
        if h2_to_ob < oh_a2:
            flags.append(f"H_free(A) closer to O_B ({h2_to_ob:.3f}) than its own O_A ({oh_a2:.3f}) -- possible proton transfer")
        if h4_to_oa < oh_b1:
            flags.append(f"H1(B) closer to O_A ({h4_to_oa:.3f}) than its own O_B ({oh_b1:.3f}) -- possible proton transfer")
        if h5_to_oa < oh_b2:
            flags.append(f"H2(B) closer to O_A ({h5_to_oa:.3f}) than its own O_B ({oh_b2:.3f}) -- possible proton transfer")

        print(f"\n  target={target:.2f} A  (frame idx {idx}, actual O-O={oo:.4f} A)")
        print(f"    O_A-H(bridge)={oh_a1:.4f}  O_A-H(free)={oh_a2:.4f}  H-O_A-H angle={hoh_a:.2f} deg")
        print(f"    O_B-H1={oh_b1:.4f}  O_B-H2={oh_b2:.4f}  H-O_B-H angle={hoh_b:.2f} deg")
        print(f"    cross: H_bridge(A)-O_B={h1_to_ob:.3f}  H_free(A)-O_B={h2_to_ob:.3f}  H1(B)-O_A={h4_to_oa:.3f}  H2(B)-O_A={h5_to_oa:.3f}")
        if flags:
            for f in flags:
                print(f"    !!! FLAG: {f}")
        else:
            print(f"    OK: no collapsed O-H, no proton transfer detected")
