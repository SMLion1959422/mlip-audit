"""Water dimer geometry construction for Test 3 (O-O potential energy scan).

Atom ordering convention used throughout this module and by test3_dimer.py:

    index 0: O  (monomer A, hydrogen-bond DONOR)
    index 1: H  (monomer A, bridging H -- points at monomer B's O)
    index 2: H  (monomer A, non-bridging)
    index 3: O  (monomer B, hydrogen-bond ACCEPTOR)
    index 4: H  (monomer B)
    index 5: H  (monomer B)

Indices 0 and 3 (the two oxygens) are the pair constrained by
`ase.constraints.FixBondLength` in the O-O scan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

# Standard gas-phase monomer geometry (used only to build the initial guess;
# every other degree of freedom is relaxed by the optimizer in test3_dimer).
WATER_OH_LENGTH_ANG = 0.9572
WATER_HOH_ANGLE_DEG = 104.52

O_INDEX_A = 0  # donor oxygen
H_INDEX_A_BRIDGE = 1
H_INDEX_A_FREE = 2
O_INDEX_B = 3  # acceptor oxygen
H_INDEX_B1 = 4
H_INDEX_B2 = 5

MONOMER_B_INDICES = (O_INDEX_B, H_INDEX_B1, H_INDEX_B2)


def build_water_dimer(oo_distance: float) -> Atoms:
    """Build an idealized hydrogen-bonded water dimer at a given O-O distance.

    Monomer A (donor) sits at the origin with one O-H bond pointing along
    +x, roughly toward monomer B's oxygen, approximating a near-linear
    O-H...O hydrogen bond. Monomer B (acceptor) sits at
    (oo_distance, 0, 0) with its bisector pointing along -x (both H's
    splayed symmetrically away from monomer A). This Cs-symmetric geometry
    is only a starting guess -- test3_dimer.py relaxes every degree of
    freedom except the O-O distance itself, so the exact acceptor
    orientation here is not load-bearing.

    At very short oo_distance (the unphysical/clashed end of the scan,
    down to 0.2 A) this geometry will have severely overlapping atoms by
    construction -- that is intentional; it is the starting point the scan
    specification asks for.

    Args:
        oo_distance: target O-O distance in Angstrom.

    Returns:
        An ase.Atoms with 6 atoms in the index order documented in this
        module's docstring.
    """
    r = WATER_OH_LENGTH_ANG
    half_angle = np.radians(WATER_HOH_ANGLE_DEG / 2.0)

    # Monomer A: O at origin, bridging H exactly on +x, free H at the H-O-H
    # angle from it in the xy-plane.
    o_a = np.array([0.0, 0.0, 0.0])
    h_a_bridge = np.array([r, 0.0, 0.0])
    theta = np.radians(WATER_HOH_ANGLE_DEG)
    h_a_free = r * np.array([np.cos(theta), np.sin(theta), 0.0])

    # Monomer B: O on the x-axis at oo_distance, bisector along -x, H's
    # splayed symmetrically about the x-axis (mirrored, i.e. pointing away
    # from monomer A).
    o_b = np.array([oo_distance, 0.0, 0.0])
    h_b1 = o_b + r * np.array([-np.cos(half_angle), np.sin(half_angle), 0.0])
    h_b2 = o_b + r * np.array([-np.cos(half_angle), -np.sin(half_angle), 0.0])

    positions = np.stack([o_a, h_a_bridge, h_a_free, o_b, h_b1, h_b2])
    symbols = ["O", "H", "H", "O", "H", "H"]

    atoms = Atoms(symbols=symbols, positions=positions)
    # Non-periodic molecule; give it a large vacuum-equivalent cell so
    # calculators that assume a cell (e.g. some ASE utilities) don't choke.
    atoms.center(vacuum=20.0)
    atoms.pbc = False
    return atoms


def get_oo_distance(
    atoms: Atoms, o_index_a: int = O_INDEX_A, o_index_b: int = O_INDEX_B
) -> float:
    """Return the current O-O distance (Angstrom) between the two oxygens."""
    return float(np.linalg.norm(atoms.positions[o_index_a] - atoms.positions[o_index_b]))


def set_oo_distance(
    atoms: Atoms,
    target_distance: float,
    mobile_indices: tuple[int, ...] = MONOMER_B_INDICES,
    o_index_a: int = O_INDEX_A,
    o_index_b: int = O_INDEX_B,
) -> Atoms:
    """Rigidly translate monomer B along the current O-O axis to a new distance.

    Used to warm-start each step of the O-O scan from the previous step's
    relaxed geometry: monomer B is slid along the (possibly rotated, since
    the dimer as a whole is free to rotate under FixBondLength) current
    O_A->O_B direction, preserving both monomers' internal geometry and
    monomer B's orientation exactly -- only the separation changes.

    Args:
        atoms: structure to modify (mutated in place).
        target_distance: desired O-O distance in Angstrom.
        mobile_indices: atom indices that make up the rigid body being
            translated (default: monomer B, i.e. O_B + its two H's).
        o_index_a: index of the fixed-side oxygen (monomer A).
        o_index_b: index of the moving-side oxygen (monomer B).

    Returns:
        The same `atoms` object, mutated in place, for chaining.
    """
    pos = atoms.positions
    vec = pos[o_index_b] - pos[o_index_a]
    current_distance = np.linalg.norm(vec)
    if current_distance < 1e-8:
        raise ValueError(
            "O-O distance is ~0; cannot determine a translation direction. "
            "This should not happen starting from build_water_dimer()."
        )
    direction = vec / current_distance
    delta = (target_distance - current_distance) * direction

    mobile = np.array(mobile_indices)
    atoms.positions[mobile] += delta
    return atoms


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random 3D rotation matrix (random axis, random angle)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def random_perturbed_water_dimer(
    oo_distance: float, rng: np.random.Generator
) -> Atoms:
    """Build a water dimer at a given O-O distance with each monomer given
    an independent random 3D orientation about its own oxygen.

    Used for multi-start optimization: at short O-O separations, the
    restrained relaxation landscape can have multiple basins (see
    test3_dimer.py's module docstring) -- a single warm-started chain can
    land in a misleadingly shallow one. Rotating each monomer rigidly about
    its own oxygen preserves the O-O distance exactly while exploring
    different relative H-bond/clash orientations at that separation.

    Args:
        oo_distance: O-O distance in Angstrom (preserved exactly).
        rng: numpy random Generator (caller controls the seed, for
            reproducibility).

    Returns:
        A new ase.Atoms, same 6-atom index convention as build_water_dimer.
    """
    atoms = build_water_dimer(oo_distance)

    o_a = atoms.positions[O_INDEX_A].copy()
    r_a = _random_rotation_matrix(rng)
    for idx in (O_INDEX_A, H_INDEX_A_BRIDGE, H_INDEX_A_FREE):
        atoms.positions[idx] = o_a + r_a @ (atoms.positions[idx] - o_a)

    o_b = atoms.positions[O_INDEX_B].copy()
    r_b = _random_rotation_matrix(rng)
    for idx in MONOMER_B_INDICES:
        atoms.positions[idx] = o_b + r_b @ (atoms.positions[idx] - o_b)

    return atoms


# Real O-H equilibrium bond length is ~0.96-0.98 A. Below OH_COLLAPSE_
# THRESHOLD_ANG we are no longer looking at a chemically intact O-H bond
# (collapsed). Above OH_STRETCH_WARN_ANG we are ALSO no longer looking at
# one (dissociated/fragmented) -- both ends were empirically necessary:
# min-only checking missed cases where EVERY O-H distance was 3-11 A (the
# whole dimer had exploded apart under an extreme restraint target, not
# "collapsed" but just as meaningless as a dimer-PES data point). is_valid
# requires ALL FOUR O-H distances -- not just the smallest -- to fall
# inside [OH_COLLAPSE_THRESHOLD_ANG, OH_STRETCH_WARN_ANG]; a single
# collapsed OR a single dissociated O-H is enough to invalidate the point.
OH_COLLAPSE_THRESHOLD_ANG = 0.85
OH_STRETCH_WARN_ANG = 1.3


@dataclass
class DimerGeometryReport:
    """Structural sanity report for a relaxed water-dimer geometry.

    is_valid is the PRIMARY gate for whether an energy at this geometry
    means anything as "a water dimer's potential energy": False iff ANY of
    the four O-H distances falls outside [OH_COLLAPSE_THRESHOLD_ANG,
    OH_STRETCH_WARN_ANG] (either collapsed or dissociated). Proton
    migration (n_proton_transfer_flags) is recorded but does NOT by itself
    flip is_valid to False -- it's a softer signal worth inspecting
    alongside, not one of the two sharp criteria (collapse/dissociation)
    the audit protocol uses to discard a point outright.
    """

    oh_distances: dict[str, float]  # e.g. {"O_A-H_bridge": 0.96, ...}
    hoh_angle_a_deg: float
    hoh_angle_b_deg: float
    min_oh_ang: float
    max_oh_ang: float
    n_proton_transfer_flags: int  # count of H's closer to the OTHER monomer's O than their own
    is_valid: bool  # False iff min_oh_ang < COLLAPSE or max_oh_ang > STRETCH_WARN
    is_oh_collapsed: bool  # True iff min_oh_ang < OH_COLLAPSE_THRESHOLD_ANG
    is_oh_dissociated: bool  # True iff max_oh_ang > OH_STRETCH_WARN_ANG


def check_dimer_geometry(atoms: Atoms) -> DimerGeometryReport:
    """Structural sanity check for a water dimer geometry (see
    DimerGeometryReport). Use this on every relaxed structure Test 3
    reports an energy for -- a deep "minimum" whose geometry fails this
    check is not evidence about the model's dimer PES.
    """
    oh = {
        "O_A-H_bridge": atoms.get_distance(O_INDEX_A, H_INDEX_A_BRIDGE),
        "O_A-H_free": atoms.get_distance(O_INDEX_A, H_INDEX_A_FREE),
        "O_B-H1": atoms.get_distance(O_INDEX_B, H_INDEX_B1),
        "O_B-H2": atoms.get_distance(O_INDEX_B, H_INDEX_B2),
    }
    min_oh = min(oh.values())
    max_oh = max(oh.values())

    def angle_deg(i_center, i1, i2):
        v1 = atoms.positions[i1] - atoms.positions[i_center]
        v2 = atoms.positions[i2] - atoms.positions[i_center]
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

    hoh_a = angle_deg(O_INDEX_A, H_INDEX_A_BRIDGE, H_INDEX_A_FREE)
    hoh_b = angle_deg(O_INDEX_B, H_INDEX_B1, H_INDEX_B2)

    n_proton_transfer = 0
    for h_idx, own_o, other_o in [
        (H_INDEX_A_BRIDGE, O_INDEX_A, O_INDEX_B),
        (H_INDEX_A_FREE, O_INDEX_A, O_INDEX_B),
        (H_INDEX_B1, O_INDEX_B, O_INDEX_A),
        (H_INDEX_B2, O_INDEX_B, O_INDEX_A),
    ]:
        if atoms.get_distance(h_idx, other_o) < atoms.get_distance(h_idx, own_o):
            n_proton_transfer += 1

    is_collapsed = min_oh < OH_COLLAPSE_THRESHOLD_ANG
    is_dissociated = max_oh > OH_STRETCH_WARN_ANG
    return DimerGeometryReport(
        oh_distances=oh,
        hoh_angle_a_deg=hoh_a,
        hoh_angle_b_deg=hoh_b,
        min_oh_ang=min_oh,
        max_oh_ang=max_oh,
        n_proton_transfer_flags=n_proton_transfer,
        is_valid=not (is_collapsed or is_dissociated),
        is_oh_collapsed=is_collapsed,
        is_oh_dissociated=is_dissociated,
    )
