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
