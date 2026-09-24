"""Molecule builders for Test 2 (MD stability) and Test 4 (condensed-phase
water).

Test 2 molecules are built reproducibly from SMILES via RDKit (3D
embedding + MMFF94 force-field optimization) rather than hand-typed
coordinates -- this mirrors the paper's own approach ("benchmark molecule
was generated based on the SMILE strings of the parent molecules using the
GAFF2 force field") and means anyone can regenerate the exact same
starting geometry from the SMILES string alone.

Test 4's water box is built with a simple grid-packing routine (no
packmol/OpenMM available in this environment) -- intentionally only a
REASONABLE starting configuration, since the protocol's own 125 ps NVT
equilibration is what actually relaxes it to a physical liquid structure.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

WATER_OH_LENGTH_ANG = 0.9572
WATER_HOH_ANGLE_DEG = 104.52
WATER_MOLAR_MASS_G_MOL = 18.015
AVOGADRO = 6.02214076e23


def build_molecule_from_smiles(smiles: str, seed: int = 0) -> Atoms:
    """Build a 3D-embedded, force-field-optimized ase.Atoms from a SMILES
    string, via RDKit.

    Args:
        smiles: SMILES string (implicit hydrogens added automatically).
        seed: RDKit embedding random seed, for reproducibility.

    Returns:
        ase.Atoms with a reasonable (MMFF94-optimized) 3D starting
        geometry, centered with vacuum padding, non-periodic.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)

    embed_result = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=True)
    if embed_result != 0:
        raise RuntimeError(f"RDKit 3D embedding failed for SMILES: {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    )

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.center(vacuum=10.0)
    atoms.pbc = False
    return atoms


def _single_water(rng: np.random.Generator) -> np.ndarray:
    """Positions (3, 3) for one water molecule (O, H, H) at the origin,
    randomly rotated. Standard gas-phase geometry."""
    r = WATER_OH_LENGTH_ANG
    theta = np.radians(WATER_HOH_ANGLE_DEG)
    o = np.array([0.0, 0.0, 0.0])
    h1 = np.array([r, 0.0, 0.0])
    h2 = r * np.array([np.cos(theta), np.sin(theta), 0.0])
    pos = np.stack([o, h1, h2])

    # random rotation
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, 2 * np.pi)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rot = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return pos @ rot.T


def build_water_box(
    n_waters: int,
    target_density_g_cm3: float = 1.0,
    seed: int = 0,
    min_separation_ang: float = 2.5,
) -> Atoms:
    """Build a periodic cubic box of n_waters water molecules at
    approximately the target density, via jittered-grid packing with
    random per-molecule orientation.

    This is only a REASONABLE starting configuration, not an
    equilibrated liquid structure -- it is expected to be relaxed by NVT
    (and then NPT) molecular dynamics before any production analysis.
    No packmol/OpenMM is used (unavailable in this environment); if a
    proper packing tool is available in your environment, prefer it.

    Args:
        n_waters: number of water molecules.
        target_density_g_cm3: target initial mass density (relaxed away
            by subsequent MD -- doesn't need to be exact).
        seed: random seed for orientations and grid jitter.
        min_separation_ang: minimum oxygen-oxygen grid spacing (Angstrom)
            used when placing molecules on the grid.

    Returns:
        ase.Atoms, periodic (pbc=True), cubic cell.
    """
    rng = np.random.default_rng(seed)

    mass_g = n_waters * WATER_MOLAR_MASS_G_MOL / AVOGADRO
    volume_cm3 = mass_g / target_density_g_cm3
    volume_ang3 = volume_cm3 * 1e24  # cm^3 -> Angstrom^3
    box_side = volume_ang3 ** (1 / 3)

    # Grid dimensions: smallest cube of grid points >= n_waters.
    n_per_side = int(np.ceil(n_waters ** (1 / 3)))
    grid_spacing = box_side / n_per_side
    if grid_spacing < min_separation_ang:
        # Density too high for this grid; expand the box to respect the
        # minimum O-O separation instead (avoids building a physically
        # absurd starting configuration that NVT can't recover from).
        box_side = n_per_side * min_separation_ang
        grid_spacing = min_separation_ang

    grid_points = []
    for i in range(n_per_side):
        for j in range(n_per_side):
            for k in range(n_per_side):
                grid_points.append(
                    np.array([i, j, k]) * grid_spacing + grid_spacing / 2
                )
    rng.shuffle(grid_points)
    grid_points = grid_points[:n_waters]

    all_symbols: list[str] = []
    all_positions: list[np.ndarray] = []
    for center in grid_points:
        jitter = rng.uniform(-0.3, 0.3, size=3)
        water_pos = _single_water(rng) + center + jitter
        all_symbols.extend(["O", "H", "H"])
        all_positions.append(water_pos)

    positions = np.concatenate(all_positions, axis=0)
    cell = np.eye(3) * box_side
    atoms = Atoms(symbols=all_symbols, positions=positions, cell=cell, pbc=True)
    return atoms
