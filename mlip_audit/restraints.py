"""Harmonic distance restraint for the Test 3 O-O scan.

Why a restraint instead of a hard constraint (ase.constraints.FixBondLength):
the original spec called for a hard geometric constraint, but that turned
out to make the scan numerically unusable at short O-O separation --
ase.constraints.FixBondLengths projects forces onto the constraint manifold
with a SHAKE/RATTLE-style fixed-point iteration, and that iteration either
raises RuntimeError("Did not converge") or, once loosened enough not to
raise, lets LBFGS diverge into nonphysical multi-million-eV energies at the
short-distance end of the scan (see git history / project notes for the
debugging trail). Neither is usable, and the second failure mode looks
superficially like "no spurious minimum," which would have been a wrong
conclusion about the models themselves, not just a numerical artifact.

Checking the actual water-dimer scan methodology in Ranasinghe et al. 2025
(arXiv:2503.11537, Section 2.2.3) shows their ML-potential scans were NOT
done with a hard constraint either: "distance restraints with a strong
force constant of 10 GJ/mol/nm^2" via OpenMM, re-optimizing every 0.01 nm.
This module reproduces that: a two-sided harmonic bias

    E_restraint = 0.5 * k * (r - r0)^2

added to the model's energy during optimization only. After the restrained
optimization converges, the reported energy is the MODEL'S energy alone at
the resulting geometry (matching what the paper plots), not
model+restraint.

Note this is NOT the same as ase.constraints.Hookean: Hookean is one-sided
(only pushes back once a distance EXCEEDS a threshold, meant to cap bond
stretching) and would apply zero force in the O-O<r0 direction relevant
here.
"""

from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

# 10 GJ/mol/nm^2, converted to eV/Angstrom^2 (the unit ASE forces/energies
# use). Derivation:
#   1e10 J/mol/nm^2
#   / (6.02214076e23 /mol)        -> J/particle/nm^2
#   * (1 eV / 1.602176634e-19 J)  -> eV/particle/nm^2
#   / 100                         -> eV/particle/A^2   (1 nm^2 = 100 A^2)
# = ~1036.4 eV/A^2
RESTRAINT_K_EV_PER_ANG2 = 1036.4


class HarmonicDistanceRestraint(Calculator):
    """ASE Calculator adding a two-sided harmonic bias on one atom pair's
    distance: E = 0.5 * k * (r - r0)^2, meant to be combined with a real
    MLIP calculator via ase.calculators.mixing.SumCalculator during
    optimization (see test3_dimer.py)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, a1: int, a2: int, k: float, r0: float):
        """
        Args:
            a1, a2: atom indices whose distance is restrained.
            k: force constant, eV/Angstrom^2.
            r0: target distance, Angstrom.
        """
        super().__init__()
        self.a1 = a1
        self.a2 = a2
        self.k = k
        self.r0 = r0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        p1, p2 = atoms.positions[self.a1], atoms.positions[self.a2]
        vec = p2 - p1
        r = float(np.linalg.norm(vec))
        direction = vec / r
        dr = r - self.r0

        energy = 0.5 * self.k * dr**2
        force_on_2 = -self.k * dr * direction  # pulls a2 toward a1 if r > r0

        forces = np.zeros((len(atoms), 3))
        forces[self.a1] -= force_on_2
        forces[self.a2] += force_on_2

        self.results = {"energy": energy, "forces": forces}
