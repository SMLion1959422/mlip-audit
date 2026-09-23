"""Sanity check that charge/spin metadata actually reaches UMA-S.

UMA's `omol` task reads `atoms.info["charge"]` / `atoms.info["spin"]` on
every call. If the wiring in mlip_audit.models is broken (e.g. the
FAIRChemCalculator silently defaults to charge=0 regardless of
atoms.info), single-point energies for two different charge states of the
same geometry would come out IDENTICAL -- a bug that would otherwise be
invisible, because the calculator does not raise an error when charge is
missing/ignored.

This test builds acetate (CH3COO-) at a fixed, non-degenerate geometry and
asserts that scoring it as charge=-1 (the real acetate anion) vs.
charge=0 (a hypothetical, chemically wrong neutral) gives DIFFERENT
energies. If they match (within a tight tolerance), the charge/spin
config is not reaching the model and Test 3 results involving UMA cannot
be trusted.

This test requires network access, a HuggingFace login with UMA access,
and (in practice) a decent amount of RAM/VRAM -- it is skipped rather than
failed if the model cannot be loaded, since that is an environment issue,
not a code issue.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mlip_audit.models import get_calc, prepare_atoms_for_model

ENERGY_DIFF_TOLERANCE_EV = 1e-4  # energies differing by less than this count as "the same"


def build_acetate() -> Atoms:
    """A fixed, chemically reasonable (not necessarily fully relaxed)
    geometry for the acetate ion CH3COO-. Atom order: C(methyl), H, H, H,
    C(carboxyl), O, O."""
    positions = np.array(
        [
            [0.000, 0.000, 0.000],   # C1 (methyl)
            [0.507, 0.879, -0.363],  # H
            [0.507, -0.879, -0.363],  # H
            [-1.014, 0.000, -0.363],  # H
            [0.000, 0.000, 1.520],   # C2 (carboxyl)
            [1.145, 0.000, 2.106],   # O
            [-1.145, 0.000, 2.106],  # O
        ]
    )
    symbols = ["C", "H", "H", "H", "C", "O", "O"]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.center(vacuum=15.0)
    atoms.pbc = False
    return atoms


@pytest.fixture(scope="module")
def uma_calc():
    try:
        return get_calc("uma-s-1p1")
    except Exception as exc:  # noqa: BLE001 -- environment-dependent skip
        pytest.skip(f"UMA-S not available in this environment: {exc}")


def _single_point_energy(uma_calc, charge: int, spin: int) -> float:
    atoms = build_acetate()
    atoms.calc = uma_calc
    prepare_atoms_for_model(atoms, "uma-s-1p1", charge=charge, spin=spin)
    return float(atoms.get_potential_energy())


def test_charge_reaches_model(uma_calc):
    """Same geometry, charge=-1 (real acetate anion) vs charge=0 (wrong)
    must give different energies."""
    e_anion = _single_point_energy(uma_calc, charge=-1, spin=1)
    e_neutral = _single_point_energy(uma_calc, charge=0, spin=1)

    assert abs(e_anion - e_neutral) > ENERGY_DIFF_TOLERANCE_EV, (
        f"Energies for charge=-1 ({e_anion} eV) and charge=0 ({e_neutral} eV) "
        f"are indistinguishable (|diff| <= {ENERGY_DIFF_TOLERANCE_EV} eV). "
        "This means atoms.info charge is NOT reaching the UMA model -- check "
        "mlip_audit.models.prepare_atoms_for_model and the FAIRChemCalculator wiring."
    )


def test_prepare_atoms_sets_info_for_uma():
    """Unit-level check independent of the model: prepare_atoms_for_model
    must actually write charge/spin into atoms.info for UMA."""
    atoms = build_acetate()
    prepare_atoms_for_model(atoms, "uma-s-1p1", charge=-1, spin=1)
    assert atoms.info.get("charge") == -1
    assert atoms.info.get("spin") == 1


def test_prepare_atoms_is_noop_for_non_uma_models():
    """ANI-2x/MACE have no charge/spin input; prepare_atoms_for_model must
    not inject anything into atoms.info for them."""
    atoms = build_acetate()
    prepare_atoms_for_model(atoms, "ani2x", charge=-1, spin=1)
    assert "charge" not in atoms.info
    assert "spin" not in atoms.info
