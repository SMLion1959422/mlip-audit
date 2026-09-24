"""Test 2: short-MD molecular stability test.

Protocol (session scope -- see RESULTS.md's deviations table for exactly
how and why this differs from Ranasinghe et al. 2025 Sec. 2.2.2):
the paper's ACTUAL 349-atom drug-like benchmark molecule (see
mlip_audit.config.TEST2_BENCHMARK_GEOMETRY_PATH and
scripts/build_drug_like_benchmark_geometry.py for how it was obtained and
validated -- its exact SMILES is deposited in the paper's SI, not a
reconstruction), 1 velocity seed, 400 K Langevin dynamics, 1 fs timestep,
100 ps production, for each of 4 models (UMA-S, eSEN-conserving,
eSEN-direct, ANI-2x as a non-fairchem control).

Each (molecule, model) pair gets a FRESH calculator loaded with
inference_settings="turbo" -- safe here because that calculator is used
for exactly one fixed-composition system (one molecule, constant
charge/spin) for its entire lifetime; see mlip_audit.models.get_calc's
docstring for why this would NOT be safe if the same calculator were
reused across different molecules.

Resumable: each (molecule, model, seed) trajectory checkpoints
independently (see mlip_audit.md_common.run_resumable_md) under
MD_ROOT/test2/<molecule>/<model>/seed<N>/. Re-running this script skips
any combination whose trajectory has already reached the target time.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ase import units
from ase.md.langevin import Langevin
from ase.optimize import LBFGS

from mlip_audit.config import (
    MD_ROOT,
    TEST2_BENCHMARK_GEOMETRY_PATH,
    TEST2_CHARGE,
    TEST2_CHECKPOINT_EVERY_PS,
    TEST2_LANGEVIN_FRICTION_PER_FS,
    TEST2_MODELS,
    TEST2_MOLECULES,
    TEST2_SEEDS,
    TEST2_SPIN,
    TEST2_TEMPERATURE_K,
    TEST2_TIMESTEP_FS,
    TEST2_TOTAL_TIME_PS,
    TEST2_TRAJ_SAVE_EVERY_PS,
)
from mlip_audit.md_common import run_resumable_md
from mlip_audit.models import get_calc, prepare_atoms_for_model
from mlip_audit.molecules import build_molecule_from_smiles

logger = logging.getLogger(__name__)


def _geometry_cache_path(molecule_name: str) -> Path:
    return MD_ROOT / "test2" / "_geometries" / f"{molecule_name}.xyz"


def get_or_build_molecule(molecule_name: str, embed_seed: int = 0):
    """Return the (cached) starting geometry for a Test 2 molecule --
    resolved once and reused across all models/seeds, so every model
    starts from the identical initial 3D structure.

    If `TEST2_MOLECULES[molecule_name]` is None, the geometry comes from
    the pre-built, validated file at `TEST2_BENCHMARK_GEOMETRY_PATH`
    (the paper's real 349-atom molecule -- see
    scripts/build_drug_like_benchmark_geometry.py). Otherwise it's built
    on the fly from the given SMILES via RDKit (used for smoke-testing /
    any smaller molecule added later).
    """
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    cache_path = _geometry_cache_path(molecule_name)
    if cache_path.exists():
        return ase_read(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    smiles = TEST2_MOLECULES[molecule_name]
    if smiles is None:
        if not TEST2_BENCHMARK_GEOMETRY_PATH.exists():
            raise FileNotFoundError(
                f"{TEST2_BENCHMARK_GEOMETRY_PATH} not found. Run "
                "scripts/build_drug_like_benchmark_geometry.py first."
            )
        atoms = ase_read(TEST2_BENCHMARK_GEOMETRY_PATH)
        atoms.center(vacuum=10.0)
        ase_write(cache_path, atoms)
        logger.info(
            "Loaded %s from pre-built geometry %s -> cached at %s",
            molecule_name, TEST2_BENCHMARK_GEOMETRY_PATH, cache_path,
        )
        return atoms

    atoms = build_molecule_from_smiles(smiles, seed=embed_seed)
    ase_write(cache_path, atoms)
    logger.info("Built and cached %s geometry (SMILES=%s) -> %s", molecule_name, smiles, cache_path)
    return atoms


def run_one(molecule_name: str, model: str, seed: int, device: str | None = None) -> Path:
    """Run (or resume) one (molecule, model, seed) MD trajectory. Returns
    the trajectory file path."""
    out_dir = MD_ROOT / "test2" / molecule_name / model / f"seed{seed}"
    traj_path = out_dir / "md.traj"
    log_path = out_dir / "md.log"

    start_atoms = get_or_build_molecule(molecule_name).copy()
    calc = get_calc(model, device=device, inference_settings="turbo")
    start_atoms.calc = calc
    prepare_atoms_for_model(start_atoms, model, charge=TEST2_CHARGE, spin=TEST2_SPIN)

    if not traj_path.exists():
        # Per paper protocol: LBFGS-optimize geometry (with THIS model's
        # own PES) before starting MD, once, per model.
        logger.info("Optimizing %s/%s geometry before MD (LBFGS)...", molecule_name, model)
        opt = LBFGS(start_atoms, logfile=None)
        opt.run(fmax=0.05, steps=500)

    def dyn_factory(atoms):
        return Langevin(
            atoms,
            timestep=TEST2_TIMESTEP_FS * units.fs,
            temperature_K=TEST2_TEMPERATURE_K,
            friction=TEST2_LANGEVIN_FRICTION_PER_FS,
            fixcm=False,
        )

    run_resumable_md(
        start_atoms,
        dyn_factory,
        total_time_ps=TEST2_TOTAL_TIME_PS,
        timestep_fs=TEST2_TIMESTEP_FS,
        traj_save_every_ps=TEST2_TRAJ_SAVE_EVERY_PS,
        checkpoint_every_ps=TEST2_CHECKPOINT_EVERY_PS,
        traj_path=traj_path,
        log_path=log_path,
        temperature_K=TEST2_TEMPERATURE_K,
        velocity_seed=seed,
    )
    return traj_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=list(TEST2_MOLECULES), default=None,
                         help="Run only this molecule (default: all, currently just the 1 benchmark molecule)")
    parser.add_argument("--model", choices=list(TEST2_MODELS), default=None,
                         help="Run only this model (default: all 4)")
    parser.add_argument("--seed", type=int, choices=list(TEST2_SEEDS), default=None,
                         help="Run only this seed (default: all, currently just seed 0)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    molecules = [args.molecule] if args.molecule else list(TEST2_MOLECULES)
    models = [args.model] if args.model else list(TEST2_MODELS)
    seeds = [args.seed] if args.seed is not None else list(TEST2_SEEDS)

    for molecule_name in molecules:
        for model in models:
            for seed in seeds:
                logger.info("=== Test 2: molecule=%s model=%s seed=%d ===", molecule_name, model, seed)
                run_one(molecule_name, model, seed, device=args.device)


if __name__ == "__main__":
    main()
