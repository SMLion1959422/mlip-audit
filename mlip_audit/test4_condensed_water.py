"""Test 4: condensed-phase (bulk liquid) water stability test.

Protocol (session scope -- see RESULTS.md's deviations table for exactly
how and why this differs from Ranasinghe et al. 2025 Sec. 2.3): a PURE
168-water periodic box (~504 atoms, no solute -- see deviations table),
125 ps NVT equilibration then 50 ps NPT production, 300 K / 1 bar,
1 fs timestep, for each of the same 4 models as Test 2.

Thermostat/barostat: ase.md.nose_hoover_chain (NoseHooverChainNVT for
equilibration, IsotropicMTKNPT for production) -- the closest available
ASE equivalent to the paper's OpenMM Nose-Hoover-thermostat +
Monte-Carlo-barostat combination (ASE has no MC barostat; MTK is the
standard deterministic alternative achieving the same NPT ensemble).

Resumable: NVT and NPT are separate checkpointed phases (see
mlip_audit.md_common.run_resumable_md) under
MD_ROOT/test4/<model>/{nvt,npt}/. NPT only starts once NVT has reached its
full target time; its starting structure is NVT's final frame.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ase import units
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT

from mlip_audit.config import (
    MD_ROOT,
    TEST4_CHARGE,
    TEST4_CHECKPOINT_EVERY_PS,
    TEST4_MODELS,
    TEST4_N_WATERS,
    TEST4_NPT_TIME_PS,
    TEST4_NVT_TIME_PS,
    TEST4_PDAMP_FS,
    TEST4_PRESSURE_BAR,
    TEST4_SPIN,
    TEST4_TARGET_DENSITY_G_CM3,
    TEST4_TDAMP_FS,
    TEST4_TEMPERATURE_K,
    TEST4_TIMESTEP_FS,
    TEST4_TRAJ_SAVE_EVERY_PS,
)
from mlip_audit.md_common import run_resumable_md
from mlip_audit.models import get_calc, prepare_atoms_for_model
from mlip_audit.molecules import build_water_box

logger = logging.getLogger(__name__)


def _box_cache_path() -> Path:
    return MD_ROOT / "test4" / "_geometry" / f"water_box_{TEST4_N_WATERS}.xyz"


def get_or_build_water_box(seed: int = 0):
    """Return the (cached) starting water box -- built once and reused
    across all models, so every model starts from the identical initial
    configuration."""
    from ase.io import read as ase_read
    from ase.io import write as ase_write

    cache_path = _box_cache_path()
    if cache_path.exists():
        return ase_read(cache_path)

    atoms = build_water_box(TEST4_N_WATERS, TEST4_TARGET_DENSITY_G_CM3, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(cache_path, atoms)
    logger.info("Built and cached %d-water box -> %s", TEST4_N_WATERS, cache_path)
    return atoms


def run_nvt(model: str, device: str | None = None) -> Path:
    out_dir = MD_ROOT / "test4" / model / "nvt"
    traj_path = out_dir / "nvt.traj"
    log_path = out_dir / "nvt.log"

    atoms = get_or_build_water_box().copy()
    calc = get_calc(model, device=device, inference_settings="turbo")
    atoms.calc = calc
    prepare_atoms_for_model(atoms, model, charge=TEST4_CHARGE, spin=TEST4_SPIN)

    def dyn_factory(atoms):
        return NoseHooverChainNVT(
            atoms,
            timestep=TEST4_TIMESTEP_FS * units.fs,
            temperature_K=TEST4_TEMPERATURE_K,
            tdamp=TEST4_TDAMP_FS * units.fs,
        )

    run_resumable_md(
        atoms, dyn_factory,
        total_time_ps=TEST4_NVT_TIME_PS, timestep_fs=TEST4_TIMESTEP_FS,
        traj_save_every_ps=TEST4_TRAJ_SAVE_EVERY_PS,
        checkpoint_every_ps=TEST4_CHECKPOINT_EVERY_PS,
        traj_path=traj_path, log_path=log_path,
        temperature_K=TEST4_TEMPERATURE_K, velocity_seed=0,
    )
    return traj_path


def run_npt(model: str, nvt_traj_path: Path, device: str | None = None) -> Path:
    out_dir = MD_ROOT / "test4" / model / "npt"
    traj_path = out_dir / "npt.traj"
    log_path = out_dir / "npt.log"

    with Trajectory(str(nvt_traj_path), "r") as traj:
        n_nvt_frames = len(traj)
    steps_per_ps_nvt = round(1000.0 / TEST4_TIMESTEP_FS)
    save_every_steps_nvt = round(TEST4_TRAJ_SAVE_EVERY_PS * steps_per_ps_nvt)
    nvt_elapsed_ps = n_nvt_frames * save_every_steps_nvt / steps_per_ps_nvt
    if nvt_elapsed_ps < TEST4_NVT_TIME_PS - 1e-6:
        raise RuntimeError(
            f"NVT phase for {model} is only at {nvt_elapsed_ps:.2f}/{TEST4_NVT_TIME_PS} ps "
            "-- run NVT to completion before starting NPT."
        )

    # Always load from NVT's last frame; if npt.traj already has frames of
    # its own, run_resumable_md overwrites positions/momenta/cell from
    # THAT file's last frame instead once it runs -- this is just the
    # object construction, not the resume decision.
    with Trajectory(str(nvt_traj_path), "r") as traj:
        atoms = traj[-1]

    calc = get_calc(model, device=device, inference_settings="turbo")
    atoms.calc = calc
    prepare_atoms_for_model(atoms, model, charge=TEST4_CHARGE, spin=TEST4_SPIN)

    pressure_au = TEST4_PRESSURE_BAR * units.bar

    def dyn_factory(atoms):
        return IsotropicMTKNPT(
            atoms,
            timestep=TEST4_TIMESTEP_FS * units.fs,
            temperature_K=TEST4_TEMPERATURE_K,
            pressure_au=pressure_au,
            tdamp=TEST4_TDAMP_FS * units.fs,
            pdamp=TEST4_PDAMP_FS * units.fs,
        )

    run_resumable_md(
        atoms, dyn_factory,
        total_time_ps=TEST4_NPT_TIME_PS, timestep_fs=TEST4_TIMESTEP_FS,
        traj_save_every_ps=TEST4_TRAJ_SAVE_EVERY_PS,
        checkpoint_every_ps=TEST4_CHECKPOINT_EVERY_PS,
        traj_path=traj_path, log_path=log_path,
        temperature_K=TEST4_TEMPERATURE_K, velocity_seed=0,
        # NPT's starting atoms are NVT's final frame -- it already carries
        # correct, physically-equilibrated momenta. Do NOT redraw them (see
        # run_resumable_md's docstring for the bug this guards against).
        draw_initial_velocities=False,
    )
    return traj_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(TEST4_MODELS), default=None,
                         help="Run only this model (default: all 4)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--phase", choices=["nvt", "npt", "both"], default="both")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    models = [args.model] if args.model else list(TEST4_MODELS)
    for model in models:
        logger.info("=== Test 4: model=%s ===", model)
        nvt_traj = MD_ROOT / "test4" / model / "nvt" / "nvt.traj"
        if args.phase in ("nvt", "both"):
            nvt_traj = run_nvt(model, device=args.device)
        if args.phase in ("npt", "both"):
            run_npt(model, nvt_traj, device=args.device)


if __name__ == "__main__":
    main()
