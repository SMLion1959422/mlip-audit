"""Shared resumable-MD infrastructure for Test 2 and Test 4.

Colab sessions disconnect. Every driver in this module is designed so that
re-running the exact same command after a disconnect picks up from the
last completed checkpoint rather than restarting -- this mirrors Test 3's
resumability design (see mlip_audit/test3_dimer.py) applied to actual time
propagation instead of a scan over independent points.

Checkpointing mechanism: the running trajectory is written to an
ASE .traj file, appended to (not overwritten) on every checkpoint. On
resume, the last frame's positions, momenta, and cell are loaded back into
a fresh Atoms/Dynamics object and integration continues from there. This
is exact for velocity-Verlet-family integrators (Langevin, NPT/NVT here)
as long as the RNG state itself isn't required to be bit-identical across
a resume -- it is not: a resumed run's Langevin/thermostat noise sequence
differs from an uninterrupted run's after the resume point, which is
physically fine (it's still a valid draw from the same ensemble), just
not bit-reproducible across a disconnect. This is disclosed, not hidden.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

logger = logging.getLogger(__name__)


def set_initial_velocities(atoms: Atoms, temperature_K: float, seed: int) -> None:
    """Draw Maxwell-Boltzmann velocities at the given temperature and
    remove net center-of-mass translation (standard practice -- otherwise
    a nonzero net momentum slowly drifts the whole system and inflates the
    apparent temperature)."""
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    Stationary(atoms)


def run_resumable_md(
    atoms: Atoms,
    dyn_factory,
    total_time_ps: float,
    timestep_fs: float,
    traj_save_every_ps: float,
    checkpoint_every_ps: float,
    traj_path: Path,
    log_path: Path,
    temperature_K: float,
    velocity_seed: int,
    draw_initial_velocities: bool = True,
) -> None:
    """Run (or resume) an MD trajectory to `total_time_ps`, checkpointing
    to `traj_path` every `checkpoint_every_ps` and appending
    thermodynamic-state log lines to `log_path`.

    Args:
        atoms: starting structure, with `atoms.calc` already attached and
            (for fairchem-family models) atoms.info charge/spin already
            set -- this function does not touch either.
        dyn_factory: callable(atoms) -> ase.md.md.MolecularDynamics,
            constructing the propagator (Langevin/NPT/etc.) for the given
            atoms object. Called fresh after loading the resume state (if
            any), since the dynamics object itself is not what's
            checkpointed -- the atoms' positions/momenta/cell are.
        total_time_ps: target total simulated time.
        timestep_fs: integration timestep.
        traj_save_every_ps: how often a frame is appended to `traj_path`.
        checkpoint_every_ps: how often progress is flushed to disk overall
            (>= traj_save_every_ps in practice, since a saved frame IS the
            checkpoint here).
        traj_path: ASE .traj file, appended to across resumes.
        log_path: plain-text log (time_ps, T_K, E_pot, E_kin, E_tot),
            appended to across resumes.
        temperature_K: only used to seed INITIAL velocities on a fresh
            (non-resumed) start; the thermostat/barostat in `dyn_factory`
            is responsible for maintaining it thereafter.
        velocity_seed: RNG seed for the initial Maxwell-Boltzmann draw
            (fresh start only; irrelevant on resume, since velocities come
            from the last checkpointed frame).
        draw_initial_velocities: on a FRESH (non-resumed) start, whether to
            draw a new Maxwell-Boltzmann velocity distribution at
            `temperature_K`. True for a genuinely new trajectory (e.g.
            Test 2's post-LBFGS-optimization start, Test 4's NVT phase).
            Set False when `atoms` already carries physically meaningful
            momenta from elsewhere (e.g. Test 4's NPT phase, whose starting
            atoms are NVT's final frame -- overwriting those velocities
            with a fresh draw would silently discard the whole point of
            chaining the two phases and was caught as a real bug during
            this project's own smoke testing: NPT's initial kinetic energy
            didn't match NVT's final frame until this flag was added).
    """
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    steps_per_ps = round(1000.0 / timestep_fs)
    total_steps = round(total_time_ps * steps_per_ps)
    save_every_steps = round(traj_save_every_ps * steps_per_ps)

    # IMPORTANT: the FIRST frame ever written (at n_completed_frames==1)
    # is the initial state (real step 0), not one full checkpoint
    # interval's worth of integration -- ASE's dyn.attach(fn, interval=N)
    # fires once immediately at step 0 in addition to every N steps after
    # (confirmed empirically: a 10-step run with interval=2 produces 6
    # frames at steps 0,2,4,6,8,10, not 5). So the number of REAL
    # completed integration steps already on disk is
    # (n_completed_frames - 1) * save_every_steps, NOT
    # n_completed_frames * save_every_steps -- using the latter was a
    # real off-by-one bug caught during this project's own smoke testing
    # (a resumed run silently fell one checkpoint interval short of its
    # requested target while logging/claiming it had reached it exactly).
    n_completed_frames = 0
    completed_steps = 0
    if traj_path.exists():
        with Trajectory(str(traj_path), "r") as traj:
            n_completed_frames = len(traj)
        if n_completed_frames > 0:
            completed_steps = (n_completed_frames - 1) * save_every_steps
            with Trajectory(str(traj_path), "r") as traj:
                last_frame = traj[-1]
            atoms.set_positions(last_frame.get_positions())
            atoms.set_momenta(last_frame.get_momenta())
            if last_frame.cell is not None:
                atoms.set_cell(last_frame.cell)
            elapsed_ps = completed_steps / steps_per_ps
            logger.info(
                "Resuming %s from frame %d (%.3f ps elapsed) of %.3f ps target",
                traj_path.name, n_completed_frames, elapsed_ps, total_time_ps,
            )
    else:
        if draw_initial_velocities:
            set_initial_velocities(atoms, temperature_K, velocity_seed)
            logger.info(
                "Starting %s fresh: %.3f ps target, T0=%.1f K (seed=%d)",
                traj_path.name, total_time_ps, temperature_K, velocity_seed,
            )
        else:
            logger.info(
                "Starting %s fresh: %.3f ps target, using ALREADY-SET "
                "momenta on the passed-in atoms (not drawing new ones)",
                traj_path.name, total_time_ps,
            )

    remaining_steps = total_steps - completed_steps
    if remaining_steps <= 0:
        logger.info("%s already complete (%d/%d steps).", traj_path.name, completed_steps, total_steps)
        return

    dyn: MolecularDynamics = dyn_factory(atoms)

    traj = Trajectory(str(traj_path), "a", atoms)
    log_f = open(log_path, "a")
    if log_path.stat().st_size == 0:
        log_f.write("time_ps,temperature_K,E_pot_eV,E_kin_eV,E_tot_eV\n")

    # ASE's dyn.attach() fires every attached observer once immediately
    # when run() starts (at nsteps==0), THEN every `interval` steps after.
    # On a resumed run that step-0 firing would re-save a frame identical
    # to the last one already on disk (the just-loaded state, before any
    # new integration), under a WRONG (over-counted) timestamp. On a fresh
    # run, the step-0 firing is correct and wanted (records the initial
    # state). This flag distinguishes the two cases.
    skip_next = n_completed_frames > 0

    def _checkpoint():
        nonlocal skip_next
        if skip_next:
            return
        traj.write(atoms)

    def _log_state():
        nonlocal skip_next
        if skip_next:
            skip_next = False
            return
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        n_atoms = len(atoms)
        temp = e_kin / (1.5 * units.kB * n_atoms)
        step = dyn.nsteps
        time_ps = (completed_steps + step) / steps_per_ps
        log_f.write(f"{time_ps:.4f},{temp:.3f},{e_pot:.6f},{e_kin:.6f},{e_pot + e_kin:.6f}\n")
        log_f.flush()

    dyn.attach(_checkpoint, interval=save_every_steps)
    dyn.attach(_log_state, interval=save_every_steps)

    try:
        dyn.run(remaining_steps)
    finally:
        traj.close()
        log_f.close()

    logger.info("%s: reached %.3f ps.", traj_path.name, total_time_ps)
