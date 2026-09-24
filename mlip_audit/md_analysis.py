"""Post-hoc analysis for Test 2 (bond-length stability) and Test 4
(O-O radial distribution function), computed directly from the saved
.traj trajectory files -- no MDTraj dependency (the paper uses MDTraj;
these are lightweight ASE/numpy equivalents covering the same physical
quantities, not a byte-for-byte reimplementation of MDTraj's algorithms).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.io.trajectory import Trajectory

# A bond is "broken" if its length exceeds this multiple of its value in
# the FIRST frame (the optimized starting geometry). The paper does not
# state an exact numeric threshold for its own "stretching of chemical
# bonds" instability criterion; this is our own choice, disclosed as such
# -- treat it as a diagnostic flag to inspect, not a validated pass/fail
# bar from the literature.
BOND_BREAK_RATIO = 1.8
BOND_DETECTION_CUTOFF_ANG = 1.8  # initial-frame pairs closer than this = bonded


def infer_bonds(positions: np.ndarray, symbols: list[str]) -> list[tuple[int, int, float]]:
    """Infer a bonding list (i, j, equilibrium_distance) from one frame's
    positions via a simple distance cutoff. Good enough for organic
    molecules with normal bond lengths; not a substitute for a real
    chemistry-aware bond perception algorithm."""
    n = len(symbols)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < BOND_DETECTION_CUTOFF_ANG:
                bonds.append((i, j, float(d)))
    return bonds


def bond_length_trajectory_report(traj_path: Path, skip_first_n_frames: int = 1) -> dict:
    """Track every bond (inferred from the first frame) across the whole
    trajectory. Returns a dict with per-bond min/max/mean length and a
    list of bonds that ever exceeded BOND_BREAK_RATIO x their initial
    length (candidate instabilities).

    Args:
        traj_path: the .traj file to analyze.
        skip_first_n_frames: number of leading frames excluded from the
            reported min/max/mean/max_ratio statistics (the d0 REFERENCE
            distance always comes from frame 0 regardless of this value --
            only the statistics computed across the trajectory are
            affected). Defaults to 1, excluding frame 0 itself: per
            mlip_audit.md_common.run_resumable_md's checkpointing (ASE's
            dyn.attach fires once at step 0), frame 0 is the
            post-LBFGS-minimization structure with freshly-drawn initial
            velocities but ZERO elapsed MD time -- not a thermally sampled
            configuration, and including it in "production" bond
            statistics would be analyzing a point that isn't production
            data. If a real equilibration transient beyond frame 0 is
            confirmed (see the Test 2 equilibration diagnostic in
            notebooks/02_md_tests.ipynb / RESULTS.md), pass a larger value
            to also exclude that burn-in window -- not done by default
            here, since the burn-in length is an empirical question, not
            something to assume.

    Returns:
        {
          "n_frames": int,            # total frames on disk
          "n_frames_analyzed": int,   # frames actually used for the stats below
          "bonds": [{"i": int, "j": int, "symbols": (s_i, s_j),
                     "d0": float, "d_min": float, "d_max": float,
                     "d_mean": float, "max_ratio": float,
                     "flagged_unstable": bool}, ...],
          "any_unstable": bool,
        }
    """
    with Trajectory(str(traj_path), "r") as traj:
        frames = list(traj)
    if not frames:
        return {"n_frames": 0, "n_frames_analyzed": 0, "bonds": [], "any_unstable": False}

    symbols = frames[0].get_chemical_symbols()
    bonds0 = infer_bonds(frames[0].get_positions(), symbols)

    analyzed_frames = frames[skip_first_n_frames:]
    if not analyzed_frames:
        raise ValueError(
            f"skip_first_n_frames={skip_first_n_frames} >= n_frames={len(frames)} "
            f"for {traj_path} -- nothing left to analyze."
        )
    all_positions = np.array([f.get_positions() for f in analyzed_frames])  # (n_analyzed, n_atoms, 3)

    report_bonds = []
    any_unstable = False
    for i, j, d0 in bonds0:
        d_series = np.linalg.norm(all_positions[:, i, :] - all_positions[:, j, :], axis=1)
        max_ratio = float(d_series.max() / d0)
        flagged = max_ratio > BOND_BREAK_RATIO
        any_unstable = any_unstable or flagged
        report_bonds.append({
            "i": i, "j": j, "symbols": (symbols[i], symbols[j]),
            "d0": d0, "d_min": float(d_series.min()), "d_max": float(d_series.max()),
            "d_mean": float(d_series.mean()), "max_ratio": max_ratio,
            "flagged_unstable": flagged,
        })

    return {
        "n_frames": len(frames),
        "n_frames_analyzed": len(analyzed_frames),
        "bonds": report_bonds,
        "any_unstable": any_unstable,
    }


def oo_radial_distribution_function(
    traj_path: Path,
    r_max_ang: float = 8.0,
    n_bins: int = 200,
    skip_first_frac: float = 0.2,
    o_symbol: str = "O",
) -> dict:
    """Compute the O-O radial distribution function g(r), averaged over
    the trajectory's frames after skipping the first `skip_first_frac`
    (equilibration transient within the file itself, in addition to
    whatever separate NVT-equilibration phase already happened upstream).

    Uses the minimum-image convention (only valid for a periodic,
    orthorhombic cell, which is what mlip_audit.molecules.build_water_box
    produces).

    Returns:
        {"r": np.ndarray (bin centers, Angstrom), "g_r": np.ndarray,
         "n_frames_used": int}
    """
    with Trajectory(str(traj_path), "r") as traj:
        frames = list(traj)
    n_skip = int(len(frames) * skip_first_frac)
    frames = frames[n_skip:]
    if not frames:
        raise ValueError(f"No frames left in {traj_path} after skipping {skip_first_frac:.0%}")

    bin_edges = np.linspace(0, r_max_ang, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_sum = np.zeros(n_bins)

    for atoms in frames:
        symbols = np.array(atoms.get_chemical_symbols())
        o_idx = np.where(symbols == o_symbol)[0]
        n_o = len(o_idx)
        cell = atoms.cell.array
        volume = atoms.get_volume()

        pos = atoms.positions[o_idx]
        diffs = pos[:, None, :] - pos[None, :, :]
        # minimum-image convention, orthorhombic cell
        for d in range(3):
            L = cell[d, d]
            diffs[:, :, d] -= L * np.round(diffs[:, :, d] / L)
        dists = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(n_o, k=1)
        pair_dists = dists[iu]

        hist, _ = np.histogram(pair_dists, bins=bin_edges)
        # normalize this frame's histogram by the ideal-gas shell count,
        # standard g(r) normalization: N_pairs_ideal(r) = rho * 4*pi*r^2*dr * N_O / 2
        rho = n_o / volume
        shell_volumes = 4 * np.pi * bin_centers**2 * np.diff(bin_edges)
        n_ideal = rho * shell_volumes * n_o / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            g_frame = np.where(n_ideal > 0, hist / n_ideal, 0.0)
        hist_sum += g_frame

    g_r = hist_sum / len(frames)
    return {"r": bin_centers, "g_r": g_r, "n_frames_used": len(frames)}
