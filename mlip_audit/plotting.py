"""Plotting and quantitative analysis for Test 3: energy vs. O-O distance.

Uses oo_distance_target_ang (the restraint's scan coordinate) as the x-axis
for cross-model comparability; test3_dimer.py also records
oo_distance_actual_ang (the geometry the restrained optimization actually
converged to, which can deviate from the target at extreme clash distances
-- see that module's docstring) for diagnostic purposes.

IMPORTANT -- read before citing any "spurious minimum" number from this
module: a deep energy at short O-O distance is only evidence about the
model's DIMER potential energy surface if the underlying geometry is a
genuinely intact water dimer. This project's own diagnostic work (see
RESULTS.md) found that a naive "just take the minimum energy" analysis
gets fooled by relaxations that collapse an O-H bond or fling a hydrogen
away from both oxygens -- energetically extreme, sometimes even
technically "converged," but not a second minimum of the dimer, and its
energy is not comparable to the physical ~2.9 A minimum. This module
therefore reports TWO versions of every summary statistic:
  - "raw": every status=ok point, no geometry/convergence filtering.
  - "valid": restricted to points with geometry_valid=True (see
    mlip_audit.geometry.check_dimer_geometry) AND converged=True.
Always look at both. A large gap between them means the raw number is
probably not trustworthy as a dimer-PES finding.

Usage:
    python -m mlip_audit.plotting --csv results/test3_dimer/mace-off23-small.csv
    python -m mlip_audit.plotting --csv results/test3_dimer/*.csv   # overlay all models
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mlip_audit.config import RESULTS_DIR

EV_TO_KCAL_MOL = 23.060548

# Physical expectation window, for annotating the plot.
PHYSICAL_MINIMUM_ANG = 2.9
SPURIOUS_MINIMUM_CUTOFF_ANG = 1.0

# Window used to locate the REFERENCE physical minimum. Deliberately
# narrower than "everything >= SPURIOUS_MINIMUM_CUTOFF_ANG": the
# spurious-minimum artifact (real or not) can already be present right at
# and even somewhat above the 1.0 A cutoff, which would contaminate a
# "physical minimum" computed over the full >=1.0 A range. 2.5-3.5 A
# safely brackets the expected ~2.9 A minimum with margin.
PHYSICAL_WINDOW_ANG = (2.5, 3.5)


def load_scan_csv(csv_path: Path) -> pd.DataFrame:
    """Load one Test 3 CSV, keeping only status=ok points, and sort by
    O-O distance ascending (the CSV itself is written in the descending
    computation order; see test3_dimer.py)."""
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    df.sort_values("oo_distance_target_ang", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def relative_energy_ev(df: pd.DataFrame) -> pd.Series:
    """Energy relative to the value at the largest scanned O-O distance
    (the most dissociated / well-separated point), in eV."""
    reference = df.loc[df["oo_distance_target_ang"].idxmax(), "energy_eV"]
    return df["energy_eV"] - reference


def plot_dimer_scan(
    csv_paths: list[Path],
    out_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Plot log-scale relative energy vs. O-O distance for one or more
    models on the same axes. Points that fail the geometry-validity check
    (collapsed or dissociated O-H -- see mlip_audit.geometry.
    check_dimer_geometry) are marked with an open/hollow marker instead of
    a filled one, so a misleadingly deep "minimum" is visually flagged
    rather than looking identical to a trustworthy point.

    Args:
        csv_paths: one or more Test 3 CSV files (one per model).
        out_path: where to save the figure (PNG). Defaults to
            RESULTS_DIR/test3_dimer/dimer_scan.png.
        show: also call plt.show() (for interactive/notebook use).

    Returns:
        Path the figure was saved to.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    for i, csv_path in enumerate(csv_paths):
        csv_path = Path(csv_path)
        df = load_scan_csv(csv_path)
        if df.empty:
            print(f"WARNING: no points in {csv_path}, skipping.")
            continue
        rel_e = relative_energy_ev(df)
        model_name = csv_path.stem
        color = f"C{i}"

        has_validity = "geometry_valid" in df.columns
        if has_validity:
            valid_mask = df["geometry_valid"].astype(str).isin(["True", "true", "1"])
        else:
            valid_mask = pd.Series(True, index=df.index)

        ax.plot(df["oo_distance_target_ang"], rel_e, "-", color=color, linewidth=1, alpha=0.6, zorder=1)
        ax.scatter(df.loc[valid_mask, "oo_distance_target_ang"], rel_e[valid_mask],
                   marker="o", s=20, color=color, label=model_name, zorder=3)
        if (~valid_mask).any():
            ax.scatter(df.loc[~valid_mask, "oo_distance_target_ang"], rel_e[~valid_mask],
                       marker="o", s=30, facecolors="none", edgecolors=color, linewidths=1.3,
                       label=f"{model_name} (geometry invalid)", zorder=3)

    ax.axvline(PHYSICAL_MINIMUM_ANG, color="gray", linestyle="--", linewidth=1,
               label=f"expected minimum (~{PHYSICAL_MINIMUM_ANG} A)")
    ax.axvspan(0, SPURIOUS_MINIMUM_CUTOFF_ANG, color="red", alpha=0.06,
               label=f"O-O < {SPURIOUS_MINIMUM_CUTOFF_ANG} A")

    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("O-O distance, restraint target (A)")
    ax.set_ylabel("Energy relative to dissociated limit (eV, symlog scale)")
    ax.set_title("Test 3: water dimer O-O potential energy scan\n(hollow markers = geometry-invalid: collapsed or dissociated O-H)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    out_path = Path(out_path) if out_path else RESULTS_DIR / "test3_dimer" / "dimer_scan.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_force_convergence(
    csv_paths: list[Path],
    out_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Plot final max force vs. O-O distance (log-scale), with
    non-converged points marked distinctly (per-project-request: Check 2
    of the ANI-2x/MACE diagnostic -- non-converged points must be marked
    on the figure, not silently included).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, csv_path in enumerate(csv_paths):
        csv_path = Path(csv_path)
        df = load_scan_csv(csv_path)
        if df.empty or "final_max_force_eV_per_ang" not in df.columns:
            print(f"WARNING: {csv_path} has no final_max_force_eV_per_ang column (older schema?), skipping.")
            continue
        model_name = csv_path.stem
        color = f"C{i}"
        conv_mask = df["converged"].astype(str).isin(["True", "true", "1"])

        ax.plot(df["oo_distance_target_ang"], df["final_max_force_eV_per_ang"],
                "-", color=color, linewidth=1, alpha=0.5, zorder=1)
        ax.scatter(df.loc[conv_mask, "oo_distance_target_ang"], df.loc[conv_mask, "final_max_force_eV_per_ang"],
                   marker="o", s=20, color=color, label=f"{model_name} (converged)", zorder=3)
        if (~conv_mask).any():
            ax.scatter(df.loc[~conv_mask, "oo_distance_target_ang"], df.loc[~conv_mask, "final_max_force_eV_per_ang"],
                       marker="x", s=60, color=color, linewidths=2, label=f"{model_name} (NOT converged)", zorder=4)

    ax.axhline(0.05, color="black", linestyle=":", linewidth=1, label="fmax=0.05 target")
    ax.set_yscale("log")
    ax.set_xlabel("O-O distance, restraint target (A)")
    ax.set_ylabel("Final max force component (eV/A, log scale)")
    ax.set_title("Test 3: LBFGS convergence (max force at the reported geometry)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    out_path = Path(out_path) if out_path else RESULTS_DIR / "test3_dimer" / "force_convergence.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _valid_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that are both converged and geometry-valid -- see module
    docstring for why both gates matter."""
    if "geometry_valid" not in df.columns or "converged" not in df.columns:
        return df.iloc[0:0]  # empty: can't assess validity, don't pretend to
    conv_mask = df["converged"].astype(str).isin(["True", "true", "1"])
    valid_mask = df["geometry_valid"].astype(str).isin(["True", "true", "1"])
    return df[conv_mask & valid_mask]


def check_spurious_minimum(csv_path: Path) -> dict:
    """Quantitative analysis of the failure mode Test 3 looks for.

    Depth is reported in eV AND kcal/mol, both as "raw" (every status=ok
    point) and "valid" (converged=True AND geometry_valid=True only) --
    see the module docstring for why both are needed. depth_eV > 0 means
    the (raw/valid) global minimum is that many eV DEEPER than the
    physical ~2.9 A reference; depth_eV <= 0 means no deeper minimum was
    found under that filter (the physical one is still the lowest).
    """
    df = load_scan_csv(csv_path)
    lo, hi = PHYSICAL_WINDOW_ANG
    physical = df[(df["oo_distance_target_ang"] >= lo) & (df["oo_distance_target_ang"] <= hi)]

    result = {
        "physical_min_distance": None,
        "physical_min_energy": None,
        "n_points_total": int(len(df)),
        "n_points_geometry_valid": None,
        "raw_global_min_distance": None,
        "raw_global_min_energy": None,
        "raw_depth_eV": None,
        "raw_depth_kcal_mol": None,
        "valid_global_min_distance": None,
        "valid_global_min_energy": None,
        "valid_depth_eV": None,
        "valid_depth_kcal_mol": None,
    }

    if physical.empty:
        return result
    idx = physical["energy_eV"].idxmin()
    physical_min_distance = float(physical.loc[idx, "oo_distance_target_ang"])
    physical_min_energy = float(physical.loc[idx, "energy_eV"])
    result["physical_min_distance"] = physical_min_distance
    result["physical_min_energy"] = physical_min_energy

    if not df.empty:
        idx = df["energy_eV"].idxmin()
        raw_dist = float(df.loc[idx, "oo_distance_target_ang"])
        raw_energy = float(df.loc[idx, "energy_eV"])
        result["raw_global_min_distance"] = raw_dist
        result["raw_global_min_energy"] = raw_energy
        result["raw_depth_eV"] = physical_min_energy - raw_energy
        result["raw_depth_kcal_mol"] = result["raw_depth_eV"] * EV_TO_KCAL_MOL

    valid_df = _valid_subset(df)
    result["n_points_geometry_valid"] = int(len(valid_df))
    if not valid_df.empty:
        idx = valid_df["energy_eV"].idxmin()
        valid_dist = float(valid_df.loc[idx, "oo_distance_target_ang"])
        valid_energy = float(valid_df.loc[idx, "energy_eV"])
        result["valid_global_min_distance"] = valid_dist
        result["valid_global_min_energy"] = valid_energy
        result["valid_depth_eV"] = physical_min_energy - valid_energy
        result["valid_depth_kcal_mol"] = result["valid_depth_eV"] * EV_TO_KCAL_MOL

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSV paths/globs")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force-out", type=Path, default=None)
    args = parser.parse_args()

    csv_paths: list[Path] = []
    for pattern in args.csv:
        matched = sorted(glob.glob(pattern))
        csv_paths.extend(Path(p) for p in matched) if matched else csv_paths.append(Path(pattern))

    out_path = plot_dimer_scan(csv_paths, out_path=args.out)
    print(f"Saved energy plot to {out_path}")
    force_out_path = plot_force_convergence(csv_paths, out_path=args.force_out)
    print(f"Saved force-convergence plot to {force_out_path}")

    for csv_path in csv_paths:
        r = check_spurious_minimum(csv_path)
        print(f"\n{csv_path.stem}:  ({r['n_points_geometry_valid']}/{r['n_points_total']} points geometry-valid)")
        if r["physical_min_energy"] is None:
            print("  physical minimum: N/A")
            continue
        print(f"  physical (~2.9 A) minimum: {r['physical_min_distance']} A, {r['physical_min_energy']:.4f} eV")
        print(f"  RAW global minimum (unfiltered):  {r['raw_global_min_distance']} A, {r['raw_global_min_energy']:.4f} eV")
        print(f"    -> RAW depth below physical: {r['raw_depth_eV']:.4f} eV = {r['raw_depth_kcal_mol']:.2f} kcal/mol")
        if r["valid_global_min_energy"] is not None:
            print(f"  VALID global minimum (converged & geometry-valid only): {r['valid_global_min_distance']} A, {r['valid_global_min_energy']:.4f} eV")
            print(f"    -> VALID depth below physical: {r['valid_depth_eV']:.4f} eV = {r['valid_depth_kcal_mol']:.2f} kcal/mol")
        else:
            print("  VALID global minimum: no converged+geometry-valid points found at all (!)")


if __name__ == "__main__":
    main()
