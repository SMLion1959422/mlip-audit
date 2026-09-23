"""Plotting for Test 3: log-scale energy vs. O-O distance.

Uses oo_distance_target_ang (the restraint's scan coordinate) as the x-axis
for cross-model comparability; test3_dimer.py also records
oo_distance_actual_ang (the geometry the restrained optimization actually
converged to, which can deviate from the target at extreme clash distances
-- see that module's docstring) for diagnostic purposes.

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

# Physical expectation window, for annotating the plot.
PHYSICAL_MINIMUM_ANG = 2.9
SPURIOUS_MINIMUM_CUTOFF_ANG = 1.0


def load_scan_csv(csv_path: Path) -> pd.DataFrame:
    """Load one Test 3 CSV, keeping only successfully converged points, and
    sort by O-O distance ascending (the CSV itself is written in the
    descending computation order; see test3_dimer.py)."""
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
    """Plot log-scale |relative energy| vs. O-O distance for one or more
    models on the same axes.

    The y-axis is log-scale on the magnitude of the energy relative to the
    dissociated limit, since the point of Test 3 is to see whether energy
    rises monotonically approaching short O-O distance, or whether it dips
    into a second, spurious minimum -- which is visually obvious as a
    downward notch in an otherwise monotonic curve even on a log axis of
    |energy|, as long as sign is tracked separately (see below).

    Args:
        csv_paths: one or more Test 3 CSV files (one per model).
        out_path: where to save the figure (PNG). Defaults to
            RESULTS_DIR/test3_dimer/dimer_scan.png.
        show: also call plt.show() (for interactive/notebook use).

    Returns:
        Path the figure was saved to.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        df = load_scan_csv(csv_path)
        if df.empty:
            print(f"WARNING: no converged points in {csv_path}, skipping.")
            continue
        rel_e = relative_energy_ev(df)
        model_name = csv_path.stem
        ax.plot(df["oo_distance_target_ang"], rel_e, marker="o", markersize=3, label=model_name)

    ax.axvline(PHYSICAL_MINIMUM_ANG, color="gray", linestyle="--", linewidth=1,
               label=f"expected minimum (~{PHYSICAL_MINIMUM_ANG} A)")
    ax.axvspan(0, SPURIOUS_MINIMUM_CUTOFF_ANG, color="red", alpha=0.08,
               label=f"spurious-minimum zone (<{SPURIOUS_MINIMUM_CUTOFF_ANG} A)")

    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_xlabel("O-O distance, restraint target (A)")
    ax.set_ylabel("Energy relative to dissociated limit (eV, symlog scale)")
    ax.set_title("Test 3: water dimer O-O potential energy scan")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    out_path = Path(out_path) if out_path else RESULTS_DIR / "test3_dimer" / "dimer_scan.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path



# Window used to locate the REFERENCE physical minimum. Deliberately
# narrower than "everything >= SPURIOUS_MINIMUM_CUTOFF_ANG": empirically
# (see README), the spurious-minimum artifact can already dominate right
# at and even somewhat above the 1.0 A cutoff (e.g. MACE-OFF23-small was
# observed to already be anomalously deep AT 1.0 A), which would
# contaminate a "physical minimum" computed over the full >=1.0 A range.
# 2.5-3.5 A safely brackets the expected ~2.9 A minimum with margin while
# staying clear of that contamination.
PHYSICAL_WINDOW_ANG = (2.5, 3.5)


def check_spurious_minimum(csv_path: Path) -> dict:
    """Quantitative check for the failure mode Test 3 is designed to catch.

    Returns a dict with:
        physical_min_distance, physical_min_energy: the minimum within
            PHYSICAL_WINDOW_ANG (a window around the expected ~2.9 A
            minimum), used as the reference "correct" value.
        global_min_distance, global_min_energy: the minimum over the ENTIRE
            scanned curve (matches how Ranasinghe et al. frame the result:
            "the global energy minimum resides at [a short distance]").
        short_range_min_distance, short_range_min_energy: the minimum
            restricted to O-O < SPURIOUS_MINIMUM_CUTOFF_ANG.
        has_spurious_minimum: True iff the global minimum lies outside
            PHYSICAL_WINDOW_ANG and is deeper than the physical-window
            minimum (i.e. the true minimum of the curve is NOT the
            expected ~2.9 A one).
    """
    df = load_scan_csv(csv_path)
    lo, hi = PHYSICAL_WINDOW_ANG
    physical = df[(df["oo_distance_target_ang"] >= lo) & (df["oo_distance_target_ang"] <= hi)]
    short = df[df["oo_distance_target_ang"] < SPURIOUS_MINIMUM_CUTOFF_ANG]

    result = {
        "physical_min_distance": None,
        "physical_min_energy": None,
        "global_min_distance": None,
        "global_min_energy": None,
        "short_range_min_distance": None,
        "short_range_min_energy": None,
        "has_spurious_minimum": False,
    }

    if not physical.empty:
        idx = physical["energy_eV"].idxmin()
        result["physical_min_distance"] = float(physical.loc[idx, "oo_distance_target_ang"])
        result["physical_min_energy"] = float(physical.loc[idx, "energy_eV"])

    if not df.empty:
        idx = df["energy_eV"].idxmin()
        result["global_min_distance"] = float(df.loc[idx, "oo_distance_target_ang"])
        result["global_min_energy"] = float(df.loc[idx, "energy_eV"])

    if not short.empty:
        idx = short["energy_eV"].idxmin()
        result["short_range_min_distance"] = float(short.loc[idx, "oo_distance_target_ang"])
        result["short_range_min_energy"] = float(short.loc[idx, "energy_eV"])

    if (
        result["physical_min_energy"] is not None
        and result["global_min_energy"] is not None
        and not (lo <= result["global_min_distance"] <= hi)
        and result["global_min_energy"] < result["physical_min_energy"]
    ):
        result["has_spurious_minimum"] = True

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSV paths/globs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    csv_paths: list[Path] = []
    for pattern in args.csv:
        matched = sorted(glob.glob(pattern))
        csv_paths.extend(Path(p) for p in matched) if matched else csv_paths.append(Path(pattern))

    out_path = plot_dimer_scan(csv_paths, out_path=args.out)
    print(f"Saved plot to {out_path}")

    for csv_path in csv_paths:
        result = check_spurious_minimum(csv_path)
        print(f"\n{csv_path.stem}:")
        print(f"  physical (~2.9 A) minimum: {result['physical_min_distance']} A, {result['physical_min_energy']} eV")
        print(f"  short-range (<{SPURIOUS_MINIMUM_CUTOFF_ANG} A) minimum: {result['short_range_min_distance']} A, {result['short_range_min_energy']} eV")
        print(f"  global minimum (whole curve): {result['global_min_distance']} A, {result['global_min_energy']} eV")
        print(f"  HAS SPURIOUS MINIMUM: {result['has_spurious_minimum']}")


if __name__ == "__main__":
    main()
