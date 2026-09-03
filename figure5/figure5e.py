#!/usr/bin/env python3
"""Reproduce the IU04 random-walk cell-type composition stacked bar chart.

The script reads cell-type proportions for each glomerulus grade, retains the
15 cell types with the largest total proportion across grades, and aggregates
the rest into ``Other``. It uses the fixed custom palette from the final figure
and leaves the plot title blank.

Run from the project root, or from any directory:

    python kidney_code/plot_IU04_random_walk_cell_type_top15.py

Use ``--help`` to override the input, output, or number of retained cell types.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "kidney" / "IU04_split2_random_walk_cell_type_proportion_by_grade.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "kidney"
    / "IU04_split2_random_walk_cell_type_proportion_by_grade_top15_other_custom_colors_other_gray_notitle.png"
)
DEFAULT_SOURCE_DATA = (
    PROJECT_ROOT
    / "kidney"
    / "IU04_split2_random_walk_cell_type_proportion_by_grade_top15_other.csv"
)

# Positions 1--15 correspond to the ranked Top 15 cell types. The final color
# is reserved for the aggregated Other category.
PALETTE = [
    "#2D8875",
    "#52AADC",
    "#7FABD1",
    "#91ccc0",
    "#963B79",
    "#97D0C5",
    "#B5CE4E",
    "#BD7795",
    "#C7C1DE",
    "#C89736",
    "#D75B4E",
    "#EC6E66",
    "#EEB6D4",
    "#F39865",
    "#F7AC53",
]
OTHER_COLOR = "#7C7979"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_proportions(path: Path) -> pd.DataFrame:
    """Load and validate the long-format random-walk proportion table."""

    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")

    table = pd.read_csv(path)
    required_columns = {"cell_type", "proportion", "glomerulus_grade"}
    missing = required_columns.difference(table.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    table = table.loc[:, ["cell_type", "proportion", "glomerulus_grade"]].copy()
    table["glomerulus_grade"] = pd.to_numeric(
        table["glomerulus_grade"], errors="coerce"
    )
    table["proportion"] = pd.to_numeric(table["proportion"], errors="coerce")
    table = table.dropna(subset=["cell_type", "proportion", "glomerulus_grade"])

    if table.empty:
        raise ValueError("No valid rows remain after removing missing annotations.")
    if (table["proportion"] < 0).any():
        raise ValueError("Proportions must be non-negative.")

    table["glomerulus_grade"] = table["glomerulus_grade"].astype(int)
    return table


def aggregate_top_cell_types(table: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    """Keep globally dominant cell types and combine all others into Other."""

    if not 1 <= top_n <= len(PALETTE):
        raise ValueError(f"--top-n must be between 1 and {len(PALETTE)}.")

    top_cell_types = (
        table.groupby("cell_type")["proportion"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    plot_table = table.copy()
    plot_table["cell_type_plot"] = plot_table["cell_type"].where(
        plot_table["cell_type"].isin(top_cell_types), "Other"
    )
    plot_table = (
        plot_table.groupby(["glomerulus_grade", "cell_type_plot"], as_index=False)[
            "proportion"
        ]
        .sum()
        .sort_values(["glomerulus_grade", "cell_type_plot"])
    )
    return plot_table, top_cell_types


def plot_composition(
    plot_table: pd.DataFrame,
    top_cell_types: list[str],
    output_path: Path,
    dpi: int,
) -> None:
    """Render the title-free Top-N plus Other stacked bar chart."""

    plot_df = plot_table.pivot(
        index="glomerulus_grade", columns="cell_type_plot", values="proportion"
    ).fillna(0)
    plot_df = plot_df.sort_index()
    ordered_columns = [name for name in top_cell_types if name in plot_df.columns]
    if "Other" in plot_df.columns:
        ordered_columns.append("Other")
    plot_df = plot_df.reindex(columns=ordered_columns, fill_value=0)

    colors = PALETTE[: len(top_cell_types)]
    if "Other" in ordered_columns:
        colors.append(OTHER_COLOR)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(13, 7))
    plot_df.plot(kind="bar", stacked=True, ax=ax, width=0.8, color=colors)
    ax.set_xlabel("Glomerulus grade")
    ax.set_ylabel("Cell type proportion in random walk")
    ax.set_ylim(0, 1)
    ax.legend(title="Cell type", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    source_data_path = args.source_data.expanduser().resolve()

    table = load_proportions(input_path)
    plot_table, top_cell_types = aggregate_top_cell_types(table, args.top_n)

    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    plot_table.to_csv(source_data_path, index=False)
    plot_composition(plot_table, top_cell_types, output_path, args.dpi)

    print(f"Figure: {output_path}")
    print(f"Source data: {source_data_path}")
    print("Top cell types, in palette order:")
    for cell_type, color in zip(top_cell_types, PALETTE):
        print(f"  {cell_type}: {color}")
    print(f"  Other: {OTHER_COLOR}")


if __name__ == "__main__":
    main()
