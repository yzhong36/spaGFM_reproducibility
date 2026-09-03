#!/usr/bin/env python3
"""Calculate per-glomerulus vector-field metrics for every attention head.

The numerical definitions in the original analysis are retained, while data
loading, vector construction, metric calculation, pairwise testing, plotting,
and output are separated into testable functions. Each split is loaded once
instead of once per head.

Default outputs are written under ``kidney_vectorfiled``. Run
``python kidney_code/figure5f.py --help`` for configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import entropy as scipy_entropy
from scipy.stats import mannwhitneyu
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "kidney"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "kidney_vectorfiled"
GRADE_ORDER = ("0", "1", "2", "3")
GRADE_PALETTE = {
    "0": "#E5F5E0",
    "1": "#A1D99B",
    "2": "#41AB5D",
    "3": "#006D2C",
}
METRICS = (
    "mean_magnitude",
    "magnitude_variance",
    "neighbor_cosine",
    "polarization",
    "divergence",
    "curl",
    "entropy",
)
PAIRWISE_COMPARISONS = tuple(combinations(GRADE_ORDER, 2))
TEST_COLUMNS = (
    "metric",
    "head",
    "grade_a",
    "grade_b",
    "n_a",
    "n_b",
    "u_statistic",
    "p_value",
    "p_adjust_method",
    "p_adjusted",
    "significance",
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class AnalysisConfig:
    splits: tuple[int, ...] = (0, 2)
    minimum_cells: int = 20
    neighbor_k: int = 6
    derivative_k: int = 10
    entropy_bins: int = 4
    excluded_glomeruli: tuple[str, ...] = ("None", "Selection 2")
    p_adjust: str = "none"
    alpha: float = 0.05


@dataclass(frozen=True)
class SplitData:
    split: int
    coordinates: np.ndarray
    glomerulus: np.ndarray
    grade: np.ndarray
    attention: np.ndarray
    random_walks: np.ndarray


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_h5ad_column(handle: h5py.File, name: str) -> np.ndarray:
    """Read a plain or categorical AnnData obs column using h5py."""

    path = f"obs/{name}"
    if path not in handle:
        raise KeyError(f"Missing required h5ad column: {path}")
    node = handle[path]

    if isinstance(node, h5py.Group):
        if "categories" not in node or "codes" not in node:
            raise ValueError(f"Unsupported categorical encoding for {path}")
        categories = [decode_scalar(value) for value in node["categories"][:]]
        codes = np.asarray(node["codes"][:], dtype=int)
        values = np.empty(codes.size, dtype=object)
        values[:] = None
        valid = codes >= 0
        values[valid] = [categories[code] for code in codes[valid]]
        return values

    return np.asarray([decode_scalar(value) for value in node[:]], dtype=object)


def read_spatial_annotations(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only the spatial coordinates and two required obs columns."""

    with h5py.File(path, "r") as handle:
        if "obsm/spatial" not in handle:
            raise KeyError("Missing required h5ad field: obsm/spatial")
        coordinates = np.asarray(handle["obsm/spatial"][:], dtype=float)
        glomerulus = read_h5ad_column(handle, "glomerulus")
        grade = read_h5ad_column(handle, "glomerulus_grade")

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"Expected spatial coordinates with shape (cells, 2); got {coordinates.shape}")
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates contain non-finite values")
    if not (coordinates.shape[0] == glomerulus.size == grade.size):
        raise ValueError("Spatial coordinates and obs annotations have different cell counts")
    return coordinates, glomerulus, grade


def load_random_walks(path: Path) -> np.ndarray:
    """Load the saved PyTorch tensor on CPU and return cells × walks × steps."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to read random-walk .pt files. Install it in the active environment."
        ) from exc

    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        tensor = torch.load(path, map_location="cpu")

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}; found {type(tensor).__name__}")
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D random-walk tensor; got shape {tuple(tensor.shape)}")
    return tensor.permute(1, 0, 2).contiguous().numpy()


def split_paths(input_dir: Path, split: int) -> tuple[Path, Path, Path]:
    prefix = f"IU04_split{split}"
    return (
        input_dir / f"{prefix}_finetunescpgt_isglomerulus_attention_perhead.h5ad",
        input_dir / f"{prefix}_attention_diseased_317M.npy",
        input_dir / f"{prefix}_randomwalk_diseased_317M.pt",
    )


def validate_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_split(input_dir: Path, split: int) -> SplitData:
    h5ad_path, attention_path, walks_path = split_paths(input_dir, split)
    h5ad_path = validate_file(h5ad_path, f"split {split} h5ad")
    attention_path = validate_file(attention_path, f"split {split} attention")
    walks_path = validate_file(walks_path, f"split {split} random walks")

    coordinates, glomerulus, grade = read_spatial_annotations(h5ad_path)
    attention = np.load(attention_path, mmap_mode="r")
    random_walks = load_random_walks(walks_path)

    if attention.ndim != 3:
        raise ValueError(f"Attention must have shape (cells, heads, walks); got {attention.shape}")
    if random_walks.ndim != 3 or random_walks.shape[2] < 2:
        raise ValueError(
            "Random walks must have shape (cells, walks, steps) with at least two steps; "
            f"got {random_walks.shape}"
        )
    cell_count = coordinates.shape[0]
    if attention.shape[0] != cell_count or random_walks.shape[0] != cell_count:
        raise ValueError(
            f"Split {split} cell-count mismatch: spatial={cell_count}, "
            f"attention={attention.shape[0]}, random_walks={random_walks.shape[0]}"
        )
    if attention.shape[2] != random_walks.shape[1]:
        raise ValueError(
            f"Split {split} walk-count mismatch: attention={attention.shape[2]}, "
            f"random_walks={random_walks.shape[1]}"
        )
    if not np.issubdtype(random_walks.dtype, np.integer):
        raise TypeError(f"Random-walk indices must be integers; got {random_walks.dtype}")
    if random_walks.min() < 0 or random_walks.max() >= cell_count:
        raise IndexError(f"Split {split} random-walk indices fall outside [0, {cell_count})")

    return SplitData(
        split=split,
        coordinates=coordinates,
        glomerulus=glomerulus,
        grade=grade,
        attention=attention,
        random_walks=random_walks,
    )


def cell_density(coordinates: np.ndarray) -> float:
    """Cells per 2D convex-hull area."""

    if coordinates.shape[0] < 3:
        return math.nan
    try:
        area = float(ConvexHull(coordinates).volume)
    except QhullError:
        return math.nan
    return coordinates.shape[0] / area if area > 0 else math.nan


def mean_magnitude(vectors: np.ndarray) -> float:
    return float(np.linalg.norm(vectors, axis=1).mean())


def magnitude_variance(vectors: np.ndarray) -> float:
    return float(np.linalg.norm(vectors, axis=1).var())


def neighbor_cosine_similarity(
    coordinates: np.ndarray,
    vectors: np.ndarray,
    k: int = 6,
    epsilon: float = 1e-8,
) -> float:
    """Mean directed cosine similarity to each cell's k nearest neighbors."""

    if coordinates.shape[0] < 2:
        return math.nan
    effective_k = min(k, coordinates.shape[0] - 1)
    tree = cKDTree(coordinates)
    _, indices = tree.query(coordinates, k=effective_k + 1)
    indices = np.atleast_2d(indices)[:, 1:]

    focal = vectors[:, None, :]
    neighbor = vectors[indices]
    numerator = np.sum(focal * neighbor, axis=2)
    denominator = (
        np.linalg.norm(focal, axis=2) * np.linalg.norm(neighbor, axis=2) + epsilon
    )
    return float(np.mean(numerator / denominator))


def polarization(vectors: np.ndarray, epsilon: float = 1e-8) -> float:
    numerator = np.linalg.norm(vectors.sum(axis=0))
    denominator = np.linalg.norm(vectors, axis=1).sum() + epsilon
    return float(numerator / denominator)


def local_divergence_and_curl(
    coordinates: np.ndarray,
    vectors: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate local divergence and 2D scalar curl by least squares."""

    cell_count = coordinates.shape[0]
    divergence_values = np.zeros(cell_count, dtype=float)
    curl_values = np.zeros(cell_count, dtype=float)
    if cell_count < 4:
        return divergence_values, curl_values

    tree = cKDTree(coordinates)
    effective_k = min(k, cell_count)
    _, neighbor_matrix = tree.query(coordinates, k=effective_k)
    neighbor_matrix = np.atleast_2d(neighbor_matrix)

    for index in range(cell_count):
        neighbors = neighbor_matrix[index, 1:]
        if neighbors.size < 3:
            continue
        delta_position = coordinates[neighbors] - coordinates[index]
        delta_vx = vectors[neighbors, 0] - vectors[index, 0]
        delta_vy = vectors[neighbors, 1] - vectors[index, 1]
        try:
            gradient_vx = np.linalg.lstsq(delta_position, delta_vx, rcond=None)[0]
            gradient_vy = np.linalg.lstsq(delta_position, delta_vy, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        divergence_values[index] = gradient_vx[0] + gradient_vy[1]
        curl_values[index] = gradient_vy[0] - gradient_vx[1]
    return divergence_values, curl_values


def vector_field_entropy(vectors: np.ndarray, bins: int = 4) -> float:
    magnitudes = np.linalg.norm(vectors, axis=1)
    histogram, _ = np.histogram(magnitudes, bins=bins, density=True)
    return float(scipy_entropy(histogram + 1e-12))


def strongest_flow_vectors(
    data: SplitData,
    indices: np.ndarray,
    head: int,
) -> np.ndarray:
    """Build the strongest first-hop attention vector for selected cells."""

    selected_attention = np.asarray(data.attention[indices, head, :])
    strongest_walk = np.argmax(selected_attention, axis=1)
    row = np.arange(indices.size)
    target_indices = data.random_walks[indices, strongest_walk, 1]
    weights = selected_attention[row, strongest_walk]
    displacement = data.coordinates[target_indices] - data.coordinates[indices]
    return displacement * weights[:, None]


def modal_grade(grades: np.ndarray) -> str:
    values = pd.Series(grades, dtype="object").dropna().astype(str)
    modes = values.mode()
    if modes.empty:
        raise ValueError("A selected glomerulus has no valid grade")
    return str(modes.iloc[0])


def calculate_head_metrics(
    data: SplitData,
    head: int,
    config: AnalysisConfig,
) -> list[dict[str, object]]:
    """Calculate every metric for one split/head combination."""

    records: list[dict[str, object]] = []
    for glomerulus_id in pd.unique(data.glomerulus):
        if glomerulus_id is None or str(glomerulus_id) in config.excluded_glomeruli:
            continue
        indices = np.flatnonzero(data.glomerulus == glomerulus_id)
        if indices.size < config.minimum_cells:
            continue

        coordinates = data.coordinates[indices]
        vectors = strongest_flow_vectors(data, indices, head)
        divergence_values, curl_values = local_divergence_and_curl(
            coordinates,
            vectors,
            k=config.derivative_k,
        )
        records.append(
            {
                "split": data.split,
                "head": head,
                "glomerulus": str(glomerulus_id),
                "grade": modal_grade(data.grade[indices]),
                "n_cells": int(indices.size),
                "cell_density": cell_density(coordinates),
                "mean_magnitude": mean_magnitude(vectors),
                "magnitude_variance": magnitude_variance(vectors),
                "neighbor_cosine": neighbor_cosine_similarity(
                    coordinates,
                    vectors,
                    k=config.neighbor_k,
                ),
                "polarization": polarization(vectors),
                "divergence": float(divergence_values.mean()),
                "curl": float(np.abs(curl_values).mean()),
                "entropy": vector_field_entropy(vectors, bins=config.entropy_bins),
            }
        )
    return records


def resolve_heads(requested_heads: Sequence[int] | None, head_count: int) -> tuple[int, ...]:
    if requested_heads is None:
        return tuple(range(head_count))
    heads = tuple(dict.fromkeys(requested_heads))
    invalid = [head for head in heads if head < 0 or head >= head_count]
    if invalid:
        raise ValueError(f"Head indices out of range [0, {head_count}): {invalid}")
    return heads


def calculate_all_metrics(
    input_dir: Path,
    config: AnalysisConfig,
    requested_heads: Sequence[int] | None,
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    records: list[dict[str, object]] = []
    resolved_heads: tuple[int, ...] | None = None

    for split in config.splits:
        print(f"Loading split {split} ...", flush=True)
        data = load_split(input_dir, split)
        split_heads = resolve_heads(requested_heads, data.attention.shape[1])
        if resolved_heads is None:
            resolved_heads = split_heads
        elif split_heads != resolved_heads:
            raise ValueError("Requested head indices are inconsistent across splits")

        for head in split_heads:
            print(f"  Calculating head {head} ...", flush=True)
            records.extend(calculate_head_metrics(data, head, config))

    if resolved_heads is None:
        raise ValueError("No splits were requested")
    metrics = pd.DataFrame.from_records(records)
    if metrics.empty:
        raise ValueError("No eligible glomeruli were found")
    metrics["grade"] = metrics["grade"].astype(str)
    return metrics, resolved_heads


def adjust_pvalues(pvalues: np.ndarray, method: str) -> np.ndarray:
    """Adjust a finite vector of p-values without an extra dependency."""

    pvalues = np.asarray(pvalues, dtype=float)
    if method == "none":
        return pvalues.copy()
    if method == "bonferroni":
        return np.minimum(pvalues * pvalues.size, 1.0)
    if method == "fdr_bh":
        order = np.argsort(pvalues)
        ranked = pvalues[order]
        adjusted_ranked = ranked * pvalues.size / np.arange(1, pvalues.size + 1)
        adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
        adjusted = np.empty_like(adjusted_ranked)
        adjusted[order] = np.minimum(adjusted_ranked, 1.0)
        return adjusted
    raise ValueError(f"Unknown p-value adjustment method: {method}")


def significance_stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def pairwise_tests(
    metrics: pd.DataFrame,
    metric_names: Iterable[str],
    heads: Sequence[int],
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Run within-head grade comparisons for every requested metric."""

    records: list[dict[str, object]] = []
    for metric in metric_names:
        for head in heads:
            head_data = metrics.loc[metrics["head"] == head]
            pending: list[dict[str, object]] = []
            for grade_a, grade_b in PAIRWISE_COMPARISONS:
                values_a = head_data.loc[head_data["grade"] == grade_a, metric].dropna().to_numpy()
                values_b = head_data.loc[head_data["grade"] == grade_b, metric].dropna().to_numpy()
                if values_a.size == 0 or values_b.size == 0:
                    continue
                test = mannwhitneyu(values_a, values_b, alternative="two-sided")
                pending.append(
                    {
                        "metric": metric,
                        "head": head,
                        "grade_a": grade_a,
                        "grade_b": grade_b,
                        "n_a": int(values_a.size),
                        "n_b": int(values_b.size),
                        "u_statistic": float(test.statistic),
                        "p_value": float(test.pvalue),
                    }
                )
            if pending:
                adjusted = adjust_pvalues(
                    np.asarray([record["p_value"] for record in pending]),
                    config.p_adjust,
                )
                for record, p_adjusted in zip(pending, adjusted, strict=True):
                    record["p_adjust_method"] = config.p_adjust
                    record["p_adjusted"] = float(p_adjusted)
                    record["significance"] = (
                        significance_stars(float(p_adjusted))
                        if p_adjusted < config.alpha
                        else ""
                    )
                    records.append(record)
    return pd.DataFrame.from_records(records, columns=TEST_COLUMNS)


def annotate_significance(
    ax: plt.Axes,
    tests: pd.DataFrame,
    annotation_start: float,
    data_range: float,
) -> None:
    significant = tests.loc[tests["significance"] != ""]
    line_height = data_range * 0.018
    row_step = data_range * 0.052
    for row_index, row in enumerate(significant.itertuples(index=False)):
        x1 = GRADE_ORDER.index(row.grade_a)
        x2 = GRADE_ORDER.index(row.grade_b)
        y = annotation_start + row_step * row_index
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + line_height, y + line_height, y],
            color="black",
            linewidth=0.65,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            y + line_height,
            row.significance,
            ha="center",
            va="bottom",
            fontsize=6.5,
            clip_on=False,
        )


def plot_metric_grid(
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    metric: str,
    heads: Sequence[int],
    output: Path,
    dpi: int,
) -> None:
    columns = 4
    rows = math.ceil(len(heads) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.2, 2.15 * rows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for panel, (head, ax) in enumerate(zip(heads, flat_axes, strict=False)):
        head_data = metrics.loc[metrics["head"] == head]
        sns.violinplot(
            data=head_data,
            x="grade",
            y=metric,
            hue="grade",
            order=GRADE_ORDER,
            hue_order=GRADE_ORDER,
            palette=GRADE_PALETTE,
            legend=False,
            cut=0,
            linewidth=0.55,
            inner="box",
            ax=ax,
        )
        ax.set_title(f"Head {head}", fontsize=7, fontweight="bold")
        ax.set_xlabel("Grade", fontsize=6.5)
        ax.set_ylabel(metric.replace("_", " ") if panel % columns == 0 else "", fontsize=6.5)
        ax.tick_params(labelsize=6, length=2.5, width=0.6)

    for ax in flat_axes[len(heads) :]:
        ax.set_visible(False)

    visible_axes = flat_axes[: len(heads)]
    y_bottom = min(ax.get_ylim()[0] for ax in visible_axes)
    y_data_top = max(ax.get_ylim()[1] for ax in visible_axes)
    data_range = y_data_top - y_bottom
    if data_range <= 0:
        data_range = 1.0

    max_annotations = 0
    for head in heads:
        head_tests = tests.loc[(tests["metric"] == metric) & (tests["head"] == head)]
        max_annotations = max(max_annotations, int((head_tests["significance"] != "").sum()))
    y_top = y_data_top + data_range * (0.04 + 0.055 * max_annotations)
    annotation_start = y_top - data_range * (0.06 + 0.055 * max_annotations)

    for head, ax in zip(heads, visible_axes, strict=True):
        ax.set_ylim(y_bottom, y_top)
        head_tests = tests.loc[(tests["metric"] == metric) & (tests["head"] == head)]
        annotate_significance(ax, head_tests, annotation_start, data_range)

    fig.suptitle(
        f"{metric.replace('_', ' ').title()} across glomerulus grades",
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.2, w_pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(
    metrics: pd.DataFrame,
    tests: pd.DataFrame,
    heads: Sequence[int],
    metric_names: Sequence[str],
    output_dir: Path,
    config: AnalysisConfig,
    dpi: int,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "kidney_vector_metrics_all_heads.csv", index=False)
    for head in heads:
        metrics.loc[metrics["head"] == head].to_csv(
            output_dir / f"kidney_vector_metrics_head{head}.csv",
            index=False,
        )
    tests.to_csv(output_dir / "kidney_vector_metrics_pairwise_tests.csv", index=False)

    if make_plots:
        figure_dir = output_dir / "violin_all_heads"
        for metric in metric_names:
            print(f"Plotting {metric} ...", flush=True)
            plot_metric_grid(
                metrics,
                tests,
                metric,
                heads,
                figure_dir / f"{metric}_violin_all_heads_significance.png",
                dpi,
            )

    metadata = {
        "analysis": asdict(config),
        "heads": list(heads),
        "metrics": list(metric_names),
        "rows": int(metrics.shape[0]),
        "statistical_test": "Two-sided Mann-Whitney U within each attention head.",
        "p_adjustment_scope": "Six grade comparisons within each metric/head combination.",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": sns.__version__,
        },
    }
    (output_dir / "kidney_vector_metrics_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("value must satisfy 0 < value < 1")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate glomerulus vector-field metrics for all attention heads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 2])
    parser.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=None,
        help="Head indices to analyze; omit to use every available head.",
    )
    parser.add_argument("--minimum-cells", type=positive_int, default=20)
    parser.add_argument("--neighbor-k", type=positive_int, default=6)
    parser.add_argument("--derivative-k", type=positive_int, default=10)
    parser.add_argument("--entropy-bins", type=positive_int, default=4)
    parser.add_argument(
        "--exclude-glomerulus",
        nargs="*",
        default=["None", "Selection 2"],
        help="Exact glomerulus labels to exclude.",
    )
    parser.add_argument(
        "--p-adjust",
        choices=("none", "bonferroni", "fdr_bh"),
        default="none",
        help="Correction across six grade comparisons within each metric/head.",
    )
    parser.add_argument("--alpha", type=probability, default=0.05)
    parser.add_argument("--dpi", type=positive_int, default=600)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
        help="Metrics to test and plot; all metrics remain in the CSV output.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write tables only.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = AnalysisConfig(
        splits=tuple(dict.fromkeys(args.splits)),
        minimum_cells=args.minimum_cells,
        neighbor_k=args.neighbor_k,
        derivative_k=args.derivative_k,
        entropy_bins=args.entropy_bins,
        excluded_glomeruli=tuple(args.exclude_glomerulus),
        p_adjust=args.p_adjust,
        alpha=args.alpha,
    )
    metrics, heads = calculate_all_metrics(input_dir, config, args.heads)
    tests = pairwise_tests(metrics, args.metrics, heads, config)
    write_outputs(
        metrics,
        tests,
        heads,
        args.metrics,
        output_dir,
        config,
        args.dpi,
        make_plots=not args.no_plots,
    )
    print(f"Finished. Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
