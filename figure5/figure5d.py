#!/usr/bin/env python3
"""Reproduce the NPHS2-expression overlay on the IU04 H&E image.

Scientific role
---------------
Show the spatial localization and relative intensity of NPHS2 expression while
retaining the underlying H&E morphology.

Figure type
-----------
Image plate with a quantitative expression overlay. The script generates a raw
count panel, a library-size-normalized/log1p panel, and a side-by-side comparison.

The historical image-registration constants and plotting defaults are preserved,
but are exposed as command-line options and recorded in a JSON metadata file.
Run ``python kidney_code/kidney_h_e_enhanced.py --help`` for all options.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
import numpy as np
from PIL import Image
import scipy
from scipy.sparse import csr_matrix
import tifffile


# Pillow checks the full whole-slide canvas even though only a bounded crop is
# decoded below. The input is trusted local microscopy data.
Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5AD = PROJECT_ROOT / "kidney" / "IU04_xenium_gloms.h5ad"
DEFAULT_HE_IMAGE = PROJECT_ROOT / "kidney_img" / "IU04.TIF"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "kidney_img"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


@dataclass(frozen=True)
class AlignmentConfig:
    """Coordinate-registration constants inherited from the original analysis."""

    x_offset: float = -260.0
    y_offset: float = -110.0
    sample_x_min: float = 10_520.0
    crop_padding: float = 180.0


@dataclass(frozen=True)
class RenderConfig:
    """Rendering and expression-scale settings."""

    output_dpi: int = 300
    max_crop_pixels: int = 5_000
    zero_cell_alpha: float = 0.06
    positive_cell_alpha: float = 0.98
    marker_size: float = 8.0
    raw_gamma: float = 0.55
    raw_upper_quantile: float = 0.995
    log_lower_quantile: float = 0.05
    log_upper_quantile: float = 0.995
    saturation_mix: float = 0.12
    white_mix: float = 0.14


@dataclass(frozen=True)
class OutputPaths:
    raw: Path
    log_normalized: Path
    comparison: Path
    source_data: Path
    metadata: Path


@dataclass(frozen=True)
class ExpressionData:
    x: np.ndarray
    y: np.ndarray
    raw: np.ndarray
    log_normalized: np.ndarray


def decode_strings(values: np.ndarray) -> np.ndarray:
    """Decode HDF5 byte strings without changing non-byte values."""

    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def validate_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def read_h5ad_fields(path: Path, gene: str) -> tuple[csr_matrix, np.ndarray, int]:
    """Read the CSR expression matrix, spatial coordinates, and gene index."""

    with h5py.File(path, "r") as handle:
        required = ("X", "var/_index", "obsm/spatial")
        missing = [key for key in required if key not in handle]
        if missing:
            raise KeyError(f"Missing required h5ad field(s): {', '.join(missing)}")

        gene_names = decode_strings(handle["var/_index"][:])
        matches = np.flatnonzero(gene_names == gene)
        if matches.size == 0:
            raise ValueError(f"Gene {gene!r} is not present in {path}")
        if matches.size > 1:
            raise ValueError(f"Gene {gene!r} occurs {matches.size} times in var/_index")

        x_group = handle["X"]
        encoding = x_group.attrs.get("encoding-type")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if encoding != "csr_matrix":
            raise ValueError(f"Expected a CSR-encoded h5ad X matrix; found {encoding!r}")

        matrix = csr_matrix(
            (x_group["data"][:], x_group["indices"][:], x_group["indptr"][:]),
            shape=tuple(int(value) for value in x_group.attrs["shape"]),
        )
        spatial = np.asarray(handle["obsm/spatial"][:], dtype=float)

    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError("obsm/spatial must have at least two coordinate columns")
    if spatial.shape[0] != matrix.shape[0]:
        raise ValueError("Expression and spatial matrices contain different cell counts")
    if not np.isfinite(spatial[:, :2]).all():
        raise ValueError("Spatial coordinates contain non-finite values")

    return matrix, spatial[:, :2], int(matches[0])


def read_he_dimensions(path: Path) -> tuple[int, int]:
    """Return whole-slide image height and width without decoding the full TIFF."""

    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError(f"No image pages were found in {path}")
        height, width = tif.pages[0].shape[:2]
    return int(height), int(width)


def transform_coordinates(
    spatial: np.ndarray,
    image_width: int,
    image_height: int,
    alignment: AlignmentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Xenium spatial coordinates to the registered H&E pixel space."""

    x = spatial[:, 0].astype(float, copy=True)
    y = spatial[:, 1].astype(float, copy=True)
    x -= x.min()
    y -= y.min()

    if x.max() == 0 or y.max() == 0:
        raise ValueError("Spatial coordinates have zero range and cannot be registered")

    x *= image_width / x.max()
    y *= image_height / y.max()

    # Preserve the original display registration: horizontal reflection followed
    # by fixed pixel offsets (-250 - 10 in x; -10 - 100 in y).
    x = image_width - x + alignment.x_offset
    y = y + alignment.y_offset
    return x, y


def calculate_expression(
    matrix: csr_matrix,
    gene_index: int,
    sample_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate raw counts and Scanpy-equivalent normalize_total(1e4)+log1p."""

    selected = matrix[sample_mask]
    totals = np.asarray(selected.sum(axis=1)).ravel()
    counts = selected[:, gene_index].toarray().ravel().astype(float)
    normalized = np.divide(
        counts * 1e4,
        totals,
        out=np.zeros_like(counts, dtype=float),
        where=totals > 0,
    )
    return counts, np.log1p(normalized)


def prepare_expression_data(
    matrix: csr_matrix,
    spatial: np.ndarray,
    gene_index: int,
    image_width: int,
    image_height: int,
    alignment: AlignmentConfig,
) -> ExpressionData:
    """Register coordinates, select the original sample region, and transform expression."""

    x_all, y_all = transform_coordinates(spatial, image_width, image_height, alignment)
    sample_mask = x_all >= alignment.sample_x_min
    if not sample_mask.any():
        raise ValueError(
            f"No cells remain after sample_x_min={alignment.sample_x_min:g}; "
            "check the coordinate-registration parameters"
        )

    raw, log_normalized = calculate_expression(matrix, gene_index, sample_mask)
    if not np.any(raw > 0):
        raise ValueError("No positive expression was found in the selected sample region")

    return ExpressionData(
        x=x_all[sample_mask],
        y=y_all[sample_mask],
        raw=raw,
        log_normalized=log_normalized,
    )


def compute_bounds(
    x: np.ndarray,
    y: np.ndarray,
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[float, float, float, float]:
    """Return a padded crop constrained to the H&E canvas."""

    return (
        max(0.0, float(x.min() - padding)),
        min(float(image_width), float(x.max() + padding)),
        max(0.0, float(y.min() - padding)),
        min(float(image_height), float(y.max() + padding)),
    )


def lighten_he(rgb: np.ndarray, render: RenderConfig) -> np.ndarray:
    """Apply global, documented H&E adjustments for overlay readability."""

    image = rgb.astype(np.float32) / 255.0
    grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114])[..., None]
    image = (1.0 - render.saturation_mix) * image + render.saturation_mix * grayscale
    image = (1.0 - render.white_mix) * image + render.white_mix
    return np.clip(image, 0.0, 1.0)


def load_he_crop(
    image: Image.Image,
    bounds: Sequence[float],
    render: RenderConfig,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Decode and downsample only the requested H&E region."""

    xmin, xmax, ymin, ymax = bounds
    box = (
        max(0, int(np.floor(xmin))),
        max(0, int(np.floor(ymin))),
        min(image.width, int(np.ceil(xmax))),
        min(image.height, int(np.ceil(ymax))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid H&E crop box: {box}")

    crop = image.crop(box).convert("RGB")
    scale = min(1.0, render.max_crop_pixels / max(crop.size))
    if scale < 1.0:
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return lighten_he(np.asarray(crop), render), box


def expression_colormap() -> LinearSegmentedColormap:
    """Return a perceptually ordered viridis map with visible low values."""

    return LinearSegmentedColormap.from_list(
        "viridis_visible_low",
        plt.get_cmap("viridis")(np.linspace(0.08, 1.0, 256)),
    )


def expression_norms(
    data: ExpressionData,
    render: RenderConfig,
) -> tuple[PowerNorm, Normalize, dict[str, float]]:
    """Create robust raw and normalized colour scales from positive cells."""

    raw_positive = data.raw[data.raw > 0]
    log_positive = data.log_normalized[data.log_normalized > 0]
    raw_vmax = float(np.quantile(raw_positive, render.raw_upper_quantile))
    log_vmin, log_vmax = (
        float(value)
        for value in np.quantile(
            log_positive,
            [render.log_lower_quantile, render.log_upper_quantile],
        )
    )
    if raw_vmax <= 0 or log_vmax <= log_vmin:
        raise ValueError("Expression quantiles do not define valid colour ranges")

    raw_norm = PowerNorm(gamma=render.raw_gamma, vmin=0.0, vmax=raw_vmax, clip=True)
    log_norm = Normalize(vmin=log_vmin, vmax=log_vmax, clip=True)
    ranges = {"raw_vmin": 0.0, "raw_vmax": raw_vmax, "log_vmin": log_vmin, "log_vmax": log_vmax}
    return raw_norm, log_norm, ranges


def plot_region(
    ax: plt.Axes,
    image: Image.Image,
    bounds: Sequence[float],
    data: ExpressionData,
    expression: np.ndarray,
    norm: Normalize,
    cmap: LinearSegmentedColormap,
    render: RenderConfig,
    *,
    point_scale: float = 1.0,
    show_zero_cells: bool = True,
):
    """Draw one aligned H&E crop with low-to-high ordered expression points."""

    xmin, xmax, ymin, ymax = bounds
    background, box = load_he_crop(image, bounds, render)
    bx0, by0, bx1, by1 = box
    ax.imshow(background, extent=(bx0, bx1, by1, by0), interpolation="bilinear")

    inside = (
        (data.x >= xmin)
        & (data.x <= xmax)
        & (data.y >= ymin)
        & (data.y <= ymax)
    )
    positive = inside & (expression > 0)
    zero = inside & ~positive
    marker_size = render.marker_size * point_scale

    if show_zero_cells:
        ax.scatter(
            data.x[zero],
            data.y[zero],
            s=marker_size,
            c="#6F6F78",
            alpha=render.zero_cell_alpha,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    positive_indices = np.flatnonzero(positive)
    positive_indices = positive_indices[np.argsort(expression[positive_indices])]
    scatter = ax.scatter(
        data.x[positive_indices],
        data.y[positive_indices],
        c=expression[positive_indices],
        s=marker_size,
        cmap=cmap,
        norm=norm,
        alpha=render.positive_cell_alpha,
        edgecolors="white",
        linewidths=0.18 * np.sqrt(point_scale),
        rasterized=True,
        zorder=4,
    )

    # These limits preserve the orientation of the original analysis.
    ax.set_xlim(xmax, xmin)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return scatter


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_single_panel(
    image: Image.Image,
    bounds: Sequence[float],
    data: ExpressionData,
    expression: np.ndarray,
    norm: Normalize,
    cmap: LinearSegmentedColormap,
    render: RenderConfig,
    gene: str,
    subtitle: str,
    colorbar_label: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 11.2), dpi=180, facecolor="white")
    scatter = plot_region(ax, image, bounds, data, expression, norm, cmap, render)
    ax.set_title(f"{gene} expression", fontsize=19, pad=12, color="#202124")
    ax.text(
        0.5,
        1.004,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555B65",
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.025)
    colorbar.set_label(colorbar_label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    save_figure(fig, output, render.output_dpi)


def draw_comparison(
    image: Image.Image,
    bounds: Sequence[float],
    data: ExpressionData,
    raw_norm: PowerNorm,
    log_norm: Normalize,
    cmap: LinearSegmentedColormap,
    render: RenderConfig,
    gene: str,
    positive_note: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 10.2), dpi=180, facecolor="white")
    panels = (
        (axes[0], data.raw, raw_norm, "Raw counts", "Raw counts"),
        (
            axes[1],
            data.log_normalized,
            log_norm,
            "Normalize total + log1p",
            "log1p normalized expression",
        ),
    )
    for ax, values, norm, title, colorbar_label in panels:
        scatter = plot_region(
            ax,
            image,
            bounds,
            data,
            values,
            norm,
            cmap,
            render,
            point_scale=0.9,
        )
        ax.set_title(title, fontsize=15, pad=10, color="#202124")
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.036, pad=0.022)
        colorbar.set_label(colorbar_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{gene} expression on H&E", fontsize=20, y=0.985, color="#202124")
    fig.text(
        0.5,
        0.955,
        f"Sequential viridis scale (higher expression = brighter); {positive_note}",
        ha="center",
        va="top",
        fontsize=9,
        color="#555B65",
    )
    save_figure(fig, output, render.output_dpi)


def write_source_data(path: Path, data: ExpressionData) -> None:
    """Save the plotted cell-level values without requiring pandas."""

    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((data.x, data.y, data.raw, data.log_normalized))
    np.savetxt(
        path,
        table,
        delimiter=",",
        header="registered_x,registered_y,raw_count,log1p_normalized_expression",
        comments="",
        fmt="%.8g",
    )


def write_metadata(
    path: Path,
    *,
    h5ad_path: Path,
    he_path: Path,
    gene: str,
    alignment: AlignmentConfig,
    render: RenderConfig,
    bounds: Sequence[float],
    color_ranges: dict[str, float],
    data: ExpressionData,
    outputs: OutputPaths,
) -> None:
    """Record processing and image-integrity details needed for reproduction."""

    positive_cells = int(np.count_nonzero(data.raw > 0))
    record = {
        "input": {"h5ad": str(h5ad_path), "he_image": str(he_path), "gene": gene},
        "alignment": asdict(alignment),
        "render": asdict(render),
        "crop_bounds_pixels": [float(value) for value in bounds],
        "color_ranges": color_ranges,
        "cells": {
            "selected": int(data.raw.size),
            "positive": positive_cells,
            "positive_fraction": positive_cells / int(data.raw.size),
        },
        "image_integrity": {
            "crop": "Global rectangular crop defined by crop_bounds_pixels.",
            "brightness_contrast_gamma": (
                "H&E receives global desaturation and white blending only; raw counts use "
                "PowerNorm gamma and robust upper clipping recorded above."
            ),
            "pseudocolor": "Viridis is applied only to expression markers.",
            "stitching": "None performed by this script.",
            "scale_calibration": "No physical pixel-size metadata or scale bar is added.",
        },
        "outputs": {name: str(value) for name, value in asdict(outputs).items()},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": Image.__version__,
            "h5py": h5py.__version__,
            "tifffile": tifffile.__version__,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def run(
    h5ad_path: Path,
    he_path: Path,
    gene: str,
    alignment: AlignmentConfig,
    render: RenderConfig,
    outputs: OutputPaths,
) -> None:
    h5ad_path = validate_file(h5ad_path, "h5ad input")
    he_path = validate_file(he_path, "H&E image")
    image_height, image_width = read_he_dimensions(he_path)
    matrix, spatial, gene_index = read_h5ad_fields(h5ad_path, gene)
    data = prepare_expression_data(
        matrix,
        spatial,
        gene_index,
        image_width,
        image_height,
        alignment,
    )
    bounds = compute_bounds(
        data.x,
        data.y,
        image_width,
        image_height,
        alignment.crop_padding,
    )
    raw_norm, log_norm, color_ranges = expression_norms(data, render)
    cmap = expression_colormap()
    positive_cells = int(np.count_nonzero(data.raw > 0))
    positive_note = f"{positive_cells:,}/{data.raw.size:,} cells positive"

    with Image.open(he_path) as image:
        draw_single_panel(
            image,
            bounds,
            data,
            data.raw,
            raw_norm,
            cmap,
            render,
            gene,
            f"Raw gene counts; {positive_note}; zero-count cells de-emphasized",
            "Raw counts",
            outputs.raw,
        )
        draw_single_panel(
            image,
            bounds,
            data,
            data.log_normalized,
            log_norm,
            cmap,
            render,
            gene,
            f"normalize_total(1e4) + log1p; {positive_note}; zero-count cells de-emphasized",
            "log1p normalized expression",
            outputs.log_normalized,
        )
        draw_comparison(
            image,
            bounds,
            data,
            raw_norm,
            log_norm,
            cmap,
            render,
            gene,
            positive_note,
            outputs.comparison,
        )

    write_source_data(outputs.source_data, data)
    write_metadata(
        outputs.metadata,
        h5ad_path=h5ad_path,
        he_path=he_path,
        gene=gene,
        alignment=alignment,
        render=render,
        bounds=bounds,
        color_ranges=color_ranges,
        data=data,
        outputs=outputs,
    )

    print(f"Selected cells: {data.raw.size:,}")
    print(f"Positive cells: {positive_cells:,} ({positive_cells / data.raw.size:.2%})")
    print(
        "Raw colour range: "
        f"{color_ranges['raw_vmin']:.3f} to {color_ranges['raw_vmax']:.3f}"
    )
    print(
        "Log-normalized colour range: "
        f"{color_ranges['log_vmin']:.3f} to {color_ranges['log_vmax']:.3f}"
    )
    for label, output in (
        ("raw counts", outputs.raw),
        ("log normalized", outputs.log_normalized),
        ("comparison", outputs.comparison),
        ("source data", outputs.source_data),
        ("metadata", outputs.metadata),
    ):
        print(f"Saved {label}: {output}")


def bounded_fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number < 1.0:
        raise argparse.ArgumentTypeError("value must satisfy 0 <= value < 1")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the NPHS2-on-H&E expression figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--he-image", type=Path, default=DEFAULT_HE_IMAGE)
    parser.add_argument("--gene", default="NPHS2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=positive_int, default=300)
    parser.add_argument("--max-crop-pixels", type=positive_int, default=5_000)

    alignment_group = parser.add_argument_group("coordinate alignment")
    alignment_group.add_argument("--x-offset", type=float, default=-260.0)
    alignment_group.add_argument("--y-offset", type=float, default=-110.0)
    alignment_group.add_argument("--sample-x-min", type=float, default=10_520.0)
    alignment_group.add_argument("--crop-padding", type=float, default=180.0)

    scale_group = parser.add_argument_group("expression scales")
    scale_group.add_argument("--raw-gamma", type=float, default=0.55)
    scale_group.add_argument("--raw-upper-quantile", type=bounded_fraction, default=0.995)
    scale_group.add_argument("--log-lower-quantile", type=bounded_fraction, default=0.05)
    scale_group.add_argument("--log-upper-quantile", type=bounded_fraction, default=0.995)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.raw_gamma <= 0:
        raise ValueError("--raw-gamma must be greater than zero")
    if args.log_lower_quantile >= args.log_upper_quantile:
        raise ValueError("--log-lower-quantile must be smaller than --log-upper-quantile")

    output_dir = args.output_dir.expanduser().resolve()
    safe_gene = "".join(character if character.isalnum() else "_" for character in args.gene)
    outputs = OutputPaths(
        raw=output_dir / f"IU04_split0_{safe_gene}_he_raw_counts.png",
        log_normalized=output_dir / f"IU04_split0_{safe_gene}_he_log_normalized.png",
        comparison=output_dir / f"IU04_split0_{safe_gene}_he_raw_vs_log.png",
        source_data=output_dir / f"IU04_split0_{safe_gene}_he_source_data.csv",
        metadata=output_dir / f"IU04_split0_{safe_gene}_he_metadata.json",
    )
    alignment = AlignmentConfig(
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        sample_x_min=args.sample_x_min,
        crop_padding=args.crop_padding,
    )
    render = RenderConfig(
        output_dpi=args.dpi,
        max_crop_pixels=args.max_crop_pixels,
        raw_gamma=args.raw_gamma,
        raw_upper_quantile=args.raw_upper_quantile,
        log_lower_quantile=args.log_lower_quantile,
        log_upper_quantile=args.log_upper_quantile,
    )
    run(args.h5ad, args.he_image, args.gene, alignment, render, outputs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
