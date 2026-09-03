#!/usr/bin/env python3
"""Reproduce the IU04 split1 317M Leiden spatial result.

The pipeline is fixed to the settings used for the published figure:

1. L2-normalize the 317M embedding cell by cell.
2. Reduce it to 50 dimensions with randomized PCA.
3. Build a 15-nearest-neighbor graph.
4. Run Leiden at resolution 0.22765625.
5. Align Leiden IDs to the independently clustered NOVAE labels using maximum
   overlap (Hungarian assignment), so the cluster colors are comparable.
6. Save the aligned cluster CSV and spatial PNG.

Run from anywhere with:

    python kidney_code/reproduce_IU04_split1_317M_leiden_res0.22765625.py

Input and output paths can be overridden with command-line arguments.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

import anndata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_H5AD = PROJECT_ROOT / "kidney/IU04_split1_novae.h5ad"
DEFAULT_EMBEDDING = PROJECT_ROOT / "kidney/IU04_split1_317M.npy"
DEFAULT_NOVAE_CLUSTERS = (
    PROJECT_ROOT
    / "kidney_img/IU04_split1_embedding_clusters_k11_novae_clusters.csv"
)
DEFAULT_OUTPUT_PREFIX = (
    PROJECT_ROOT
    / "kidney_img/IU04_split1_317M_leiden_k11"
)

RESOLUTION = 0.22765625
N_CLUSTERS = 11
N_PCS = 50
N_NEIGHBORS = 15
RANDOM_STATE = 0

# Same ordered palette as the linked spatial figure. Cluster IDs are zero-based
# in this reproduction because the source figure legend is Cluster 0--10.
PALETTE = np.array(
    [
        "#7DC69B",
        "#9BD7F3",
        "#F2A1A7",
        "#B8AED8",
        "#F5C28B",
        "#D5EAD9",
        "#D8EEFB",
        "#EEF0A7",
        "#DCD7EB",
        "#FCE6CF",
        "#5FAFCB",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--embedding", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument(
        "--novae-clusters", type=Path, default=DEFAULT_NOVAE_CLUSTERS
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output path without the _spatial.png or _clusters.csv suffix.",
    )
    parser.add_argument("--point-size", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def validate_embedding(embedding: np.ndarray, n_cells: int) -> np.ndarray:
    embedding = np.asarray(embedding)
    if embedding.ndim != 2:
        raise ValueError(f"Embedding must be 2D; got {embedding.shape}.")
    if embedding.shape[0] != n_cells:
        raise ValueError(
            f"Embedding has {embedding.shape[0]} rows but h5ad has {n_cells} cells."
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinite values.")
    return embedding.astype(np.float32, copy=False)


def preprocess_embedding(embedding: np.ndarray) -> np.ndarray:
    normalized = normalize(embedding, norm="l2", axis=1)
    n_components = min(N_PCS, normalized.shape[0] - 1, normalized.shape[1])
    if n_components < 2:
        raise ValueError("At least two PCA components are required.")
    return PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
    ).fit_transform(normalized).astype(np.float32, copy=False)


def leiden_labels(reduced_embedding: np.ndarray) -> np.ndarray:
    graph_adata = sc.AnnData(X=reduced_embedding)
    sc.pp.neighbors(
        graph_adata,
        n_neighbors=min(N_NEIGHBORS, graph_adata.n_obs - 1),
        use_rep="X",
        random_state=RANDOM_STATE,
    )
    sc.tl.leiden(
        graph_adata,
        resolution=RESOLUTION,
        random_state=RANDOM_STATE,
        key_added="leiden",
        flavor="leidenalg",
        n_iterations=-1,
        directed=False,
    )
    categorical = graph_adata.obs["leiden"].astype("category")
    labels = categorical.cat.codes.to_numpy(dtype=np.int16)
    observed = int(categorical.cat.categories.size)
    if observed != N_CLUSTERS:
        raise RuntimeError(
            f"Expected {N_CLUSTERS} clusters at resolution {RESOLUTION}; got {observed}. "
            "Check the package versions printed by this script."
        )
    return labels


def load_novae_labels(path: Path, cell_ids: pd.Index) -> np.ndarray:
    table = pd.read_csv(path, dtype={"cell_id": str})
    required = {"cell_id", "novae_cluster"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"NOVAE CSV is missing columns: {sorted(missing)}")
    if table["cell_id"].duplicated().any():
        raise ValueError("NOVAE CSV contains duplicate cell IDs.")
    labels = table.set_index("cell_id")["novae_cluster"].reindex(cell_ids.astype(str))
    if labels.isna().any():
        raise ValueError(f"NOVAE CSV is missing {int(labels.isna().sum())} cells.")
    labels = labels.to_numpy(dtype=np.int16)
    if set(np.unique(labels)) != set(range(N_CLUSTERS)):
        raise ValueError("NOVAE reference labels must contain cluster IDs 0--10.")
    return labels


def align_to_novae(
    novae_labels: np.ndarray, raw_labels: np.ndarray
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    overlap = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.int64)
    np.add.at(overlap, (novae_labels, raw_labels), 1)
    reference_ids, raw_ids = linear_sum_assignment(-overlap)
    mapping = {
        int(raw_id): int(reference_id)
        for reference_id, raw_id in zip(reference_ids, raw_ids)
    }
    aligned = np.array([mapping[int(label)] for label in raw_labels], dtype=np.int16)
    return aligned, mapping, overlap


def plot_spatial(
    spatial: np.ndarray, labels: np.ndarray, output: Path, point_size: float, dpi: int
) -> None:
    x_range = max(float(np.ptp(spatial[:, 0])), np.finfo(float).eps)
    y_range = max(float(np.ptp(spatial[:, 1])), np.finfo(float).eps)
    width = float(np.clip(7.0 * x_range / y_range, 3.0, 6.0))
    fig, ax = plt.subplots(figsize=(width + 1.8, 7.0), constrained_layout=True)
    ax.scatter(
        spatial[:, 0],
        spatial[:, 1],
        c=PALETTE[labels],
        s=point_size,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("317M Leiden clusters (resolution=0.227656)", fontsize=11)
    ax.set_xlabel("spatial x")
    ax.set_ylabel("spatial y")
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=PALETTE[cluster],
            markeredgecolor="none",
            markersize=6,
            label=f"Cluster {cluster}",
        )
        for cluster in range(N_CLUSTERS)
    ]
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_versions() -> None:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "anndata": anndata.__version__,
        "scanpy": sc.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }
    print("Package versions:")
    for name, version in versions.items():
        print(f"  {name}: {version}")


def main() -> None:
    args = parse_args()
    print_versions()

    adata = sc.read_h5ad(args.h5ad)
    embedding = validate_embedding(np.load(args.embedding), adata.n_obs)
    reduced_embedding = preprocess_embedding(embedding)
    raw_labels = leiden_labels(reduced_embedding)
    novae_labels = load_novae_labels(args.novae_clusters, adata.obs_names)
    aligned_labels, mapping, overlap = align_to_novae(novae_labels, raw_labels)

    if "spatial" not in adata.obsm:
        raise KeyError("The h5ad does not contain adata.obsm['spatial'].")
    spatial = np.asarray(adata.obsm["spatial"])[:, :2]

    output_prefix = args.output_prefix.resolve()
    spatial_path = output_prefix.parent / f"{output_prefix.name}_spatial.png"
    cluster_path = output_prefix.parent / f"{output_prefix.name}_clusters.csv"
    overlap_path = output_prefix.parent / f"{output_prefix.name}_overlap_matrix.csv"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    plot_spatial(spatial, aligned_labels, spatial_path, args.point_size, args.dpi)
    pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "317M_leiden_cluster_raw": raw_labels,
            "317M_leiden_cluster_aligned": aligned_labels,
            "leiden_resolution": RESOLUTION,
        }
    ).to_csv(cluster_path, index=False)
    pd.DataFrame(
        overlap,
        index=[f"novae_{i}" for i in range(N_CLUSTERS)],
        columns=[f"317M_raw_{i}" for i in range(N_CLUSTERS)],
    ).to_csv(overlap_path)

    print(f"Raw-to-aligned mapping: {mapping}")
    print(f"Number of clusters: {np.unique(aligned_labels).size}")
    print(f"Saved spatial plot: {spatial_path}")
    print(f"Saved cluster CSV: {cluster_path}")
    print(f"Saved overlap matrix: {overlap_path}")


if __name__ == "__main__":
    main()
