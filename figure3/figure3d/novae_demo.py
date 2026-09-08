#!/usr/bin/env python
# coding: utf-8

## the env is under: /fs/ess/PAS1475/yzhong/sf_project/conda_env/novae

from pathlib import Path
import argparse
import gc

import novae
import scanpy as sc
from scipy import sparse


DATA_FOLDER = Path(r"/fs/ess/PAS1475/Fei/data/spaGFM/TLS_data/LC_5_fold")
MODEL_NAME = "MICS-Lab/novae-human-0"


def iter_subfolder_h5ad_files(data_folder):
    """Return h5ad files stored below fold/sample subfolders, not root-level h5ad files."""
    return sorted(
        file_path
        for file_path in data_folder.rglob("*.h5ad")
        if file_path.parent != data_folder
    )


def prepare_adata(adata):
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()

    adata.var["gene_name"] = adata.var_names

    rename_map = {}
    if "pxl_row" in adata.obs and "x_pixel" not in adata.obs:
        rename_map["pxl_row"] = "x_pixel"
    if "pxl_col" in adata.obs and "y_pixel" not in adata.obs:
        rename_map["pxl_col"] = "y_pixel"
    if rename_map:
        adata.obs.rename(columns=rename_map, inplace=True)

    missing_coord_keys = {"x_pixel", "y_pixel"} - set(adata.obs.columns)
    if missing_coord_keys:
        raise KeyError(
            f"Missing spatial coordinate columns: {', '.join(sorted(missing_coord_keys))}"
        )


def add_novae_representation(file_path, model):
    print(f"Processing {file_path}")
    adata = sc.read_h5ad(file_path)
    prepare_adata(adata)

    novae.spatial_neighbors(adata, technology="visium")
    model.compute_representations(adata, zero_shot=True)

    adata.write_h5ad(file_path)
    del adata
    gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Add Novae zero-shot representations to h5ad files in new_model_5fold subfolders."
    )
    parser.add_argument(
        "--data_folder",
        type=Path,
        default=DATA_FOLDER,
        help="Folder containing fold subfolders with h5ad files.",
    )
    parser.add_argument(
        "--model_name",
        default=MODEL_NAME,
        help="Novae pretrained model name or path.",
    )
    args = parser.parse_args()

    h5ad_files = iter_subfolder_h5ad_files(args.data_folder)
    if not h5ad_files:
        raise FileNotFoundError(f"No h5ad files found under subfolders of {args.data_folder}")

    model = novae.Novae.from_pretrained(args.model_name)
    for file_path in h5ad_files:
        add_novae_representation(file_path, model)


if __name__ == "__main__":
    main()
