#!/usr/bin/env python3
"""Run repeated sample-level KNN evaluation for AnnData embeddings."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier


DEFAULT_INPUT_DIR = Path(__file__).parent / "corpus_full_317M_05_19_wl"
DEFAULT_EMBEDDINGS = {
    "spaGFM_8": "X_subgraph_emb_wl_8",
    "spaGFM_64": "X_subgraph_emb_wl_64",
    "Novae": "novae_latent",
}


def parse_embedding_specs(specs):
    """Convert ``MODEL=OBSM_KEY`` CLI values to an embedding mapping."""
    if specs is None:
        return DEFAULT_EMBEDDINGS.copy()

    embeddings = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --embedding value {spec!r}; use MODEL=OBSM_KEY")
        model_name, obsm_key = (value.strip() for value in spec.split("=", maxsplit=1))
        if not model_name or not obsm_key:
            raise ValueError(f"Invalid --embedding value {spec!r}; use MODEL=OBSM_KEY")
        if model_name in embeddings:
            raise ValueError(f"Duplicate model name in --embedding: {model_name!r}")
        embeddings[model_name] = obsm_key
    return embeddings


def load_embeddings(input_dir, embedding_keys, file_pattern):
    """Read files and return one concatenated AnnData object per embedding."""
    adata_files = sorted(input_dir.glob(file_pattern))
    if not adata_files:
        raise FileNotFoundError(f"No files matching {file_pattern!r} found in {input_dir}")

    embeddings = {model_name: [] for model_name in embedding_keys}
    for adata_file in adata_files:
        source_adata = sc.read(adata_file)
        for model_name, embedding_key in embedding_keys.items():
            if embedding_key not in source_adata.obsm:
                raise KeyError(f"{embedding_key!r} is missing from {adata_file}")
            embeddings[model_name].append(
                ad.AnnData(source_adata.obsm[embedding_key], obs=source_adata.obs.copy())
            )
        print(f"Processed {adata_file}")

    return {model_name: sc.concat(adata_list) for model_name, adata_list in embeddings.items()}


def run_knn_benchmark(
    input_dir,
    embedding_keys=None,
    output_dir=None,
    file_pattern="*.h5ad",
    train_sample_ratio=0.5,
    split_name="sample",
    label_name="CNiche",
    k=5,
    n_runs=10,
    seed=0,
):
    """Evaluate embeddings and return metrics plus saved output paths."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "knn_predictions"
    embedding_keys = DEFAULT_EMBEDDINGS.copy() if embedding_keys is None else embedding_keys
    if not embedding_keys:
        raise ValueError("embedding_keys must contain at least one model mapping")
    if not 0 < train_sample_ratio < 1 or k < 1 or n_runs < 1:
        raise ValueError("Require 0 < train_sample_ratio < 1, k >= 1, and n_runs >= 1")

    embeddings = load_embeddings(input_dir, embedding_keys, file_pattern)
    output_files = {
        model_name: output_dir / f"{model_name}_knn_predictions.h5ad"
        for model_name in embeddings
    }
    reference_adata = next(iter(embeddings.values()))
    for model_name, embedding_adata in embeddings.items():
        if split_name not in embedding_adata.obs:
            raise KeyError(f"{split_name!r} is missing from obs for {model_name}")
        if label_name not in embedding_adata.obs:
            raise KeyError(f"{label_name!r} is missing from obs for {model_name}")
        print(f"{model_name} shape: {embedding_adata.shape}")

    sample_names = reference_adata.obs[split_name].unique()
    train_count = int(train_sample_ratio * len(sample_names))
    if train_count == 0 or train_count == len(sample_names):
        raise ValueError("The split must contain at least one training and one test sample")
    metrics = []
    for run_number in range(n_runs):
        run_seed = seed + run_number
        rng = np.random.default_rng(run_seed)
        train_sample_names = rng.choice(sample_names, size=train_count, replace=False)
        test_sample_names = np.setdiff1d(sample_names, train_sample_names)
        print(f"Run {run_number + 1}/{n_runs} (seed={run_seed})")

        for model_name, embedding_adata in embeddings.items():
            train_mask = embedding_adata.obs[split_name].isin(train_sample_names).to_numpy()
            test_mask = embedding_adata.obs[split_name].isin(test_sample_names).to_numpy()
            train_adata = embedding_adata[train_mask]
            test_adata = embedding_adata[test_mask]
            if k > train_adata.n_obs:
                raise ValueError(f"k ({k}) exceeds the {train_adata.n_obs} training observations")

            model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
            model.fit(train_adata.X, train_adata.obs[label_name])
            probabilities = model.predict_proba(test_adata.X)
            predicted_labels = model.classes_[probabilities.argmax(axis=1)]
            true_labels = test_adata.obs[label_name].values
            key = f"{model_name}_knn_pred_seed_{run_seed}"
            predictions = np.full((embedding_adata.n_obs, probabilities.shape[1]), np.nan)
            predictions[test_mask] = probabilities
            embedding_adata.obsm[key] = predictions
            embedding_adata.uns[f"{key}_classes"] = model.classes_.tolist()

            result = {
                "model": model_name,
                "seed": run_seed,
                "f1_macro": f1_score(true_labels, predicted_labels, average="macro"),
                "mcc": matthews_corrcoef(true_labels, predicted_labels),
                "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
            }
            metrics.append(result)
            print(f"Train adata shape ({model_name}): {train_adata.shape}")
            print(f"Test adata shape ({model_name}): {test_adata.shape}")
            print(
                f"{model_name} F1-Macro: {result['f1_macro']}, "
                f"MCC: {result['mcc']}, Accuracy: {result['balanced_accuracy']}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, embedding_adata in embeddings.items():
        output_file = output_files[model_name]
        embedding_adata.write_h5ad(output_file)
        print(f"Saved predictions to {output_file}")

    return metrics, list(output_files.values())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for concatenated embedding AnnData files with predictions",
    )
    parser.add_argument("--file-pattern", default="*.h5ad", help="Input file glob")
    parser.add_argument(
        "--embedding",
        action="append",
        metavar="MODEL=OBSM_KEY",
        help=(
            "Embedding to evaluate; repeat for multiple embeddings. Omitting this "
            "option uses the spaGFM_8, spaGFM_64, and Novae defaults."
        ),
    )
    parser.add_argument("--train-sample-ratio", type=float, default=0.5)
    parser.add_argument("--split-name", default="sample", help="obs column used for the split")
    parser.add_argument("--label-name", default="CNiche", help="obs column to predict")
    parser.add_argument("--k", type=int, default=5, help="Number of KNN neighbors")
    parser.add_argument("--n-runs", type=int, default=10, help="Number of sample splits")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the first split")
    return parser.parse_args()


def main():
    args = parse_args()
    run_knn_benchmark(
        input_dir=args.input_dir,
        embedding_keys=parse_embedding_specs(args.embedding),
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        train_sample_ratio=args.train_sample_ratio,
        split_name=args.split_name,
        label_name=args.label_name,
        k=args.k,
        n_runs=args.n_runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
