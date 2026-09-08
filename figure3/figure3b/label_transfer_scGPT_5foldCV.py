from pathlib import Path

import faiss
import numpy as np
import scanpy as sc


DATA_FOLDER = Path(r"C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data\new_model_5fold")
FOLD_COUNT = 5
LABEL_KEY = "sub_tls"
K_NEIGHBORS = 10


def _as_faiss_array(values):
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {values.shape}")
    return np.ascontiguousarray(values, dtype=np.float32)


def _get_label_key(train_adata):
    for label_key in LABEL_KEY_CANDIDATES:
        if label_key in train_adata.obs:
            return label_key
    raise KeyError(
        "No supported label key found in train.h5ad obs. "
        f"Expected one of: {', '.join(LABEL_KEY_CANDIDATES)}"
    )


def label_transfer(query_adata, train_adata, fold_num, label_key, embed_key, k=10):
    ref_cell_embeddings = _as_faiss_array(train_adata.obsm[embed_key])
    test_embed = _as_faiss_array(query_adata.obsm[embed_key])

    if ref_cell_embeddings.shape[1] != test_embed.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch for {embed_key}: "
            f"train has {ref_cell_embeddings.shape[1]}, test has {test_embed.shape[1]}"
        )

    k = min(k, ref_cell_embeddings.shape[0])
    index = faiss.IndexFlatL2(ref_cell_embeddings.shape[1])
    index.add(ref_cell_embeddings)

    _, labels = index.search(test_embed, k)
    train_labels = train_adata.obs[label_key].reset_index(drop=True)
    preds = [train_labels.iloc[idx].value_counts().index[0] for idx in labels]

    query_adata.obs[f"fold{fold_num}_{embed_key}_predict"] = preds


def transfer_fold(fold_num):
    fold_folder = DATA_FOLDER / f"fold{fold_num}"
    train_file = fold_folder / "train.h5ad"
    test_file = fold_folder / "test.h5ad"

    if not train_file.exists():
        raise FileNotFoundError(train_file)
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    train_adata = sc.read_h5ad(train_file)
    query_adata = sc.read_h5ad(test_file)

    embed_keys = sorted(set(train_adata.obsm.keys()) & set(query_adata.obsm.keys()))
    # embed_keys = ['novae_latent']
    if not embed_keys:
        raise KeyError(f"No shared obsm keys found in {fold_folder}")

    missing_from_train = sorted(set(query_adata.obsm.keys()) - set(train_adata.obsm.keys()))
    missing_from_query = sorted(set(train_adata.obsm.keys()) - set(query_adata.obsm.keys()))
    if missing_from_train:
        print(f"fold{fold_num}: skipping query-only obsm keys: {missing_from_train}")
    if missing_from_query:
        print(f"fold{fold_num}: skipping train-only obsm keys: {missing_from_query}")

    for embed_key in embed_keys:
        print(f"fold{fold_num}: transferring labels with obsm['{embed_key}']")
        label_transfer(
            query_adata=query_adata,
            train_adata=train_adata,
            fold_num=fold_num,
            label_key=LABEL_KEY,
            embed_key=embed_key,
            k=K_NEIGHBORS,
        )

    query_adata.write_h5ad(test_file)


def main():
    for fold_num in range(FOLD_COUNT):
        transfer_fold(fold_num)


if __name__ == "__main__":
    main()
