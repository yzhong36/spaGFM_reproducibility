import scanpy as sc
import os.path as Path
import os
import faiss


def label_transfer(query_adata, train_adata,label_key,predict_binary_map, embed_key = 'X_scGPT', k=10):
    ref_cell_embeddings = train_adata.obsm[embed_key]
    test_embed = query_adata.obsm[embed_key]


    index = faiss.IndexFlatL2(ref_cell_embeddings.shape[1])
    index.add(ref_cell_embeddings)

    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    distances, labels = index.search(test_embed, k)

    idx_list = [i for i in range(test_embed.shape[0])]
    preds = []

    for k in idx_list:
        idx = labels[k]
        pred = train_adata.obs[label_key][idx].value_counts()
        preds.append(pred.index[0])

    query_adata.obs[f'{embed_key}_predict'] = preds
    query_adata.obs[f'{embed_key}_binary_predict'] = query_adata.obs[f'{embed_key}_predict'].map(predict_binary_map)

    del preds





ref_adata = sc.read_h5ad(r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data\train\train.h5ad')
folder = r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data\pick'
predict_binary_map = {0:0, 1:0, 2:2}
#read the h5ad files for reference and test datasets

for file_name in os.listdir(folder):
    if not file_name.endswith(".h5ad"):
        continue
    if file_name == 'train.h5ad':
        continue

    # query_embed = torch.load(Path.join(file_folder, query_file_name[:-5] +'.pt'), map_location=torch.device('cpu'), weights_only=False)
    query_adata = sc.read_h5ad(Path.join(folder, file_name))
    # query_adata.obsm['x_scGPT'] = query_embed.x.numpy()
    # del query_embed

    # label_transfer(query_adata, ref_adata, predict_binary_map = predict_binary_map, label_key="label_id",  embed_key='X_scGPT', k=10)
    label_transfer(query_adata, ref_adata, predict_binary_map=predict_binary_map, label_key="label_id",
                   embed_key='spatial', k=10)
    # label_transfer(query_adata, ref_adata, predict_binary_map = predict_binary_map, label_key="label_id", embed_key='X_finetune_scGPT',  k=10)
    # query_adata.obs['X_finetune_scGPT_binary_predict'] = query_adata.obs['X_finetune_scGPT_predict'].map(predict_binary_map)
    # label_transfer(query_adata, ref_adata, predict_binary_map = predict_binary_map, label_key="label_id", embed_key='X_NicheCompass',  k=10)

    query_adata.write_h5ad(Path.join(folder, file_name))

    del query_adata