import scanpy as sc
import anndata as ad
import os.path as Path
from sklearn.metrics import matthews_corrcoef,precision_score, recall_score, f1_score

result_folder = r'M:\backup\spaGFM\TLS_results'

# agg_result = ad.AnnData()
#
# #read 5-fold cross validation results
# for model_name in ['36M', '317M']:
#     for fold_num in range(1):
#         result = sc.read_h5ad(Path.join(result_folder, 'unfrozen_b128_lr0.01_result.h5ad'))
#         # mask = sc.read_h5ad(Path.join(result_folder, f'LC{fold_num +1}.h5ad'))
#         result = result[result.obs['is_assigned']]
#
#         best_walk = 1
#         best_mcc = 0
#         for walk_length in range(1, 9):
#             current_mcc = matthews_corrcoef(result.obs['labels'], result.obs[f'fold{fold_num}_unfrozen_{model_name}_b128_lr0.01_{walk_length}_prediction'])
#             if current_mcc > best_mcc:
#                 best_mcc = current_mcc
#                 best_walk = walk_length
#         agg_result.obs[f'fold{fold_num}_{model_name}_prediction'] = result.obs[f'fold{fold_num}_unfrozen_{model_name}_b128_lr0.01_{walk_length}_prediction']
#         agg_result.obs['labels'] = result.obs['labels']
#         del result
#
# agg_result.write_h5ad(Path.join(result_folder, f'unfrozen_b128_lr0.01_result.h5ad'))


result = sc.read_h5ad(Path.join(result_folder, 'unfrozen_b128_lr0.01_result.h5ad'))

for model_name in ['36M', '317M']:
    precision_list = []
    rcall_list = []
    mcc_list = []
    f1_list = []
    for fold_num in range(1):
        precision_list.append(precision_score(result.obs['labels'], result.obs[f'fold{fold_num}_{model_name}_prediction'], average='macro'))

        rcall_list.append(recall_score(result.obs['labels'], result.obs[f'fold{fold_num}_{model_name}_prediction'], average='macro'))

        mcc_list.append(matthews_corrcoef(result.obs['labels'], result.obs[f'fold{fold_num}_{model_name}_prediction']))

        f1_list.append(f1_score(result.obs['labels'], result.obs[f'fold{fold_num}_{model_name}_prediction'], average='macro'))

    print(f'{model_name} Precision: {precision_list}')
    print(f'{model_name} Recall: {rcall_list}')
    print(f'{model_name} MCC: {mcc_list}')
    print(f'{model_name} F1: {f1_list}')


