import scanpy as sc
import os.path as Path
from sklearn.metrics import matthews_corrcoef,precision_score, recall_score, f1_score

result_folder = r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data\new_model_5fold'

key_to_label = 'sub_tls'
# keys_to_predict = ['PCA','x_scGPT',  'NicheCompass']
keys_to_predict = ['spatial', 'x_scGPT','X_NicheCompass','X_finetune_scGPT','novae_latent']
for key in keys_to_predict:
    precision_list = []
    rcall_list = []
    mcc_list = []
    f1_list = []
    for fold_num in range(5):
        result = sc.read_h5ad(Path.join(result_folder,f'fold{fold_num}' ,'test.h5ad'))
        precision_list.append(
            precision_score(result.obs[key_to_label], result.obs[f'fold{fold_num}_{key}_predict'],
                            average='macro'))
        rcall_list.append(
            recall_score(result.obs[key_to_label], result.obs[f'fold{fold_num}_{key}_predict'], average='macro'))

        mcc_list.append(matthews_corrcoef(result.obs[key_to_label], result.obs[f'fold{fold_num}_{key}_predict']))

        f1_list.append(
            f1_score(result.obs[key_to_label], result.obs[f'fold{fold_num}_{key}_predict'], average='macro'))

        del result

    print(f'{key} Precision: {precision_list}')
    print(f'{key} Recall: {rcall_list}')
    print(f'{key} MCC: {mcc_list}')
    print(f'{key} F1: {f1_list}')

