import scanpy as sc
import os.path as Path
from sklearn.metrics import matthews_corrcoef,precision_score, recall_score, f1_score
import os


folder = r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data\pick\probability'

key_to_label = 'label'

# keys_to_predict = ['PCA','x_scGPT', 'X_finetune_scGPT', 'X_NicheCompass']
keys_to_predict = ['PCA_binary', 'spaGFM_binary', 'X_finetune_scGPT_binary', 'X_NicheCompass_binary', 'X_scGPT_binary','novae_latent_binary','spatial_binary']

for key in keys_to_predict:
    precision_list = []
    rcall_list = []
    mcc_list = []
    f1_list = []

    for file_name in os.listdir(folder):
        if not file_name.endswith('.h5ad'):
            continue


        result = sc.read_h5ad(Path.join(folder, file_name))

        precision_list.append(
            precision_score(result.obs[key_to_label], result.obs[f'{key}_predict'],
                            average='macro'))
        rcall_list.append(
            recall_score(result.obs[key_to_label], result.obs[f'{key}_predict'], average='macro'))

        mcc_list.append(matthews_corrcoef(result.obs[key_to_label], result.obs[f'{key}_predict']))

        f1_list.append(
            f1_score(result.obs[key_to_label], result.obs[f'{key}_predict'], average='macro'))
        del result

    print(f'{key} Precision: {precision_list}')
    print(f'{key} Recall: {rcall_list}')
    print(f'{key} MCC: {mcc_list}')
    print(f'{key} F1: {f1_list}')

