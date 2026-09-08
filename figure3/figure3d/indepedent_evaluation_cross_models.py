import scanpy as sc
import os.path as Path
from sklearn.metrics import f1_score
import os


folder = r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data\pick\new_model_results'

key_to_label = 'label'

# keys_to_predict = ['PCA','x_scGPT', 'X_finetune_scGPT', 'X_NicheCompass']
# keys_to_predict = ['PCA_binary', 'spaGFM_binary', 'X_finetune_scGPT_binary', 'X_NicheCompass_binary', 'X_scGPT_binary','novae_latent_binary','spatial_binary']

models = ['3M', '15M', '36M', '317M']


for model in models:
    f1_list = []
    for file_name in os.listdir(folder):
        if not file_name.endswith('.h5ad'):
            continue
        result = sc.read_h5ad(Path.join(folder, file_name))

        f1_list.append(
            f1_score(result.obs[key_to_label], result.obs[f'{model}_binary_predict'], average='macro'))
        del result

    print(f'{model} F1: {f1_list}')

