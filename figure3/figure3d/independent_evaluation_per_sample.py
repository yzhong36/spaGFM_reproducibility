import  numpy as np
import  scanpy as sc
import pandas as pd
from scipy.special import softmax
import sklearn.metrics as sklmetrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as Path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def plot_roc_curve(save_folder,true_labels, predicted_probabilities, transfer_labels):
    """
    Plot the ROC curve for the predicted labels in the AnnData object.

    :param adata: AnnData object containing the original data and predicted labels.
    """
    # Get true labels and predicted probabilities
    # true_labels = adata.obs['true_TLS'].astype(int)  # Assuming 'true_TLS' is the column with true labels
    # predicted_probabilities = adata.obsm['logits'][:, 1]  # Assuming binary classification and logits are in obsm

    # Calculate false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = sklmetrics.roc_curve(true_labels, predicted_probabilities)

    # Calculate AUC
    auc_score = sklmetrics.auc(fpr, tpr)

    # For binary classification, with labels [0, 1]
    tn, fp, fn, tp = confusion_matrix(true_labels, transfer_labels, labels=[0, 1]).ravel()

    # Calculate TPR (Recall or Sensitivity): TP / (TP + FN)
    baselin_TPR = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Calculate FPR: FP / (FP + TN)
    baselin_FPR = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.plot(baselin_FPR, baselin_TPR, marker='*', ms=15, mfc='red', label='Baseline')  # Baseline point


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    # Save Figure with 300 DPI
    plt.savefig(Path.join(save_folder, "roc_curve.png"), dpi=300, bbox_inches='tight')

def plot_precision_recall_curve(save_folder,true_labels, predicted_probabilities, transfer_labels):
    """
    Plot the Precision-Recall curve for the predicted labels in the AnnData object.

    :param adata: AnnData object containing the original data and predicted labels.
    """
    # Get true labels and predicted probabilities
    # true_labels = adata.obs['true_TLS'].astype(int)  # Assuming 'true_TLS' is the column with true labels
    # predicted_probabilities = adata.obsm['logits'][:, 1]  # Assuming binary classification and logits are in obsm

    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = sklmetrics.precision_recall_curve(true_labels, predicted_probabilities)


    baseline_precision = precision_score(true_labels, transfer_labels, average="macro"),
    baseline_recall = recall_score(true_labels, transfer_labels, average="macro"),


    # Calculate AUC for Precision-Recall curve
    auc_score = sklmetrics.auc(recall, precision)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (AUC = {auc_score:.2f})')
    plt.plot(baseline_recall, baseline_precision, marker='*', ms=15, mfc='red', label='Baseline')  # Baseline point
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper left')
    # Save Figure with 300 DPI
    plt.savefig(Path.join(save_folder, "precision_recall_curve.png"), dpi=300, bbox_inches='tight')



def convert_logits_to_predictions(adata):
    """
    Convert model logits to predicted labels and add them to the AnnData object.

    :param adata: AnnData object containing the original data.

    :return: Updated AnnData object with predicted labels.
    """


    # Get the index of the max log-probability (the predicted class)
    probabilities = softmax(adata.obsm['logits'], axis=1)
    predicted_codes = np.argmax(probabilities, axis=1)

    # Add predicted labels to the AnnData object
    adata.obs['predicted_binary_TLS'] = predicted_codes

    adata.obs['predicted_binary_probabilities'] = np.max(probabilities, axis=1)

    return adata


def plot_spatial_distribution(adata, save_folder, plot_keys, x_key='x_centroid',  y_key = 'y_centroid'):
    """
    Plot the spatial distribution of predicted labels in the AnnData object.

    :param adata: AnnData object containing the original data and predicted labels.
    """
    # custom_colors = {
    #     'ETLS': '#FF0000',  # Red
    #     'GC': '#00FF00',  # Green
    #     'INFL': 'blue',  # Blue (using a color name)
    #     'MTLS' : 'orange',
    #     'NOR' : 'yellow',
    #     'TUM' : 'brown',
    #     'UNASSIGNED': 'pink'  # Light gray (using an RGB tuple)
    # }

    for plot_key in plot_keys:

        sc.pl.scatter(
            adata, x=x_key, y=y_key, color=plot_key, palette= 'tab10',
            show=False,
            title=f'Spatial Distribution of {plot_key}',
        )
        # Save the figure with specific parameters
        plt.savefig(Path.join(save_folder, f"{plot_key}.png") , dpi=300, bbox_inches="tight")
        plt.cla()

    # sc.pl.scatter(
    #     adata, x=x_key, y=y_key, color='predicted_TLS',
    #     show=False,
    #     title=f'Spatial Distribution of Predicted TLS',
    # )
    # # Save the figure with specific parameters
    # plt.savefig(Path.join(save_folder, "predicted_TLS.png"), dpi=300, bbox_inches="tight")
    # plt.cla()
    # sc.pl.scatter(
    #     adata, x=x_key, y=y_key, color='is_TLS_filtered',
    #     show=False,
    #     title=f'Spatial Distribution of labeled TLS',
    # )
    # plt.savefig(Path.join(save_folder, "labels.png"), dpi=300, bbox_inches="tight")


def plot_confusion_matrix(true_labels, predicted_labels, save_file_path):
    """
    Plot the confusion matrix for the predicted labels in the AnnData object.

    :param adata: AnnData object containing the original data and predicted labels.
    """
    # Get true labels and predicted labels
    # true_labels = adata.obs['true_TLS'].astype(int)  # Assuming 'true_TLS' is the column with true labels
    # predicted_labels = adata.obs['predicted_TLS'].astype(int)  # Assuming 'predicted_TLS' is the column with predicted labels


    cell_type_list = np.unique(true_labels)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=cell_type_list)
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    df = pd.DataFrame(matrix, index=cell_type_list[:matrix.shape[0]], columns=cell_type_list[:matrix.shape[1]])

    ax = sns.clustermap(df,
                        cmap='Purples',
                        annot=True,
                        fmt=".2f",
                        annot_kws={'size': 25},
                        vmin=0,
                        vmax=1,
                        row_cluster=False,
                        col_cluster=False,
                        figsize=(14, 14))

    ax.savefig(save_file_path, dpi=300, bbox_inches='tight')

import os

folder = r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data\pick\probability'
save_folder = r'.\results'
os.makedirs(save_folder, exist_ok=True)
label_number = {'TLS':0, 'NO_TLS':2}
predict_binary_map = {0:0, 1:0, 2:2}
probability_cutoffs = [0.6, 0.65, 0.7]
# models = ['3M', '15M', '36M', '317M']
models = ['317M']

for file_name in os.listdir(folder):
    file_path = os.path.join(folder, file_name)
    if not file_name.endswith(".h5ad"):
       continue
    adata = sc.read_h5ad(file_path)
    # train_adata = sc.read_h5ad(r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\data\LC3.h5ad')

    # convert label into id
    adata.obs['binary_tls'] = adata.obs['TLS_2_cat'].map(label_number)

    #
    # print('****************************************************\n')
    # print('now evaluate spaGFM\n')
    #
    # for model_name in models:
    #     best_f1 = 0
    #     best_walk = 0
    #     for walk_length in range(1, 9):
    #         print(f"walk_length: {walk_length}")
    #         # adata.obsm['logits'] = adata.obsm[f"{str(walk_length)}_binary_logits"]
    #         # convert_logits_to_predictions(adata)
    #
    #         # plot_roc_curve(save_folder=r'.\results', true_labels=adata.obs['is_TLS_filtered'], predicted_probabilities=adata.obs['predicted_probabilities'], transfer_labels=adata.obs['predicted_TLS'])
    #         # plot_precision_recall_curve(save_folder=r'.\results', true_labels=adata.obs['is_TLS_filtered'], predicted_probabilities=adata.obs['predicted_probabilities'], transfer_labels=adata.obs['predicted_TLS'])
    #         # plot_spatial_distribution(adata, save_folder=r'.\results')
    #         adata.obs[f"{model_name}_{str(walk_length)}_binary_predict"] = adata.obs[f"{model_name}_{str(walk_length)}_prediction"].map(predict_binary_map)
    #         res_dict = {
    #             "accuracy": accuracy_score(adata.obs['binary_tls'],  adata.obs[f"{model_name}_{str(walk_length)}_binary_predict"]),
    #             "precision": precision_score(adata.obs['binary_tls'],  adata.obs[f"{model_name}_{str(walk_length)}_binary_predict"], average="macro"),
    #             "recall": recall_score(adata.obs['binary_tls'],  adata.obs[f"{model_name}_{str(walk_length)}_binary_predict"], average="macro"),
    #             "macro_f1": f1_score(adata.obs['binary_tls'],  adata.obs[f"{model_name}_{str(walk_length)}_binary_predict"], average="macro"),
    #         }
    #
    #         print(res_dict)
    #         # plot_confusion_matrix(adata.obs['label'], adata.obsm[f"{str(walk_length)}_binary_tls_predict"], save_file_path=f".\\results\\{walk_length}_confusion_matrix_clustermap.png")
    #         if best_f1 < res_dict['macro_f1']:
    #             best_f1 = res_dict['macro_f1']
    #             best_walk = walk_length
    #     # save the best walk result as final
    #     adata.obs[f'{model_name}_predict'] = adata.obs[f'{model_name}_{best_walk}_prediction']
    #     adata.obs[f'{model_name}_binary_predict'] = adata.obs[f"{model_name}_{str(walk_length)}_binary_predict"]
    #
    # adata.write_h5ad(file_path)

    for model in models:
        print('****************************************************')
        print(f'now evaluate {model}\n')
        prediction_key = f"{model}_6_prediction"
        probability_key = f"{model}_6_probability"
        fallback_probability_key = f"{model}_6_prediction_probability"

        if probability_key not in adata.obs and fallback_probability_key in adata.obs:
            probability_key = fallback_probability_key

        if prediction_key not in adata.obs:
            raise KeyError(f"Missing prediction column: {prediction_key}")
        if probability_key not in adata.obs:
            raise KeyError(f"Missing probability column: {probability_key}")

        base_predict = adata.obs[prediction_key].map(predict_binary_map)

        if {"x_pixel", "y_pixel"}.issubset(adata.obs.columns):
            spatial = adata.obs[["x_pixel", "y_pixel"]].to_numpy()
        elif {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
            spatial = adata.obs[["x_centroid", "y_centroid"]].to_numpy()
        elif "spatial" in adata.obsm:
            spatial = adata.obsm["spatial"]
        else:
            raise KeyError(
                "sc.pl.spatial requires adata.obsm['spatial'] or coordinate "
                "columns x_pixel/y_pixel or x_centroid/y_centroid."
            )

        adata.obsm["spatial"] = spatial
        x_min, y_min = np.nanmin(spatial, axis=0)
        x_max, y_max = np.nanmax(spatial, axis=0)
        x_span = x_max - x_min
        y_span = y_max - y_min
        plot_margin = max(x_span, y_span) * 0.03
        plot_spot_size = max(max(x_span, y_span) / 250, 10)

        print(f"spatial x range: {x_min:.2f}-{x_max:.2f}, y range: {y_min:.2f}-{y_max:.2f}, spot_size: {plot_spot_size:.2f}")

        for probability_cutoff in probability_cutoffs:
            plot_key = f"{model}_cutoff_{str(probability_cutoff).replace('.', '_')}_predict"
            cutoff_predict = base_predict.copy()
            cutoff_predict.loc[adata.obs[probability_key] < probability_cutoff] = label_number['NO_TLS']
            adata.obs[plot_key] = cutoff_predict.astype("Int64").astype("category")
            if probability_cutoff == 0.65:
                adata.obs["317M_predict"] = cutoff_predict

            print(f"{plot_key} value counts:")
            print(adata.obs[plot_key].value_counts(dropna=False))

            sc.pl.spatial(
                adata,
                color=plot_key,
                spot_size=plot_spot_size,
                frameon=False,  # Removes the outer box frame
                img_key=None,
                crop_coord=(
                    x_min - plot_margin,
                    x_max + plot_margin,
                    y_min - plot_margin,
                    y_max + plot_margin,
                ),
                title=f"{model} cutoff {probability_cutoff}",
                show=False,
            )
            plt.savefig(Path.join(save_folder, f"{file_name}_{plot_key}_spatial.png"), dpi=300, bbox_inches="tight")
            plt.close()
            # accuracy = sklmetrics.accuracy_score(adata.obs['is_TLS_filtered'], adata.obs['predicted_TLS'])
            # roc = sklmetrics.roc_auc_score(adata.obs['is_TLS_filtered'], adata.obsm['logits'][:, 1])  # Assuming binary classification and logits[:, 1] corresponds to the positive class
            # f1 = sklmetrics.f1_score(adata.obs['is_TLS_filtered'], adata.obs['predicted_TLS'], average='binary')  # Use 'weighted' for multi-class classification
            res_dict = {
                "cutoff": probability_cutoff,
                "accuracy": accuracy_score(adata.obs['binary_tls'],  adata.obs[plot_key]),
                "precision": precision_score(adata.obs['binary_tls'],  adata.obs[plot_key],pos_label=0, average="binary"),
                "recall": recall_score(adata.obs['binary_tls'],  adata.obs[plot_key],pos_label=0, average="binary"),
                "macro_f1": f1_score(adata.obs['binary_tls'],  adata.obs[plot_key], average="macro"),
            }
            print(res_dict)

    adata.write_h5ad(file_path)
    # plot_confusion_matrix(adata.obs['binary_tls'], adata.obs['scGPT_binary_predict'], save_file_path=r'.\results\finetune_scGPT_binary_confusion_matrix_clustermap.png')

    # print('****************************************************')
    # print('now evaluate scGPT\n')
    # # accuracy = sklmetrics.accuracy_score(adata.obs['is_TLS_filtered'], adata.obs['predicted_TLS'])
    # # roc = sklmetrics.roc_auc_score(adata.obs['is_TLS_filtered'], adata.obsm['logits'][:, 1])  # Assuming binary classification and logits[:, 1] corresponds to the positive class
    # # f1 = sklmetrics.f1_score(adata.obs['is_TLS_filtered'], adata.obs['predicted_TLS'], average='binary')  # Use 'weighted' for multi-class classification
    # res_dict = {
    #     "accuracy": accuracy_score(adata.obs['binary_tls'],  adata.obs['scGPT_binary_predict']),
    #     "precision": precision_score(adata.obs['binary_tls'],  adata.obs['scGPT_binary_predict'],pos_label=1.0, average="binary"),
    #     "recall": recall_score(adata.obs['binary_tls'],  adata.obs['scGPT_binary_predict'],pos_label=1.0, average="binary"),
    #     "macro_f1": f1_score(adata.obs['binary_tls'],  adata.obs['scGPT_binary_predict'],pos_label=1.0, average="binary"),
    # }
    # print(res_dict)
    # plot_confusion_matrix(adata.obs['binary_tls'], adata.obs['scGPT_binary_predict'], save_file_path=r'.\results\finetune_scGPT_binary_confusion_matrix_clustermap.png')


    # print('****************************************************')
    # print('now evaluate PCA\n')
    # res_dict = {
    #     "accuracy": accuracy_score(adata.obs['binary_tls'],  adata.obs['PCA_binary_predict']),
    #     "precision": precision_score(adata.obs['binary_tls'],  adata.obs['PCA_binary_predict'], pos_label=1.0, average="binary"),
    #     "recall": recall_score(adata.obs['binary_tls'],  adata.obs['PCA_binary_predict'], pos_label=1.0, average="binary"),
    #     "macro_f1": f1_score(adata.obs['binary_tls'],  adata.obs['PCA_binary_predict'], pos_label=1.0, average="binary"),
    # }
    # print(res_dict)
    # plot_confusion_matrix(adata.obs['binary_tls'], adata.obs['PCA_binary_predict'], save_file_path=r'.\results\PCA_binary_confusion_matrix_clustermap.png')

    # plot_spatial_distribution(adata, save_folder=r'.\results', x_key='x_pixel',  y_key = 'y_pixel',
    #                           plot_keys=['binary_tls', 'predicted_TLS', 'scGPT_predict', 'PCA_predict'])
