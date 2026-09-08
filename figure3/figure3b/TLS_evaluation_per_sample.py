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



def convert_logits_to_predictions(adata, train_adata):
    """
    Convert model logits to predicted labels and add them to the AnnData object.

    :param adata: AnnData object containing the original data.

    :return: Updated AnnData object with predicted labels.
    """


    # Get the index of the max log-probability (the predicted class)
    probabilities = softmax(adata.obsm['logits'], axis=1)
    predicted_codes = np.argmax(probabilities, axis=1)

    # Add predicted labels to the AnnData object
    adata.obs['predicted_TLS'] = pd.Categorical.from_codes(predicted_codes, categories=train_adata.obs['manual_anno_tls'].cat.categories, ordered=True)

    adata.obs['predicted_probabilities'] = np.max(probabilities, axis=1)

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
                        annot_kws={'size': 16},
                        vmin=0,
                        vmax=1,
                        row_cluster=False,
                        col_cluster=False,
                        figsize=(14, 14))

    ax.savefig(save_file_path, dpi=300, bbox_inches='tight')




adata = sc.read_h5ad(r"C:\Users\hef\Downloads\spaGFM\G2PM_finetune\results\KC1.h5ad")
train_adata = sc.read_h5ad(r'C:\Users\hef\Downloads\spaGFM\G2PM_finetune\results\KC3.h5ad')
convert_logits_to_predictions(adata, train_adata)

# remove outside predictions
adata = adata[adata.obs['predicted_TLS'].isin(adata.obs['manual_anno_tls'].cat.categories)]
adata = adata[adata.obs['PCA_predict'].isin(adata.obs['manual_anno_tls'].cat.categories)]
adata = adata[adata.obs['scGPT_predict'].isin(adata.obs['manual_anno_tls'].cat.categories)]

print('****************************************************\n')
print('now evaluate spaGFM\n')
# plot_roc_curve(save_folder=r'.\results', true_labels=adata.obs['is_TLS_filtered'], predicted_probabilities=adata.obs['predicted_probabilities'], transfer_labels=adata.obs['predicted_TLS'])
# plot_precision_recall_curve(save_folder=r'.\results', true_labels=adata.obs['is_TLS_filtered'], predicted_probabilities=adata.obs['predicted_probabilities'], transfer_labels=adata.obs['predicted_TLS'])
# plot_spatial_distribution(adata, save_folder=r'.\results')
res_dict = {
    "accuracy": accuracy_score(adata.obs['manual_anno_tls'],  adata.obs['predicted_TLS']),
    "precision": precision_score(adata.obs['manual_anno_tls'],  adata.obs['predicted_TLS'], average="macro"),
    "recall": recall_score(adata.obs['manual_anno_tls'],  adata.obs['predicted_TLS'], average="macro"),
    "macro_f1": f1_score(adata.obs['manual_anno_tls'],  adata.obs['predicted_TLS'], average="macro"),
}

print(res_dict)
plot_confusion_matrix(adata.obs['manual_anno_tls'], adata.obs['predicted_TLS'], save_file_path=r'.\results\spaGFM_confusion_matrix_clustermap.png')


print('****************************************************')
print('now evaluate scGPT\n')
# accuracy = sklmetrics.accuracy_score(adata.obs['is_TLS_filtered'], adata.obs['predicted_TLS'])
# roc = sklmetrics.roc_auc_score(adata.obs['is_TLS_filtered'], adata.obsm['logits'][:, 1])  # Assuming binary classification and logits[:, 1] corresponds to the positive class
# f1 = sklmetrics.f1_score(adata.obs['is_TLS_filtered'], adata.obs['predicted_TLS'], average='binary')  # Use 'weighted' for multi-class classification
res_dict = {
    "accuracy": accuracy_score(adata.obs['manual_anno_tls'],  adata.obs['scGPT_predict']),
    "precision": precision_score(adata.obs['manual_anno_tls'],  adata.obs['scGPT_predict'], average="macro"),
    "recall": recall_score(adata.obs['manual_anno_tls'],  adata.obs['scGPT_predict'], average="macro"),
    "macro_f1": f1_score(adata.obs['manual_anno_tls'],  adata.obs['scGPT_predict'], average="macro"),
}
print(res_dict)
plot_confusion_matrix(adata.obs['manual_anno_tls'], adata.obs['scGPT_predict'], save_file_path=r'.\results\scGPT_confusion_matrix_clustermap.png')


print('****************************************************')
print('now evaluate PCA\n')
res_dict = {
    "accuracy": accuracy_score(adata.obs['manual_anno_tls'],  adata.obs['PCA_predict']),
    "precision": precision_score(adata.obs['manual_anno_tls'],  adata.obs['PCA_predict'], average="macro"),
    "recall": recall_score(adata.obs['manual_anno_tls'],  adata.obs['PCA_predict'], average="macro"),
    "macro_f1": f1_score(adata.obs['manual_anno_tls'],  adata.obs['PCA_predict'], average="macro"),
}
print(res_dict)
plot_confusion_matrix(adata.obs['manual_anno_tls'], adata.obs['PCA_predict'], save_file_path=r'.\results\PCA_confusion_matrix_clustermap.png')

plot_spatial_distribution(adata, save_folder=r'.\results', x_key='x_pixel',  y_key = 'y_pixel',
                          plot_keys=['manual_anno_tls', 'predicted_TLS', 'scGPT_predict', 'PCA_predict'])