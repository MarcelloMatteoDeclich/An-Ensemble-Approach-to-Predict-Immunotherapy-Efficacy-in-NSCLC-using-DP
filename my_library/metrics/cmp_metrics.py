import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def load_and_average_predictions(parent_dir, n_classes):
    """
    Load and average predictions from multiple parquet files across subdirectories.
    
    Args:
        parent_dir (str): Path to directory containing prediction files
        n_classes (int): Number of prediction classes
        
    Returns:
        pd.DataFrame: DataFrame with averaged predictions per slide
    """
    all_predictions = []

    # Traverse through all subdirectories
    for subfolder in os.listdir(parent_dir):
        subfolder_path = os.path.join(parent_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            # Load all parquet files in subdirectory
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".parquet"):
                    file_path = os.path.join(subfolder_path, file_name)
                    df = pd.read_parquet(file_path)
                    all_predictions.append(df)

    # Combine all predictions and average by slide
    combined_df = pd.concat(all_predictions, ignore_index=True)
    
    # Create aggregation dictionary for predictions and true labels
    agg_dict = {f"y_pred{i}": "mean" for i in range(n_classes)}
    agg_dict["y_true"] = "first"  # Take first true label per slide
    
    averaged_df = combined_df.groupby("slide").agg(agg_dict).reset_index()
    
    return averaged_df

def ld_avg_pred_majority(parent_dir, n_classes):
    """
    Load predictions and perform majority voting by slide.
    
    Args:
        parent_dir (str): Path to directory containing prediction files
        n_classes (int): Number of prediction classes
        
    Returns:
        pd.DataFrame: DataFrame with majority vote predictions per slide
    """
    all_predictions = []

    # Load all prediction files
    for subfolder in os.listdir(parent_dir):
        subfolder_path = os.path.join(parent_dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith(".parquet"):
                    file_path = os.path.join(subfolder_path, file_name)
                    df = pd.read_parquet(file_path)
                    all_predictions.append(df)

    combined_df = pd.concat(all_predictions, ignore_index=True)

    # Determine predicted class for each row
    def get_predicted_class(row):
        pred_columns = [col for col in row.index if col.startswith('y_pred')]
        max_col = row[pred_columns].idxmax(axis=0)
        return int(max_col.split('y_pred')[-1])
 
    combined_df['predicted_class'] = combined_df.apply(get_predicted_class, axis=1)
    
    # Perform majority voting by slide
    majority_voting_df = combined_df.groupby('slide')['predicted_class'].agg(lambda x: x.mode()[0]).reset_index()
    majority_voting_df['y_true'] = combined_df['y_true'].apply(lambda x: int(x))

    return majority_voting_df

def compute_metrics_with_class_accuracy(df, n_classes):
    """
    Compute comprehensive classification metrics including per-class statistics.
    
    Args:
        df (pd.DataFrame): DataFrame with true labels and predictions
        n_classes (int): Number of classes
        
    Returns:
        tuple: (dict of overall metrics, dict of per-class metrics)
    """
    true_y = df['y_true']
    y_preds = df[[f"y_pred{i}" for i in range(n_classes)]].values
    predicted_classes = np.argmax(y_preds, axis=1)

    # Compute overall metrics
    metrics = {
        "Accuracy": accuracy_score(true_y, predicted_classes),
        "Balanced_Accuracy": balanced_accuracy_score(true_y, predicted_classes),
        "Precision": precision_score(true_y, predicted_classes, average="weighted", zero_division=0),
        "Recall": recall_score(true_y, predicted_classes, average="weighted", zero_division=0),
        "F1-Score": f1_score(true_y, predicted_classes, average="weighted", zero_division=0),
        "Confusion Matrix": confusion_matrix(true_y, predicted_classes, labels=range(n_classes)).tolist(),
    }

    # Compute per-class metrics
    class_metrics = {}
    for cls in range(n_classes):
        cls_indices = (true_y == cls)
        
        if cls_indices.any():
            y_true_cls = (true_y == cls).astype(int)
            y_pred_cls = (predicted_classes == cls).astype(int)
            
            class_metrics[f"Class_{cls}"] = {
                "Accuracy": accuracy_score(true_y[cls_indices], predicted_classes[cls_indices]),
                "Precision": precision_score(y_true_cls, y_pred_cls, zero_division=0),
                "Recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
                "F1-score": f1_score(y_true_cls, y_pred_cls, zero_division=0)
            }
        else:
            class_metrics[f"Class_{cls}"] = {
                "Accuracy": 0.0,
                "Precision": 0.0,
                "Recall": 0.0,
                "F1-score": 0.0
            }

    return metrics, class_metrics

def compute_metrics_with_class_accuracy_maority(df, n_classes):
    """
    Compute metrics using majority voting predictions (note: contains typo in function name).
    
    Args:
        df (pd.DataFrame): DataFrame with true labels and majority predictions
        n_classes (int): Number of classes
        
    Returns:
        dict: Dictionary of classification metrics
    """
    true_y = df['y_true']
    predicted_classes = df['predicted_class']

    metrics = {
        "Accuracy": accuracy_score(true_y, predicted_classes),
        "Precision": precision_score(true_y, predicted_classes, average="weighted", zero_division=0),
        "Recall": recall_score(true_y, predicted_classes, average="weighted", zero_division=0),
        "F1-Score": f1_score(true_y, predicted_classes, average="weighted", zero_division=0),
        "Confusion Matrix": confusion_matrix(true_y, predicted_classes, labels=range(n_classes)).tolist(),
    }

    # Compute per-class accuracy
    class_accuracies = {}
    for cls in range(n_classes):
        cls_indices = (true_y == cls)
        class_accuracies[f"Accuracy_Class_{cls}"] = (
            accuracy_score(true_y[cls_indices], predicted_classes[cls_indices]) 
            if cls_indices.any() else 0.0
        )

    metrics["Class Accuracies"] = class_accuracies
    return metrics

def plot_confusion_matrix(conf_matrix, class_names, cmap="Blues"):
    """
    Plot a confusion matrix with counts and percentages.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix array
        class_names (list): List of class names
        cmap (str): Color map for visualization
    """
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    
    annotations = np.array([
        [f"{conf_matrix[i, j]}\n({conf_matrix_percent[i, j]:.1f}%)" 
         for j in range(conf_matrix.shape[1])]
        for i in range(conf_matrix.shape[0])
    ])

    sns.heatmap(conf_matrix, annot=annotations, fmt="", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Best Response Classification Matrix', fontsize=14, fontweight='bold')
    plt.show()

def plot_confusion_matrix_complex(conf_matrix, class_names, model_name="", setup="", 
                                cmap="Blues", working_directory=""):
    """
    Enhanced confusion matrix plot with saving capability.
    
    Args:
        conf_matrix (np.ndarray): Confusion matrix array
        class_names (list): List of class names
        model_name (str): Name of model for title
        setup (str): Setup description for title
        cmap (str): Color map for visualization
        working_directory (str): Directory to save plot
    """
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))
    
    annotations = np.array([
        [f"{conf_matrix[i, j]}\n({conf_matrix_percent[i, j]:.1f}%)" 
         for j in range(conf_matrix.shape[1])]
        for i in range(conf_matrix.shape[0])
    ])

    sns.heatmap(conf_matrix, annot=annotations, fmt="", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{model_name} - {setup} Classification Matrix', 
              fontsize=14, fontweight='bold')
    
    if working_directory:
        os.makedirs(f"{working_directory}/metriche/plots", exist_ok=True)
        plt.savefig(
            f"{working_directory}/metriche/plots/{model_name}-{setup}-classification-Matrix.png", 
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()

def find_classes(directory_path):
    """
    Extract class labels from MIL parameters JSON file.
    
    Args:
        directory_path (str): Path to model directory
        
    Returns:
        tuple: (outcome_labels, outcomes) from JSON config
    """
    model_folder = [item for item in Path(directory_path).iterdir() if item.is_dir()][0]
    json_file_path = os.path.join(directory_path, model_folder, "mil_params.json")

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    return data['mil_params']['outcome_labels'], data['mil_params']['outcomes']

def compute_metrics(parent_directory, n_classes, base_folder):
    """
    Main function to compute and display all metrics.
    
    Args:
        parent_directory (str): Path to prediction files
        n_classes (int): Number of classes
        base_folder (str): Base directory for model config
        
    Returns:
        tuple: (metrics, class_metrics, conf_matrix, class_names)
    """
    out_lab, out = find_classes(base_folder)
    n_classes = len(out_lab)

    averaged_predictions = load_and_average_predictions(parent_directory, n_classes)
    metrics, class_metrics = compute_metrics_with_class_accuracy(averaged_predictions, n_classes)

    # Display metrics
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(metric + ":")
            for sub_metric, sub_value in value.items():
                print(f"  {sub_metric}: {sub_value}")
        else:
            print(f"{metric}: {value}")

    # Prepare confusion matrix data
    class_values = out_lab.values()
    class_names = [f"Class {i}" for i in class_values]
    conf_matrix = np.array(metrics["Confusion Matrix"])

    return metrics, class_metrics, conf_matrix, class_names