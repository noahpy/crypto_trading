import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

"""
    This file contains functions for evaluation during and after training
"""

## Metrics for Training ##


def trend_accuracy_metric(y_pred, y_true):
    # Flatten and move to CPU if needed
    y_true = y_true.flatten().cpu() if y_true.is_cuda else y_true.flatten()
    y_pred = y_pred.flatten().cpu() if y_pred.is_cuda else y_pred.flatten()
    
    # Find non-zero values in target data
    target_trends = y_true != 0
    if torch.sum(target_trends) == 0:
        return "No trends in data"
    
    # Get signs for relevant indices
    pred_signs = torch.sign(y_pred[target_trends])
    target_signs = torch.sign(y_true[target_trends])
    
    # Basic metrics
    total = len(target_signs)
    correct = torch.sum(pred_signs == target_signs).item()
    accuracy = correct / total
    
    # Confidence-based accuracy metrics
    conf = torch.abs(y_pred[target_trends])
    sorted_idx = torch.argsort(conf, descending=True)
    sorted_pred = pred_signs[sorted_idx]
    sorted_true = target_signs[sorted_idx]
    
    # Calculate accuracy at 10% intervals
    intervals = {}
    for i in range(1, 6):
        cutoff = int(total * i/20)
        if cutoff > 0:
            acc = torch.sum(sorted_pred[:cutoff] == sorted_true[:cutoff]).item() / cutoff
            intervals[i*5] = round(acc, 3)
    
    conf_str = " ".join([f"{k}%={v:.3f}" for k, v in intervals.items()])
    
    return f"Acc={accuracy:.3f} | {conf_str}"



## Final Evaluation ##


def print_enhanced_confusion_matrix(tn, fp, fn, tp):
    """
    Print a simple confusion matrix showing only the percentage of total samples in each category.
    
    Args:
        tn: True negatives (correctly predicted down)
        fp: False positives (incorrectly predicted up)
        fn: False negatives (incorrectly predicted down)
        tp: True positives (correctly predicted up)
    """
    # Calculate total
    total = tn + fp + fn + tp
    
    # Calculate percentages of total samples
    tn_pct = tn / total * 100 if total > 0 else 0
    fp_pct = fp / total * 100 if total > 0 else 0
    fn_pct = fn / total * 100 if total > 0 else 0
    tp_pct = tp / total * 100 if total > 0 else 0
    
    # Calculate row and column totals
    row_down_pct = (tn_pct + fp_pct)
    row_up_pct = (fn_pct + tp_pct)
    col_down_pct = (tn_pct + fn_pct)
    col_up_pct = (fp_pct + tp_pct)
    
    # Print the simple percentage matrix
    print("\nConfusion Matrix (% of Total Samples):")
    print("┌────────────┬─────────────────┬─────────────────┬─────────────┐")
    print("│            │ Predicted Down  │  Predicted Up   │    Total    │")
    print("├────────────┼─────────────────┼─────────────────┼───────── ───┤")
    print(f"│ Actual Down│     {tn_pct:5.1f}%      │     {fp_pct:5.1f}%      │   {row_down_pct:5.1f}%    │")
    print("├────────────┼─────────────────┼─────────────────┼─────────────┤")
    print(f"│ Actual Up  │     {fn_pct:5.1f}%      │     {tp_pct:5.1f}%      │   {row_up_pct:5.1f}%    │")
    print("├────────────┼─────────────────┼─────────────────┼─────────────┤")
    print(f"│ Total      │     {col_down_pct:5.1f}%      │     {col_up_pct:5.1f}%      │   100.0%    │")
    print("└────────────┴─────────────────┴─────────────────┴─────────────┘")
    
    # Print accuracy
    accuracy = (tp + tn) / total if total > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.1%}")


def evaluate_softmax_prediction(prediction, targets):
    """
    Evaluate softmax predictions and display percentage-based confusion matrix
    """
    # Detach tensors from computation graph and convert to numpy
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.detach().cpu().numpy()
    
    # Convert predictions to class indices
    pred_classes = np.argmax(prediction, axis=1)
    
    # Calculate accuracy
    acc = accuracy_score(targets, pred_classes)
    
    # Get confusion matrix
    cm = confusion_matrix(targets, pred_classes)
    
    # Convert to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Print results
    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix (percentages):")
    print(cm_percentage)
    
    return acc

def evaluate_mse_prediction(predictions, targets):
    """
    Evaluate model's trend prediction accuracy for time series data with auxiliary features.
    Args:
    predictions: Predicted values as numpy array
    targets: Ground truth target values as numpy array
    Returns:
    dict: Dictionary containing evaluation metrics
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Ensure proper shape
    y_true = targets
    y_pred = predictions

    # Filter out neutral trends
    non_zero = y_true != 0
    y_true = y_true[non_zero]
    y_pred = y_pred[non_zero]

    # Convert to binary trends (1 for positive, 0 for negative)
    y_true_trend = (y_true > 0).astype(int)
    y_pred_trend = (y_pred > 0).astype(int)

    # Calculate confusion matrix components
    tn = np.sum((y_true_trend == 0) & (y_pred_trend == 0))
    fp = np.sum((y_true_trend == 0) & (y_pred_trend == 1))
    fn = np.sum((y_true_trend == 1) & (y_pred_trend == 0))
    tp = np.sum((y_true_trend == 1) & (y_pred_trend == 1))

    # Create confusion matrix
    cm = [[tn, fp], [fn, tp]]

    # Calculate metrics
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Additional trend stats
    true_up = tp
    false_up = fp
    true_down = tn
    false_down = fn
    correct = true_up + true_down
    up_acc = true_up / (true_up + false_down) if (true_up + false_down) > 0 else 0
    down_acc = true_down / (true_down + false_up) if (true_down + false_up) > 0 else 0

    # Print results in a well-formatted way
    print_enhanced_confusion_matrix(tn, fp, fn, tp)
    print("\nTrend Prediction:")
    print(f"Correct trends: {correct}/{total} ({correct/total:.2%})")
    print(f"Up trend accuracy: {up_acc:.2%}")
    print(f"Down trend accuracy: {down_acc:.2%}")

    # Sort by prediction confidence (absolute value)
    indices = np.argsort(np.abs(y_pred))[::-1]  # Descending order
    sorted_pred = y_pred[indices]
    sorted_true = y_true[indices]

    # Convert to binary trends
    sorted_pred_trend = np.where(sorted_pred > 0, 1, 0)
    sorted_true_trend = np.where(sorted_true > 0, 1, 0)

    # Calculate accuracy at different percentage thresholds
    percentages = []
    accuracies = []
    total_samples = len(sorted_pred)

    for i in range(1, 101):  # 1% to 100%
        # Calculate samples to include
        n_samples = int(i * total_samples / 100)
        if n_samples == 0:
            continue

        # Calculate accuracy
        correct = np.sum(sorted_pred_trend[:n_samples] == sorted_true_trend[:n_samples])
        accuracy = correct / n_samples
        percentages.append(i/100)
        accuracies.append(accuracy)

    # Create and save the graph
    plt.figure(figsize=(8, 5))
    plt.plot(percentages, accuracies, 'b-')
    plt.xlabel('Proportion of Most Confident Predictions')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Confidence Level')
    plt.grid(True, alpha=0.3)
    plt.savefig('accuracy_by_confidence.png')
    plt.show()

    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'up_accuracy': up_acc,
        'down_accuracy': down_acc
    }


