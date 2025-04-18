import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def signed_log_transform(x):
    """Apply sign-preserving log transformation to data."""
    # Get the sign of the data
    signs = np.sign(x)
    log_abs = np.log1p(np.abs(x))
    return signs * log_abs

def signed_root_transform(x):
    """Apply sign-preserving log transformation to data."""
    # Get the sign of the data
    signs = np.sign(x)
    root_abs = np.sqrt(np.abs(x))
    return signs * root_abs

def normalise(x_train, x_val):
    # Compute mean and std along the samples axis (axis 0)
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    # Normalize while preserving 3D structure
    x_train = (x_train - train_mean) / train_std
    x_val = (x_val - train_mean) / train_std

    return x_train, x_val

def normalise_std(x_train, x_val):
    # Compute mean and std along the samples axis (axis 0)
    
    train_std = np.std(x_train, axis=0)

    # Normalize while preserving 3D structure
    x_train = x_train / train_std
    x_val = x_val / train_std

    return x_train, x_val



def build_data_set_from_features(
        input_features,
        output_features,
        window_len,
        horizon,
        steps_between):
    
    X_data = []
    Y_data = []

    for i in range(0, len(input_features) - window_len - horizon-1, steps_between):
        
        X_data.append(input_features[i:i + window_len])
        Y_data.append(output_features[i + horizon])

    return np.array(X_data), np.array(Y_data)



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



def evaluate_mse_prediction(predictions, targets):
    """
    Evaluate model's trend prediction accuracy for time series data with auxiliary features.
    
    Args:
        model: The prediction model to evaluate
        features: Primary time series features tensor or array
        aux_features: Auxiliary features tensor or array
        targets: Ground truth target values tensor or array
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    import torch
    
    
    
    # Make predictions - process one sample at a time

    
    # Convert predictions to tensor
    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)
    
    # Ensure proper shape
    y_true = targets
    y_pred = predictions

    # Filter out neutral trends
    non_zero = y_true != 0
    y_true = y_true[non_zero]
    y_pred = y_pred[non_zero]

    
    # Convert to binary trends (1 for positive, 0 for negative)
    y_true_trend = (y_true > 0).int()
    y_pred_trend = (y_pred > 0).int()
    
    # Calculate confusion matrix components
    tn = torch.sum((y_true_trend == 0) & (y_pred_trend == 0)).item()
    fp = torch.sum((y_true_trend == 0) & (y_pred_trend == 1)).item()
    fn = torch.sum((y_true_trend == 1) & (y_pred_trend == 0)).item()
    tp = torch.sum((y_true_trend == 1) & (y_pred_trend == 1)).item()
    
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
    print(f"Up trend accuracy:   {up_acc:.2%}")
    print(f"Down trend accuracy: {down_acc:.2%}")

    

    # Sort by prediction confidence (absolute value)
    zipped_lists = zip(y_pred, y_true)
    sorted_pairs = sorted(zipped_lists, key=lambda pair: abs(pair[0]), reverse=True)
    sorted_pred, sorted_true = zip(*sorted_pairs)

    # Convert to binary trends
    sorted_pred_trend = [1 if p > 0 else 0 for p in sorted_pred]
    sorted_true_trend = [1 if t > 0 else 0 for t in sorted_true]

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
        correct = sum(p == t for p, t in zip(sorted_pred_trend[:n_samples], sorted_true_trend[:n_samples]))
        accuracy = correct / n_samples
        
        percentages.append(i/100)
        accuracies.append(accuracy)

    # Create and save the graph
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(percentages, accuracies, 'b-')
    plt.xlabel('Proportion of Most Confident Predictions')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Confidence Level')
    plt.grid(True, alpha=0.3)
    plt.savefig('accuracy_by_confidence.png')
    plt.show()



