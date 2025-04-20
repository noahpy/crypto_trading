import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

## DATA TRANSFORM ## 

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


## PYTORCH TRAINING ##


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

def proper_init_weights(model, loader, device):
    """Properly initialize model weights with careful scaling to avoid NaN issues"""

    # Initialize lazy parameters with a dummy forward pass
    try:
        # Get a sample input from the loader
        for batch_x, _ in loader:
            # Just need one sample to initialize parameters
            sample_x = batch_x[:1].to(device)
            with torch.no_grad():
                _ = model(sample_x)  # This initializes lazy parameters
            break
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return None


    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'norm' in name:  # Layer norm weights
                nn.init.ones_(param)
            elif param.dim() >= 2:  # For matrices (linear layers)
                # Xavier/Glorot initialization with a smaller scale
                nn.init.xavier_uniform_(param, gain=0.5)
            else:  # For vectors
                nn.init.uniform_(param, -0.1, 0.1)
        elif 'bias' in name:
            nn.init.zeros_(param)  # Always initialize biases to zero

    return model

def train(model, loader_train, loader_val, epochs=10, lr=0.001, init_params=True, metrics=[]):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)

    if init_params:
      model = proper_init_weights(model, loader_train, device)


    # Loss function setup
    if model.loss_function == "mse":
        criterion = nn.MSELoss()
    elif model.loss_function == "ce":
        criterion = nn.CrossEntropyLoss()
        # Ensure model doesn't apply softmax if using CE loss
        if model.use_softmax:
            print("Warning: use_softmax=True with CrossEntropyLoss may cause issues.")
    elif model.loss_function == "mle":
        criterion = GaussianNLLLoss()
    else:
        raise ValueError(f"Unknown loss function: {model.loss_function}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        
        train_batch_count = 0
        val_batch_count = 0

        train_pred = []
        train_target = []

        val_pred = []
        val_target = []

        for batch_x, batch_y in loader_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)


            # Loss calculation
            if model.use_softmax:
                loss = criterion(outputs, batch_y)
            else:
                loss = criterion(outputs.squeeze(), batch_y.squeeze())

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batch_count += 1

            train_pred.append(outputs.detach().cpu())
            train_target.append(batch_y.detach().cpu())

        train_pred = torch.cat(train_pred, dim=0)
        train_target = torch.cat(train_target, dim=0)

        for batch_x, batch_y in loader_val:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            # Loss calculation
            if model.use_softmax:
                loss = criterion(outputs, batch_y)
            else:
                loss = criterion(outputs.squeeze(), batch_y.squeeze())

            val_loss += loss.item()
            val_batch_count += 1
            val_pred.append(outputs.detach().cpu())
            val_target.append(batch_y.detach().cpu())

        val_pred = torch.cat(val_pred, dim=0)
        val_target = torch.cat(val_target, dim=0)
        

        avg_loss_train = train_loss / len(loader_train)
        avg_loss_val = val_loss / len(loader_train)
        print(f'Epoch {epoch+1}/{epochs}, train loss: {avg_loss_train:.4f}, val loss {avg_loss_val:.4f}')
        
        metric_string = "TRAIN: "
        for metric in metrics:
            metric_string += f"{metric(train_pred, train_target)} "
        print(metric_string)

        metric_string = "VAL:   "
        for metric in metrics:
            metric_string += f"{metric(val_pred, val_target)} "
        print(metric_string)

    return model


## LOAD AND STORE MODEL ##

def store_model(
    coin,
    time_delta_ms,
    window_length,
    target,
    model_name,
    model,
    input_feature_creator,
    description):
    
    if len(description) < 20:
        print("Please write a longer description")
        return  # Exit the function if description is too short
    
    import os
    
    # Create the model path
    model_path = f"ml_models/{coin}/time_delta={time_delta_ms}ms/window_length={window_length}/target={target}/{model_name}"
    
    # Create the directory structure if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Save the model
    torch.save(model, f'{model_path}/model.pth')
    
    # Save the feature creator
    fc_path = f'{model_path}/input_feature_creator.pkl'
    with open(fc_path, 'wb') as f:  # 'wb' for write binary
        pickle.dump(input_feature_creator, f)  # Note: using 'f' not 'fc_path'
    
    # Save the description
    desc_file_path = f'{model_path}/description.txt'
    with open(desc_file_path, 'w') as file:
        file.write(description)
    
    print(f"Model, feature creator, and description saved to {model_path}")


def load_model(
        coin,
        time_delta_ms, 
        window_length,
        target,
        model_name):
    """
    Load a saved model and its input feature creator.
    
    Parameters:
    - coin: The cryptocurrency symbol
    - time_delta_ms: Time delta in milliseconds
    - target: Target variable name
    - model_name: Name of the model
    
    Returns:
    - model: The loaded PyTorch model
    - input_feature_creator: The loaded feature creator
    """
    # Construct the paths
    model_path = f"ml_models/{coin}/time_delta={time_delta_ms}ms/window_length={window_length}/target={target}/{model_name}"
    model_file = f'{model_path}/model.pth'
    fc_path = f'{model_path}/input_feature_creator.pkl'
    
    # Load the model
    model = torch.load(model_file, map_location=torch.device('cpu'))
    model.eval()  # Set to evaluation mode
    
    # Load the input feature creator
    with open(fc_path, 'rb') as f:  # 'rb' for read binary
        input_feature_creator = pickle.load(f)
    
    print(f"Successfully loaded model and feature creator from {model_path}")
    return model, input_feature_creator

## EVALUATION ##


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


