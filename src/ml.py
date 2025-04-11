import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim





# Custom dataset
class OrderBookDataset(Dataset):
    def __init__(self, features, aux_features, targets):
        self.features = torch.FloatTensor(features)
        self.aux_features = torch.FloatTensor(aux_features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.aux_features[idx], self.targets[idx]



def train_model(model, train_loader, criterion, optimizer, num_epochs=100):

    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, aux_features, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(features, aux_features)
            loss = criterion(outputs, targets)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        #if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    return model, losses



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



def evaluate_trend_prediction(model, features, aux_features, targets):
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

    # Prepare data and make predictions
    model.eval()
    
    # Convert inputs to tensors if they aren't already
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features)
    if not isinstance(aux_features, torch.Tensor):
        aux_features = torch.FloatTensor(aux_features)
    if not isinstance(targets, torch.Tensor):
        targets = torch.FloatTensor(targets)
    
    print(f"Processing {features.shape[0]} samples...")
    
    # Make predictions - process one sample at a time
    all_predictions = []
    with torch.no_grad():
        for i in range(features.shape[0]):
            # Get single sample with time steps
            sample = features[i:i+1]  # Add batch dimension (1, 50, 20)
            aux_sample = aux_features[i:i+1]  # Add batch dimension (1, 50, 5)
            
            # Make prediction for this sample
            pred = model(sample, aux_sample)
            
            # Store the prediction
            all_predictions.append(
                pred.item() if pred.numel() == 1 else pred.squeeze().mean().item()
            )
    
    # Convert predictions to tensor
    predictions = torch.tensor(all_predictions)
    
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

    return y_pred


