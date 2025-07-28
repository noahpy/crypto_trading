import torch
import torch.nn as nn
import torch.optim as optim






## PYTORCH TRAINING ##

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

