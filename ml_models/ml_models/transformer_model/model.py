import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleOrderBookModel(nn.Module):
    """
    A simple baseline model for order book prediction that handles variable-length inputs
    and provides flexible output options.
    
    This model:
    1. Flattens the input order book states
    2. Processes them through an MLP
    3. Outputs either scalar values or softmax probabilities
    """
    def __init__(
        self,
        hidden_dim=128,
        output_dim=1,
        use_softmax=False,
        dtype=torch.float32
    ):
        super(SimpleOrderBookModel, self).__init__()
        
        self.output_dim = output_dim
        self.use_softmax = use_softmax
        self.dtype = dtype
        
        # Layers to process flattened input
        self.fc_layers = nn.Sequential(
            nn.LazyLinear(hidden_dim),  # LazyLinear infers input size dynamically
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.LazyLinear(output_dim)
        )
        
        # Convert model parameters to specified dtype
        self.to(dtype)
        
    def forward(self, x):
        """
        Args:
            x: Order book states tensor [batch_size, time_steps, order_book_features]
                where order_book_features is a 2D representation of the order book
        
        Returns:
            Predictions [batch_size, output_dim]
        """
        # Ensure input is a tensor and convert to the model's dtype
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.dtype)
        else:
            x = x.to(dtype=self.dtype)
            
        # Use shape property instead of size() method
        batch_size = x.shape[0]
        
        # Flatten all dimensions except batch
        x_flat = x.reshape(batch_size, -1)
        
        # Process through network
        output = self.fc_layers(x_flat)
        
        # Apply softmax if needed
        if self.use_softmax:
            output = F.softmax(output, dim=1)
            
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer models.
    Adds information about position in the sequence to each token.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Calculate positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.shape[1], :]
        return x



class GaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        mean = y_pred[:, 0]
        var = y_pred[:, 1]
        var = torch.clamp(var, min=self.eps)
        
        nll = 0.5 * (torch.log(2 * math.pi * var) + (y_true - mean)**2 / var)
        
        return torch.mean(nll)
        

class BasicTransformerModel(nn.Module):
    """
    A simple transformer-based model for order book prediction.
    
    This model:
    1. Projects order book states to embedding dimension
    2. Adds positional encoding
    3. Processes them through transformer encoder layers
    4. Outputs either scalar values or softmax probabilities
    """
    def __init__(
        self,
        d_model=64,          # Embedding dimension
        nhead=4,             # Number of attention heads
        num_layers=2,        # Number of transformer layers
        output_dim=1,        # Number of output dimensions
        use_softmax=False,    # Whether to apply softmax to output
        loss_function="mse"
    ):
        super(BasicTransformerModel, self).__init__()
        
        self.output_dim = output_dim
        self.use_softmax = use_softmax
        self.d_model = d_model
        self.loss_function = loss_function

        # Initial projection layer for order book features
        self.input_projection = nn.LazyLinear(d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
    
    


    def forward(self, x):
        """
        Args:
            x: Order book states tensor [batch_size, time_steps, order_book_features]
                where order_book_features is a 2D representation of the order book
        
        Returns:
            Predictions [batch_size, output_dim]
        """
        # Ensure input is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Use shape property instead of size() method
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Reshape if needed to ensure each timestep has a flattened order book representation
        if len(x.shape) > 3:
            # If we have [batch, time, rows, cols], flatten the last two dimensions
            x = x.reshape(batch_size, seq_len, -1)
        
        # Project to embedding dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Process through transformer
        transformer_output = self.transformer_encoder(x)
        
        # Use the last timestep's output for prediction
        final_hidden = transformer_output[:, -1, :]
        
        # Project to output dimension
        output = self.output_projection(final_hidden)
        
        # Apply softmax if needed
        if self.use_softmax:
            output = F.softmax(output, dim=1)
            
        return output



import torch
import torch.nn as nn
import torch.optim as optim


def train_simple(model, loader, epochs=10, batch_size=32, lr=0.001):
    """Minimal training function for order book models"""

    
    # Loss function and optimizer
    if model.loss_function == "mse":
        criterion = nn.MSELoss()
    if model.loss_function == "ce":
        criterion = nn.CrossEntropyLoss()
    if model.loss_function == "mle":
        criterion = GaussianNLLLoss()


    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in loader:
            # Forward pass
            outputs = model(batch_x)
            
            # Calculate loss
            if model.use_softmax:
                loss = criterion(outputs, batch_y)
            else:
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model