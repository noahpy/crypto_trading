import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class SimpleModel(nn.Module):
    def __init__(self, input_size=50*20, aux_size=50*5, output_size=1):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size + aux_size, 128)
        self.linear2 = nn.Linear(128, output_size)
        
    def forward(self, x, aux_features):
        # Flatten both inputs
        x = self.flatten(x)
        aux_features = self.flatten(aux_features)
        # Concatenate
        combined = torch.cat([x, aux_features], dim=1)
        # Process through linear layers
        x = F.relu(self.linear1(combined))
        x = self.linear2(x)
        return x.view(x.size(0), 1, -1)

class ImprovedNN(nn.Module):
    def __init__(self, input_size=50*20, aux_size=50*5, hidden_size=128, output_size=1, dropout_rate=0.1):
        super(ImprovedNN, self).__init__()
        
        # Main feature processing path with more layers and batch normalization
        self.main_path = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Auxiliary feature processing path
        self.aux_path = nn.Sequential(
            nn.Linear(aux_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU()
        )
        
        # Combined layers for final prediction
        self.combined_path = nn.Sequential(
            nn.Linear(hidden_size//2 + hidden_size//4, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//4, output_size)
        )

    def forward(self, x, aux_features):
        # Flatten inputs if necessary
        x = x.view(x.size(0), -1) if len(x.shape) > 2 else x
        aux = aux_features.view(aux_features.size(0), -1) if len(aux_features.shape) > 2 else aux_features
        
        # Process through respective paths
        x = self.main_path(x)
        aux = self.aux_path(aux)
        
        # Combine features
        combined = torch.cat((x, aux), dim=1)
        
        # Final prediction
        output = self.combined_path(combined)
        
        return output
    



class OrderBookModel(nn.Module):
    def __init__(self, input_size=50*20, aux_size=50*5, hidden_size=128, output_size=1, dropout_rate=0.0):
        super(OrderBookModel, self).__init__()
        
        # Set parameters based on inputs
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.horizons_backward = 50 - 1  # Assuming 50 timesteps
        self.num_order_buckets = 20      # Based on input shape
        self.num_order_book_features = aux_size // 50  # Features per timestep
        
        # First convolutional block
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=(2, 1), padding='same')
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(2, 1), padding='same')
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2), padding=0)
        
        # Second convolutional block
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=(2, 1), padding='same')
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=(2, 1), padding='same')
        self.conv2_3 = nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2), padding=0)
        
        # Third convolutional block
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=(2, 1), padding='same')
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=(2, 1), padding='same')
        self.conv3_3 = nn.Conv2d(32, 32, kernel_size=(1, 5), padding=0)
        
        # Inception module
        # Branch 1
        self.inc1_1 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding=0)
        self.inc1_2 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        
        # Branch 2
        self.inc2_1 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding=0)
        self.inc2_2 = nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0))
        
        # Branch 3
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.inc3 = nn.Conv2d(32, 64, kernel_size=(1, 1), padding=0)
        
        # Equivalent to SpatialDropout1D in TensorFlow
        self.spatial_dropout = nn.Dropout2d(dropout_rate)
        
        # Estimate LSTM input size (64*3 from inception module + auxiliary features)
        estimated_lstm_input_size = 192 + self.num_order_book_features
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=estimated_lstm_input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Dense layers
        self.dense1 = nn.Linear(hidden_size, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 32)
        
        # Output layer
        self.output_layer = nn.Linear(32, output_size)
        
    def convolution_layer(self, x):
        # Ensure input has the right shape (add channel dimension if needed)
        if len(x.shape) < 4:
            x = x.view(x.size(0), 1, -1, 20)  # Assuming last dimension is 20
            
        # First conv block
        x = F.leaky_relu(self.conv1_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv1_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv1_3(x), negative_slope=0.01)
        
        # Second conv block
        x = F.leaky_relu(self.conv2_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2_3(x), negative_slope=0.01)
        
        # Third conv block
        x = F.leaky_relu(self.conv3_1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3_3(x), negative_slope=0.01)
        
        # Inception module
        # Branch 1
        inc1 = F.leaky_relu(self.inc1_1(x), negative_slope=0.01)
        inc1 = F.leaky_relu(self.inc1_2(inc1), negative_slope=0.01)
        
        # Branch 2
        inc2 = F.leaky_relu(self.inc2_1(x), negative_slope=0.01)
        inc2 = F.leaky_relu(self.inc2_2(inc2), negative_slope=0.01)
        
        # Branch 3
        inc3 = self.maxpool(x)
        inc3 = F.leaky_relu(self.inc3(inc3), negative_slope=0.01)
        
        # Concatenate branches along channel dimension
        concatenated = torch.cat([inc1, inc2, inc3], dim=1)
        
        # Reshape to (batch_size, time_steps, features)
        batch_size = concatenated.size(0)
        time_steps = concatenated.size(2)
        features = concatenated.size(1) * concatenated.size(3)
        
        reshaped = concatenated.permute(0, 2, 1, 3).contiguous()
        reshaped = reshaped.view(batch_size, time_steps, features)
        
        return reshaped
        
    def forward(self, x, aux_features):
        # Process through convolutional layers
        conv_output = self.convolution_layer(x)
        
        # Apply spatial dropout equivalent to TF's SpatialDropout1D
        # PyTorch's Dropout2d works on the channel dimension, so we reshape temporarily
        conv_output = self.spatial_dropout(conv_output.unsqueeze(1)).squeeze(1)
        
        # Reshape aux_features if needed
        if len(aux_features.shape) == 2:
            batch_size = aux_features.size(0)
            aux_features = aux_features.view(batch_size, self.horizons_backward+1, self.num_order_book_features)
        
        # Concatenate with auxiliary features along the feature dimension (dim=2)
        lstm_inputs = torch.cat([conv_output, aux_features], dim=2)
        
        # Dynamically adjust LSTM input size if needed
        if self.lstm.input_size != lstm_inputs.size(2):
            new_lstm = nn.LSTM(
                input_size=lstm_inputs.size(2),
                hidden_size=self.hidden_size,
                batch_first=True
            ).to(lstm_inputs.device)
            
            # Initialize the new LSTM weights
            for name, param in new_lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            
            self.lstm = new_lstm
        
        # Process through LSTM - get last output only to match TensorFlow behavior
        lstm_output, _ = self.lstm(lstm_inputs)
        lstm_output = lstm_output[:, -1, :]
        
        # Dense layers with ReLU activation, adding L2 regularization during training
        dense1 = F.relu(self.dense1(lstm_output))
        dense1 = F.relu(self.dense2(dense1))
        dense1 = F.relu(self.dense3(dense1))
        
        # Output layer with activation based on your TF model
        # In TF you used 'linear' activation which is just identity, so no activation here
        output = self.output_layer(dense1)
        
        # Reshape to match TensorFlow's output shape
        output = output.view(output.size(0), 1, -1)
        
        return output