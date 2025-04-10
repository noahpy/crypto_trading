import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class AttentionModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # Initial CNN layers
        self.conv_first = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(0.01)
        )
        
        # Inception module
        self.inception_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(0.01)
        )
        
        self.inception_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(0.01)
        )
        
        self.inception_3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01)
        )
        
        # LSTM and Dense layers
        self.encoder_lstm = nn.LSTM(192, latent_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(latent_dim + 2, latent_dim, batch_first=True)
        
        # Separate heads for mean and variance
        self.mean_head = nn.Linear(latent_dim + latent_dim, 1)
        self.log_var_head = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, 1),
            nn.Tanh(),  # Limit range of log variance
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x, decoder_input):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.conv_first(x)
        
        # Inception module
        branch1 = self.inception_1(x)
        branch2 = self.inception_2(x)
        branch3 = self.inception_3(x)
        
        x = torch.cat([branch1, branch2, branch3], dim=1)
        
        # Reshape for LSTM
        x = x.reshape(batch_size, -1, x.size(1))
        
        # Encoder LSTM
        encoder_outputs, (state_h, state_c) = self.encoder_lstm(x)
        
        # Initialize decoder input
        decoder_states = (state_h, state_c)
        
        # For collecting outputs
        all_means = []
        all_vars = []
        all_attention = []
        
        # Reshape state_h for concatenation
        state_h_reshaped = state_h.transpose(0, 1).reshape(batch_size, 1, -1)
        inputs = torch.cat([decoder_input, state_h_reshaped], dim=2)
        
        # Decoder loop
        for _ in range(5):
            # Decoder LSTM
            outputs, decoder_states = self.decoder_lstm(inputs, decoder_states)
            
            # Attention mechanism
            attention = torch.bmm(outputs, encoder_outputs.transpose(1, 2))
            attention = F.softmax(attention, dim=2)
            
            # Context vector
            context = torch.bmm(attention, encoder_outputs)
            context = F.batch_norm(context.transpose(1, 2), running_mean=None, running_var=None, 
                                 training=self.training, momentum=0.6).transpose(1, 2)
            
            # Combine and predict
            decoder_combined_context = torch.cat([context, outputs], dim=2)
            
            # Predict mean and variance
            mean = self.mean_head(decoder_combined_context)
            log_var = self.log_var_head(decoder_combined_context)
            var = torch.exp(log_var).clamp(min=1e-6, max=1e6)  # Ensure positive variance
            
            all_means.append(mean)
            all_vars.append(var)
            all_attention.append(attention)
            
            # Prepare next input (using mean prediction)
            inputs = torch.cat([mean, context], dim=2)
            
        # Stack all outputs
        means = torch.cat(all_means, dim=1)
        vars = torch.cat(all_vars, dim=1)
        attention = torch.cat(all_attention, dim=1)
        
        return means, vars

# Usage:
# model = AttentionModel(latent_dim=128)
# x = torch.randn(batch_size, 1, 50, 40)
# decoder_input = torch.randn(batch_size, 1, 1)  # Changed to 1 feature since we're predicting scalar
# mean, var = model(x, decoder_input)





# Custom dataset
class OrderBookDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    


    
    
    
class PricePredictor(nn.Module):
    class AddConstant(nn.Module):
        def __init__(self, constant=1e-6):
            super().__init__()
            self.constant = constant
            
        def forward(self, x):
            return x + self.constant
            
    def __init__(self, input_shape):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * input_shape[0] * input_shape[1], 32),
            nn.ReLU()
        )
        self.mean = nn.Linear(32, 1)
        self.log_var = nn.Sequential(
            nn.Linear(32, 1),
            self.AddConstant(-10)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension [batch, 1, height, width]
        features = self.shared(x)
        return self.mean(features), torch.exp(self.log_var(features))
    
    

def train_network(model, train_dataset, val_dataset, criterion, epochs=100, batch_size=32, lr=0.0001):
   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            mean, var = model(batch_features)
            loss = criterion(mean.squeeze(), var.squeeze(), batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            train_loss += loss.item()
           
        # Validate    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                mean, var = model(batch_features)
                loss = criterion(mean.squeeze(), var.squeeze(), batch_targets)
                val_loss += loss.item()
               
        print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}')
        
        
class LL_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mean, var, targets):
        # Ensure variance is positive and not too small
        var = torch.clamp(var, min=1e-6)
        
        # Calculate loss components separately for better numerical stability
        log_var = torch.log(var)
        squared_error = (targets - mean)**2
        standardized_error = squared_error / var
        
        # Combine components with careful attention to signs
        # Note: we want to maximize likelihood, so minimize negative log likelihood
        loss = 0.5 * (log_var + standardized_error)
        
        return loss.mean()
    
class MSE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, mean, var, targets):
        # Simplified negative log likelihood
        loss = (targets - mean)**2
        return loss.mean()