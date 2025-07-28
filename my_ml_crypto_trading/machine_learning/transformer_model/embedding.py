import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEncoding(nn.Module):
    """Positional encoding for transformer sequence inputs"""
    def __init__(self, d_model, len):
        super(PositionalEncoding, self).__init__()
        
        # creates array of (input length x embedding_dimensionality)
        pe = torch.zeros(len, d_model)

        # creates a tensor [0, 1, 2, .., len-1]
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # This goes though all rows and sets all even terms to sin(i * div_term)
        pe[:, 0::2] = torch.sin(position * div_term)

        # This goes though all rows and sets all uneven terms to cos(i * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    




