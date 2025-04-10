import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim




class ImprovedNN(nn.Module):
    def __init__(self, input_size=50*20, aux_size=50*5, hidden_size=128, output_size=1, dropout_rate=0.2):
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
    




    def __init__(self, input_size=50*20, aux_size=50*5, hidden_size=128, output_size=1, dropout_rate=0.2):
        super(OrderBookModel, self).__init__()
        
        # Calculate parameters
        num_order_book_features = aux_size // 50  # Features per timestep
        
        # Initialize the model
        self.model = OrderBookLSTM(
            latent_dim=hidden_size,
            dropout_rate=dropout_rate,
            num_order_book_features=num_order_book_features,
            num_outputs=output_size,
            in_channels=8
        )
    
    def forward(self, x, aux_features):
        # Reshape inputs if needed
        if len(x.shape) == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 8, 50, 20)
        
        if len(aux_features.shape) == 2:
            batch_size = aux_features.size(0)
            features_per_step = aux_features.size(1) // 50
            aux_features = aux_features.view(batch_size, 50, features_per_step)
        
        # Forward through the model
        return self.model(x, aux_features)


import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Reshape, Concatenate, Flatten, Layer
from tensorflow.keras import regularizers
from tensorflow.keras import layers

class OrderBookModel(tf.keras.Model):
    def __init__(self, input_size=50*20, aux_size=50*5, hidden_size=128, output_size=1, dropout_rate=0.2):
        super(OrderBookModel, self).__init__()
        
        # Calculate parameters
        self.input_size = input_size
        self.aux_size = aux_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Determine the dimensions
        self.horizons_backward = 50 - 1  # 50 timesteps, but zero-indexed
        self.num_order_buckets = 20  # From input size dimensions
        self.num_order_book_features = aux_size // 50  # Features per timestep

        # Build model layers using the get_model_lstm function
        self.l2_regularizer = 0.0001
        
        # We'll initialize internal layers that will be connected in the call method
        self.conv_layer = self._create_conv_layer
        self.spatial_dropout = layers.SpatialDropout1D(dropout_rate)
        self.lstm_layer = layers.LSTM(hidden_size, kernel_regularizer=regularizers.l2(self.l2_regularizer))
        
        # Dense layers for processing the LSTM output
        self.dense1 = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(self.l2_regularizer))
        self.dense2 = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(self.l2_regularizer))
        self.dense3 = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(self.l2_regularizer))
        
        # Output layer
        self.output_layer = layers.Dense(output_size, activation="linear", name='output_layer')
        self.reshape_layer = layers.Reshape((1, output_size))

    @property
    def _create_conv_layer(self):
        """Returns a function that creates a convolutional layer structure when called."""
        def apply_convolution(input_train):
            conv_first1 = layers.Conv2D(32, (2, 1), padding='same')(input_train)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = layers.Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            
            conv_first1 = layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = layers.Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            
            conv_first1 = layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            conv_first1 = layers.Conv2D(32, (1, 8))(conv_first1)
            conv_first1 = layers.LeakyReLU(alpha=0.01)(conv_first1)
            
            # build the inception module
            convsecond_1 = layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
            convsecond_1 = layers.LeakyReLU(alpha=0.01)(convsecond_1)
            convsecond_1 = layers.Conv2D(64, (3, 1), padding='same')(convsecond_1)
            convsecond_1 = layers.LeakyReLU(alpha=0.01)(convsecond_1)
            
            convsecond_2 = layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
            convsecond_2 = layers.LeakyReLU(alpha=0.01)(convsecond_2)
            convsecond_2 = layers.Conv2D(64, (5, 1), padding='same')(convsecond_2)
            convsecond_2 = layers.LeakyReLU(alpha=0.01)(convsecond_2)
            
            convsecond_3 = layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
            convsecond_3 = layers.Conv2D(64, (1, 1), padding='same')(convsecond_3)
            convsecond_3 = layers.LeakyReLU(alpha=0.01)(convsecond_3)
            
            convsecond_output = layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
            conv_reshape = layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)
            
            return conv_reshape
        
        return apply_convolution

    def call(self, x, aux_features, training=False):
        # Reshape inputs if needed
        if len(tf.shape(x)) == 2:
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, [batch_size, 50, 20, 1])  # Reshape to expected format
        elif len(tf.shape(x)) == 4 and tf.shape(x)[-1] != 1:
            # If 4D but last dimension isn't 1, reshape it
            x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])
            
        if len(tf.shape(aux_features)) == 2:
            batch_size = tf.shape(aux_features)[0]
            features_per_step = self.num_order_book_features
            aux_features = tf.reshape(aux_features, [batch_size, 50, features_per_step])
            
        # Process the orderbook buckets through convolution layers
        conv_output = self.conv_layer(x)
        
        # Apply spatial dropout
        conv_output = self.spatial_dropout(conv_output, training=training)
        
        # Concatenate with auxiliary features
        lstm_inputs = layers.concatenate([conv_output, aux_features], axis=2)
        
        # Pass through LSTM
        lstm_output = self.lstm_layer(lstm_inputs)
        
        # Dense layers
        dense1 = self.dense1(lstm_output)
        dense1 = self.dense2(dense1)
        dense1 = self.dense3(dense1)
        
        # Output layer
        output = self.output_layer(dense1)
        output = self.reshape_layer(output)
        
        return output

# Create a function to instantiate the model with the same interface as PyTorch
def create_tensorflow_model(input_size=50*20, aux_size=50*5, hidden_size=128, output_size=1, dropout_rate=0.2):
    return OrderBookModel(
        input_size=input_size,
        aux_size=aux_size,
        hidden_size=hidden_size,
        output_size=output_size,
        dropout_rate=dropout_rate
    )