
import tensorflow as tf 
from tensorflow import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Reshape, Concatenate, Flatten, Layer, Attention
from tensorflow.keras import regularizers
import pickle
from tensorflow.keras import layers, regularizers



def convolution_layer(input_train):
    
    """
    Apply a series of convolutional layers to the input data.

    This function creates a complex convolutional network structure,
    including multiple Conv2D layers with LeakyReLU activations,
    and an inception-style module.

    Args:
        input_train: Input tensor, expected to be a 4D tensor
                     representing order book data.

    Returns:
        tf.Tensor: Reshaped output after applying convolutions,
                   with shape (batch_size, time_steps, features).
    """

    conv_first1 = keras.layers.Conv2D(32, (2, 1), padding='same')(input_train)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    conv_first1 = keras.layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    
    conv_first1 = keras.layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (2, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (1, 8))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    
    # build the inception module
    convsecond_1 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = keras.layers.Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = keras.layers.Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = keras.layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = keras.layers.Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = keras.layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    return conv_reshape

    
def get_model_lstm(
    latent_dim,
    dropout_rate = 0,
    horizons_backward = 19,
    num_order_buckets = 32,
    num_order_book_features = 6,
    num_outputs=1,
    last_activation_function="linear",
    l2_regularizer = 0):

    """
    Create a LSTM-based model for order book analysis and prediction.

    This model:
    1. Processes order book buckets using convolutional layers.
    2. Combines the output with order data for each time step.
    3. Passes the combined data through an LSTM layer.
    4. Uses the LSTM output for final prediction.

    Args:
        latent_dim (int): Dimensionality of the LSTM layer.
        dropout_rate (float): Dropout rate for regularization.
        horizons_backward (int): Number of past time steps to consider.
        num_order_buckets (int): Number of order book buckets.
        num_order_book_features (int): Number of features in order book data.
        num_outputs (int): Number of output dimensions.
        last_activation_function (str): Activation function for the output layer.
        l2_regularizer (float): L2 regularization parameter.

    Returns:
        keras.Model: Compiled Keras model ready for training.
    """
    
    # Regularizer
    regularizer = regularizers.l2(l2_regularizer)
    
    # process order book buckets as image
    input_order_book_buckets = keras.layers.Input(shape=(horizons_backward+1, num_order_buckets, 1), name="ob_bucket_input")
    
    conv_output = convolution_layer(input_order_book_buckets)
    # conv_output = convolution_layer_dropout(input_order_book_buckets)
    
    
    conv_output = layers.SpatialDropout1D(dropout_rate)(conv_output)
    

    input_order_data = keras.layers.Input(shape=(horizons_backward+1, num_order_book_features),name="ob_data_input")
    lstm_inputs = keras.layers.concatenate([conv_output, input_order_data], axis=2)

    # LSTM
    lstm_output = keras.layers.LSTM(latent_dim, kernel_regularizer=regularizer)(lstm_inputs)
        
    dense1 = keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizer)(lstm_output)
    dense1 = keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(dense1)
    dense1 = keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer)(dense1)
    
    output = keras.layers.Dense(num_outputs, activation=last_activation_function, name='output_layer')(dense1)
    output = keras.layers.Reshape((1, num_outputs))(output)

    model = keras.models.Model([input_order_book_buckets, input_order_data], output)
    return model




