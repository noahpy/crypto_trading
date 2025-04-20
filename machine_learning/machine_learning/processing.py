import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

## DATA TRANSFORM ## 

class NormalizedModel(nn.Module):

    def __init__(
            self,
            base_model,
            x_train,
            y_train,
            norm_input_std=True,
            norm_input_mean=True,
            norm_output_std=True,
            norm_output_mean=False):
        
        super().__init__()
        self.base_model = base_model
        
        self.input_mean = torch.mean(x_train, axis=0) if norm_input_mean else 0
        self.input_std = torch.std(x_train, axis=0) if norm_input_std else 1

        self.output_mean = torch.mean(y_train, axis=0) if norm_output_mean else 0
        self.output_std = torch.std(y_train, axis=0) if norm_output_std else 1

        self.use_softmax = base_model.use_softmax
        self.loss_function = base_model.loss_function

    def forward(self, x):

        x_norm = (x - self.input_mean) / self.input_std
        y = self.base_model(x_norm)
        y_denorm =  y * self.output_std + self.output_mean

        return y_denorm






def build_mp_change_prediction_data_set(
        features,
        feature_creator,
        mp_f_index,
        horizon,
        input_length,
        steps_between):
    
    X_data = []
    Y_data = []

    mp_data = feature_creator.get_feature(features, mp_f_index)

    for i in range(0, len(features) - input_length - horizon-1, steps_between):
        
        X_data.append(features[i:i + input_length])

        mp_future = mp_data[i+input_length+horizon-1]
        mp_now = mp_data[i+input_length-1]

        Y_data.append(mp_future - mp_now)
        
    return np.array(X_data), np.array(Y_data)



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
