import torch
import pickle
import numpy as np


class PyTorchWrapper():


    def __init__(self, model_path, input_length):
        """
        At the path the init function expects a 
        - feature_creator.pkl
        - model.pth
        """
        self.input_length = input_length

        model_file = f'{model_path}/model.pth'
        fc_path = f'{model_path}/feature_creator.pkl'
        
        # Load the model
        self.model = torch.load(model_file, map_location=torch.device('cpu'))
        self.model.eval()  # Set to evaluation mode
        
        # Load the input feature creator
        with open(fc_path, 'rb') as f:  # 'rb' for read binary
            self.feature_creator = pickle.load(f)

        self.feature_data = []


    def feed_snapshot(self, snapshot):
        self.feature_creator.feed_datapoint(snapshot)
        if self.feature_creator.is_ready():
            self.feature_data.append(self.feature_creator.create_features())
            self.feature_data = self.feature_data[-self.input_length:]

        return len(self.feature_data) == self.input_length
            

    def predict(self, snapshot):

        if not self.feed_snapshot(snapshot):
            print("Not enough data to make predictions")
            return None
    
        features_tensor = torch.tensor(np.array(self.feature_data), dtype=torch.float32)
        
        # Make prediction using the model
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to track gradients for inference
            prediction = self.model(features_tensor.unsqueeze(0))  # Add batch dimension
            prediction_value = prediction.item()  # Extract the scalar value
            return prediction_value