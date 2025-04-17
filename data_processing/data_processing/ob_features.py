
from data_processing.FeatureCreation import Feature
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LevelOBFeature(Feature):

    def __init__(
            self,
            num_levels=10,
            change=True,
            include_prices=False):
        

        self.num_levels = num_levels
        self.change = change
        self.include_prices = include_prices

        if change:
            self.timesteps_back = 2
        else:
            self.timesteps_back = 1

    def get_min_timesteps(self):
        return self.timesteps_back

    def get_feature_size(self):
        
        if self.include_prices:
            return self.num_levels * 4
        else:
            return self.num_levels * 2
        

    def get_level_data(self, bids, asks):

        max_bids = sorted(bids.keys(), reverse=True)[:self.num_levels]
        min_asks = sorted(asks.keys(), reverse=False)[:self.num_levels]
        
        level_data = np.zeros(self.get_feature_size())
        
        for i in range(self.num_levels):
            level_data[i]                  = asks[min_asks[self.num_levels - 1 - i]]
            level_data[self.num_levels+i]  = bids[max_bids[i]]
            
            if self.include_prices:
                level_data[2*self.num_levels+i]  = min_asks[i]
                level_data[3*self.num_levels+i]  = max_bids[i]
            


        return np.array(level_data)

    def create_feature(self, buffer: List[dict]) -> List[float]:
        
        level_data = self.get_level_data(buffer[-1]["bids"], buffer[-1]["asks"])
        

        if not self.change:
            return level_data
        
        past_level_data = self.get_level_data(buffer[-2]["bids"], buffer[-2]["asks"])
        
        return level_data - past_level_data
    

    def visualize_feature(self, features):
        
        plt.figure(figsize=(15, 5))
        sns.heatmap(
            features.T,
            cmap='viridis_r',
            cbar=False) #, vmin=-5, vmax=100)
        
        plt.gca()
        plt.title("Order Book Levels", fontsize=14)
        plt.tight_layout() # This helps with overall spacing
        plt.show()





class TrendFeature(Feature):

    def __init__(self,timesteps_back):
        
        self.timesteps_back = timesteps_back

    def get_min_timesteps(self):
        return self.timesteps_back + 1

    def get_feature_size(self):
        return 3

    def create_feature(self, buffer: List[dict]) -> List[float]:
        
        mp_current = (max(buffer[-1]["bids"].keys()) + min(buffer[-1]["asks"].keys()))/2

        mp_past = (max(buffer[-1-self.timesteps_back]["bids"].keys()) 
                   + min(buffer[-1-self.timesteps_back]["asks"].keys()))/2
        
        
        if mp_past < mp_current:
            return [1,0,0]
        elif mp_past == mp_current:
            return [0,1,0]
        else:
            return [0,0,1]
    

    def visualize_feature(self, features):
        
        plt.figure(figsize=(15, 2))
        sns.heatmap(
            features.T,
            cmap='viridis_r',
            cbar=False  # This removes the color bar/legend
        ) #, vmin=-5, vmax=100)
        plt.gca()
        plt.title("Price Trends", fontsize=14)
        plt.tight_layout() # This helps with overall spacing
        plt.show()




