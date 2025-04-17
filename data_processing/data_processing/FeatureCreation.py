from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Feature(ABC):
    
    @abstractmethod
    def get_min_timesteps(self) -> int:
        """Returns minimum number of timesteps needed for this feature."""
        pass

    def create_feature(self, buffer: List[dict]) -> List[float]:
        """Extracts feature values from the data buffer."""
        pass

    def get_feature_size() -> int:
        """Returns the number of values this feature needs."""
        pass

    def visualize_feature(self, features: np.ndarray) -> None:
        """Visualizes this feature's data over time"""
        pass


class FeatureCreator:

    def __init__(self, features: List[Feature]):
        self.features = features
        self.buffer = []
        self.buffer_len = max([f.get_min_timesteps() for f in features] + [1])

    def feed_datapoint(self, data_frame: dict) -> None:
        self.buffer.append(data_frame)
        self.buffer = self.buffer[-self.buffer_len:]

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.buffer_len

    def create_features(self) -> List[float]:
        if len(self.buffer) < self.buffer_len:
            raise Exception("Not enough data!")
        feature_vector = np.array([])
        for f in self.features:
            feature_vector = np.concatenate((feature_vector, f.create_feature(self.buffer)))
        return feature_vector


    def visualize(self, features_list: np.ndarray, f_idx: int) -> None:
        start_idx = 0
        for i in range(f_idx):
            start_idx += self.features[i].get_feature_size()
        
        end_idx = start_idx + self.features[f_idx].get_feature_size()
        print(f"{start_idx} - {end_idx}")
        self.features[f_idx].visualize_feature(features_list[:, start_idx:end_idx])


        