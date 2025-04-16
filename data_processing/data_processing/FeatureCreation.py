
from abc import ABC
from typing import List


class Feature(ABC):

    def get_min_timesteps(self):
        pass

    def create_feature(self, buffer: List[dict]) -> List[float]:
        pass


class FeatureCreator:

    def __init__(self, features: List[Feature]):
        self.features = features
        self.buffer = []
        self.buffer_len = max([f.get_min_timesteps() for f in features] + [1])

    def feed_datapoint(self, data_frame: dict):
        self.buffer.append(data_frame)
        self.buffer = self.buffer[-self.buffer_len:]

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.buffer_len

    def create_features(self) -> List[float]:
        if len(self.buffer) < self.buffer_len:
            raise Exception("Not enough data!")
        feature_vector = []
        for f in self.features:
            feature_vector += f.create_feature(self.buffer)
        return feature_vector
