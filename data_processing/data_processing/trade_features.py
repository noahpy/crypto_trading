
from data_processing.FeatureCreation import Feature
from typing import List


class VWAPFeature(Feature):

    def __init__(self, timesteps_back=1):
        self.timesteps_back = timesteps_back

    def get_min_timesteps(self):
        return self.timesteps_back

    def create_feature(self, buffer: List[dict]) -> List[float]:
        vwap = 0
        total_size = 0
        for i in range(len(buffer)):
            trades = buffer[i]["trades"]
            for trade in trades:
                vwap += trade["price"] * trade["size"]
                total_size += trade["size"]

        if vwap == 0:
            mid_price = max(buffer[-1]["bids"].keys()) + \
                min(buffer[-1]["asks"].keys()) / 2
            return [mid_price]

        return [vwap / total_size]


class VolumeFeatures(Feature):

    def __init__(self, include_delta=False):
        self.include_delta = include_delta

    def get_min_timesteps(self):
        return 1 if not self.include_delta else 2

    def create_feature(self, buffer: List[dict]) -> List[float]:

        vol = sum([trade["size"] for trade in buffer[-1]["trades"]])

        if not self.include_delta:
            return [vol]

        vol_past = sum([trade["size"] for trade in buffer[-2]["trades"]])
        return [vol, vol - vol_past]
