
from data_processing.FeatureCreation import Feature
from data_processing.trade_features import create_feature_subplot
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class LevelOBFeature(Feature):

    def __init__(
            self,
            num_levels=10,
            change=False,
            include_prices=True):

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

    def get_subfeature_names_and_toggle_function(self):
        return []

    def get_level_data(self, bids, asks):

        max_bids = sorted(bids.keys(), reverse=True)[:self.num_levels]
        min_asks = sorted(asks.keys(), reverse=False)[:self.num_levels]

        level_data = np.zeros(self.get_feature_size())

        for i in range(self.num_levels):
            level_data[i] = asks[min_asks[self.num_levels - 1 - i]]
            level_data[self.num_levels+i] = bids[max_bids[i]]

            if self.include_prices:
                level_data[2*self.num_levels+i] = min_asks[i]
                level_data[3*self.num_levels+i] = max_bids[i]

        return np.array(level_data)

    def create_feature(self, buffer: List[dict]) -> List[float]:

        level_data = self.get_level_data(
            buffer[-1]["bids"], buffer[-1]["asks"])

        if not self.change:
            return level_data

        past_level_data = self.get_level_data(
            buffer[-2]["bids"], buffer[-2]["asks"])

        return level_data - past_level_data

    def visualize_feature(self, features, ax=None):

        if ax is None:
            plt.figure(figsize=(15, 5))
            sns.heatmap(
                features.T,
                cmap='viridis_r',
                cbar=False)  # , vmin=-5, vmax=100)

            plt.gca()
            plt.title("Order Book Levels", fontsize=14)
            plt.tight_layout()  # This helps with overall spacing
            plt.show()
        else:
            sns.heatmap(
                features.T,
                cmap='viridis_r',
                cbar=False,
                ax=ax
            )
            ax.set_title("Order Book Levels", fontsize=14)

    def turn_all_subfeatures_on(self):
        pass

    def turn_all_subfeatures_off(self):
        pass



class MidPriceFeature(Feature):
    """
    Simple feature class to calculate the mid_price
    """

    def __init__(self, timesteps_back=1, inc_mp_change=False, inc_mp=False):

        self.inc_mp_change = inc_mp_change
        self.inc_mp = inc_mp
        self.timesteps_back = timesteps_back

    def get_min_timesteps(self):
        if not self.inc_mp_change:
            return 1

        return self.timesteps_back + 1

    def get_feature_size(self):
        return sum([self.inc_mp_change, self.inc_mp])

    def get_subfeature_names_and_toggle_function(self):
        return [
            ["mid_price", self.toggle_mp],
            ["mid_price_change", self.toggle_mp_change],
        ]

    def toggle_mp(self):
        self.inc_mp = not self.inc_mp

    def toggle_mp_change(self):
        self.inc_mp_change = not self.inc_mp_change

    def turn_all_subfeatures_on(self):
        self.inc_mp = True
        self.inc_mp_change = True

    def turn_all_subfeatures_off(self):
        self.inc_mp = False
        self.inc_mp_change = False

    def create_feature(self, buffer: List[dict]) -> List[float]:
        
        

        mid_price_curr = buffer[-1]["mid_price"]
    
        feature = []
        if self.inc_mp:
            feature.append(mid_price_curr)
        if self.inc_mp_change:
            mid_price_past = buffer[-self.timesteps_back-1]["mid_price"]
            feature.append(mid_price_curr - mid_price_past)

        return feature

    def visualize_feature(self, features, ax=None):
        """
        Visualize the volatility feature with y-axis labels on the inside of the plot
        and increased height for better visibility.

        Args:
            features: A numpy array of feature values where each feature's values form a time series
        """
        if not ax:
            if self.get_feature_size() == 1:
                plt.figure(figsize=(15, 5))
                plt.plot(features, linewidth=1)
                plt.show()
                return

            plt.figure(figsize=(15, 5))
            plt.plot(features[:, 0], linewidth=1, color='b', label='mid_price')
            plt.plot(features[:, 1], linewidth=1,
                     color='g', label='mid_price_change')
            plt.legend(loc='upper right', fontsize=8)
            plt.show()
            return

        if self.get_feature_size() == 1:
            ax.plot(features, linewidth=1)
            return

        ax.plot(features[:, 0], linewidth=1, color='b', label='mid_price')
        ax.plot(features[:, 1], linewidth=1,
                color='g', label='mid_price_change')
        ax.legend(loc='upper right', fontsize=8)


class TrendFeature(Feature):

    def __init__(self, timesteps_back=1):

        self.timesteps_back = timesteps_back

    def get_min_timesteps(self):
        return self.timesteps_back + 1

    def get_feature_size(self):
        return 3

    def get_feature_name(self):
        return "trend"

    def get_subfeature_names_and_toggle_function(self):
        return []

    def create_feature(self, buffer: List[dict]) -> List[float]:

        mp_current = (max(buffer[-1]["bids"].keys()) +
                      min(buffer[-1]["asks"].keys()))/2

        mp_past = (max(buffer[-1-self.timesteps_back]["bids"].keys())
                   + min(buffer[-1-self.timesteps_back]["asks"].keys()))/2

        if mp_past < mp_current:
            return [1, 0, 0]
        elif mp_past == mp_current:
            return [0, 1, 0]
        else:
            return [0, 0, 1]

    def visualize_feature(self, features, ax=None):

        if ax is not None:
            sns.heatmap(
                features.T,
                cmap='viridis_r',
                cbar=False,
                ax=ax
            )
            ax.set_title("Price Trends", fontsize=14)
            return
        plt.figure(figsize=(15, 2))
        sns.heatmap(
            features.T,
            cmap='viridis_r',
            cbar=False  # This removes the color bar/legend
        )  # , vmin=-5, vmax=100)
        plt.gca()
        plt.title("Price Trends", fontsize=14)
        plt.tight_layout()  # This helps with overall spacing
        plt.show()

    def turn_all_subfeatures_on(self):
        pass

    def turn_all_subfeatures_off(self):
        pass

