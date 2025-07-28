
from data_processing.FeatureCreation import Feature
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def create_feature_subplot(ax, data, title, color, label=None, second_data=None, second_color=None, second_label=None):
    """
    Create a single feature subplot with consistent styling and inside y-axis labels.

    Args:
        ax: The matplotlib axis to plot on
        data: The data to plot
        title: The title for the subplot
        color: The color for the line
        label: Optional label for the first line (for legend)
        second_data: Optional second data series to plot on the same axis
        second_color: Optional color for the second line
        second_label: Optional label for the second line (for legend)
    """
    # Plot the main data
    ax.plot(data, linewidth=1, color=color, label=label)

    # Plot the optional second data if provided
    if second_data is not None and second_color is not None:
        ax.plot(second_data, linewidth=1,
                color=second_color, label=second_label)

    # Add legend if we have labels
    if label is not None or second_label is not None:
        ax.legend(loc='upper right', fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Move the y-axis tick labels to the inside of the plot
    # First, hide the original labels
    ax.tick_params(axis='y', which='both', labelleft=False)

    # Create custom tick labels on the inside
    yticks = ax.get_yticks()
    ylim = ax.get_ylim()

    # Filter ticks that are within the plot range
    valid_ticks = [tick for tick in yticks if ylim[0] <= tick <= ylim[1]]

    # Format the tick labels with fewer decimal places for cleaner display
    for tick in valid_ticks:
        # Use a more compact format for small numbers
        if abs(tick) < 0.01:
            label = f"{tick:.2e}"
        else:
            label = f"{tick:.4f}"

        # Add the text annotation inside the plot
        ax.text(0.02, tick, label, transform=ax.get_yaxis_transform(),
                verticalalignment='center', horizontalalignment='left',
                fontsize=8, color='black', alpha=0.7)


class TradeFeature(Feature):

    def __init__(
            self,
            timesteps_back=1,
            inc_vwap=False,
            inc_vwap_change=False,
            inc_vol=False,
            inc_vol_change=False,
            inc_taker_vol=False,
            inc_taker_vol_change=False
    ):

        self.timesteps_back = timesteps_back

        self.inc_vwap = inc_vwap
        self.inc_vwap_change = inc_vwap_change

        self.inc_vol = inc_vol
        self.inc_vol_change = inc_vol_change

        self.inc_taker = inc_taker_vol
        self.inc_taker_change = inc_taker_vol_change

        self.inc_change = (
            self.inc_vwap_change or self.inc_vol_change or self.inc_taker_change)
        self.timesteps_back = timesteps_back

    def get_min_timesteps(self):

        if self.inc_change:
            return self.timesteps_back + 1
        else:
            return self.timesteps_back

    def get_feature_size(self):
        return sum([self.inc_vwap,
                    self.inc_vwap_change,
                    self.inc_vol,
                    self.inc_vol_change,
                    self.inc_taker*2,
                    self.inc_taker_change*2])

    def get_feature_name(self):
        return "trade_features"

    def get_subfeature_names_and_toggle_function(self):
        return [
            ["vawp", self.toggle_vwap],
            ["vawp_change", self.toggle_vwap_change],
            ["vol", self.toggle_vol],
            ["vol_change", self.toggle_vol_change],
            ["taker_vol", self.toggle_taker],
            ["taker_vol_change", self.toggle_taker_change]
        ]

    def toggle_vwap(self):
        self.inc_vwap = not self.inc_vwap

    def toggle_vwap_change(self):
        self.inc_vwap_change = not self.inc_vwap_change

    def toggle_vol(self):
        self.inc_vol = not self.inc_vol

    def toggle_vol_change(self):
        self.inc_vol_change = not self.inc_vol_change

    def toggle_taker(self):
        self.inc_taker = not self.inc_taker

    def toggle_taker_change(self):
        self.inc_taker_change = not self.inc_taker_change

    def turn_all_subfeatures_on(self):
        self.inc_vwap = True
        self.inc_vwap_change = True
        self.inc_vol = True
        self.inc_vol_change = True
        self.inc_taker = True
        self.inc_taker_change = True

    def turn_all_subfeatures_off(self):
        self.inc_vwap = False
        self.inc_vwap_change = False
        self.inc_vol = False
        self.inc_vol_change = False
        self.inc_taker = False
        self.inc_taker_change = False

    def create_feature(self, buffer: List[dict]) -> List[float]:

        feature = []
        tmp_features = []

        for i in range(len(buffer)):

            size_bid_takers = size_ask_takers = 0
            vwap = 0
            for trade in buffer[i]["trades"]:

                if trade["side"] == "Buy":
                    size_bid_takers += trade["size"]
                else:
                    size_ask_takers += trade["size"]

                vwap += trade["price"] * trade["size"]

            tmp_features.append({
                "vwap": vwap,
                "size_bid_takers": size_bid_takers,
                "size_ask_takers": size_ask_takers,
                "size": size_bid_takers + size_ask_takers
            })

        vwap_curr = sum([t["vwap"]
                        for t in tmp_features[-self.timesteps_back:]])
        vol_curr = sum([t["size"]
                       for t in tmp_features[-self.timesteps_back:]])

        if vol_curr == 0:
            vwap_curr = (max(buffer[-1]["bids"].keys()) +
                         min(buffer[-1]["asks"].keys()))/2
        else:
            vwap_curr = vwap_curr / vol_curr

        bid_takers_curr = sum([t["size_bid_takers"]
                              for t in tmp_features[-self.timesteps_back:]])
        ask_takers_curr = sum([t["size_ask_takers"]
                              for t in tmp_features[-self.timesteps_back:]])

        vwap_past = sum([t["vwap"]
                        for t in tmp_features[-self.timesteps_back-1:-1]])
        vol_past = sum([t["size"]
                       for t in tmp_features[-self.timesteps_back-1:-1]])

        if vol_past == 0:
            vwap_past = (max(buffer[-2]["bids"].keys()) +
                         min(buffer[-2]["asks"].keys()))/2
        else:
            vwap_past = vwap_past / vol_past

        bid_takers_past = sum([t["size_bid_takers"]
                              for t in tmp_features[-self.timesteps_back-1:-1]])
        ask_takers_past = sum([t["size_ask_takers"]
                              for t in tmp_features[-self.timesteps_back-1:-1]])

        if self.inc_vwap:
            feature.append(vwap_curr)
        if self.inc_vwap_change:
            feature.append(vwap_curr - vwap_past)

        if self.inc_vol:
            feature.append(vol_curr)

        if self.inc_vol_change:
            feature.append(vol_curr - vol_past)

        if self.inc_taker:
            feature.append(bid_takers_curr)
            feature.append(ask_takers_curr)

        if self.inc_taker_change:
            feature.append(bid_takers_curr - bid_takers_past)
            feature.append(ask_takers_curr - ask_takers_past)

        return feature

    def visualize_feature(self, features, ax=None):
        """
        Visualize activated features stacked vertically inside a given ax's figure.
        """

        # Count active plots
        num_plots = sum([
            self.inc_vwap,
            self.inc_vwap_change,
            self.inc_vol,
            self.inc_vol_change,
            self.inc_taker,
            self.inc_taker_change
        ])

        if num_plots == 0:
            print("No features selected for visualization")
            return

        # Use provided ax and clear it
        if ax is None:
            fig, axs = plt.subplots(num_plots, 1, figsize=(
                12, 2.0 * num_plots), sharex=True)
            # Ensure axs is iterable
            if num_plots == 1:
                axs = [axs]
        else:
            fig = ax.figure
            ax.clear()
            gs = gridspec.GridSpec(num_plots, 1, figure=fig, hspace=0.4)
            axs = [fig.add_subplot(gs[i]) for i in range(num_plots)]

        plot_idx = 0
        feature_idx = 0

        if self.inc_vwap:
            create_feature_subplot(
                axs[plot_idx],
                features[:, feature_idx],
                'VWAP',
                'blue'
            )
            plot_idx += 1
            feature_idx += 1

        if self.inc_vwap_change:
            create_feature_subplot(
                axs[plot_idx],
                features[:, feature_idx],
                'VWAP Change',
                'cyan'
            )
            plot_idx += 1
            feature_idx += 1

        if self.inc_vol:
            create_feature_subplot(
                axs[plot_idx],
                features[:, feature_idx],
                'Volume',
                'green'
            )
            plot_idx += 1
            feature_idx += 1

        if self.inc_vol_change:
            create_feature_subplot(
                axs[plot_idx],
                features[:, feature_idx],
                'Volume Change',
                'lime'
            )
            plot_idx += 1
            feature_idx += 1

        if self.inc_taker:
            bid = features[:, feature_idx]
            feature_idx += 1
            ask = features[:, feature_idx]
            feature_idx += 1
            create_feature_subplot(
                axs[plot_idx],
                bid,
                'Taker Volumes',
                'red',
                'Bid Takers',
                ask,
                'orange',
                'Ask Takers'
            )
            plot_idx += 1

        if self.inc_taker_change:
            bid = features[:, feature_idx]
            feature_idx += 1
            ask = features[:, feature_idx]
            feature_idx += 1
            create_feature_subplot(
                axs[plot_idx],
                bid,
                'Taker Volume Changes',
                'darkred',
                'Bid Δ',
                ask,
                'darkorange',
                'Ask Δ'
            )

        fig.tight_layout()


class VolatilityFeature(Feature):
    """
    Simple feature class to calculate price volatility from trade data.
    """

    def __init__(self, timesteps_back=10):
        """
        Initialize the volatility feature calculator.

        Args:
            timesteps_back: Number of previous timesteps to use for volatility calculation
        """
        self.timesteps_back = timesteps_back

    def get_min_timesteps(self):
        """Return minimum required timesteps for this feature"""
        return self.timesteps_back

    def get_feature_size(self):
        """Return the number of features this class produces"""
        return 1

    def get_feature_name(self):
        """Return the name of this feature"""
        return "Volatility"

    def get_subfeature_names_and_toggle_function(self):
        return []

    def create_feature(self, buffer: List[dict]) -> List[float]:
        """
        Create volatility feature from the buffer data.

        Args:
            buffer: List of order book and trade data

        Returns:
            List containing the volatility value
        """
        # Extract price data for each timestep in the buffer
        prices = []

        for snapshot in buffer[-self.timesteps_back:]:
            # Get trades in this period
            period_trades = snapshot["trades"]

            if not period_trades:
                # If no trades, use mid price from order book
                mid_price = (max(snapshot["bids"].keys()) +
                             min(snapshot["asks"].keys())) / 2
                prices.append(mid_price)
            else:
                # Calculate VWAP for this period
                vwap = sum(trade["price"] * trade["size"]
                           for trade in period_trades)
                total_vol = sum(trade["size"] for trade in period_trades)
                if total_vol > 0:
                    vwap = vwap / total_vol
                else:
                    vwap = (max(snapshot["bids"].keys()) +
                            min(snapshot["asks"].keys())) / 2

                prices.append(vwap)

        # Calculate volatility as standard deviation of prices
        volatility = np.std(prices) if len(prices) > 1 else 0

        return [volatility]

    def visualize_feature(self, features, ax=None):
        """
        Visualize the volatility feature with y-axis labels on the inside of the plot
        and increased height for better visibility.

        Args:
            features: A numpy array of feature values where each feature's values form a time series
        """
        # Create figure with subplots - with increased height (2.0 instead of 1.5)
        num_plots = 1
        if not ax:
            fig, axes = plt.subplots(num_plots, 1, figsize=(
                10, 2.0 * num_plots), sharex=True)
        else:
            axes = ax

        # Handle case with only one subplot
        if num_plots == 1:
            axes = [axes]

        # Ensure features is properly shaped
        if len(features.shape) == 1:
            # If features is 1D, reshape it to 2D
            features = features.reshape(-1, 1)

        # Use the shared function to create the subplot
        create_feature_subplot(
            axes[0],
            features[:, 0],
            'Price Volatility',
            'purple'
        )

        if ax is None:
            # Set common labels and adjust layout
            plt.xlabel('Time Step')
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)  # Add space between subplots
            plt.show()

    def turn_all_subfeatures_on(self):
        pass

    def turn_all_subfeatures_off(self):
        pass

