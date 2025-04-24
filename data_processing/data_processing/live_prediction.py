
from data_retrieving.PeriodicLiveRetriever import PeriodicLiveRetriever
from machine_learning.wrapper import PyTorchWrapper
import sys
from typing import List
from datetime import datetime
from threading import Thread
import traceback
import time
import queue
import json
import math
from scipy.stats import norm
import numpy as np
from tabulate import tabulate


def probability_random_achieves_accuracy(accuracy, attempts, p=0.5):
    """
    Calculate the probability that a random strategy with success rate p
    achieves at least the given accuracy over a number of attempts.
    Parameters:
    - accuracy (float): Target accuracy to reach (e.g. 0.7137)
    - attempts (int): Number of predictions made
    - p (float): Probability of success for random strategy (default is 0.5)
    Returns:
    - float: Probability of reaching or exceeding the accuracy by chance
    """
    # Calculate number of correct predictions needed
    k = int(accuracy * attempts)
    # Mean and standard deviation under binomial distribution
    mu = attempts * p
    sigma = math.sqrt(attempts * p * (1 - p))
    # Continuity correction
    z = (k - 0.5 - mu) / sigma
    # Probability of getting k or more correct predictions
    return 1 - norm.cdf(z)

def probability_table_for_accuracy(accuracy, attempts, granularity=0.05):
    # Generate p values from 0.5 to accuracy (inclusive)
    p_values = np.arange(0.5, min(accuracy + 0.1, 1.0), granularity)
    results = []

    for p in p_values:
        prob = probability_random_achieves_accuracy(accuracy, attempts, p)
        results.append([round(p, 2), f"{prob:.10f}"])

    # Display table
    print(tabulate(results, headers=["p (random success rate)", f"P(≥ {accuracy*100:.2f}% accuracy)"], tablefmt="pretty"))


def convert_bybit_ob_to_snapshot(order_book):
    """Convert order book data to snapshot format more efficiently"""
    ts = datetime.fromtimestamp(order_book['ts'] / 1000)
    # Use dictionary comprehension only once for each
    bids = {float(price): float(size) for price, size in order_book['b']}
    asks = {float(price): float(size) for price, size in order_book['a']}
    mid_price = (min(asks.keys()) + max(bids.keys())) / 2
    return {"ts": ts, "mid_price": mid_price, "bids": bids, "asks": asks, "original_ts": order_book['ts']}


class ModelPredictionLiveTester():
    def __init__(self, api_key_path: str, model_path: str, model_input_len: int, horizon: int, trend_threshhold: float):

        self.trend_threshhold = trend_threshhold
        self.model_horizon = horizon

        # Create data retriever
        self.pld = PeriodicLiveRetriever(
            api_key_path, 1000, "CAKEUSDT", "linear")
        self.pld.start()

        self.last_snapshot = None

        self.model = PyTorchWrapper(model_path, model_input_len)
        self.model_ready = False
        self.past_midprices = []
        self.past_predictions = []
        self.past_best_bid_ask = []

        # Start update thread
        self.run_update_thread = True
        self._start_update_thread()

    def _start_update_thread(self):
        """Start the background thread for processing data updates"""
        self.updating_process = Thread(
            target=self.run_feature_update, daemon=True)
        self.updating_process.start()

    def run_feature_update(self):
        """Background thread for processing data updates"""
        print("Starting feature update process...")
        acc_mse = 0
        num_predictions = 0
        trend_acc = 0
        trend_acc_sum = 0
        profit = 0
        while self.run_update_thread:
            try:
                # Get data with timeout to avoid blocking forever
                try:
                    data = self.pld.data_queue.get(block=True)
                except queue.Empty:
                    continue

                # Process data
                snapshot_data = convert_bybit_ob_to_snapshot(data["ob"])
                snapshot_data["trades"] = self.calculate_trades_since_last_timestep(
                    data["trades"]["list"])
                self.last_snapshot = snapshot_data

                if not self.model_ready:
                    self.model_ready = self.model.feed_snapshot(snapshot_data)
                    continue

                prediction = self.model.predict(snapshot_data)

                self.past_midprices.append(snapshot_data["mid_price"])
                self.past_predictions.append(prediction)

                best_bid = max(self.last_snapshot["bids"].keys())
                best_ask = min(self.last_snapshot["asks"].keys())
                self.past_best_bid_ask.append((best_bid, best_ask))

                if len(self.past_midprices) > self.model_horizon:
                    prediction_before = self.past_predictions[-self.model_horizon-1]
                    midprice_before = self.past_midprices[-self.model_horizon-1]
                    midprice_current = self.past_midprices[-1]
                    past_best_bid_ask = self.past_best_bid_ask[-self.model_horizon-1]
                    current_best_bid_ask = self.past_best_bid_ask[-1]
                    predicted_midprice_current = prediction_before + midprice_before
                    mse = pow(
                        (midprice_current - predicted_midprice_current), 2)

                    predicted_trend = "NEUTRAL"
                    if prediction_before > 0:
                        predicted_trend = "UP"
                    if prediction_before < 0:
                        predicted_trend = "DOWN"

                    actual_trend = "NEUTRAL"
                    if midprice_current - midprice_before > 0:
                        actual_trend = "UP"
                    if midprice_current - midprice_before < 0:
                        actual_trend = "DOWN"

                    profit_step = 0
                    profit_string = ""
                    if actual_trend != "NEUTRAL":
                        if actual_trend == predicted_trend:
                            trend_acc += 1
                        trend_acc_sum += 1
                    if predicted_trend == "UP":
                        # buy in the past and sell now
                        # profit += current best bid - past best ask
                        sell_price = current_best_bid_ask[0]
                        buy_price = past_best_bid_ask[1]
                        profit_string = f"by buying at {buy_price:.5f} and selling at {sell_price:.5f}"
                    elif predicted_trend == "DOWN":
                        # sell in the past and buy now
                        # profit += past best ask - current best bid
                        sell_price = past_best_bid_ask[1]
                        buy_price = current_best_bid_ask[0]
                        profit_string = f"by selling at {sell_price:.5f} and buying at {buy_price:.5f}"
                    profit_step = sell_price - buy_price

                    profit += profit_step
                    acc_mse += mse
                    num_predictions += 1

                    try:
                        accuracy = float(trend_acc) / trend_acc_sum 
                        data = {
                            "prediction_before": prediction_before,
                            "midprice_current": midprice_current,
                            "midprice_before": midprice_before,
                            "actual_trend": actual_trend,
                            "predicted_trend": predicted_trend,
                            "accuracy": accuracy,
                            "trend_acc_sum": trend_acc_sum,
                            "profit": profit,
                            "timestamp": snapshot_data["original_ts"],
                            "bids": self.last_snapshot["bids"],
                            "asks": self.last_snapshot["asks"]
                        }
                        # append file with data json as line
                        with open("prediction.data", "a") as f:
                            f.write(json.dumps(data) + "\n")
                        msg = f"Prediction Before: {prediction_before:.5f}, Mid Price: {midprice_current:.5f} Mid Price Before: {midprice_before:.5f} Actual Trend: {actual_trend}, Predicted Trend: {predicted_trend} Accuracy: {accuracy:.5f}, Trend Count: {trend_acc_sum}, Profit: {profit_step:.5f} {profit_string} Total profit: {profit:.5f}"
                        print(msg)
                    except Exception as e:
                        print(f"No trend yet: Mid Price: {midprice_current:.5f} Mid Price Before: {midprice_before:.5f} Actual Trend: {actual_trend}, Predicted Trend: {predicted_trend}")
                        print(e)

            except Exception as e:
                print(f"Error in feature update process: {str(e)}")
                traceback.print_exc()
        else:
            # Sleep when not updating to reduce CPU usage
            time.sleep(0.1)

    def calculate_trades_since_last_timestep(self, new_trades: List[dict]):
        """Calculate trades since last timestamp"""
        if self.last_snapshot is None:
            for trade in new_trades:
                trade["price"] = float(trade["price"])
                trade["size"] = float(trade["size"])
            return new_trades

        last_timestep = self.last_snapshot["original_ts"]

        trades_since_last_timestep = [
            trade for trade in new_trades if int(trade["time"]) > last_timestep]

        for trade in trades_since_last_timestep:
            trade["price"] = float(trade["price"])
            trade["size"] = float(trade["size"])

        # print(trades_since_last_timestep)

        return trades_since_last_timestep


if __name__ == "__main__":
    api_key_path = "api_key.json"
    if len(sys.argv) > 1:
        api_key_path = sys.argv[1]
    app = ModelPredictionLiveTester(api_key_path, "ml_models", 30, 10, 0.002)
