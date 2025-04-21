
from data_retrieving.PeriodicLiveRetriever import PeriodicLiveRetriever
from machine_learning.wrapper import PyTorchWrapper
import tkinter as tk
import sys
from typing import List
from datetime import datetime
from threading import Thread
import traceback
import time
import queue
import json


def convert_bybit_ob_to_snapshot(order_book):
    """Convert order book data to snapshot format more efficiently"""
    ts = datetime.fromtimestamp(order_book['ts'] / 1000)
    # Use dictionary comprehension only once for each
    bids = {float(price): float(size) for price, size in order_book['b']}
    asks = {float(price): float(size) for price, size in order_book['a']}
    mid_price = (min(asks.keys()) + max(bids.keys())) / 2
    return {"ts": ts, "mid_price": mid_price, "bids": bids, "asks": asks, "original_ts": order_book['ts']}


class ModelPredictionLiveTester(tk.Tk):
    def __init__(self, api_key_path: str, model_path: str, model_input_len: int, horizon: int, trend_threshhold: float):
        super().__init__()
        self.title("Crypto Live Feature Viewer")
        self.geometry("1200x900")

        self.trend_threshhold = trend_threshhold
        self.model_horizon = horizon

        # Create data retriever
        self.pld = PeriodicLiveRetriever(
            api_key_path, 1000, "CAKEUSDT", "linear")
        self.pld.start()

        self.last_snapshot = None

        # Start update thread
        self.run_update_thread = True
        self.after(100, self._start_update_thread)

        self.model = PyTorchWrapper(model_path, model_input_len)
        self.model_ready = False
        self.past_midprices = []
        self.past_predictions = []

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

                if len(self.past_midprices) > self.model_horizon:
                    prediction_before = self.past_predictions[-self.model_horizon-1]
                    midprice_before = self.past_midprices[-self.model_horizon-1]
                    midprice_current = self.past_midprices[-1]
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

                    if actual_trend != "NEUTRAL":
                        if actual_trend == predicted_trend:
                            trend_acc += 1
                            profit += abs(midprice_current - midprice_before)
                        else:
                            profit -= abs(midprice_current - midprice_before)
                        trend_acc_sum += 1

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
                            "timestamp": snapshot_data["original_ts"]
                        }
                        # append file with data json as line
                        with open("prediction.data", "a") as f:
                            f.write(json.dumps(data) + "\n")
                        msg = f"Prediction Before: {prediction_before:.5f}, Mid Price: {midprice_current:.5f} Mid Price Before: {midprice_before:.5f} Actual Trend: {actual_trend}, Predicted Trend: {predicted_trend} Accuracy: {accuracy:.5f}, Trend Count: {trend_acc_sum}, Profit: {profit:.5f}"
                        print(msg)
                    except Exception as e:
                        print(f"No trend yet: Mid Price: {midprice_current:.5f} Mid Price Before: {midprice_before:.5f} Actual Trend: {actual_trend}, Predicted Trend: {predicted_trend}")
                        print(e)

                self.update()

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
    app.mainloop()
