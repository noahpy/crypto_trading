

from data_retrieving.PeriodicLiveRetriever import PeriodicLiveRetriever
import sys
from typing import List
from datetime import datetime
from threading import Thread
import queue
import os
import json
import msgpack
import gzip


def convert_bybit_ob_to_snapshot(order_book):
    """Convert order book data to snapshot format more efficiently"""
    # Use dictionary comprehension only once for each
    bids = {float(price): float(size) for price, size in order_book['b']}
    asks = {float(price): float(size) for price, size in order_book['a']}
    mid_price = (min(asks.keys()) + max(bids.keys())) / 2
    return {"mid_price": mid_price, "bids": bids, "asks": asks, "ts": order_book['ts']}


class LiveSnapshotDownloader():
    def __init__(self, api_key_path: str, folder_path: str, symbol: str, category: str):

        self.folder_path = folder_path + "/" + category + "/" + symbol
        self.symbol = symbol
        self.category = category

        # Create data retriever
        self.pld = PeriodicLiveRetriever(
            api_key_path, 1000, self.symbol, self.category)
        self.pld.start()

        self.last_snapshot = None

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

                # make directory if not exists
                os.makedirs(self.folder_path, exist_ok=True)
                # download and append snapshot to folder/{date}_snapshot.data
                today = datetime.now().strftime("%Y-%m-%d")

                file_path = f"{self.folder_path}/{today}_snapshots.msgpack.gz"
                packed_data = msgpack.packb(snapshot_data, use_bin_type=True)

                # Open with gzip in append binary mode ("ab")
                with gzip.open(file_path, "ab") as f:
                    f.write(packed_data)

            except Exception as e:
                print("Passing exception", e)

    def calculate_trades_since_last_timestep(self, new_trades: List[dict]):
        """Calculate trades since last timestamp"""
        if self.last_snapshot is None:
            for trade in new_trades:
                trade["price"] = float(trade["price"])
                trade["size"] = float(trade["size"])
            return new_trades

        last_timestep = self.last_snapshot["ts"]

        trades_since_last_timestep = [
            trade for trade in new_trades if int(trade["time"]) > last_timestep]

        for trade in trades_since_last_timestep:
            trade["price"] = float(trade["price"])
            trade["size"] = float(trade["size"])

        # print(trades_since_last_timestep)

        return trades_since_last_timestep


if __name__ == "__main__":
    api_key_path = "api_key.json"
    folder_path = "live_snapshots"
    symbol = "CAKEUSDT"
    category = "spot"
    if len(sys.argv) > 1:
        api_key_path = sys.argv[1]
    if len(sys.argv) > 4:
        folder_path = sys.argv[2]
        symbol = sys.argv[3]
        category = sys.argv[4]
    app = LiveSnapshotDownloader(api_key_path, folder_path, symbol, category)
