from datetime import timedelta, datetime
import numpy as np
import json
import pandas as pd
from typing import List
from data_processing.FeatureCreation import FeatureCreator


class DataLoader:
    def __init__(self):
        pass

    def zero_pad(self, x):
        if x < 10:
            return f"0{x}"
        else:
            return x

    def update_order_book(self, current_bids, current_asks, new_bids, new_asks, ob_type):
        """
        Given a line of the historical orderbook data, update the order book.
        """

        # update order book
        if ob_type == "snapshot":
            current_bids = new_bids
            current_asks = new_asks

        else:
            # update bids
            for price in new_bids:
                if new_bids[price] == 0:
                    if price in current_bids:
                        del current_bids[price]
                else:
                    current_bids[price] = new_bids[price]

            # update asks
            for price in new_asks:
                if new_asks[price] == 0:
                    if price in current_asks:
                        del current_asks[price]
                else:
                    current_asks[price] = new_asks[price]

        return current_bids, current_asks

    def load_features_from_data(self,
                                folder: str,
                                contract: str,
                                start_date: datetime,
                                end_date: datetime,
                                time_delta: timedelta,
                                input_feature_creator: FeatureCreator,
                                output_feature_creator: FeatureCreator,
                                category: str = "linear"):
        """
        Calculate the input and output features between the start and end date,
        from the trade and order book data in the specified folder.
        """

        feature_data = []
        target_data = []

        curr_date = start_date
        next_ts = start_date + time_delta

        while curr_date <= end_date:
            trade_id = 0

            print(f"{curr_date.year}-{self.zero_pad(curr_date.month)}-{self.zero_pad(curr_date.day)}")

            filepath_ob = f"{folder}/ob/{category}/{contract}/{curr_date.year}-{self.zero_pad(curr_date.month)}-{self.zero_pad(curr_date.day)}_{contract}_ob500.data"
            filepath_trades = f"{folder}/td/{contract}/{contract}{curr_date.year}-{self.zero_pad(curr_date.month)}-{self.zero_pad(curr_date.day)}.csv.gz"

            hist_trade_data = pd.read_csv(filepath_trades, compression='gzip')
            current_bids = None
            current_asks = None
            current_trades = []

            with open(filepath_ob, 'rb') as f:
                for line_number, line in enumerate(f, 1):

                    # read data from line
                    data = json.loads(line.decode('utf-8'))
                    ts = datetime.fromtimestamp(data['ts']/1000)
                    new_bids = {float(price): float(size)
                                for price, size in data['data']['b']}
                    new_asks = {float(price): float(size)
                                for price, size in data['data']['a']}

                    # update order book
                    current_bids, current_asks = self.update_order_book(
                        current_bids, current_asks,
                        new_bids, new_asks,
                        data['type'])

                    # collect trades
                    while trade_id < len(hist_trade_data) and hist_trade_data.iloc[trade_id]["timestamp"] <= data["ts"]/1000:
                        trade = hist_trade_data.iloc[trade_id]
                        current_trades.append(trade)
                        trade_id += 1

                    # skip until next_ts is reached
                    if ts < next_ts:
                        continue
                    next_ts += time_delta

                    # store formatted data
                    mid_price = (max(current_bids.keys()) +
                                 min(current_asks.keys()))/2
                    # print(mid_price)

                    snapshot = {
                        'timestamp': ts,
                        'mid_price': mid_price,
                        'bids': current_bids.copy(),
                        'asks': current_asks.copy(),
                        'trades': current_trades
                    }
                    current_trades = []

                    input_feature_creator.feed_datapoint(snapshot)
                    output_feature_creator.feed_datapoint(snapshot)

                    if input_feature_creator.is_ready() and output_feature_creator.is_ready():
                        feature_data.append(input_feature_creator.create_features())
                        target_data.append(output_feature_creator.create_features())

            curr_date += timedelta(days=1)

        return np.array(feature_data), np.array(target_data)


def convert_bybit_ob_to_snapshot(order_book):
    """
    This function takes a live bybit order book and converts it into a format that is better to work with
    """

    ts = datetime.fromtimestamp(order_book['ts']/1000)

    # Convert bids list to dictionary with validation
    bids = {float(price): float(size) for price, size in order_book['b']}

    # Convert asks list to dictionary with validation
    asks = {float(price): float(size) for price, size in order_book['a']}

    mid_price = (min(asks.keys()) + max(bids.keys()))/2

    return {"ts": ts, "mid_price": mid_price, "bids": bids, "asks": asks}
