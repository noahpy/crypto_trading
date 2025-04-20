from datetime import timedelta, datetime
import numpy as np
import json
import pandas as pd
from typing import List
from data_processing.FeatureCreation import FeatureCreator
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pytz
import time


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






    @staticmethod
    def _process_single_day(args):
        """Worker function to process a single day and return snapshots"""
        folder, contract, curr_date, time_delta, category, zero_pad_func, update_order_book_func = args
        
        snapshots = []
        
        day_str = f"{curr_date.year}-{zero_pad_func(curr_date.month)}-{zero_pad_func(curr_date.day)}"
        print(f"Start loading snapshots for {day_str}")
        
        filepath_ob = f"{folder}/ob/{category}/{contract}/{day_str}_{contract}_ob500.data"
        filepath_trades = f"{folder}/td/{contract}/{contract}{curr_date.year}-{zero_pad_func(curr_date.month)}-{zero_pad_func(curr_date.day)}.csv.gz"
        
        # Check if files exist
        if not os.path.exists(filepath_ob) or not os.path.exists(filepath_trades):
            print(f"Missing files for {day_str}, skipping...")
            return []
        
        try:
            hist_trade_data = pd.read_csv(filepath_trades, compression='gzip')
            current_bids = None
            current_asks = None
            current_trades = []
            next_ts = curr_date + time_delta
            trade_id = 0
            
            with open(filepath_ob, 'rb') as f:
                for line_number, line in enumerate(f, 1):
                    # read data from line
                    data = json.loads(line.decode('utf-8'))
                    ts = datetime.fromtimestamp(data['ts']/1000, tz=pytz.UTC)
                    
                    new_bids = {float(price): float(size) for price, size in data['data']['b']}
                    new_asks = {float(price): float(size) for price, size in data['data']['a']}
                    
                    # update order book
                    current_bids, current_asks = update_order_book_func(
                        current_bids, current_asks,
                        new_bids, new_asks,
                        data['type'])
                            
                    
                    if ts < next_ts:# and peek:
                        continue
                        
                    # collect 
                    next_ts_unix = next_ts.timestamp()
                    while trade_id < len(hist_trade_data) and hist_trade_data.iloc[trade_id]["timestamp"] <= next_ts_unix:
                        trade = hist_trade_data.iloc[trade_id]
                        current_trades.append({"side": trade["side"], "size": trade["size"], "price": trade["price"]})
                        trade_id += 1

                    next_ts += time_delta
                    
                    # store formatted data
                    mid_price = (max(current_bids.keys()) + min(current_asks.keys()))/2
                        
                    snapshot = {
                            'timestamp': ts,
                            'mid_price': mid_price,
                            'bids': current_bids.copy(),
                            'asks': current_asks.copy(),
                            'trades': current_trades.copy()  # Important to copy
                        }
                    snapshots.append(snapshot)
                    current_trades = []

            print(f"End loading snapshots for {day_str}")


            return snapshots
            
        except Exception as e:
            print(f"Error processing {day_str}: {str(e)}")
            return []

    def load_features_from_data_parallel(self,
                                     folder: str,
                                     contract: str,
                                     start_date: datetime,
                                     end_date: datetime,
                                     time_delta: timedelta,
                                     input_feature_creator,
                                     output_feature_creator,
                                     category: str = "linear",
                                     max_workers: int = None):
        """
        Parallelized version to calculate features from order book data.
        Worker processes only generate snapshots, while the main thread processes features.
        """

        

        # If max_workers not specified, use CPU count
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        # Generate list of dates to process
        curr_date = start_date
        dates = []
        while curr_date <= end_date:
            dates.append(curr_date)
            curr_date += timedelta(days=1)
        
        # Create arguments for each day's processing
        # Pass functions as arguments to avoid pickling issues
        args_list = [
            (folder, contract, date, time_delta, category, self.zero_pad, self.update_order_book)
            for date in dates
        ]
        
        
        # Process days in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures_list = [executor.submit(self._process_single_day, args) for args in args_list]
            print("submitted all data loading jobs")


            feature_data = []
            target_data = []

            for future in futures_list:
                while not future.done():
                    time.sleep(0.1)
            

                day_snapshots = future.result()
                print(f"creating features for {day_snapshots[0]['timestamp']}")

                for snapshot in day_snapshots:
                    
                    input_feature_creator.feed_datapoint(snapshot)
                    output_feature_creator.feed_datapoint(snapshot)
                    
                    if input_feature_creator.is_ready() and output_feature_creator.is_ready():
                        feature_data.append(input_feature_creator.create_features())
                        target_data.append(output_feature_creator.create_features())
        
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
