from datetime import timedelta, datetime
import numpy as np
import json
import pandas as pd
from data_format.order_book_format import *
from data_format.trade_format import *



def zero_pad(x):
    if x < 10:
        return f"0{x}"
    else:
        return x

def signed_log_transform(x):
    """Apply sign-preserving log transformation to data."""
    # Get the sign of the data
    signs = np.sign(x)
    log_abs = np.log1p(np.abs(x))
    return signs * log_abs

def signed_root_transform(x):
    """Apply sign-preserving log transformation to data."""
    # Get the sign of the data
    signs = np.sign(x)
    root_abs = np.sqrt(np.abs(x))
    return signs * root_abs

def normalise(x_train, x_val):
    # Compute mean and std along the samples axis (axis 0)
    train_mean = np.mean(x_train, axis=0)
    train_std = np.std(x_train, axis=0)

    # Normalize while preserving 3D structure
    x_train = (x_train - train_mean) / train_std
    x_val = (x_val - train_mean) / train_std

    return x_train, x_val

def normalise_std(x_train, x_val):
    # Compute mean and std along the samples axis (axis 0)
    
    train_std = np.std(x_train, axis=0)

    # Normalize while preserving 3D structure
    x_train = x_train / train_std
    x_val = x_val / train_std

    return x_train, x_val


def update_order_book(current_bids, current_asks, new_bids, new_asks, ob_type):

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


def load_ob_and_trade_data(

        folder,
        contract,
        start_date,
        end_date,
        td,
        ob_format_func,
        trade_format_func,
        category="linear"):

        ob_data = []
        trade_data = []
        mp_data = []

        curr_date = start_date
        next_ts = start_date + td
        trade_id = 0

        while curr_date <= end_date:
            

            print(f"{curr_date.year}-{zero_pad(curr_date.month)}-{zero_pad(curr_date.day)}")
            
            filepath_ob = f"{folder}/ob/{category}/{contract}/{curr_date.year}-{zero_pad(curr_date.month)}-{zero_pad(curr_date.day)}_{contract}_ob500.data"
            filepath_trades = f"{folder}/td/{contract}/{contract}{curr_date.year}-{zero_pad(curr_date.month)}-{zero_pad(curr_date.day)}.csv.gz"

            hist_trade_data = pd.read_csv(filepath_trades, compression='gzip')
            current_bids = None
            current_asks = None


            with open(filepath_ob, 'rb') as f:
                for line_number, line in enumerate(f, 1):
                    
                    # read data from line
                    data = json.loads(line.decode('utf-8'))
                    ts = datetime.fromtimestamp(data['ts']/1000)
                    new_bids = {float(price): float(size) for price, size in data['data']['b']}
                    new_asks = {float(price): float(size) for price, size in data['data']['a']}
                    
                    # update order book
                    current_bids, current_asks = update_order_book(
                                                            current_bids, current_asks, 
                                                            new_bids, new_asks, 
                                                            data['type'])
                    
                    mid_price = (max(current_bids.keys()) + min(current_asks.keys()))/2

                    # skip until next_ts is reached
                    if ts < next_ts:
                        continue
                    next_ts += td

                    
                    # calculate trade statistics between the last two order books
                    num_bid_takers = num_ask_takers = 0
                    size_bid_takers = size_ask_takers = 0
                    vwap = 0
                    while trade_id < len(hist_trade_data) and hist_trade_data.iloc[trade_id]["timestamp"] <= data["ts"]/1000:
                        trade = hist_trade_data.iloc[trade_id]
                        trade_id += 1


                while trade_id < len(df) and df.iloc[trade_id]["timestamp"] <= data["ts"]/1000:
                    trade = df.iloc[trade_id]
                    trade_id += 1

                    if trade["side"] == "Buy":
                        num_bid_takers += 1
                        size_bid_takers += trade["size"]
                    else:
                        num_ask_takers += 1
                        size_ask_takers += trade["size"]

                    vwap += trade["price"] * trade["size"]



                    # store formatted data
                    snapshot = {
                        'timestamp': ts,
                        'mid_price': mid_price,
                        'bids': current_bids.copy(),
                        'asks': current_asks.copy(),
                        'num_bid_takers': num_bid_takers,
                        'num_ask_takers': num_ask_takers,
                        'size_bid_takers': size_bid_takers,
                        'size_ask_takers': size_ask_takers,
                        'vwap': vwap
                    }

                    ob_data.append(ob_format_func(snapshot))
                    trade_data.append(trade_format_func(snapshot))
                    mp_data.append(mid_price)
            curr_date += timedelta(days=1)

        return ob_data, trade_data, mp_data
    

def build_data_set(ob_data, trade_data, mp_data, horizon, window_len, steps_between):

    X_ob = []
    X_trade = []
    Y_data = []

    # Create sequences up to the last possible complete sequence
    for i in range(0, len(ob_data) - window_len - horizon, steps_between):
        
        X_ob.append(ob_data[i:i + window_len])
        X_trade.append(trade_data[i:i + window_len])
        Y_data.append(mp_data[i + window_len + horizon] - mp_data[i + window_len])

    return np.array(X_ob), np.array(X_trade), np.array(Y_data)


def convert_bybit_ob_to_snapshot(order_book):
    """
    This function takes a bybit order book and converts it into a format that is better to work with
    """

    ts = datetime.fromtimestamp(order_book['ts']/1000)

    # Convert bids list to dictionary with validation
    bids = {float(price): float(size) for price, size in order_book['b']}

    # Convert asks list to dictionary with validation
    asks = {float(price): float(size) for price, size in order_book['a']}

    mid_price = (min(asks.keys()) + max(bids.keys()))/2

    return {"ts": ts, "mid_price": mid_price, "bids": bids, "asks": asks}

