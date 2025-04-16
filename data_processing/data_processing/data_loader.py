from datetime import timedelta, datetime
import numpy as np
import json
import pandas as pd
from data_processing.order_book_format import *
from data_processing.trade_format import *



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
    








def load_ob_and_trade_data_new(
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
                        'trades': [],
                        
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





#### OLD ####


def load_ob_data_old(folder, contract, start_date, end_date, exchange, category="linear"):

    if exchange == "bybit":
        snapshots = []

        curr_date = start_date
        while curr_date <= end_date:

            filepath_ob = f"{folder}/ob/{category}/{contract}/{curr_date.year}-{zero_pad(curr_date.month)}-{zero_pad(curr_date.day)}_{contract}_ob500.data"
            filepath_trades = f"{folder}/td/{contract}/{contract}{curr_date.year}-{zero_pad(curr_date.month)}-{zero_pad(curr_date.day)}.csv.gz"

            print(f"{curr_date.year}-{zero_pad(curr_date.month)}-{zero_pad(curr_date.day)}")

            df = pd.read_csv(filepath_trades, compression='gzip')
            trade_id = 0

            with open(filepath_ob, 'rb') as f:
                for line_number, line in enumerate(f, 1):

                    data = json.loads(line.decode('utf-8'))
                    ts = datetime.fromtimestamp(data['ts']/1000)

                    # Convert bids list to dictionary with validation
                    bids = {float(price): float(size)
                            for price, size in data['data']['b']}

                    # Convert asks list to dictionary with validation
                    asks = {float(price): float(size)
                            for price, size in data['data']['a']}

                    num_bid_takers = 0
                    num_ask_takers = 0
                    size_bid_takers = 0
                    size_ask_takers = 0
                    vwap = 0

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

                    if vwap != 0:
                        vwap /= (size_bid_takers + size_ask_takers)

                    snapshot = {
                        'timestamp': ts,
                        'bids': bids,
                        'asks': asks,
                        'type': data['type'],
                        'seq': data['data'].get('seq'),
                        'update_id': data['data'].get('u'),
                        'num_bid_takers': num_bid_takers,
                        'num_ask_takers': num_ask_takers,
                        'size_bid_takers': size_bid_takers,
                        'size_ask_takers': size_ask_takers,
                        'vwap': vwap
                    }

                    snapshots.append(snapshot)

            curr_date += timedelta(days=1)

        return snapshots

    else:
        print(f"{exchange} not implemented")
        return None


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


def get_bucket_representation(ob_snapshot, num_buckets, bucket_size):

    curr_bids = ob_snapshot["bids"]
    curr_asks = ob_snapshot["asks"]

    max_bid = max(curr_bids.keys())
    min_ask = min(curr_asks.keys())
    # print(f"min_ask: {min_ask}, max_bid: {max_bid}")
    mid_price = (max_bid + min_ask)/2

    bid_buckets = [0 for i in range(num_buckets)]
    ask_buckets = [0 for i in range(num_buckets)]

    for level in curr_bids:
        bucket = min(int((mid_price - level)/mid_price /
                     bucket_size), num_buckets - 1)
        bid_buckets[bucket] += curr_bids[level]

    for level in curr_asks:
        bucket = min(int((level - mid_price)/mid_price /
                     bucket_size), num_buckets - 1)
        # print(f"level: {level}, midprice: {mid_price}")
        # print(f"bucket: {bucket}")
        bucket = num_buckets - 1 - bucket
        # print(f"len: {len(ask_buckets)}, index: {bucket}")
        ask_buckets[bucket] += curr_asks[level]

    return {
        "midprice": mid_price,
        "bids": bid_buckets,
        "asks": ask_buckets
    }


def get_level_representation(ob_snapshot, num_levels):
    return None


def build_bucket_snapshots(ob_data, td, num_buckets, bucket_size):

    if ob_data[0]["type"] != "snapshot":
        print("first element must be of type snapshot")

    next_ts = ob_data[0]["timestamp"] + td
    snapshots = []

    vwap = 0
    num_bid_takers = 0
    num_ask_takers = 0
    size_bid_takers = 0
    size_ask_takers = 0
    curr_bids = ob_data[0]["bids"].copy()
    curr_asks = ob_data[0]["asks"].copy()

    for i in range(1, len(ob_data)):

        curr_ts = ob_data[i]["timestamp"]

        if ob_data[i]["type"] == 'snapshot':
            curr_bids = ob_data[i]["bids"]
            curr_asks = ob_data[i]["asks"]

        elif ob_data[i]["type"] == "delta":

            # update bids
            new_bids = ob_data[i]["bids"]
            for price in new_bids:
                if new_bids[price] == 0:
                    if price in curr_bids:
                        del curr_bids[price]
                else:
                    curr_bids[price] = new_bids[price]

            # update asks
            new_asks = ob_data[i]["asks"]
            for price in new_asks:
                if new_asks[price] == 0:
                    if price in curr_asks:
                        del curr_asks[price]
                else:
                    curr_asks[price] = new_asks[price]

        # update aux features
        num_bid_takers += ob_data[i]["num_bid_takers"]
        num_ask_takers += ob_data[i]["num_ask_takers"]
        size_bid_takers += ob_data[i]["size_bid_takers"]
        size_ask_takers += ob_data[i]["size_ask_takers"]
        vwap += ob_data[i]["vwap"] * \
            (ob_data[i]["size_bid_takers"] + ob_data[i]["size_ask_takers"])

        # store snapshot
        if curr_ts >= next_ts:

            max_bid = max(curr_bids.keys())
            min_ask = min(curr_asks.keys())
            # print(f"min_ask: {min_ask}, max_bid: {max_bid}")
            mid_price = (max_bid + min_ask)/2

            bid_buckets = [0 for i in range(num_buckets)]
            ask_buckets = [0 for i in range(num_buckets)]

            for level in curr_bids:
                bucket = min(int((mid_price - level)/mid_price /
                             bucket_size), num_buckets - 1)
                bid_buckets[bucket] += curr_bids[level]

            for level in curr_asks:
                bucket = min(int((level - mid_price)/mid_price /
                             bucket_size), num_buckets - 1)
                # print(f"level: {level}, midprice: {mid_price}")
                # print(f"bucket: {bucket}")
                bucket = num_buckets - 1 - bucket
                # print(f"len: {len(ask_buckets)}, index: {bucket}")
                ask_buckets[bucket] += curr_asks[level]

            snapshot = {
                "timestamp": curr_ts,
                "midprice": mid_price,
                "bids": bid_buckets,
                "asks": ask_buckets,
                "num_bid_takers": num_bid_takers,
                "num_ask_takers": num_ask_takers,
                "size_bid_takers": size_bid_takers,
                "size_ask_takers": size_ask_takers,
                "vwap": 0 if vwap == 0 else vwap / (size_ask_takers + size_bid_takers)
            }
            snapshots.append(snapshot)
            num_bid_takers = num_ask_takers = size_bid_takers = size_ask_takers = vwap = 0

            next_ts = curr_ts + td

    return snapshots




def build_bucket_data_set(ob_data, trade_data, horizon, window_len, steps_between):

    ob_buckets = []
    aux_features = []
    mid_price_changes = []

    for i in range(len(bucket_snapshots) - horizon):

        # /bucket_snapshots[i]["midprice"]
        mid_price_change = (
            bucket_snapshots[i+horizon]["midprice"] - bucket_snapshots[i]["midprice"])


        bid_buckets = bucket_snapshots[i]["bids"]
        ask_buckets = bucket_snapshots[i]["asks"]

        # Concatenate both arrays
        ob = np.concatenate([bid_buckets, ask_buckets])

        ob_buckets.append(ob)
        aux_features.append(np.array([
            bucket_snapshots[i]["num_bid_takers"],
            bucket_snapshots[i]["num_ask_takers"],
            bucket_snapshots[i]["size_bid_takers"],
            bucket_snapshots[i]["size_ask_takers"],
            bucket_snapshots[i]["vwap"] - bucket_snapshots[i-1]["vwap"]
        ]))
        mid_price_changes.append(mid_price_change)

    ob_buckets = np.array(ob_buckets)
    mid_price_changes = np.array(mid_price_changes)

    X_data = []
    X_aux = []
    Y_data = []

    # Create sequences up to the last possible complete sequence
    for i in range(0, len(ob_buckets) - window_len + 1, steps_between):
        sequence = ob_buckets[i:i + window_len]
        X_data.append(sequence)
        X_aux.append(aux_features[i:i + window_len])
        Y_data.append(mid_price_changes[i + window_len - 1])

    return np.array(X_data), np.array(X_aux), np.array(Y_data)


def build_bucket_change_data_set(bucket_snapshots, horizon, window_len, steps_between):

    bid_buckets = bucket_snapshots[0]["bids"]
    ask_buckets = bucket_snapshots[0]["asks"]

    ob_bucket_changes = []
    mid_price_changes = []
    aux_features = []

    for i in range(1, len(bucket_snapshots) - horizon):

        # /bucket_snapshots[i]["midprice"]
        mid_price_change = (
            bucket_snapshots[i+horizon]["midprice"] - bucket_snapshots[i]["midprice"])

        curr_bid_buckets = bucket_snapshots[i]["bids"]
        curr_ask_buckets = bucket_snapshots[i]["asks"]

        # / bid_buckets
        bid_change = (curr_bid_buckets - np.array(bid_buckets))
        # / ask_buckets
        ask_change = (curr_ask_buckets - np.array(ask_buckets))

        # Concatenate both arrays
        ob_changes = np.concatenate([bid_change, ask_change])

        ob_bucket_changes.append(ob_changes)
        mid_price_changes.append(mid_price_change)

        aux_features.append(np.array([
            bucket_snapshots[i]["num_bid_takers"],
            bucket_snapshots[i]["num_ask_takers"],
            bucket_snapshots[i]["size_bid_takers"],
            bucket_snapshots[i]["size_ask_takers"],
            bucket_snapshots[i]["vwap"] - bucket_snapshots[i-1]["vwap"]
        ]))

        bid_buckets = curr_bid_buckets
        ask_buckets = curr_ask_buckets

    ob_bucket_changes = np.array(ob_bucket_changes)
    mid_price_changes = np.array(mid_price_changes)

    X_data = []
    X_aux = []
    Y_data = []

    # Create sequences up to the last possible complete sequence
    for i in range(0, len(ob_bucket_changes) - window_len + 1, steps_between):
        sequence = ob_bucket_changes[i:i + window_len]
        X_data.append(sequence)
        X_aux.append(aux_features[i:i + window_len])
        Y_data.append(mid_price_changes[i + window_len - 1])

    return np.array(X_data), np.array(X_aux), np.array(Y_data)


def build_level_data_set(snapshots, horizon, step_size):

    bid_levels, bid_sizes = map(
        list, zip(*sorted(snapshots[0]["bids"].items())))
    ask_levels, ask_sizes = map(
        list, zip(*sorted(snapshots[0]["asks"].items())))

    ob_changes = []
    targets = []

    for i in range(step_size, len(snapshots) - horizon, step_size):

        diff = (snapshots[i]["midprice"] - snapshots[i+horizon]
                ["midprice"])/snapshots[i+horizon]["midprice"]

        curr_bid_levels, curr_bid_sizes = map(
            list, zip(*sorted(snapshots[i]["bids"].items())))
        curr_ask_levels, curr_ask_sizes = map(
            list, zip(*sorted(snapshots[i]["asks"].items())))

        bid_levels_change = (
            curr_bid_levels - np.array(bid_levels))/curr_bid_levels
        bid_sizes_change = (
            curr_bid_sizes - np.array(bid_sizes))/curr_bid_sizes

        ask_levels_change = (
            curr_ask_levels - np.array(ask_levels))/curr_ask_levels
        ask_sizes_change = (
            curr_ask_sizes - np.array(ask_sizes))/curr_ask_sizes

        # First interleave bid changes
        bid_changes = np.ravel(
            [bid_levels_change, bid_sizes_change], order='F')

        # Then interleave ask changes
        ask_changes = np.ravel(
            [ask_levels_change, ask_sizes_change], order='F')

        # Concatenate both arrays
        all_changes = np.concatenate([ask_changes, bid_changes])

        ob_changes.append(all_changes)
        targets.append(diff)

        bid_levels, bid_sizes = curr_bid_levels, curr_bid_sizes
        ask_levels, ask_sizes = curr_ask_levels, curr_ask_sizes

    return np.array(ob_changes), np.array(targets)
