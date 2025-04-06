from datetime import timedelta, datetime
import numpy as np
import json
import pandas as pd


def zero_pad(x):
    if x < 10:
        return f"0{x}"
    else:
        return x


def load_ob_data(
        folder,
        contract,
        start_date,
        end_date,
        exchange):

    if exchange == "bybit":
        snapshots = []

        curr_date = start_date
        while curr_date <= end_date:

            filepath_ob = f"{folder}ob/{curr_date.year}-{zero_pad(curr_date.month)}-{
                zero_pad(curr_date.day)}_{contract}_ob500.data"
            filepath_trades = f"{folder}trades/{contract}{curr_date.year}-{
                zero_pad(curr_date.month)}-{zero_pad(curr_date.day)}.csv.gz"

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


def build_bucket_snapshots(ob_data, td, num_buckets, bucket_size):

    if ob_data[0]["type"] != "snapshot":
        print("first element must be of type snapshot")

    curr_bids = ob_data[0]["bids"].copy()
    curr_asks = ob_data[0]["asks"].copy()

    curr_ts = ob_data[0]["timestamp"]
    next_ts = curr_ts

    snapshots = []

    for i in range(1, len(ob_data)):

        # store snapshot
        if curr_ts >= next_ts:

            max_bid = max(curr_bids.keys())
            min_ask = min(curr_asks.keys())

            mid_price = (max_bid + min_ask)/2

            bid_buckets = [0 for i in range(num_buckets)]
            ask_buckets = [0 for i in range(num_buckets)]

            for level in curr_bids:
                bucket = min(int((mid_price - level)/mid_price /
                             bucket_size), num_buckets - 1)
                if bucket < 0:
                    print(f"{i}, {mid_price}, {level} -> {bucket}")
                bid_buckets[bucket] += curr_bids[level]

            for level in curr_asks:
                bucket = min(int((level - mid_price)/mid_price /
                             bucket_size), num_buckets - 1)
                bucket = num_buckets - 1 - bucket
                ask_buckets[bucket] += curr_asks[level]

            snapshot = {
                "timestamp": curr_ts,
                "midprice": mid_price,
                "bids": bid_buckets,
                "asks": ask_buckets,
                "num_bid_takers": ob_data[i-1]["num_bid_takers"],
                "num_ask_takers": ob_data[i-1]["num_ask_takers"],
                "size_bid_takers": ob_data[i-1]["size_bid_takers"],
                "size_ask_takers": ob_data[i-1]["size_ask_takers"],
                "vwap": ob_data[i-1]["vwap"]  # TODO: this is not correct
            }
            snapshots.append(snapshot)

            next_ts = curr_ts + td

        # update order book snapshot
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

    return snapshots


def build_level_snapshots(ob_data, td, levels):

    if ob_data[0]["type"] != "snapshot":
        print("first element must be of type snapshot")

    curr_bids = ob_data[0]["bids"].copy()
    curr_asks = ob_data[0]["asks"].copy()

    curr_ts = ob_data[0]["timestamp"]
    next_ts = curr_ts

    snapshots = []

    for i in range(1, len(ob_data)):

        # store snapshot
        if curr_ts >= next_ts:

            top_bids = dict(sorted(curr_bids.items(), reverse=True)[:levels])
            top_asks = dict(sorted(curr_asks.items())[:levels])
            min_ask = min(top_asks)
            max_bid = max(top_bids)

            snapshot = {
                "timestamp": curr_ts,
                "midprice": (max_bid + min_ask)/2,
                "bids": top_bids,
                "asks": top_asks,
                "num_bid_takers": ob_data[i-1]["num_bid_takers"],
                "num_ask_takers": ob_data[i-1]["num_ask_takers"],
                "size_bid_takers": ob_data[i-1]["size_bid_takers"],
                "size_ask_takers": ob_data[i-1]["size_ask_takers"],
                "vwap": ob_data[i-1]["vwap"]


            }
            snapshots.append(snapshot)

            next_ts = curr_ts + td

        # update order book snapshot
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

    return snapshots


def build_bucket_data_set(bucket_snapshots, horizon, step_size, window_len):

    ob_buckets = []
    aux_features = []
    mid_price_changes = []

    for i in range(0, len(bucket_snapshots) - horizon, step_size):

        mid_price_change = (bucket_snapshots[i]["midprice"] - bucket_snapshots[i +
                            horizon]["midprice"])/bucket_snapshots[i+horizon]["midprice"]

        bid_buckets = bucket_snapshots[i]["bids"]
        ask_buckets = bucket_snapshots[i]["asks"]

        # Concatenate both arrays
        ob = np.concatenate([bid_buckets, ask_buckets])

        ob_buckets.append(ob)
        mid_price_changes.append(mid_price_change)

    ob_buckets = np.array(ob_buckets)
    mid_price_changes = np.array(mid_price_changes)

    X_data = []
    Y_data = []

    # Create sequences up to the last possible complete sequence
    for i in range(len(ob_buckets) - window_len + 1):
        sequence = ob_buckets[i:i + window_len]
        X_data.append(sequence)
        Y_data.append(mid_price_changes[i + window_len - 1])

    return np.array(X_data), np.array(Y_data)


def build_bucket_change_data_set(bucket_snapshots, horizon, step_size, window_len):

    bid_buckets = bucket_snapshots[0]["bids"]
    ask_buckets = bucket_snapshots[0]["asks"]

    ob_bucket_changes = []
    mid_price_changes = []

    for i in range(step_size, len(bucket_snapshots) - horizon, step_size):

        mid_price_change = (bucket_snapshots[i]["midprice"] - bucket_snapshots[i +
                            horizon]["midprice"])/bucket_snapshots[i+horizon]["midprice"]

        curr_bid_buckets = bucket_snapshots[i]["bids"]
        curr_ask_buckets = bucket_snapshots[i]["asks"]

        bid_change = (curr_bid_buckets - np.array(bid_buckets)) / \
            curr_bid_buckets
        ask_change = (curr_ask_buckets - np.array(ask_buckets)) / \
            curr_ask_buckets

        # Concatenate both arrays
        ob_changes = np.concatenate([bid_change, ask_change])

        ob_bucket_changes.append(ob_changes)
        mid_price_changes.append(mid_price_change)

        bid_buckets = curr_bid_buckets
        ask_buckets = curr_ask_buckets

    ob_bucket_changes = np.array(ob_bucket_changes)
    mid_price_changes = np.array(mid_price_changes)

    X_data = []
    Y_data = []

    # Create sequences up to the last possible complete sequence
    for i in range(len(ob_bucket_changes) - window_len + 1):
        sequence = ob_bucket_changes[i:i + window_len]
        X_data.append(sequence)
        Y_data.append(mid_price_changes[i + window_len - 1])

    return np.array(X_data), np.array(Y_data)


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
