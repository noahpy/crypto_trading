from datetime import timedelta, datetime
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Union, Any


def ob_bucket_format(snapshot: Dict[str, Any], num_buckets: int = 10, bucket_size: float = 0.005) -> List[float]:
    """
    Transforms order book data into fixed-size buckets based on price distance from mid-price.
    
    @Params:
    - snapshot: Dictionary containing 'bids', 'asks', and 'mid_price' keys
    - num_buckets: Number of buckets to create for each side of the book
    - bucket_size: Size of each bucket as a percentage of mid-price
    
    Returns:
    - List containing aggregated volume in each bucket (ask buckets followed by bid buckets)
    """

    curr_bids = snapshot["bids"]
    curr_asks = snapshot["asks"]
    mid_price = snapshot["mid_price"]

    bid_buckets = [0 for i in range(num_buckets)]
    ask_buckets = [0 for i in range(num_buckets)]

    for level in curr_bids:
        bucket = min(int((mid_price - level)/mid_price / bucket_size), num_buckets - 1)
        bid_buckets[bucket] += curr_bids[level]

    for level in curr_asks:
        bucket = min(int((level - mid_price)/mid_price /bucket_size), num_buckets - 1)
        bucket = num_buckets - 1 - bucket
        ask_buckets[bucket] += curr_asks[level]

    return ask_buckets + bid_buckets



def ob_top_level_format(snapshot: Dict[str, Dict[float, float]], num_levels: int = 10, shuffle: bool = True) -> np.ndarray:
    """
    Records the best >num_levels< on each side.
    @Params:
        - snapshot: dict must contain "asks" and "bids"
        - num_levels: number of levels for each side
        - shuffle:
            True:  -> [ask_vol_9, ask_price_9, ... ,ask_vol_0, ask_price_0, bid_vol_0, bid_price_0, ... , bid_vol_9, bid_price_9]
            False: -> [ask_vol_9, ... ,ask_vol_0, bid_vol_9, ... , bid_vol_0, ask_price_9, ... ,ask_vol_0, bid_price_0, bid_price_9]
    """

    bids = snapshot["bids"]
    asks = snapshot["asks"]

    max_bids = sorted(bids.keys(), reverse=False)[:num_levels]
    min_asks = sorted(asks.keys(), reverse=True)[:num_levels]

    level_data = np.zeros(4 * num_levels)

    for i in range(num_levels):
        if shuffle:
            level_data[2*i]                  = min_asks[i]
            level_data[2*i+1]                = asks[min_asks[i]]
            level_data[2*num_levels + 2*i]   = max_bids[i]
            level_data[2*num_levels + 2*i+1] = bids[max_bids[i]]
        
        else:
            level_data[i]               = min_asks[i]
            level_data[1*num_levels+i]  = max_bids[i]
            level_data[2*num_levels+i]  = asks[min_asks[i]]
            level_data[3*num_levels+i]  = bids[max_bids[i]]

    return level_data




    