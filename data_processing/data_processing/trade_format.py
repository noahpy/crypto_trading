from datetime import timedelta, datetime
import numpy as np
import json
import pandas as pd



def trade_format(snapshot):

    return [
        snapshot['num_bid_takers'],
        snapshot['num_ask_takers'],
        snapshot['size_bid_takers'],
        snapshot['size_ask_takers'],
        snapshot['vwap']
    ]