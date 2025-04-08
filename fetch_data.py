# fetch data using pybit (https://github.com/bybit-exchange/pybit)
#
# Example usage: python3 fetch_data.py api_key.json CAKEUSDT
# 


from pybit.unified_trading import HTTP as BybitSession
import json
import sys
import requests
from datetime import datetime, timedelta
import zipfile
import os
from multiprocessing.pool import ThreadPool

MAX_THREADS = 5  # <--- tweak this


def zero_pad_time(x: int):
    if x < 10:
        return f"0{x}"
    else:
        return x


def create_session(key_file_path: str) -> BybitSession:
    """
    Create a BybitSession object from a JSON key file.
    """
    with open(key_file_path) as f:
        keys = json.load(f)
    session = BybitSession(api_key=keys['key'], api_secret=keys['secret'])
    return session


def fetch_current_orderbook(session: BybitSession, symbol: str,
                            category: str = "spot", limit: int = 200) -> dict:
    """
    Fetch current orderbook data.
    category can be: spot, inverse, linear
    """
    if (limit < 0 or limit > 200):
        limit = 200
    try:
        orderbook = session.get_orderbook(
            category=category, symbol=symbol, limit=limit)
        print(type(orderbook))
        return orderbook
    except Exception as e:
        print("Error fetching orderbook!")
        print(e)
        

def fetch_recent_trading_history(session: BybitSession, symbol: str, category: str = "spot", limit: int=500):
    """
    Fetch recent trading history.
    category can be: spot, inverse, linear, option
    """

    if (category == "spot"):
        if (limit < 0 or limit > 60):
            limit = 60
    else:
        if (limit < 0 or limit > 1000):
            limit = 1000
    try:
        trades = session.get_public_trade_history(
            category=category, symbol=symbol, limit=limit)
        return trades
    except Exception as e:
        print("Error fetching recent trades!")
        print(e)


def fetch_historical_orderbook_day_data(day: datetime, symbol: str, category: str = "spot"):
    """
    Fetch historical orderbook data.
    The .zip file is downloaded to ./data/ob/{category}/{symbol}/{year}-{month}-{day}_{symbol}_ob500.zip
    and then inflated to ./data/ob/{category}/{symbol}/{year}-{month}-{day}_{symbol}_ob500.data
    """
    url = f"https://quote-saver.bycsi.com/orderbook/{category}/{symbol}/{day.year}-{zero_pad_time(day.month)}-{zero_pad_time(day.day)}_{symbol}_ob500.data.zip"
    local_filename = url.split("/")[-1]
    print(f"Downloading {local_filename}")

    # skip if file already exists
    if os.path.exists(f"data/ob/{category}/{symbol}/{local_filename}"):
        print(f"Skipping {local_filename}, as file already exists!")
        return

    # Create the directory if it doesn't exist
    if not os.path.exists(f"data/ob/{category}/{symbol}"):
        os.makedirs(f"data/ob/{category}/{symbol}")

    # Extend filename with directory path
    local_filename = f"data/ob/{category}/{symbol}/{local_filename}"

    # Download the file
    response = requests.get(url)
    if response.status_code == 404:
        print(f"Skipping {url}, as resource does not exist.")
        return
    if response.status_code != 200:
        print(f"Error downloading {url}")
        return
    with open(local_filename, "wb") as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(f"data/ob/{category}/{symbol}")


def fetch_historical_orderbook_period_data(startday: datetime, endday: datetime, symbol: str, category: str = "spot"):
    """
    Fetch historical orderbook data for a given period.
    The .zip file is downloaded to ./data/ob/{category}/{symbol}/{year}-{month}-{day}_{symbol}_ob500.zip
    and then inflated to ./data/ob/{category}/{symbol}/{year}-{month}-{day}_{symbol}_ob500.data
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(f"data/ob/{category}/{symbol}"):
        os.makedirs(f"data/ob/{category}/{symbol}")

    day = startday
    tasks = []

    while day <= endday:
        tasks.append((day, symbol, category))
        day += timedelta(days=1)

    pool = ThreadPool(MAX_THREADS)
    pool.starmap(fetch_historical_orderbook_day_data, tasks)
    pool.close()
    pool.join()


def fetch_historical_trading_day_data(day: datetime, symbol: str, category: str = "spot"):
    """
    Fetch historical trading history data, given a date.
    The .zip file is downloaded to ./data/td/{symbol}/{symbol}{year}-{month}-{day}.csv.gz
    """
    url = f"https://public.bybit.com/trading/{symbol}/{symbol}{day.year}-{zero_pad_time(day.month)}-{zero_pad_time(day.day)}.csv.gz"
    local_filename = url.split("/")[-1]
    print(f"Downloading {local_filename}")

    # skip if file already exists
    if os.path.exists(f"data/td/{symbol}/{local_filename}"):
        print(f"Skipping {local_filename}, as file already exists!")
        return

    # Create the directory if it doesn't exist
    if not os.path.exists(f"data/td/{symbol}"):
        os.makedirs(f"data/td/{symbol}")

    # Extend filename with directory path
    local_filename = f"data/td/{symbol}/{local_filename}"

    print(f"Downloading {url}")

    # Download the file
    response = requests.get(url)
    if response.status_code == 404:
        print(f"Skipping {url}, as resource does not exist.")
        return
    if response.status_code != 200:
        print(f"Error downloading {url}")
        return
    with open(local_filename, "wb") as f:
        f.write(response.content)


def fetch_historical_trading_period_data(startday: datetime, endday: datetime, symbol: str, category: str = "spot"):
    """
    Fetch historical trading history data for a given period.
    The .zip file is downloaded to ./data/td/{symbol}/{symbol}{year}-{month}-{day}.csv.gz
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(f"data/td/{symbol}"):
        os.makedirs(f"data/td/{symbol}")

    day = startday
    tasks = []

    while day <= endday:
        tasks.append((day, symbol, category))
        day += timedelta(days=1)

    pool = ThreadPool(MAX_THREADS)
    pool.starmap(fetch_historical_trading_day_data, tasks)
    pool.close()
    pool.join()


if __name__ == "__main__":
    key_file_path = sys.argv[1]
    symbol = sys.argv[2]
    session = create_session(key_file_path)
    trades = fetch_recent_trading_history(session, symbol, category="linear")
    print(trades)

    # orderbook = fetch_current_orderbook(session, symbol)
    # print(orderbook)

    # download historical data from the last week
    # day = datetime.today() - timedelta(days=7)
    # fetch_historical_trading_period_data(
    #     day, day + timedelta(days=7), symbol, category="linear")

    day = datetime.today() - timedelta(days=7)
    fetch_historical_trading_period_data(day, day + timedelta(days=7), symbol, category="linear")
    fetch_historical_orderbook_period_data(day, day + timedelta(days=7), symbol, category="linear")