
import requests
from datetime import datetime, timedelta
import zipfile
import os
from multiprocessing.pool import ThreadPool
from io import StringIO
import pandas as pd
import io
import gzip

class HistoricalDataRetriever:

    def __init__(self, donwload_path: str = ".", max_threads: int = 5):
        self.download_path = donwload_path
        if (self.download_path == ""):
            self.download_path = "."
        if (self.download_path.endswith("/")):
            self.download_path = self.download_path[:-1]
        self.max_threads = max_threads

    def zero_pad_time(self, x: int):
        if x < 10:
            return f"0{x}"
        else:
            return x

    @staticmethod
    def check_category_symbol_exists(data_type: str, symbol: str, category: str) -> bool:
        if (data_type == "orderbook"):
            response = requests.get(
                "https://quote-saver.bycsi.com/orderbook/" + category + "/" + symbol)
        else:
            response = requests.get(
                "https://public.bybit.com/trading/" + symbol)
        if (response.status_code == 200):
            return True
        return False

    def fetch_historical_orderbook_day_data(self, day: datetime, symbol: str, category: str = "spot"):
        """
        Fetch historical orderbook data.
        The .zip file is downloaded to ./data/ob/{category}/{symbol}/{year}-{month}-{day}_{symbol}_ob500.zip
        and then inflated to ./data/ob/{category}/{symbol}/{year}-{month}-{day}_{symbol}_ob500.data
        """
        url = f"https://quote-saver.bycsi.com/orderbook/{category}/{symbol}/{day.year}-{self.zero_pad_time(day.month)}-{self.zero_pad_time(day.day)}_{symbol}_ob500.data.zip"
        local_filename = url.split("/")[-1]
        print(f"Downloading {local_filename}")

        total_download_path = f"{self.download_path}/data/ob/{category}/{symbol}/"

        # skip if file already exists
        if os.path.exists(f"{total_download_path}{local_filename}"):
            print(f"Skipping {local_filename}, as file already exists!")
            return

        # Create the directory if it doesn't exist
        if not os.path.exists(total_download_path):
            os.makedirs(f"data/ob/{category}/{symbol}")

        # Extend filename with directory path
        local_filename = total_download_path + local_filename

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

    def get_historical_orderbook_day_data(self, day: datetime, symbol: str, category: str = "spot"):
        """
        Fetch historical orderbook data and return it in serialized form.
        Returns:
            bytes: The deserialized orderbook data, or None if there was an error downloading
        """
        url = f"https://quote-saver.bycsi.com/orderbook/{category}/{symbol}/{day.year}-{self.zero_pad_time(day.month)}-{self.zero_pad_time(day.day)}_{symbol}_ob500.data.zip"
        
        
        # Download the file
        try:
            
            response = requests.get(url, timeout=30)  # Add timeout to prevent hanging
            
            if response.status_code == 404:
                print(f"Skipping {url}, as resource does not exist.")
                return None
            
            if response.status_code != 200:
                print(f"Error downloading {url}, status code: {response.status_code}")
                print(f"Response content: {response.text[:200]}...")  # Print start of response content
                return None
            
            
            # Extract data from zip in memory
            try:
                zip_bytes = io.BytesIO(response.content)
                
                with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
                    # Print the contents of the zip file
                    file_list = zip_ref.namelist()
                
                    
                    # Extract the data file
                    if not file_list:
                        print("ZIP file is empty!")
                        return None
                        
                    data_file_name = file_list[0]  # Using first file
                
                    orderbook_data = zip_ref.read(data_file_name)

                    
                return orderbook_data.decode('utf-8').splitlines()
            
            except Exception as e:
                print(f"Error extracting zip content: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None



    def fetch_historical_orderbook_period_data(self, startday: datetime, endday: datetime, symbol: str, category: str = "linear"):
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

        pool = ThreadPool(self.max_threads)
        pool.starmap(self.fetch_historical_orderbook_day_data, tasks)
        pool.close()
        pool.join()

    def fetch_historical_trading_day_data(self, day: datetime, symbol: str, category: str = "spot"):
        """
        Fetch historical trading history data, given a date.
        The .zip file is downloaded to ./data/td/{symbol}/{symbol}{year}-{month}-{day}.csv.gz
        """
        url = f"https://public.bybit.com/trading/{symbol}/{symbol}{day.year}-{self.zero_pad_time(day.month)}-{self.zero_pad_time(day.day)}.csv.gz"
        local_filename = url.split("/")[-1]
        print(f"Downloading {local_filename}")

        total_download_path = f"{self.download_path}/data/td/{symbol}/"

        # skip if file already exists
        if os.path.exists(f"{total_download_path}{local_filename}"):
            print(f"Skipping {local_filename}, as file already exists!")
            return

        # Create the directory if it doesn't exist
        if not os.path.exists(total_download_path):
            os.makedirs(f"data/td/{symbol}")

        # Extend filename with directory path
        local_filename = total_download_path + local_filename

        # Download the file
        response = requests.get(url)
        if response.status_code == 404:
            print(f"Skipping {url}, as resource does not exist.")
            return
        if response.status_code != 200:
            print(f"Error downloading {url}: {response}")
            return
        with open(local_filename, "wb") as f:
            f.write(response.content)

    def get_historical_trading_day_data(self, day: datetime, symbol: str, category: str = "spot"):
        """
        Get historical trading history data, given a date. Returns pandas data frame
        """
        url = f"https://public.bybit.com/trading/{symbol}/{symbol}{day.year}-{self.zero_pad_time(day.month)}-{self.zero_pad_time(day.day)}.csv.gz"

        # Download the file
        response = requests.get(url)
        
        decompressed_content = gzip.decompress(response.content)
    
        # Create a file-like object and read it with pandas
        df = pd.read_csv(io.BytesIO(decompressed_content))

        return df

    


    def fetch_historical_trading_period_data(self, startday: datetime, endday: datetime, symbol: str, category: str = "linear"):
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

        pool = ThreadPool(self.max_threads)
        pool.starmap(self.fetch_historical_trading_day_data, tasks)
        pool.close()
        pool.join()





if __name__ == "__main__":
    retriever = HistoricalDataRetriever()
    retriever.fetch_historical_orderbook_period_data(
        datetime(2025, 4, 10), datetime(2025, 4, 16), "CAKEUSDT", "linear")
    retriever.fetch_historical_trading_period_data(
        datetime(2025, 4, 10), datetime(2025, 4, 16), "CAKEUSDT", "linear")
