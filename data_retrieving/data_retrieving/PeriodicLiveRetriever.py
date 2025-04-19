
from data_retrieving.LiveDataRetriever import LiveDataRetriever
from datetime import timedelta, datetime
from multiprocessing import Queue, Process
from signal import signal, SIGINT
import time


class PeriodicLiveRetriever():
    """
    Uses LiveDataRetriever to fetch live data periodically, amassing a queue of data
    """

    def __init__(self, key_file_path: str, update_time_interval: timedelta,
                 symbol: str, category: str):
        self.ld = LiveDataRetriever(key_file_path)
        self.update_time_interval = update_time_interval
        self.ob_data_queue = Queue()
        self.trades_data_queue = Queue()
        self.symbol = symbol
        self.category = category
        self.ob_process = Process(target=self.get_ob_process)
        self.trades_process = Process(target=self.get_trades_process)
        self.ob_process.start()
        self.trades_process.start()

    def set_time_interval(self, update_time_interval: timedelta):
        self.update_time_interval = update_time_interval

    def set_currency(self, symbol: str, category: str):
        self.symbol = symbol
        self.category = category

    def get_ob_process(self):
        def interpret_sigint(signum, frame):
            print("Received SIGINT at PeriodicLiveRetriever subprocess, cleaning up...")
            del self.ld
            exit(0)

        signal(SIGINT, interpret_sigint)
        while True:
            now = datetime.now()

            ob_data = self.ld.fetch_current_orderbook(
                self.symbol, self.category, limit=50)
            self.ob_data_queue.put(ob_data)
            time_passed_requesting = datetime.now() - now
            time_left_to_sleep_seconds = float(
                self.update_time_interval.microseconds - time_passed_requesting.microseconds) / 1000000
            if (time_left_to_sleep_seconds < 0):
                print(
                    "WARNING: Requesting ob data took more time than update time interval!")
                continue
            time.sleep(time_left_to_sleep_seconds)

    def get_trades_process(self):
        def interpret_sigint(signum, frame):
            print("Received SIGINT at PeriodicLiveRetriever subprocess, cleaning up...")
            del self.ld
            exit(0)

        signal(SIGINT, interpret_sigint)
        while True:
            now = datetime.now()

            trade_data = self.ld.fetch_recent_trading_history(
                self.symbol, self.category)
            self.trades_data_queue.put(trade_data)
            time_passed_requesting = datetime.now() - now
            time_left_to_sleep_seconds = float(
                self.update_time_interval.microseconds - time_passed_requesting.microseconds) / 1000000
            if (time_left_to_sleep_seconds < 0):
                print(
                    "WARNING: Requesting ob data took more time than update time interval!")
                continue
            time.sleep(time_left_to_sleep_seconds)


if __name__ == "__main__":
    ld = PeriodicLiveRetriever(
        "apiKey.json", timedelta(milliseconds=650), "CAKEUSDT", "linear")
    while True:
        pass
        ld.ob_data_queue.get(block=True)
        ld.trades_data_queue.get(block=True)
    del ld
