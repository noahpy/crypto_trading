
from data_retrieving.LiveDataRetriever import LiveDataRetriever
from datetime import timedelta, datetime
from multiprocessing import Queue, Process, Value
from signal import signal, SIGINT
import time


class PeriodicLiveRetriever():
    """
    Uses LiveDataRetriever to fetch live data periodically, amassing a queue of data
    """

    def __init__(self, key_file_path: str, update_time_interval: timedelta,
                 symbol: str, category: str, start=False):
        self.ld = LiveDataRetriever(key_file_path)
        self.update_time_interval = update_time_interval
        self.ob_data_queue = Queue()
        self.trades_data_queue = Queue()
        self.symbol = symbol
        self.category = category
        self.ob_process = Process(target=self.get_ob_process)
        self.trades_process = Process(target=self.get_trades_process)
        self.run_ob_process = Value('i', 0)
        self.run_trades_process = Value('i', 0)
        if start:
            self.run_ob_process.value = 1
            self.run_trades_process.value = 1
            self.ob_process.start()
            self.trades_process.start()

    def set_time_interval(self, update_time_interval: timedelta):
        self.update_time_interval = update_time_interval

    def set_currency(self, symbol: str, category: str):
        self.symbol = symbol
        self.category = category

    def start(self):
        self.run_ob_process.value = 1
        self.run_trades_process.value = 1
        if not self.ob_process.is_alive():
            self.ob_process.start()

        if not self.trades_process.is_alive():
            self.trades_process.start()

    def pause(self):
        self.run_ob_process.value = 0
        self.run_trades_process.value = 0

    def stop(self):
        self.ob_process.terminate()
        self.trades_process.terminate()

    def __del__(self):
        print("Cleaning up PeriodicLiveRetriever...")
        self.ob_process.terminate()
        self.trades_process.terminate()
        try:
            del self.ld
        except:
            pass

    def get_ob_process(self):
        def interpret_sigint(signum, frame):
            print("Received SIGINT at PeriodicLiveRetriever subprocess, cleaning up...")
            # frame.f_locals['self.ld'].__del__()
            del self.ld
            exit(0)

        signal(SIGINT, interpret_sigint)
        while True:
            if self.run_ob_process.value:
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
            if self.run_trades_process.value:
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

    # def interpret_sigint(signum, frame):
    #     print("Received SIGINT at main process, cleaning up...")
    #     print(frame.f_locals.keys())
    #     frame.f_locals['self'].ld.__del__()
    #     exit(0)

    # signal(SIGINT, interpret_sigint)
    t = time.time()
    ld.start()
    while time.time() - t < 10:
        pass
        a = ld.ob_data_queue.get(block=True)
        # print(a.keys())
        b = ld.trades_data_queue.get(block=True)
        print(b)

        ld.pause()
        ld.start()
    del ld
