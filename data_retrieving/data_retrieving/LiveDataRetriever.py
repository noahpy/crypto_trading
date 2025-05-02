
from pybit.unified_trading import HTTP as BybitSession
import json
import time
from multiprocessing import Process, Queue, Value
import signal

RATELIMIT_PER_FIVE_SECONDS = 600
request_reset_process_started = Value('b', False)
# intialize a queue with max size of ratelimit
request_queue = Queue(RATELIMIT_PER_FIVE_SECONDS - 1)

WAIT_TIME_LIMIT = 0.1

REPORT_REFRESH_RATE = 3

class LiveDataRetriever:
    """Class for fetching live data.
       Has respective methods to fetch current orderbook and recent trading history data.
       Implements rate limiting.

    @params:
        key_file_path: path to JSON key file

    """

    def __init__(self, key_file_path: str):
        self.session = self.create_session(key_file_path)
        self.loop_refresh = True
        self.refresh_process = None
        if not request_reset_process_started.value:
            self.refresh_process = Process(
                target=self.ratelimit_queue_clearance)
            self.refresh_process.start()
            request_reset_process_started.value = True

    def __del__(self):
        self.loop_refresh = False
        if self.refresh_process:
            self.refresh_process.terminate()
        time.sleep(0.1)

    def ratelimit_queue_clearance(self):
        def handle_sigint(signum, frame):
            print("Received SIGINT at queue process, cleaning up...")
            exit(0)

        signal.signal(signal.SIGINT, handle_sigint)
        report_time = time.time()
        while self.loop_refresh:
            if time.time() - report_time >= REPORT_REFRESH_RATE:
                print(f"Made {request_queue.qsize()} request in past {REPORT_REFRESH_RATE} seconds.")
                report_time = time.time()
            time.sleep(0.01)
            # requests before 5 seconds can be dismissed
            dismiss_threshhold = time.time() - 5
            req = request_queue.get()
            while req <= dismiss_threshhold:
                dismiss_threshhold = time.time() - 5
                req = request_queue.get()
            time.sleep(req - dismiss_threshhold)

    def create_session(self, key_file_path: str) -> BybitSession:
        """
        Create a BybitSession object from a JSON key file.
        """
        with open(key_file_path) as f:
            keys = json.load(f)
        session = BybitSession(
            api_key=keys['key'], api_secret=keys['secret'])
        return session

    def fetch_current_orderbook(self, symbol: str,
                                category: str = "spot", limit: int = 200) -> dict:
        """
        Fetch current orderbook data.
        category can be: spot, inverse, linear
        """
        if (limit < 0 or limit > 200):
            limit = 200
        try:
            start_time = time.time()
            request_queue.put(time.time())
            wait_time = time.time() - start_time
            if wait_time >= WAIT_TIME_LIMIT:
                print("Wait time too long: " + str(wait_time))

            orderbook = self.session.get_orderbook(
                category=category, symbol=symbol, limit=limit)
            return orderbook["result"]
        except Exception as e:
            print("Error fetching orderbook!")
            print(e)

    def fetch_recent_trading_history(self, symbol: str, category: str = "spot", limit: int = 500):
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
            start_time = time.time()
            request_queue.put(time.time())
            wait_time = time.time() - start_time
            if wait_time >= WAIT_TIME_LIMIT:
                print("Wait time too long: " + str(wait_time))
            trades = self.session.get_public_trade_history(
                category=category, symbol=symbol, limit=limit)
            return trades["result"]
        except Exception as e:
            print("Error fetching recent trades!")
            print(e)


if __name__ == "__main__":

    ld = LiveDataRetriever("api_key.json")

    def interpret_sigint(signum, frame):
        print("Received SIGINT at subprocess, cleaning up...")
        exit(0)

    def loop():
        signal.signal(signal.SIGINT, interpret_sigint)
        ld = LiveDataRetriever("api_key.json")
        while (True):
            ld.fetch_current_orderbook("BTCUSDT")

    def loop2():
        signal.signal(signal.SIGINT, interpret_sigint)
        ld = LiveDataRetriever("api_key.json")
        while (True):
            ld.fetch_recent_trading_history("BTCUSDT")

    # increasing the number of processes will increase the number of requests linearly
    p1 = Process(target=loop2)
    p2 = Process(target=loop2)
    p3 = Process(target=loop2)
    p1.start()
    p2.start()
    p3.start()

    def interpret_sigint2(signum, frame):
        print("Received SIGINT at main process, cleaning up...")
        frame.f_locals['ld'].__del__()
        exit(0)

    signal.signal(signal.SIGINT, interpret_sigint2)
    time.sleep(30)

    p1.terminate()
    p2.terminate()
    p3.terminate()

    del ld
