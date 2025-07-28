
from pybit.unified_trading import HTTP as BybitSession
import json
import time
from threading import Thread
from multiprocessing import Process, Lock, Value
import signal

request_mutex = Lock()
request_count = Value('i', 0)

WAIT_TIME_LIMIT = 0.1

class LiveDataRetriever:
    """Singleton class for fetching live data.
       There can only exist one instace of this class per process
       (See https://code.activestate.com/recipes/52558/).
       Across multiple processes, the mutex and request count will be shared,
       ensuring that the rate limit is not exceeded.
       After usage, this class should be deleted using del ld, to ensure a
       graceful exit.

    @params:
        key_file_path: path to JSON key file

    """

    class LiveDataRetrieverImpl:

        def __init__(self, key_file_path: str):
            self.session = self.create_session(key_file_path)
            self.RATELIMIT_PER_FIVE_SECONDS = 600
            self.loop_refresh = True
            self.refresh_thread = Thread(
                target=self.periodic_reset_request_count)
            self.refresh_thread.start()

        def __del__(self):
            self.loop_refresh = False
            time.sleep(0.1)

        def periodic_reset_request_count(self):
            last_time = time.time()
            while self.loop_refresh:
                time.sleep(0.02)
                if (time.time() - last_time >= 4.9):
                    last_time = time.time()
                    with request_mutex:
                        print("Reset rate limit counter from: ",
                              request_count.value)
                        request_count.value = 0

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
                loop = True
                start_time = time.time()
                while loop:
                    with request_mutex:
                        if (request_count.value < self.RATELIMIT_PER_FIVE_SECONDS):
                            request_count.value += 1
                            loop = False
                            break
                    time.sleep(0.01)
                wait_time = time.time() - start_time
                if wait_time >= WAIT_TIME_LIMIT:
                    raise Exception("Wait time too long: " + str(wait_time))

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
                loop = True
                start_time = time.time()
                while loop:
                    with request_mutex:
                        if (request_count.value < self.RATELIMIT_PER_FIVE_SECONDS):
                            request_count.value += 1
                            loop = False
                            break
                    time.sleep(0.01)
                wait_time = time.time() - start_time
                if wait_time >= WAIT_TIME_LIMIT:
                    raise Exception("Wait time too long: " + str(wait_time))
                trades = self.session.get_public_trade_history(
                    category=category, symbol=symbol, limit=limit)
                return trades["result"]
            except Exception as e:
                print("Error fetching recent trades!")
                print(e)

    __instance = None

    def __init__(self, key_file_path: str):
        """ Create singleton instance """
        # Check whether we already have an instance
        if LiveDataRetriever.__instance is None:
            # Create and remember instance
            LiveDataRetriever.__instance = LiveDataRetriever.LiveDataRetrieverImpl(
                key_file_path)

        # Store instance reference as the only member in the handle
        self.__dict__[
            '_LiveDataRetriever__instance'] = LiveDataRetriever.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)

    def __del__(self):
        self.__instance.__del__()


if __name__ == "__main__":

    ld = LiveDataRetriever("api_key.json")

    def interpret_sigint(signum, frame):
        print("Received SIGINT at subprocess, cleaning up...")
        exit(0)

    def loop():
        signal.signal(signal.SIGINT, interpret_sigint)
        ld = LiveDataRetriever("apiKey.json")
        while (True):
            ld.fetch_current_orderbook("BTCUSDT")

    def loop2():
        signal.signal(signal.SIGINT, interpret_sigint)
        ld = LiveDataRetriever("apiKey.json")
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
