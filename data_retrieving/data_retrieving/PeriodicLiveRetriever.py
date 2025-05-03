
from data_retrieving.LiveDataRetriever import LiveDataRetriever
from datetime import timedelta, datetime
from multiprocessing import Queue, Process, Value, Manager
from signal import signal, SIGINT
import time
from ctypes import c_char_p


def get_ob_process(ld, symbol, category, ob_data_queue, rq_time_ms, limit=50):
    # sleep until request to be done
    time.sleep((rq_time_ms - int(datetime.now().timestamp() * 1000)) / 1000)

    ob_data = ld.fetch_current_orderbook(
        symbol, category, limit=50)
    if ob_data is not None:
        time_passed_requesting = int(
            datetime.now().timestamp() * 1000) - rq_time_ms
        ob_data_queue.put([rq_time_ms, time_passed_requesting, ob_data])


def get_trades_process(ld, symbol, category, trades_data_queue, rq_time_ms):
    # sleep until request to be done
    time.sleep((rq_time_ms - int(datetime.now().timestamp() * 1000)) / 1000)

    trade_data = ld.fetch_recent_trading_history(
        symbol, category, )
    if trade_data is not None:
        time_passed_requesting = int(
            datetime.now().timestamp() * 1000) - rq_time_ms
        trades_data_queue.put([rq_time_ms, time_passed_requesting, trade_data])


class PeriodicLiveRetriever():
    """
    Uses LiveDataRetriever to fetch live data periodically, amassing a queue of data
    """

    def __init__(self, key_file_path: str, update_time_interval: int,
                 symbol: str, category: str, start=False):
        self.ld = LiveDataRetriever(key_file_path)
        self.update_time_interval_ms = Value('i', update_time_interval)
        self.ob_data_queue = Queue()
        self.trades_data_queue = Queue()
        self.data_queue = Queue()
        self.manager = Manager()
        self.symbol = self.manager.Value(c_char_p, symbol)
        self.category = self.manager.Value(c_char_p, category)
        self.start_time_ms = int(datetime.now().timestamp() * 1000)
        self.run_spawner = Value('b', False)
        self.spawner_process = Process(
            target=self.run_spawner_process, args=(self.start_time_ms,))
        self.digester_process = Process(
            target=self.run_digester_process, args=(self.start_time_ms,))
        if start:
            self.spawner_process.start()
            self.digester_process.start()

    def set_time_interval(self, update_time_interval_ms: int):
        self.update_time_interval_ms.value = update_time_interval_ms

    def set_currency(self, symbol: str, category: str):
        self.symbol.value = symbol
        self.category.value = category

    def start(self):
        self.run_spawner.value = True
        if not self.spawner_process.is_alive():
            self.spawner_process.start()
        if not self.digester_process.is_alive():
            self.digester_process.start()

    def pause(self):
        self.run_spawner.value = False

    def stop(self):
        self.spawner_process.terminate()
        self.digester_process.terminate()

    def __del__(self):
        print("Cleaning up PeriodicLiveRetriever...")
        self.stop()
        try:
            del self.ld
        except Exception:
            pass

    def run_spawner_process(self, start_time_ms):
        """
        Spawns two data retrieval processes every time interval, giving them the timestamp in ms
        when the request should happen.
        """

        # How early do we want to activate the processes
        # NOTE: Feedback loop might be possible from the digester
        ACTIVATION_THESHHOLD_MS = 100

        def interpret_sigint(signum, frame):
            print("Received SIGINT at PeriodicLiveRetriever subprocess, cleaning up...")
            exit(0)

        signal(SIGINT, interpret_sigint)

        print("Starting spawner process at: ", start_time_ms)
        next_time_ms = start_time_ms + self.update_time_interval_ms.value

        while True:
            current_time_ms = int(datetime.now().timestamp() * 1000)
            if next_time_ms - current_time_ms < ACTIVATION_THESHHOLD_MS:
                # only start if required
                if self.run_spawner.value:
                    ob_process = Process(
                        target=get_ob_process, args=(self.ld, self.symbol.value,
                                                     self.category.value, self.ob_data_queue, next_time_ms))
                    trade_process = Process(
                        target=get_trades_process, args=(self.ld, self.symbol.value,
                                                         self.category.value, self.trades_data_queue, next_time_ms))

                    ob_process.start()
                    trade_process.start()
                    pass
                next_time_ms += self.update_time_interval_ms.value

    def run_digester_process(self, start_time_ms):
        """
        Digests the data put into the queue by the retrieval processes and outputs a chornological
        and robust data sequence.
        """

        def interpret_sigint(signum, frame):
            print("Received SIGINT at PeriodicLiveRetriever subprocess, cleaning up...")
            exit(0)

        signal(SIGINT, interpret_sigint)

        print("Starting digest process at: ", start_time_ms)

        current_time_ms = start_time_ms

        while True:
            # Rule 1: the data fetched must be newer than current_time_ms
            # Rule 2: if the timesteps of the popped data are not equivalent, discard the older one
            #         and pop a new one
            # This sequence ensures a chronological timeline of data for any combination of these events:
            #   - data could not be retrieved by subprocess and nothing is put into Queue
            #   - some workers finishing faster than chronologically assigned
            ob_data = self.ob_data_queue.get(block=True)
            trade_data = self.trades_data_queue.get(block=True)

            while ob_data[0] < current_time_ms:
                ob_data = self.ob_data_queue.get(block=True)

            while trade_data[0] < current_time_ms:
                trade_data = self.trades_data_queue.get(block=True)

            while trade_data[0] != ob_data[0]:
                if trade_data[0] < ob_data[0]:
                    trade_data = self.trades_data_queue.get(block=True)
                else:
                    ob_data = self.ob_data_queue.get(block=True)

            current_time_ms = trade_data[0]
            current_delay = int(datetime.now().timestamp()
                                * 1000) - current_time_ms

            data = {
                "ts": current_time_ms,
                "delay_after_request": {"trades": trade_data[1], "ob": ob_data[1]},
                "delay_after_digesting": current_delay,
                "ob": ob_data[2],
                "trades": trade_data[2]
            }

            self.data_queue.put(data)


if __name__ == "__main__":
    interval = 100
    ld = PeriodicLiveRetriever(
        "api_key.json", interval, "BTCUSDT", "spot")

    def handle_sigint(signum, frame):
        print("Received SIGINT at PeriodicLiveRetriever, cleaning up...")
        exit(0)

    signal(SIGINT, handle_sigint)

    t = time.time()
    ld.start()
    skip_count = 0
    total_req_count = 0
    last_request_time = 0
    total_throughout_count = 0
    total_trade_before_ob_count = 0
    total_unique_skip_count = 0
    total_skip_amount = 0
    
    while time.time() - t < 200:
        d = ld.data_queue.get(block=True)
        req_time = d["ts"]
        delay_req_time = d["delay_after_request"]
        delay_digest_time = d["delay_after_digesting"]
        ob_time = d["ob"]["ts"]
        trades_time = d["trades"]["list"][0]["time"]

        diff = int(req_time) - int(last_request_time)
        if diff < 0:
            print("Negative time difference, something is wrong!")
            exit(0)
        if (diff > interval) and last_request_time != 0:
            skip_count += 1
            total_unique_skip_count += 1
            total_skip_amount += diff
            total_req_count += (diff - interval) / interval
        total_req_count += 1
        total_throughout_count += 1
        last_request_time = req_time

        if (int(trades_time) > int(ob_time)):
            total_trade_before_ob_count += 1

        trade_before_ob_rate = total_trade_before_ob_count / total_throughout_count
        skip_rate = skip_count / total_req_count
        if total_unique_skip_count == 0:
            mean_skip_amount = 0
        else:
            mean_skip_amount = total_skip_amount / total_unique_skip_count
        print(f"Request time: {req_time}, delay after request: {delay_req_time}, delay after digesting: {delay_digest_time}, ob time: {ob_time}, trades time: {trades_time}, skip rate: {skip_rate:.2f}, mean skip: {mean_skip_amount:.2f}, trade before ob rate: {trade_before_ob_rate:.2f}")


    del ld
