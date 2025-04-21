from data_retrieving.LiveDataRetriever import LiveDataRetriever
from datetime import timedelta, datetime
from multiprocessing import Queue, Process, Value, Manager
from signal import signal, SIGINT
import time
from ctypes import c_char_p
import multiprocessing as mp


def get_ob_process(key_file_path, symbol, category, ob_data_queue, rq_time_ms, limit=50):
    ld = LiveDataRetriever(key_file_path)
    ob_data = ld.fetch_current_orderbook(symbol, category, limit=limit)
    if ob_data is not None:
        time_passed_requesting = int(datetime.now().timestamp() * 1000) - rq_time_ms
        ob_data_queue.put([rq_time_ms, time_passed_requesting, ob_data])


def get_trades_process(key_file_path, symbol, category, trades_data_queue, rq_time_ms):
    ld = LiveDataRetriever(key_file_path)
    trade_data = ld.fetch_recent_trading_history(symbol, category)
    if trade_data is not None:
        time_passed_requesting = int(datetime.now().timestamp() * 1000) - rq_time_ms
        trades_data_queue.put([rq_time_ms, time_passed_requesting, trade_data])


class PeriodicLiveRetriever():
    """
    Uses LiveDataRetriever to fetch live data periodically, amassing a queue of data
    """

    def __init__(self, key_file_path: str, update_time_interval: timedelta,
                 symbol: str, category: str, start=False):
        self.key_file_path = key_file_path
        self.ld = LiveDataRetriever(key_file_path)
        self.update_time_interval_ms = Value('i', int(update_time_interval.total_seconds() * 1000))
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
                if self.run_spawner.value:
                    ob_process = Process(
                        target=get_ob_process, args=(
                            self.key_file_path, self.symbol.value,
                            self.category.value, self.ob_data_queue, next_time_ms)
                    )
                    trade_process = Process(
                        target=get_trades_process, args=(
                            self.key_file_path, self.symbol.value,
                            self.category.value, self.trades_data_queue, next_time_ms)
                    )

                    ob_process.start()
                    trade_process.start()
                next_time_ms += self.update_time_interval_ms.value

    def run_digester_process(self, start_time_ms):
        def interpret_sigint(signum, frame):
            print("Received SIGINT at PeriodicLiveRetriever subprocess, cleaning up...")
            exit(0)

        signal(SIGINT, interpret_sigint)

        print("Starting digest process at: ", start_time_ms)

        current_time_ms = start_time_ms

        while True:
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
            current_delay = int(datetime.now().timestamp() * 1000) - current_time_ms

            data = {
                "ts": current_time_ms,
                "delay_after_request": {"trades": trade_data[1], "ob": ob_data[1]},
                "delay_after_digesting": current_delay,
                "ob": ob_data[2],
                "trades": trade_data[2]
            }

            self.data_queue.put(data)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)  # 👈 Important for macOS!

    ld = PeriodicLiveRetriever(
        "api_key.json", timedelta(milliseconds=650), "BTCUSDT", "spot"
    )

    t = time.time()
    ld.start()
    while time.time() - t < 100:
        d = ld.data_queue.get(block=True)
        print(d["ts"], d["delay_after_request"], d["delay_after_digesting"],
              d["ob"]["ts"], d["trades"]["list"][0]["time"])

    del ld
