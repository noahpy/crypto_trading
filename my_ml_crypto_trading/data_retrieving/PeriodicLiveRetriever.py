from my_ml_crypto_trading.data_retrieving.LiveDataRetriever import LiveDataRetriever
from datetime import timedelta, datetime
from multiprocessing import Queue, Process, Value, Manager
from signal import signal, SIGINT
import time
from ctypes import c_char_p
import multiprocessing as mp
import os


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


def run_spawner_process(key_file_path, update_time_interval_ms, symbol, category, 
                        ob_data_queue, trades_data_queue, run_spawner, start_time_ms):
    """Separate function to avoid class method pickling issues"""
    ACTIVATION_THESHHOLD_MS = 100

    def interpret_sigint(signum, frame):
        print("Received SIGINT at spawner process, cleaning up...")
        exit(0)

    signal(SIGINT, interpret_sigint)
    
    print("Starting spawner process at: ", start_time_ms)
    next_time_ms = start_time_ms + update_time_interval_ms.value

    while True:
        current_time_ms = int(datetime.now().timestamp() * 1000)
        if next_time_ms - current_time_ms < ACTIVATION_THESHHOLD_MS:
            if run_spawner.value:
                # Direct function calls instead of creating new processes
                try:
                    # Get orderbook data
                    rq_time_ms = next_time_ms
                    ld = LiveDataRetriever(key_file_path)
                    ob_data = ld.fetch_current_orderbook(symbol.value, category.value, limit=50)
                    if ob_data is not None:
                        time_passed_requesting = int(datetime.now().timestamp() * 1000) - rq_time_ms
                        ob_data_queue.put([rq_time_ms, time_passed_requesting, ob_data])
                    
                    # Get trade data
                    trade_data = ld.fetch_recent_trading_history(symbol.value, category.value)
                    if trade_data is not None:
                        time_passed_requesting = int(datetime.now().timestamp() * 1000) - rq_time_ms
                        trades_data_queue.put([rq_time_ms, time_passed_requesting, trade_data])
                except Exception as e:
                    print(f"Error in spawner data fetch: {e}")
                    
            next_time_ms += update_time_interval_ms.value
        
        # Sleep a bit to avoid high CPU usage
        time.sleep(0.01)


def run_digester_process(ob_data_queue, trades_data_queue, data_queue, start_time_ms):
    """Separate function to avoid class method pickling issues"""
    def interpret_sigint(signum, frame):
        print("Received SIGINT at digester process, cleaning up...")
        exit(0)

    signal(SIGINT, interpret_sigint)

    print("Starting digest process at: ", start_time_ms)

    current_time_ms = start_time_ms

    while True:
        try:
            ob_data = ob_data_queue.get(block=True, timeout=1.0)
            trade_data = trades_data_queue.get(block=True, timeout=1.0)

            while ob_data[0] < current_time_ms:
                ob_data = ob_data_queue.get(block=True, timeout=1.0)

            while trade_data[0] < current_time_ms:
                trade_data = trades_data_queue.get(block=True, timeout=1.0)

            while trade_data[0] != ob_data[0]:
                if trade_data[0] < ob_data[0]:
                    trade_data = trades_data_queue.get(block=True, timeout=1.0)
                else:
                    ob_data = ob_data_queue.get(block=True, timeout=1.0)

            current_time_ms = trade_data[0]
            current_delay = int(datetime.now().timestamp() * 1000) - current_time_ms

            data = {
                "ts": current_time_ms,
                "delay_after_request": {"trades": trade_data[1], "ob": ob_data[1]},
                "delay_after_digesting": current_delay,
                "ob": ob_data[2],
                "trades": trade_data[2]
            }

            data_queue.put(data)
        except Exception as e:
            # Add timeout to avoid blocking indefinitely
            # print(f"Error in digester process: {e}")
            time.sleep(0.1)
            continue


class PeriodicLiveRetriever():
    """
    Uses LiveDataRetriever to fetch live data periodically, amassing a queue of data
    """

    def __init__(self, key_file_path: str, update_time_interval: timedelta,
                 symbol: str, category: str, start=False):
        # Make sure to set multiprocessing start method early
        self._setup_multiprocessing()
        
        self.key_file_path = key_file_path
        self.update_time_interval_ms = Value('i', int(update_time_interval.total_seconds() * 1000))
        # Use mp.Manager().Queue() instead of Queue() for better macOS compatibility
        manager = Manager()
        self.ob_data_queue = manager.Queue()
        self.trades_data_queue = manager.Queue()
        self.data_queue = manager.Queue()
        self.symbol = manager.Value(c_char_p, symbol)
        self.category = manager.Value(c_char_p, category)
        self.start_time_ms = int(datetime.now().timestamp() * 1000)
        self.run_spawner = Value('b', False)
        # Don't create processes in __init__, only create them when start() is called
        self.spawner_process = None
        self.digester_process = None
        if start:
            self.start()
    
    def _setup_multiprocessing(self):
        """Setup multiprocessing with the right context"""
        if 'fork' in mp.get_all_start_methods():
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError:
                pass  # Already set

    def set_time_interval(self, update_time_interval_ms: int):
        self.update_time_interval_ms.value = update_time_interval_ms

    def set_currency(self, symbol: str, category: str):
        self.symbol.value = symbol
        self.category.value = category

    def start(self):
        self.run_spawner.value = True
        
        # Create new processes if they don't exist or are not alive
        if self.spawner_process is None or not self.spawner_process.is_alive():
            self.spawner_process = Process(
                target=run_spawner_process,  # Use global function instead of class method
                args=(
                    self.key_file_path, 
                    self.update_time_interval_ms,
                    self.symbol,
                    self.category,
                    self.ob_data_queue,
                    self.trades_data_queue,
                    self.run_spawner,
                    self.start_time_ms
                ),
                daemon=False  # Not a daemon so it can spawn processes
            )
            self.spawner_process.start()
            
        if self.digester_process is None or not self.digester_process.is_alive():
            self.digester_process = Process(
                target=run_digester_process,  # Use global function instead of class method
                args=(
                    self.ob_data_queue,
                    self.trades_data_queue,
                    self.data_queue,
                    self.start_time_ms
                ),
                daemon=False  # Not a daemon
            )
            self.digester_process.start()

    def pause(self):
        self.run_spawner.value = False

    def stop(self):
        self.run_spawner.value = False
        
        if self.spawner_process is not None and self.spawner_process.is_alive():
            self.spawner_process.terminate()
            self.spawner_process.join(timeout=1.0)  # Wait for process to terminate
            
        if self.digester_process is not None and self.digester_process.is_alive():
            self.digester_process.terminate()
            self.digester_process.join(timeout=1.0)  # Wait for process to terminate

    def __del__(self):
        print("Cleaning up PeriodicLiveRetriever...")
        try:
            self.stop()
        except:
            pass  # Ignore errors during cleanup


if __name__ == "__main__":
    ld = PeriodicLiveRetriever(
        "api_key.json", timedelta(milliseconds=650), "BTCUSDT", "spot"
    )

    t = time.time()
    ld.start()
    try:
        while time.time() - t < 100:
            try:
                d = ld.data_queue.get(block=True, timeout=1.0)
                print(d["ts"], d["delay_after_request"], d["delay_after_digesting"],
                    d["ob"]["ts"], d["trades"]["list"][0]["time"])
            except Exception as e:
                # Add timeout to the queue get to avoid blocking indefinitely
                continue
    finally:
        # Ensure cleanup happens
        ld.stop()