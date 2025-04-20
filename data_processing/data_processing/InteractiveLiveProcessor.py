from data_retrieving.PeriodicLiveRetriever import PeriodicLiveRetriever
from data_processing.FeatureCreation import Feature
import data_processing.ob_features
import data_processing.trade_features
from data_processing.FeatureCreation import FeatureCreator
import tkinter as tk
from tkinter import ttk
import inspect
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import timedelta
from typing import List
from datetime import datetime
from multiprocessing import Value
from threading import Thread
import traceback


def convert_bybit_ob_to_snapshot(order_book):
    ts = datetime.fromtimestamp(order_book['ts'] / 1000)
    bids = {float(price): float(size) for price, size in order_book['b']}
    asks = {float(price): float(size) for price, size in order_book['a']}
    mid_price = (min(asks.keys()) + max(bids.keys())) / 2
    return {"ts": ts, "mid_price": mid_price, "bids": bids, "asks": asks}


class InteractiveLiveProcessor(tk.Tk):
    def __init__(self, api_key_path: str):
        super().__init__()
        self.title("Crypto Live Feature Viewer")
        self.geometry("1200x900")

        self.pld = PeriodicLiveRetriever(api_key_path, timedelta(milliseconds=650), "CAKEUSDT", "linear")

        self.feature_instances = {}
        self.feature_data = []
        self.last_snapshot = None
        self.active_features = {}
        self.subfeature_buttons = {}

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=1)

        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(self.left_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.right_frame = tk.Frame(self)
        self.right_frame.grid(row=0, column=1, sticky="ns")

        self.button_canvas = tk.Canvas(self.right_frame, width=220)
        self.button_canvas.pack(side="left", fill="y", expand=True)

        self.button_scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.button_canvas.yview)
        self.button_scrollbar.pack(side="right", fill="y")

        self.button_canvas.configure(yscrollcommand=self.button_scrollbar.set)
        self.button_canvas.bind('<Configure>', lambda e: self.button_canvas.configure(
            scrollregion=self.button_canvas.bbox("all")))

        self.button_container = tk.Frame(self.button_canvas)
        self.button_canvas.create_window((0, 0), window=self.button_container, anchor="nw")

        self.feature_frames = {}

        self._load_features_from_module(data_processing.ob_features)
        self._load_features_from_module(data_processing.trade_features)

        self.feature_creator = FeatureCreator(list(self.feature_instances.values()))

        for name, feature_obj in self.feature_instances.items():
            btn = tk.Button(self.button_container, text=name, width=20,
                            command=lambda n=name: self.toggle_feature(n))
            btn.pack(pady=(6, 2), padx=5)
            default_bg = btn.cget("bg")
            self.active_features[name] = {"active": False, "button": btn, "default_bg": default_bg}

            subfeatures = feature_obj.get_subfeature_names_and_toggle_function()
            self.subfeature_buttons[name] = []
            if subfeatures:
                for sub_name, toggle_fn in subfeatures:
                    sub_btn = tk.Button(self.button_container, text=f"↳ {sub_name}", width=18,
                                        relief="ridge", bg="#f0f0f0", anchor="w")
                    sub_btn.pack(padx=20, pady=(0, 4), anchor="w")

                    sub_btn_state = {
                        "active": False,
                        "button": sub_btn,
                        "default_bg": sub_btn.cget("bg")
                    }

                    def make_toggle_handler(fn, update_render, state=sub_btn_state):
                        def handler():
                            fn()
                            state["active"] = not state["active"]
                            state["button"].config(
                                bg="lightgreen" if state["active"] else state["default_bg"]
                            )
                            update_render()
                        return handler

                    def update_render(current_name=name): return self.update_visualization(current_name)

                    sub_btn.config(command=make_toggle_handler(toggle_fn, update_render, sub_btn_state))
                    self.subfeature_buttons[name].append(sub_btn_state)

        self.control_frame = tk.Frame(self.right_frame, bd=2, relief="groove", padx=5, pady=5)
        self.control_frame.pack(side="bottom", fill="x", pady=(10, 0))

        tk.Label(self.control_frame, text="Control Space", font=("Arial", 10, "bold")).pack(pady=(0, 5))

        tk.Label(self.control_frame, text="Coin Name:").pack(anchor="w")
        self.coin_name_var = tk.StringVar(value="CAKEUSDT")
        tk.Entry(self.control_frame, textvariable=self.coin_name_var).pack(fill="x")

        tk.Label(self.control_frame, text="Coin Category:").pack(anchor="w", pady=(5, 0))
        self.coin_category_var = tk.StringVar(value="linear")
        tk.Entry(self.control_frame, textvariable=self.coin_category_var).pack(fill="x")

        tk.Label(self.control_frame, text="Timedelta (ms):").pack(anchor="w", pady=(5, 0))
        self.timedelta_var = tk.StringVar(value="650")
        tk.Entry(self.control_frame, textvariable=self.timedelta_var).pack(fill="x")

        self.updating = Value("b", False)
        self.toggle_button = tk.Button(self.control_frame, text="Start Updates", bg="lightgreen", command=self.toggle_updates)
        self.toggle_button.pack(fill="x", pady=(10, 0))

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.run_update_thread = True
        self.after(100, self._start_update_thread)

    def _start_update_thread(self):
        self.updating_process = Thread(target=self.run_feature_update, daemon=True)
        self.updating_process.start()

    def toggle_updates(self):
        if self.updating.value:
            self.toggle_button.config(text="Start Updates", bg="lightgreen")
            self.pld.pause()
        else:
            self.toggle_button.config(text="Stop Updates", bg="lightcoral")
            self.pld.start()
        self.updating.value = not self.updating.value

    def on_close(self):
        print("Cleaning up InteractiveLiveProcessor...")
        self.updating.value = False
        self.run_update_thread = False
        self.pld.__del__()
        self.destroy()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _load_features_from_module(self, module):
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Feature) and name != "Feature":
                print(f"Loaded feature: {name}")
                instance = obj()
                self.feature_instances[name] = instance

    def toggle_feature(self, name):
        feature_state = self.active_features[name]
        feature_obj = self.feature_instances[name]

        if not feature_state["active"]:
            feature_state["active"] = True
            feature_state["button"].config(bg="lightgreen")
            self.embed_visualization(name, feature_obj)
        else:
            feature_state["active"] = False
            feature_state["button"].config(bg=feature_state["default_bg"])
            self.remove_visualization(name)

    def embed_visualization(self, name, feature_obj):
        print(f"Embedding feature: {name}")
        try:
            height = 3
            fig, ax = plt.subplots(figsize=(8, height), constrained_layout=True)

            feature_index = list(self.feature_instances.keys()).index(name)
            data = self.feature_creator.get_feature(np.array(self.feature_data), feature_index)

            if hasattr(feature_obj, 'visualize_feature'):
                feature_obj.visualize_feature(data, ax)

            canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            self.feature_frames[name] = {
                "widget": widget,
                "figure": fig,
                "axes": ax,
                "canvas": canvas
            }

            plt.close(fig)

        except Exception:
            print(f"Error visualizing {name}:")
            traceback.print_exc()

    def remove_visualization(self, name):
        frame_data = self.feature_frames.pop(name, None)
        if frame_data:
            frame_data["widget"].destroy()

    def update_visualization(self, name):
        if name not in self.active_features or not self.active_features[name]["active"]:
            return

        frame_data = self.feature_frames.get(name)
        feature_obj = self.feature_instances[name]

        if not frame_data:
            return

        fig = frame_data["figure"]
        ax = frame_data["axes"]
        canvas = frame_data["canvas"]

        try:
            ax.clear()

            feature_index = list(self.feature_instances.keys()).index(name)
            data = self.feature_creator.get_feature(np.array(self.feature_data), feature_index)

            if hasattr(feature_obj, 'visualize_feature'):
                feature_obj.visualize_feature(data, ax)

            canvas.draw()

        except Exception:
            print(f"Error updating visualization {name}:")
            traceback.print_exc()

    def run_feature_update(self):
        print("Starting feature update process...")
        while self.run_update_thread:
            if self.updating.value:
                try:
                    ob_data = self.pld.ob_data_queue.get(block=True)
                    trades_data = self.pld.trades_data_queue.get(block=True)

                    snapshot_data = convert_bybit_ob_to_snapshot(ob_data)
                    snapshot_data["trades"] = self.calculate_trades_since_last_timestep(trades_data["list"])
                    self.last_snapshot = snapshot_data
                    self.feature_creator.feed_datapoint(snapshot_data)

                    if self.feature_creator.is_ready():
                        self.feature_data.append(self.feature_creator.create_features())
                        for name, feature_obj in self.feature_instances.items():
                            self.after(0, self.update_visualization, name)

                except Exception:
                    print("Error in feature update process:")
                    traceback.print_exc()

    def calculate_trades_since_last_timestep(self, new_trades: List[dict]):
        if self.last_snapshot is None:
            for trade in new_trades:
                trade["price"] = float(trade["price"])
                trade["size"] = float(trade["size"])
            return new_trades
        last_timestep = int(self.last_snapshot["ts"].timestamp() * 1000)
        trades_since_last_timestep = [trade for trade in new_trades if int(trade["time"]) > last_timestep]
        for trade in trades_since_last_timestep:
            trade["price"] = float(trade["price"])
            trade["size"] = float(trade["size"])
        # print(trades_since_last_timestep)
        return trades_since_last_timestep


if __name__ == "__main__":
    api_key_path = "api_key.json"
    if len(sys.argv) > 1:
        api_key_path = sys.argv[1]
    app = InteractiveLiveProcessor(api_key_path)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
