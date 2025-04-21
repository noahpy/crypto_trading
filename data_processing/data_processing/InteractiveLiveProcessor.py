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
import time
import queue


def convert_bybit_ob_to_snapshot(order_book):
    """Convert order book data to snapshot format more efficiently"""
    ts = datetime.fromtimestamp(order_book['ts'] / 1000)
    # Use dictionary comprehension only once for each
    bids = {float(price): float(size) for price, size in order_book['b']}
    asks = {float(price): float(size) for price, size in order_book['a']}
    mid_price = (min(asks.keys()) + max(bids.keys())) / 2
    return {"ts": ts, "mid_price": mid_price, "bids": bids, "asks": asks, "original_ts": order_book['ts']}


class InteractiveLiveProcessor(tk.Tk):
    def __init__(self, api_key_path: str):
        super().__init__()
        self.title("Crypto Live Feature Viewer")
        self.geometry("1200x900")

        # Initialize UI update throttling
        self.update_interval = 500  # ms between UI updates
        self.last_update_time = 0
        self.update_pending = False
        self.ui_update_queue = queue.Queue()

        # Create data retriever
        self.pld = PeriodicLiveRetriever(
            api_key_path, timedelta(milliseconds=1000), "CAKEUSDT", "linear")

        self.feature_instances = {}
        self.feature_data = []
        self.last_snapshot = None
        self.active_features = {}
        self.subfeature_buttons = {}

        self.FEATURE_WINDOW_LEN = 30

        # Setup UI
        self._setup_ui()

        # Load features
        self._load_features()

        # Start update thread
        self.run_update_thread = True
        self.after(100, self._start_update_thread)

        # Start UI update schedule
        self.after(50, self._process_ui_updates)

    def _setup_ui(self):
        """Setup UI components - separated for better organization"""
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=5)
        self.grid_columnconfigure(1, weight=1)

        # Setup left frame (visualization area)
        self._setup_left_frame()

        # Setup right frame (controls)
        self._setup_right_frame()

    def _setup_left_frame(self):
        """Setup the left frame containing visualizations"""
        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(self.left_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = ttk.Scrollbar(
            self.left_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas_frame = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all"))
        )

        # Optimize scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _setup_right_frame(self):
        """Setup the right frame containing buttons and controls"""
        self.right_frame = tk.Frame(self)
        self.right_frame.grid(row=0, column=1, sticky="ns")

        # Button area
        self.button_canvas = tk.Canvas(self.right_frame, width=220)
        self.button_canvas.pack(side="left", fill="y", expand=True)

        self.button_scrollbar = ttk.Scrollbar(
            self.right_frame, orient="vertical", command=self.button_canvas.yview)
        self.button_scrollbar.pack(side="right", fill="y")

        self.button_canvas.configure(yscrollcommand=self.button_scrollbar.set)
        self.button_canvas.bind('<Configure>', lambda e: self.button_canvas.configure(
            scrollregion=self.button_canvas.bbox("all")))

        self.button_container = tk.Frame(self.button_canvas)
        self.button_canvas.create_window(
            (0, 0), window=self.button_container, anchor="nw")

        self.feature_frames = {}

        # Control area
        self._setup_control_frame()

    def _setup_control_frame(self):
        """Setup the control frame with settings"""
        self.control_frame = tk.Frame(
            self.right_frame, bd=2, relief="groove", padx=5, pady=5)
        self.control_frame.pack(side="bottom", fill="x", pady=(10, 0))

        tk.Label(self.control_frame, text="Control Space",
                 font=("Arial", 10, "bold")).pack(pady=(0, 5))

        tk.Label(self.control_frame, text="Coin Name:").pack(anchor="w")
        self.coin_name_var = tk.StringVar(value="CAKEUSDT")
        tk.Entry(self.control_frame,
                 textvariable=self.coin_name_var).pack(fill="x")

        tk.Label(self.control_frame, text="Coin Category:").pack(
            anchor="w", pady=(5, 0))
        self.coin_category_var = tk.StringVar(value="linear")
        tk.Entry(self.control_frame,
                 textvariable=self.coin_category_var).pack(fill="x")

        tk.Label(self.control_frame, text="Timedelta (ms):").pack(
            anchor="w", pady=(5, 0))
        self.timedelta_var = tk.StringVar(value="650")
        tk.Entry(self.control_frame,
                 textvariable=self.timedelta_var).pack(fill="x")

        # Add UI refresh rate control
        tk.Label(self.control_frame, text="UI Refresh Rate (ms):").pack(
            anchor="w", pady=(5, 0))
        self.refresh_rate_var = tk.StringVar(value="500")
        refresh_entry = tk.Entry(
            self.control_frame, textvariable=self.refresh_rate_var)
        refresh_entry.pack(fill="x")
        refresh_entry.bind("<Return>", self._update_refresh_rate)

        self.updating = Value("b", False)
        self.toggle_button = tk.Button(
            self.control_frame, text="Start Updates", bg="lightgreen", command=self.toggle_updates)
        self.toggle_button.pack(fill="x", pady=(10, 0))

        self.update_settings_button = tk.Button(
            self.control_frame, text="Apply Coin Settings", bg="lightblue", command=self.apply_settings)
        self.update_settings_button.pack(fill="x", pady=(5, 0))

    def _update_refresh_rate(self, event=None):
        """Update the UI refresh rate based on user input"""
        try:
            new_rate = int(self.refresh_rate_var.get())
            if new_rate >= 100:  # Minimum 100ms to prevent freezing
                self.update_interval = new_rate
                print(f"UI refresh rate updated to {new_rate}ms")
        except ValueError:
            print("Invalid refresh rate value")

    def _load_features(self):
        """Load and initialize features"""
        self._load_features_from_module(data_processing.ob_features)
        self._load_features_from_module(data_processing.trade_features)

        for feature in list(self.feature_instances.values()):
            feature.turn_all_subfeatures_on()

        try:
            self.feature_instances["MidPriceFeature"].inc_mp_change = False
        except:
            pass

        self.feature_creator = FeatureCreator(
            list(self.feature_instances.values()))

        self._setup_feature_buttons()

    def _setup_feature_buttons(self):
        """Setup feature buttons in the UI"""
        for name, feature_obj in self.feature_instances.items():
            btn = tk.Button(self.button_container, text=name, width=20,
                            command=lambda n=name: self.toggle_feature(n))
            btn.pack(pady=(6, 2), padx=5)
            default_bg = btn.cget("bg")
            btn.config(activebackground=default_bg, activeforeground="black")
            self.active_features[name] = {
                "active": False, "button": btn, "default_bg": default_bg}

            subfeatures = feature_obj.get_subfeature_names_and_toggle_function()
            self.subfeature_buttons[name] = []
            if subfeatures:
                for sub_name, toggle_fn in subfeatures:
                    sub_btn = tk.Button(self.button_container, text=f"↳ {sub_name}", width=18,
                                        relief="ridge", bg="#f0f0f0", anchor="w")
                    sub_btn.pack(padx=20, pady=(0, 4), anchor="w")
                    sub_btn.config(activebackground="#f0f0f0",
                                   activeforeground="black")

                    sub_btn_state = {
                        "active": False,
                        "button": sub_btn,
                        "default_bg": sub_btn.cget("bg")
                    }

                    def make_toggle_handler(fn, update_render, state=sub_btn_state):
                        def handler():
                            state["active"] = not state["active"]
                            state["button"].config(
                                bg="lightgreen" if state["active"] else state["default_bg"]
                            )
                            update_render()
                        return handler

                    def update_render(
                        current_name=name): return self.update_visualization(current_name)

                    sub_btn.config(command=make_toggle_handler(
                        toggle_fn, update_render, sub_btn_state))
                    self.subfeature_buttons[name].append(sub_btn_state)

    def _start_update_thread(self):
        """Start the background thread for processing data updates"""
        self.updating_process = Thread(
            target=self.run_feature_update, daemon=True)
        self.updating_process.start()

    def _process_ui_updates(self):
        """Process UI updates at a throttled rate to improve performance"""
        current_time = time.time() * 1000  # Convert to ms

        # Process all pending updates if enough time has passed
        if current_time - self.last_update_time >= self.update_interval:
            try:
                # Process all queued updates at once
                updates_to_process = {}

                # Collect all updates, keeping only the latest for each feature
                while not self.ui_update_queue.empty():
                    name = self.ui_update_queue.get_nowait()
                    updates_to_process[name] = True

                # Apply updates
                for name in updates_to_process:
                    self._update_visualization_immediate(name)

                self.last_update_time = current_time
                self.update_pending = False
            except queue.Empty:
                pass

        # Schedule next check
        self.after(50, self._process_ui_updates)

    def toggle_updates(self):
        """Toggle data updates on/off"""
        if self.updating.value:
            self.toggle_button.config(text="Start Updates", bg="lightgreen")
            self.pld.pause()
        else:
            self.toggle_button.config(text="Stop Updates", bg="lightcoral")
            self.pld.start()
        self.updating.value = not self.updating.value

    def on_close(self):
        """Clean up resources when closing the app"""
        print("Cleaning up InteractiveLiveProcessor...")
        self.updating.value = False
        self.run_update_thread = False
        self.pld.__del__()
        self.destroy()

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def _load_features_from_module(self, module):
        """Load features from a module"""
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Feature) and name != "Feature":
                print(f"Loaded feature: {name}")
                instance = obj()
                self.feature_instances[name] = instance

    def toggle_feature(self, name):
        """Toggle a feature on/off"""
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
        """Embed a visualization for a feature"""
        print(f"Embedding feature: {name}")
        if not self.feature_creator.is_ready():
            print("Feature creator not ready!")
            return
        try:
            height = 3
            if name == "TradeFeature":
                height = 6

            # Create figure with toolbar disabled by default and tight_layout disabled
            plt.rcParams['toolbar'] = 'None'
            # Use constrained_layout instead of tight_layout
            fig, ax = plt.subplots(figsize=(8, height),
                                   constrained_layout=True)

            # Get feature data
            feature_index = list(self.feature_instances.keys()).index(name)
            data = self.feature_creator.get_feature(
                np.array(self.feature_data), feature_index)

            # Add warning suppression for tight_layout
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Visualize the feature
                if hasattr(feature_obj, 'visualize_feature'):
                    feature_obj.visualize_feature(data, ax)

            # Performance optimization: Use a lower DPI for faster rendering
            fig.set_dpi(80)

            # Create canvas without toolbar and disconnect observers that might try to update toolbar
            canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)

            # Disconnect the axes change event callback that tries to update toolbar
            for cid, callback in list(fig.canvas.callbacks.callbacks.get('_axes_change_event', {}).items()):
                fig.canvas.mpl_disconnect(cid)

            # If a tight_layout is called in visualize_feature, it might conflict with constrained_layout
            # Set constrained_layout to False to avoid warnings if tight_layout is used
            fig.set_constrained_layout(False)

            # Draw the canvas
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            self.feature_frames[name] = {
                "widget": widget,
                "figure": fig,
                "axes": ax,
                "canvas": canvas
            }

            # Close the matplotlib figure to free resources
            plt.close(fig)

            # Reset toolbar setting
            plt.rcParams['toolbar'] = 'toolbar2'

        except Exception:
            print(f"Error visualizing {name}:")
            traceback.print_exc()
            # Reset toolbar setting in case of error
            plt.rcParams['toolbar'] = 'toolbar2'

    def remove_visualization(self, name):
        """Remove a visualization"""
        frame_data = self.feature_frames.pop(name, None)
        if frame_data:
            try:
                # First disconnect any remaining callbacks
                fig = frame_data.get("figure")
                if fig:
                    for event_name in list(fig.canvas.callbacks.callbacks.keys()):
                        for cid in list(fig.canvas.callbacks.callbacks.get(event_name, {}).keys()):
                            try:
                                fig.canvas.mpl_disconnect(cid)
                            except:
                                pass

                # Then destroy the widget
                frame_data["widget"].destroy()

                # Clean up canvas if possible
                try:
                    if hasattr(frame_data["canvas"], "_tkcanvas"):
                        frame_data["canvas"]._tkcanvas.destroy()
                except tk.TclError:
                    pass

                # Free memory explicitly
                if fig is not None:
                    plt.close(fig)

            except Exception as e:
                print(f"Canvas cleanup error for {name}:", e)

    def update_visualization(self, name):
        """Queue a visualization update for throttled processing"""
        if name not in self.active_features or not self.active_features[name]["active"] \
                or not self.feature_creator.is_ready():
            return

        # Add to update queue instead of updating immediately
        self.ui_update_queue.put(name)

    def _update_visualization_immediate(self, name):
        """Internal method to actually update the visualization"""
        if name not in self.active_features or not self.active_features[name]["active"] \
                or not self.feature_creator.is_ready():
            return

        frame_data = self.feature_frames.get(name)
        feature_obj = self.feature_instances[name]

        if not frame_data:
            return

        fig = frame_data["figure"]
        ax = frame_data["axes"]
        canvas = frame_data["canvas"]

        try:
            # Clear existing plot
            ax.clear()

            # Get updated feature data
            feature_index = list(self.feature_instances.keys()).index(name)
            data = self.feature_creator.get_feature(
                np.array(self.feature_data), feature_index)

            # Add warning suppression for tight_layout
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Visualize updated data
                if hasattr(feature_obj, 'visualize_feature'):
                    feature_obj.visualize_feature(data, ax)

            # Performance optimization: Try to render directly
            # Regular draw is safer but slower
            canvas.draw()

        except Exception:
            print(f"Error updating visualization {name}:")
            traceback.print_exc()

    def run_feature_update(self):
        """Background thread for processing data updates"""
        print("Starting feature update process...")
        while self.run_update_thread:
            if self.updating.value:
                try:
                    # Get data with timeout to avoid blocking forever
                    try:
                        data = self.pld.data_queue.get(block=True)
                    except queue.Empty:
                        continue

                    # Process data
                    snapshot_data = convert_bybit_ob_to_snapshot(data["ob"])
                    snapshot_data["trades"] = self.calculate_trades_since_last_timestep(
                        data["trades"]["list"])
                    self.last_snapshot = snapshot_data
                    self.feature_creator.feed_datapoint(snapshot_data)

                    if self.feature_creator.is_ready():
                        self.feature_data.append(
                            self.feature_creator.create_features())
                        # Keep limited history to prevent memory bloat
                        self.feature_data = self.feature_data[-self.FEATURE_WINDOW_LEN:]

                        # Queue updates for active features
                        for name in self.active_features:
                            if self.active_features[name]["active"]:
                                self.ui_update_queue.put(name)

                except Exception as e:
                    print(f"Error in feature update process: {str(e)}")
                    traceback.print_exc()
            else:
                # Sleep when not updating to reduce CPU usage
                time.sleep(0.1)

    def calculate_trades_since_last_timestep(self, new_trades: List[dict]):
        """Calculate trades since last timestamp"""
        if self.last_snapshot is None:
            for trade in new_trades:
                trade["price"] = float(trade["price"])
                trade["size"] = float(trade["size"])
            return new_trades

        last_timestep = self.last_snapshot["original_ts"]

        trades_since_last_timestep = [
            trade for trade in new_trades if int(trade["time"]) > last_timestep]

        for trade in trades_since_last_timestep:
            trade["price"] = float(trade["price"])
            trade["size"] = float(trade["size"])

        print(trades_since_last_timestep)

        return trades_since_last_timestep

    def apply_settings(self):
        """Apply the settings from the control panel inputs"""
        new_coin = self.coin_name_var.get().strip().upper()
        new_category = self.coin_category_var.get().strip().lower()
        try:
            new_timedelta = int(self.timedelta_var.get())
        except ValueError:
            print("Invalid timedelta value")
            return

        print(f"Applying new settings: {new_coin}, {new_category}, {new_timedelta}ms")

        # Restart retriever with new settings
        self.pld.pause()

        self.pld.set_time_interval(new_timedelta)
        self.pld.set_currency(new_coin, new_category)

        self.pld.start()


if __name__ == "__main__":
    api_key_path = "api_key.json"
    if len(sys.argv) > 1:
        api_key_path = sys.argv[1]
    app = InteractiveLiveProcessor(api_key_path)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
