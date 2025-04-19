from data_retrieving.PeriodicLiveRetriever import PeriodicLiveRetriever
from data_processing.FeatureCreation import Feature
import data_processing.ob_features
import data_processing.trade_features
import tkinter as tk
from tkinter import ttk
import inspect
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import timedelta


class InteractiveLiveProcessor(tk.Tk):
    def __init__(self, api_key_path: str):
        super().__init__()
        self.title("Crypto Live Feature Viewer")
        self.geometry("1000x800")
        self.pld = PeriodicLiveRetriever(api_key_path, timedelta(
            milliseconds=650), "CAKEUSDT", "linear")


        self.feature_instances = {}
        self.active_features = {}

        # Main layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # --- Top: Toggle Buttons ---
        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=0, column=0, sticky="ew")

        # --- Bottom: Scrollable Visualization Frame ---
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=1, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=1, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_frame = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")

        # Track visualizations
        self.feature_frames = {}

        # Load features
        self._load_features_from_module(data_processing.ob_features)
        self._load_features_from_module(data_processing.trade_features)

        # Create buttons
        for name, feature_obj in self.feature_instances.items():
            btn = tk.Button(self.button_frame, text=name, width=25,
                            command=lambda n=name: self.toggle_feature(n))
            btn.pack(side=tk.LEFT, padx=5, pady=5)
            default_bg = btn.cget("bg")
            self.active_features[name] = {
                "active": False, "button": btn, "default_bg": default_bg}

        # Make scrollable on mousewheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def on_close(self):
        print("Cleaning up InteractiveLiveProcessor...")
        self.pld.__del__()  # Or better: self.pld.cleanup() if you define one
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
            # Activate and visualize
            feature_state["active"] = True
            feature_state["button"].config(bg="lightgreen")
            self.embed_visualization(name, feature_obj)
        else:
            # Deactivate and remove plot
            feature_state["active"] = False
            feature_state["button"].config(bg=feature_state["default_bg"])
            self.remove_visualization(name)

    def embed_visualization(self, name, feature_obj):
        """
        Create and embed a matplotlib plot in the scrollable frame.
        """
        print(f"Embedding feature: {name}")  # Debug print

        try:
            fig, ax = plt.subplots(figsize=(10, 2.5))

            # TEMP: Hardcoded test plot
            fake_data = np.random.rand(100, feature_obj.get_feature_size())

            if hasattr(feature_obj, 'visualize_feature'):
                try:
                    feature_obj.visualize_feature(fake_data, ax)
                except Exception as e:
                    print(f"Error visualizing {name}: {e}")
                    return

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
            canvas.draw()

            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Store reference to avoid garbage collection
            self.feature_frames[name] = widget

        except Exception as e:
            print(f"Error visualizing {name}: {e}")

    def remove_visualization(self, name):
        frame = self.feature_frames.pop(name, None)
        if frame:
            frame.destroy()


if __name__ == "__main__":
    api_key_path = "apiKey.json"
    if len(sys.argv) > 1:
        api_key_path = sys.argv[1]
    app = InteractiveLiveProcessor(api_key_path)
    # handle window closing
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
