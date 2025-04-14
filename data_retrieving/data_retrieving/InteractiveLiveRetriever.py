import tkinter as tk
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from data_retrieving.LiveDataRetriever import LiveDataRetriever
from matplotlib.colors import LinearSegmentedColormap
from data_processing.order_book import *
import seaborn as sns
import sys

api_key_path = "./api_key.json"


def zero_pad(x, tick_size):
    if len(str(x).split(".")[1]) < len(str(tick_size).split(".")[1]):
        return str(x) + "0"
    else:
        return str(x)


class InteractiveLiveRetriever(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Crypto Live Data Viewer")
        self.geometry("600x500")  # Larger window to accommodate visualization

        # Initialize any class variables you might need later
        self.task_count = 0
        self.running = False
        self.ob_history = []  # Store history as instance variable

        # Input frame for all settings
        input_frame = tk.Frame(self)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create a 2x5 grid for the inputs (2 columns, 5 rows)
        # Column 1
        tk.Label(input_frame, text="Symbol").grid(
            row=0, column=0, sticky='w', padx=5, pady=2)
        self.token_entry = tk.Entry(input_frame, width=12)
        self.token_entry.insert(0, "CAKEUSDT")  # Default value
        self.token_entry.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        tk.Label(input_frame, text="Time Delta (Seconds)").grid(
            row=1, column=0, sticky='w', padx=5, pady=2)
        self.time_delta_entry = tk.Entry(input_frame, width=12)
        self.time_delta_entry.insert(0, "1")  # Default value
        self.time_delta_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)

        tk.Label(input_frame, text="Tick Size").grid(
            row=2, column=0, sticky='w', padx=5, pady=2)
        self.tick_size_entry = tk.Entry(input_frame, width=12)
        self.tick_size_entry.insert(0, "0.001")  # Default value
        self.tick_size_entry.grid(row=2, column=1, sticky='w', padx=5, pady=2)

        # Column 2
        tk.Label(input_frame, text="Num. Time Steps").grid(
            row=0, column=2, sticky='w', padx=5, pady=2)
        self.num_time_steps_entry = tk.Entry(input_frame, width=12)
        self.num_time_steps_entry.insert(0, "30")  # Default value
        self.num_time_steps_entry.grid(
            row=0, column=3, sticky='w', padx=5, pady=2)

        tk.Label(input_frame, text="Num. Levels").grid(
            row=1, column=2, sticky='w', padx=5, pady=2)
        self.num_levels_entry = tk.Entry(input_frame, width=12)
        self.num_levels_entry.insert(0, "10")  # Default value
        self.num_levels_entry.grid(row=1, column=3, sticky='w', padx=5, pady=2)

        # Button frame for Start and Stop buttons
        button_frame = tk.Frame(input_frame)
        button_frame.grid(row=2, column=2, columnspan=2,
                          sticky='w', padx=5, pady=5)

        # Start button
        self.start_button = tk.Button(
            button_frame, text="Start", command=self.start_data_retrieval)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Stop button
        self.stop_button = tk.Button(
            button_frame, text="Stop", command=self.stop_data_retrieval, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = tk.Label(
            self, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        # Frame for matplotlib figure
        self.fig_frame = tk.Frame(self)
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create initial figure
        self.setup_plot()

        # save reference to LiveDataRetriever later
        self.ld = None

    def __del__(self):
        if self.ld:
            del self.ld

    def setup_plot(self):
        # Create a new figure and canvas
        plt.close('all')  # Close any existing figures
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Set up an empty initial plot
        empty_data = np.zeros((20, 200))  # Adjust dimensions as needed
        sns.heatmap(empty_data.T, cmap='viridis_r', ax=self.ax, cbar=False)
        self.ax.invert_yaxis()
        self.ax.set_title("Order Book Heatmap")
        self.fig.tight_layout()
        self.canvas.draw()

    def start_data_retrieval(self):
        # This method will be called when the Start button is clicked
        symbol = self.token_entry.get().strip().upper()
        try:
            time_delta = int(self.time_delta_entry.get())

            # Reset history when starting new session
            self.ob_history = []

            self.status_label.config(text=f"Starting retrieval for {
                                     symbol} with time delta {time_delta}s ")

            # Update button states
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)

            # Start a simple loop in a separate thread
            self.running = True

            # Start data processing in a separate thread
            thread = threading.Thread(
                target=self.data_processing_loop, args=(symbol, time_delta))
            thread.daemon = True  # Thread will close when main program exits
            thread.start()

        except ValueError:
            self.status_label.config(
                text="Please enter a valid time delta (integer)")

    def stop_data_retrieval(self):
        # Stop the update loop
        self.running = False

        # Update button states
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Stopped data retrieval")

    def data_processing_loop(self, symbol, time_delta):
        """Process data in a separate thread and schedule visualization on main thread"""
        try:
            global api_key_path
            self.ld = LiveDataRetriever(api_key_path)

            while self.running:
                try:
                    # Load order book
                    current_ob = self.ld.fetch_current_orderbook(symbol)
                    ob_snapshot = convert_bybit_ob_to_snapshot(current_ob)

                    # Add to history and limit size
                    self.ob_history.append(ob_snapshot)
                    if len(self.ob_history) > 2000:  # Keep more than needed for visualization
                        self.ob_history = self.ob_history[-2000:]

                    # Pre-process data for visualization
                    num_levels = int(self.num_levels_entry.get())
                    tick_size = float(self.tick_size_entry.get())
                    time_steps = int(self.num_time_steps_entry.get())

                    # Create 2d array of size time_steps x 2*ticks
                    bid_data = np.zeros((time_steps, 2*num_levels))
                    ask_data = np.zeros((time_steps, 2*num_levels))

                    mid_price = self.ob_history[-1]["mid_price"]
                    mid_price = round(mid_price / tick_size) * tick_size
                    min_level = mid_price - num_levels * tick_size
                    levels = [" " for _ in range(2*num_levels)]

                    # Fill array with information
                    for i in range(time_steps, 0, -1):
                        if i > len(self.ob_history):
                            continue

                        bids = self.ob_history[-i]["bids"]
                        asks = self.ob_history[-i]["asks"]

                        # Convert bid and ask prices to float (if they're strings)
                        if isinstance(next(iter(bids.keys()), "0"), str):
                            bid_prices = {
                                float(price): volume for price, volume in bids.items()}
                            ask_prices = {
                                float(price): volume for price, volume in asks.items()}
                        else:
                            bid_prices = bids
                            ask_prices = asks

                        for j in range(2 * num_levels):
                            level = round(
                                (min_level + j * tick_size) / tick_size) * tick_size

                            # Find matching price levels with a small tolerance
                            tolerance = tick_size / 2  # Use a fraction of tick size as tolerance

                            # Look for matching prices in bids
                            for price in bid_prices:

                                if abs(price - level) < tolerance:
                                    if i == 1:
                                        levels[j] = f"{
                                            zero_pad(price, tick_size)} - {round(bid_prices[price])}"

                                    bid_data[-i][j] += bid_prices[price]
                                    break

                            # Look for matching prices in asks
                            for price in ask_prices:

                                if abs(price - level) < tolerance:
                                    if i == 1:
                                        levels[j] = f"{
                                            zero_pad(price, tick_size)} - {round(ask_prices[price])}"
                                    ask_data[-i][j] += ask_prices[price]
                                    break

                    # Schedule visualization on main thread with both bid and ask data
                    self.after(0, lambda b=bid_data, a=ask_data,
                               l=levels: self.update_visualization(b, a, l, symbol))

                    # Update status
                    self.after(0, lambda: self.status_label.config(
                        text=f"Updated at {time.strftime(
                            '%H:%M:%S')} - {symbol} - {len(self.ob_history)} snapshots"
                    ))

                    # Sleep for time delta
                    time.sleep(time_delta)

                except Exception as e:
                    self.after(0, lambda err=str(e): self.status_label.config(
                        text=f"Error in data processing: {err}"
                    ))
                    time.sleep(time_delta)  # Continue loop even after error

        except Exception as e:
            self.after(0, lambda err=str(e): self.status_label.config(
                text=f"Failed to initialize LiveDataRetriever: {err}"
            ))

    def update_visualization(self, bid_data, ask_data, levels, symbol):
        """Update the visualization with pre-processed data on the main thread"""
        try:
            # Clear previous plot content
            self.ax.clear()

            # Remove any existing colorbars
            for cbar in self.fig.axes:
                if cbar is not self.ax:
                    cbar.remove()

            # Create custom colormaps that start with white
            from matplotlib.colors import LinearSegmentedColormap

            # Custom green colormap (white to green)
            green_cmap = LinearSegmentedColormap.from_list(
                'white_to_green', ['white', 'green'])

            # Custom red colormap (white to red)
            red_cmap = LinearSegmentedColormap.from_list(
                'white_to_red', ['white', 'red'])

            # Plot bids with green colormap - set vmin=0 to ensure zero values are white
            sns.heatmap(bid_data.T, cmap=green_cmap,
                        ax=self.ax, cbar=False, vmin=0)

            # Plot asks with red colormap on top of bids - set vmin=0 to ensure zero values are white
            sns.heatmap(ask_data.T, cmap=red_cmap, ax=self.ax,
                        cbar=False, alpha=0.5, vmin=0)

            # Customize plot
            self.ax.invert_yaxis()

            # Remove default y-ticks
            self.ax.set_yticks([])
            self.ax.set_yticklabels([])

            # Add price labels from the provided levels
            for j, level_text in enumerate(levels):
                # Position text in the center of each cell (j + 0.5) instead of at the bottom edge (j)
                self.ax.text(bid_data.shape[0] + 0.2, j + 0.5, level_text,
                             fontsize=5, ha='left', va='center')

            # Remove x ticks
            self.ax.set_xticks([])

            # Remove all labels
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")

            # Make sure the price labels are visible by adjusting the layout
            self.fig.tight_layout()
            plt.subplots_adjust(right=0.82)  # Leave space on right for labels

            self.canvas.draw_idle()

        except Exception as e:
            self.status_label.config(text=f"Visualization error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        api_key_path = sys.argv[1]
    app = InteractiveLiveRetriever()
    app.mainloop()
