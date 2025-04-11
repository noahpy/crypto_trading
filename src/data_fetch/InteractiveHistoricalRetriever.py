
import tkinter as tk
from tkinter import filedialog
from datetime import date
from HistoricalDataRetriever import HistoricalDataRetriever
import io
import sys
import threading
import datetime


class InteractiveHistoricalRetriever(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crypto Data Downloader")
        self.geometry("400x400")

        self.task_count = 1

        # self.api_key_path = None
        self.download_dir = None

        # Coin symbol input
        tk.Label(self, text="Crypto Coin Symbol (e.g., BTC, ETH)").pack(
            pady=(10, 0))
        self.coin_entry = tk.Entry(self)
        self.coin_entry.pack(pady=5)

        # Data Type
        tk.Label(self, text="Data Type").pack()
        self.data_type = tk.StringVar(value="orderbook")  # default value
        data_type_menu = tk.OptionMenu(
            self, self.data_type, "orderbook", "trades")
        data_type_menu.pack(pady=5)

        # Category dropdown (linear or inverse)
        tk.Label(self, text="Category").pack()
        self.category_var = tk.StringVar(value="linear")  # default value
        category_menu = tk.OptionMenu(
            self, self.category_var, "linear", "inverse")
        category_menu.pack(pady=5)

        # Start Date
        tk.Label(self, text="Start Date").pack()
        self.start_frame = self.create_date_widgets()

        # End Date
        tk.Label(self, text="End Date").pack()
        self.end_frame = self.create_date_widgets()

        # # File and Directory Selectors
        # tk.Button(self, text="Choose API Key File",
        #           command=self.choose_api_key).pack(pady=5)
        # self.file_label = tk.Label(self, text="No file selected")
        # self.file_label.pack()

        tk.Button(self, text="Choose Download Directory",
                  command=self.choose_directory).pack(pady=5)
        self.dir_label = tk.Label(self, text="No directory selected")
        self.dir_label.pack()

        # Submit
        tk.Button(self, text="Submit", command=self.submit).pack(pady=10)
        self.result_label = tk.Label(self, text="", wraplength=350)
        self.result_label.pack(pady=5)

        # Live Log Output
        tk.Label(self, text="Logs:").pack()
        self.log_text = tk.Text(self, height=15, wrap="word")
        self.log_text.pack(padx=10, pady=5, fill="both", expand=True)

        self.after(100, self.update_log)  # Periodically update log area
        self.log_stream = io.StringIO()

    def create_date_widgets(self):
        frame = tk.Frame(self)
        frame.pack(pady=5)

        year = tk.Spinbox(frame, from_=2023, to=date.today().year, width=5)
        year.pack(side="left")
        tk.Label(frame, text="-").pack(side="left")

        month = tk.Spinbox(frame, from_=1, to=12, width=3)
        month.pack(side="left")
        tk.Label(frame, text="-").pack(side="left")

        day = tk.Spinbox(frame, from_=1, to=31, width=3)
        day.pack(side="left")

        return {'year': year, 'month': month, 'day': day}

    def get_date_from_widgets(self, widget_set):
        try:
            y = int(widget_set['year'].get())
            m = int(widget_set['month'].get())
            d = int(widget_set['day'].get())
            return datetime.datetime(y, m, d)
        except ValueError:
            return None

    def choose_api_key(self):
        path = filedialog.askopenfilename(title="Select API Key File")
        if path:
            self.api_key_path = path
            self.file_label.config(text=path)

    def choose_directory(self):
        path = filedialog.askdirectory(title="Select Download Directory")
        if path:
            self.download_dir = path
            self.dir_label.config(text=path)

    def submit(self):
        coin = self.coin_entry.get().strip().upper()
        category = self.category_var.get()
        data_type = self.data_type.get()
        start = self.get_date_from_widgets(self.start_frame)
        end = self.get_date_from_widgets(self.end_frame)

        if not coin:
            self.result_label.config(
                text="Please enter a crypto coin symbol.", fg="red")
            return
        if category not in ["linear", "inverse"]:
            self.result_label.config(
                text="Invalid category selected.", fg="red")
            return
        # if not self.api_key_path:
        #     self.result_label.config(
        #         text="Please select an API key file.", fg="red")
        #     return
        if not self.download_dir:
            self.result_label.config(
                text="Please select a download directory.", fg="red")
            return
        if not start or not end or start > end:
            self.result_label.config(text="Invalid date range.", fg="red")
            return

        if not HistoricalDataRetriever.check_category_symbol_exists(data_type, coin, category):
            self.result_label.config(
                text="Coin type + category does not exist for this data type.", fg="red")
            return

        self.result_label.config(
            text=(
                f"Downloading {data_type} data for: \n"
                f"✅ Coin: {coin}\n"
                f"✅ Category: {category}\n"
                f"✅ Range: {start} to {end}\n"
                f"✅ Download directory: {self.download_dir}"
            ),
            fg="green"
        )

        # Clear previous logs
        self.log_text.delete("1.0", tk.END)
        self.log("Starting task...\n")

        # Run in a separate thread
        thread = threading.Thread(
            target=self.run_async_task,
            args=(self.task_count, coin, category, start, end, self.download_dir, data_type),
            daemon=True
        )
        thread.start()
        self.task_count += 1

    def run_async_task(self, task_id,  coin, category, start, end, out_dir, data_type):
        old_stdout = sys.stdout
        sys.stdout = self.log_stream
        try:
            d = HistoricalDataRetriever(out_dir)
            print("Starting download for Task ID:", task_id)
            if data_type == "orderbook":
                d.fetch_historical_orderbook_period_data(
                    start, end, coin, category)
            elif data_type == "trades":
                d.fetch_historical_trading_period_data(
                    start, end, coin, category)
            else:
                print("Invalid data type!")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            print("Done downloading for Task ID:", task_id)
            sys.stdout = old_stdout

    def update_log(self):
        new_log = self.log_stream.getvalue()
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("end", new_log)
        self.log_text.see("end")
        self.after(200, self.update_log)

    def log(self, message):
        print(message)


if __name__ == "__main__":
    app = InteractiveHistoricalRetriever()
    app.mainloop()
