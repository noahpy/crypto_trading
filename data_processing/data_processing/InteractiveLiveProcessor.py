
from data_retrieving.PeriodicLiveRetriever import PeriodicLiveRetriever
from data_processing.FeatureCreation import Feature
import data_processing.ob_features 
import data_processing.trade_features
import tkinter as tk
import inspect
import sys


class InteractiveLiveProcessor(tk.Tk):
    def __init__(self, api_key_path: str):
        super().__init__()
        self.title("Crypto Live Feature Viewer")
        self.geometry("600x500") 


        instances = {}

        for name, obj in inspect.getmembers(data_processing.ob_features):
            if inspect.isclass(obj) and name != "Feature":
                print(name)
                instances[name] = obj()

        for name, obj in inspect.getmembers(data_processing.trade_features):
            if inspect.isclass(obj) and name != "Feature":
                print(name)
                instances[name] = obj()

        for name, obj in instances.items():
            button = tk.Button(self, text=name)
            button.pack()


if __name__ == "__main__":
    api_key_path = "apiKey.json"
    if len(sys.argv) > 1:
        api_key_path = sys.argv[1]
    app = InteractiveLiveProcessor(api_key_path)
    app.mainloop()

