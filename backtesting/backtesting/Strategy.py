from abc import ABC, abstractmethod
from typing import List, Callable
from machine_learning.ml import *

class Strategy(ABC):

    @abstractmethod
    def get_orders(self, data: dict) -> None:
        """
        returns orders and the timestamp when it should be called next
        
        Input:
            {
                "Coin_1": Dict containing timestamp, mid_price, bids, asks and trades (same as for FeatureCreator)
                "Coin_2": ...
            }
        Returns:
            - {"Coin_X" : [orders]}
            - timestamp
            - [Coin_1, Coin_2] data that it expects when the function is called next
        """
        pass


class SimpleStrategy(Strategy):

    def __init__(
            self,
            coin,
            time_delta_ms,
            window_length,
            target,
            model_name
            ):
        
        self.coin = coin
        self.time_delta_ms = time_delta_ms
        self.window_length = window_length
        self.target = target
        self.model_name = model_name

        model_path = f"ml_models/{coin}/time_delta={time_delta_ms}/target={target}/{model_name}"

        self.model, self.feature_creator = load_model(
                                                coin,
                                                time_delta_ms, 
                                                window_length,
                                                target,
                                                model_name)
        


    def get_orders(self, data: dict) -> None:
        return None
