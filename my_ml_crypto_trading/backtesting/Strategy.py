from abc import ABC, abstractmethod
from typing import List, Callable
from machine_learning.wrapper import *
from datetime import datetime, timedelta
import torch

class Strategy(ABC):

    @abstractmethod
    def get_orders(
        self,
        data: dict,
        positions: dict) -> tuple[dict, datetime]:
        
        """
        returns orders and the timestamp when it should be called next
        
        Input:
            {
                "Coin_1": Dict containing timestamp, mid_price, bids, asks and trades (same as for FeatureCreator)
                "Coin_2": ...
            }
        Returns:
            - orders [{"catogory": "linear", "symbol" : "CAKEUSDT", "side": "buy", "price": 1.99, "type" : "FOK" / "GUC" / "IOC"  }]
            - timestamp
        """
        pass

    @abstractmethod
    def set_up(self) -> tuple[dict, List[dict], datetime]:
        """
        {
            "perpetual": {
                "CAKEUSDT" : ["bids", "asks", "trades"], 
            },
            "spot": {

            },
            "truth_social": {
                "@RealDonaldTrump" : 
            }
                
            
        }
        frist function to be called. Returns a symbol for which the data should be collected and a timestamp when get_orders should be called next in milliseconds
        
        Returns:
        - data requirements
        - contracts that it wants to trade
        - next timestep to be queried
        
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
        self.td = timedelta(milliseconds=time_delta_ms)
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
        
        
        self.feature_buffer = []




    def set_up(self, time_stamp: datetime):
        
        self.feature_buffer = []
        next_ts = time_stamp + self.td
        coins = [self.coin]
        
        return next_ts, coins


    def get_orders(self, snapshot: dict, current_position: int) -> None:
        
        orders = []
        next_ts = snapshot["timestamp"] + self.td
        coins = [self.coin]

        self.feature_creator.feed_datapoint(snapshot)
        
        if not self.feature_creator.is_ready():
            return orders, next_ts
        
        self.feature_buffer.append(self.feature_creator.create_features())
        
        if len(self.feature_buffer) < self.window_length:
            return orders, next_ts

        # Collected enough features. Start trading

        # make prediction using self.feature_buffer[-self.window_length:]
        # Get the most recent window_length features
        features = self.feature_buffer[-self.window_length:]
        # Convert to tensor with appropriate shape
        features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
        
        # Make prediction using the model
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # No need to track gradients for inference
            prediction = self.model(features_tensor.unsqueeze(0))  # Add batch dimension
            prediction_value = prediction.item()  # Extract the scalar value


        # check if prediction is larger than the spread
        min_ask = min(snapshot["asks"].keys())
        max_bid = max(snapshot["bids"].keys())
        spread = min_ask - max_bid
        if abs(prediction) < 0.2:
            return orders, next_ts

        # build orders
        if prediction > 0:
            if current_position < 100:
                orders.append({
                    "side": "buy",
                    "price": min_ask,
                    "quantity": 100 - current_position
                    })
        if prediction < 0:
            if current_position > -100:
                orders.append({
                    "side": "sell",
                    "price": max_bid,
                    "quantity": 100 + current_position
                    })

        return orders, next_ts
