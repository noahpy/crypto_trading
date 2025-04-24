from backtesting.Strategy import Strategy
from datetime import datetime, timedelta
import pandas as pd
from data_retrieving.HistoricalDataRetriever import *
import json
import pytz
from machine_learning.wrapper import PyTorchWrapper


class PeriodicDataLoader():

    def __init__(self, data_requrements, start_date):
        self.data_requrements = data_requrements
        self.start_date = start_date
        self.retriever = HistoricalDataRetriever()

        self.init_buffer()
        

    def init_buffer(self):
        self.buffered_data = {}
        
        if "linear" in self.data_requrements:
            self.buffered_data["linear"] = {}

            for symbol in self.data_requrements["linear"]:
                self.buffered_data["linear"][symbol] = {}

                if "bids" in self.data_requrements["linear"][symbol]:
                    ob_data = self.retriever.get_historical_orderbook_day_data(self.start_date, symbol, "linear")
                    self.buffered_data["linear"][symbol]["ob"] = {"ob_data": ob_data, "line": -1, "bids": None, "asks": None}

                if "trades" in self.data_requrements["linear"][symbol]:
                    
                    trade_data = self.retriever.get_historical_trading_day_data(self.start_date, symbol, "linear")
                    self.buffered_data["linear"][symbol]["trades"] = {"trade_data": trade_data, "trade_id": 0}

    def update_order_book(self, current_bids, current_asks, new_bids, new_asks, ob_type):
        """
        Given a line of the historical orderbook data, update the order book.
        """

        # update order book
        if ob_type == "snapshot":
            current_bids = new_bids
            current_asks = new_asks

        else:
            # update bids
            for price in new_bids:
                if new_bids[price] == 0:
                    if price in current_bids:
                        del current_bids[price]
                else:
                    current_bids[price] = new_bids[price]

            # update asks
            for price in new_asks:
                if new_asks[price] == 0:
                    if price in current_asks:
                        del current_asks[price]
                else:
                    current_asks[price] = new_asks[price]

        return current_bids, current_asks

    def get_ob_data(self, category, symbol, next_ts):
        """
        Get orderbook data for the specified category and symbol at or after the given timestamp.
        Loads new data when a new day is reached.
        
        Args:
            category (str): Market category (e.g., 'linear')
            symbol (str): Trading symbol (e.g., 'CAKEUSDT')
            next_ts (datetime): Timestamp to get data for or after
            
        Returns:
            tuple: (current_bids, current_asks) - dictionaries of price -> size
        """
        # Access buffer data
        buffer_data = self.buffered_data[category][symbol]["ob"]
        ob_data = buffer_data["ob_data"]
        idx = buffer_data.get("idx", -1)
        current_bids = buffer_data["bids"]
        current_asks = buffer_data["asks"]
        
        # Check if we need to load a new day's data
        if idx >= len(ob_data) - 1 and idx != -1:
            # Calculate the next day to load
            last_data = json.loads(ob_data[-1])
            last_ts = datetime.fromtimestamp(last_data['ts']/1000, tz=pytz.UTC)
            current_date = last_ts.date() + timedelta(days=1)
                
            # Load the new day's data
            new_date = datetime(current_date.year, current_date.month, current_date.day)
            ob_data = self.retriever.get_historical_orderbook_day_data(new_date, symbol, category)
            self.buffered_data[category][symbol]["ob"]["ob_data"] = ob_data
            idx = -1  # Reset index for the new data
        
        # If this is the first data point or we've reset, initialize orderbook
        if idx == -1:
            if len(ob_data) > 0:
                data = json.loads(ob_data[0])
                current_bids = {float(price): float(size) for price, size in data['data']['b']}
                current_asks = {float(price): float(size) for price, size in data['data']['a']}
                idx = 0
            else:
                # Handle case where no data is available
                return {}, {}
        
        # Process orderbook updates until we reach or exceed the target timestamp
        while idx < len(ob_data):
            data = json.loads(ob_data[idx])
            ts = datetime.fromtimestamp(data['ts']/1000, tz=pytz.UTC)
            
            # If we've reached data at or after the requested timestamp, stop updating
            if ts >= next_ts:
                break
                
            # Update the orderbook with this data point
            new_bids = {float(price): float(size) for price, size in data['data']['b']}
            new_asks = {float(price): float(size) for price, size in data['data']['a']}
            current_bids, current_asks = self.update_order_book(
                current_bids, current_asks, new_bids, new_asks, data['type'])
            
            idx += 1
        
        # Update the buffer with our current position and order book state
        self.buffered_data[category][symbol]["ob"]["idx"] = idx
        self.buffered_data[category][symbol]["ob"]["bids"] = current_bids
        self.buffered_data[category][symbol]["ob"]["asks"] = current_asks
        
        return current_bids, current_asks
            
    def get_trade_data(self, category, symbol, next_ts):
        """
        Get all trades that happened between the last timestamp and next_ts
        
        Args:
            category (str): Market category (e.g., 'linear')
            symbol (str): Trading symbol (e.g., 'CAKEUSDT')
            next_ts (datetime): Timestamp to get trades up to
            
        Returns:
            list: List of trade dictionaries
        """
        # Make sure next_ts has timezone info
        if next_ts.tzinfo is None:
            next_ts = next_ts.replace(tzinfo=pytz.UTC)
        
        # Access buffer data
        buffer_data = self.buffered_data[category][symbol]["trades"]
        trade_data = buffer_data["trade_data"]
        trade_id = buffer_data["trade_id"]
        
        # Initialize return list
        trades = []
        
        # Get the last timestamp we processed - use a class variable to store this
        if not hasattr(self, 'last_trade_ts'):
            self.last_trade_ts = {}
        
        if category not in self.last_trade_ts:
            self.last_trade_ts[category] = {}
        
        if symbol not in self.last_trade_ts[category]:
            # Initialize with start date
            self.last_trade_ts[category][symbol] = self.start_date
        
        last_ts = self.last_trade_ts[category][symbol]
        
        # Check if we need to load a new day's data
        if trade_id >= len(trade_data) and len(trade_data) > 0:
            # Calculate the next day to load
            last_trade = trade_data.iloc[-1]
            last_trade_ts = datetime.fromtimestamp(last_trade["timestamp"], tz=pytz.UTC)
            current_date = last_trade_ts.date() + timedelta(days=1)
            
            # Load the new day's data
            new_date = datetime(current_date.year, current_date.month, current_date.day)
            new_trade_data = self.retriever.get_historical_trading_day_data(new_date, symbol, category)
            
            self.buffered_data[category][symbol]["trades"]["trade_data"] = new_trade_data
            self.buffered_data[category][symbol]["trades"]["trade_id"] = 0
            
            # Update our references
            trade_data = new_trade_data
            trade_id = 0
        
        # Process trades until we reach the next timestamp
        while trade_id < len(trade_data):
            trade = trade_data.iloc[trade_id]
            trade_ts = datetime.fromtimestamp(trade["timestamp"], tz=pytz.UTC)
            
            # Skip trades that are before our last processed timestamp
            if trade_ts <= last_ts:
                trade_id += 1
                continue
                
            # Stop if we've reached the next timestamp
            if trade_ts > next_ts:
                break
                
            # Add this trade to our list
            trades.append({
                "side": trade["side"],
                "size": trade["size"],
                "price": trade["price"],
                "timestamp": trade_ts
            })
            
            trade_id += 1
        
        # Update our buffer and last processed timestamp
        self.buffered_data[category][symbol]["trades"]["trade_id"] = trade_id
        self.last_trade_ts[category][symbol] = next_ts
        
        return trades

    
    def get_data(self, time_stamp):
        data = {}
        if "linear" in self.data_requrements:
            data["linear"] = {}
            for symbol in self.data_requrements["linear"]:
                data["linear"][symbol] = {}

                if "bids" in self.data_requrements["linear"][symbol]:
                    bids, asks = self.get_ob_data("linear", symbol, time_stamp)
                    data["linear"][symbol]["bids"] = bids
                    data["linear"][symbol]["asks"] = asks
                    

                if "trades" in self.data_requrements["linear"][symbol]:
                    trades = self.get_trade_data("linear", symbol, time_stamp)
                    data["linear"][symbol]["trades"] = trades


                
        return data

        


class HistoricMarketSimulator():


    def __init__(self):
        pass

    def zero_pad(self, x):
        if x < 10:
            return f"0{x}"
        else:
            return x


    def update_order_book(self, current_bids, current_asks, new_bids, new_asks, ob_type):
        """
        Given a line of the historical orderbook data, update the order book.
        """

        # update order book
        if ob_type == "snapshot":
            current_bids = new_bids
            current_asks = new_asks

        else:
            # update bids
            for price in new_bids:
                if new_bids[price] == 0:
                    if price in current_bids:
                        del current_bids[price]
                else:
                    current_bids[price] = new_bids[price]

            # update asks
            for price in new_asks:
                if new_asks[price] == 0:
                    if price in current_asks:
                        del current_asks[price]
                else:
                    current_asks[price] = new_asks[price]

        return current_bids, current_asks

    
    def initialize_positions(self, tradable_assets):
        
        positions = {}

        for asset in tradable_assets:
            
            category = asset["category"]
            symbol = asset["symbol"]
            base = asset["base"]

            positions[symbol] = {
                "balance" : 0,
                "unmatched_orders" : [],
                "category" : category,
                "base" : base
            }
            
            if not base in positions:
                positions[base] = {
                    "balance" : 0
                }
        
        self.positions = positions


    def update_positions(self, orders, data):

        # 1. Add orders to unmatched orders 
        for order in orders:
            symbol = order["symbol"]
            self.positions[symbol]["unmatched_orders"].append(order)


        # 2. Go though the order book and match orders
        for symbol in self.positions:
            if not "unmatched_orders" in self.positions[symbol]:
                # non tradable asset. continue (usually USDT)
                continue 
            
            if not self.positions[symbol]["unmatched_orders"]:
                continue

            category = self.positions[symbol]["category"]
            
            unmatched_orders = self.positions[symbol]["unmatched_orders"]
            bids = data[category][symbol]["bids"]
            asks = data[category][symbol]["asks"]
            
            for order in unmatched_orders:
                side = order["side"]
                price = order["price"]
                qty = order["qty"]
                if order["type"] != "FOK":
                    print("Only use FOK orders implemented at the moment")
                    return None
                
                if side.lower() == "buy":
                    min_ask = min(asks.keys())
                    if min_ask <= price:
                        match_qty = min(asks[min_ask], qty)
                        base = self.positions[symbol]["base"]
                        self.positions[symbol]["balance"] += match_qty
                        self.positions[base]["balance"] -= min_ask * match_qty
                
                elif side.lower() == "sell":
                    max_bid = max(bids.keys())
                    if max_bid >= price:
                        match_qty = min(bids[max_bid], qty)
                        base = self.positions[symbol]["base"]
                        self.positions[symbol]["balance"] -= match_qty
                        self.positions[base]["balance"] += max_bid * match_qty

            self.positions[symbol]["unmatched_orders"] = []




    def backtest_strategy2(
            self,
            strategy: Strategy,
            start_date: datetime,
            end_date: datetime):
        
        import sys
        
        current_ts = start_date
        data_requirements, tradable_assets, next_ts = strategy.set_up()
        
        periodic_dl = PeriodicDataLoader(data_requirements, current_ts)
        self.initialize_positions(tradable_assets)
        
        print(f"positions: {self.positions}")
        print(f"periodic_dl: {periodic_dl}")
        num_trades = 0
        while current_ts <= end_date:
            # Write progress to the same line without creating a new line
            
            #sys.stdout.flush()
            
            data = periodic_dl.get_data(next_ts)
            orders, next_ts = strategy.get_orders(data, self.positions, current_ts)
            
            self.update_positions(orders, data)
            
            
            if orders:    # Print orders on new lines only when they occur
                num_trades += 1
                sys.stdout.write(f"\rPositions: {self.positions}  -  {num_trades}  -  {current_ts}                                                ")
                sys.stdout.flush()
            
            current_ts = next_ts
        
        # Add a final newline to ensure the command prompt appears correctly
        print()
        
        return None




class TestStrategy(Strategy):

    def set_up(self):
        
        data_requirements = {"linear": {"CAKEUSDT": ["bids", "asks", "trades"]}}
        tradeable_assets = [{"category": "linear", "symbol": "CAKEUSDT", "base": "USDT"}]
        self.model = PyTorchWrapper("ml_models", 30)
        self.model_ready = False

        return data_requirements, tradeable_assets, datetime(2025,1,1, tzinfo=pytz.UTC)

    def get_orders(self, data, positions, current_ts=None):

        
        bids = data["linear"]["CAKEUSDT"]["bids"]
        asks = data["linear"]["CAKEUSDT"]["asks"]
        max_bid = max(bids.keys())
        min_ask = min(asks.keys())
        mid_price = (max_bid + min_ask)/2
        spread = min_ask - max_bid

        # reduce size of bids and asks
        max_bids = sorted(bids.keys(), reverse=True)[:30]
        min_asks = sorted(asks.keys(), reverse=False)[:30]


        snapshot = {
            "ts": current_ts,
            "mid_price": mid_price,
            'bids': {level: bids[level] for level in max_bids},
            'asks': {level: asks[level] for level in min_asks},
            "trades": data["linear"]["CAKEUSDT"]["trades"]
            }
        
        if not self.model_ready:
            self.model_ready = self.model.feed_snapshot(snapshot)
            return [], current_ts + timedelta(seconds=1)
        
        prediction = self.model.predict(snapshot)
        

        if abs(prediction) < 0.5*spread:
            return [], current_ts + timedelta(seconds=1)

        cake_balance = positions["CAKEUSDT"]["balance"]

        if cake_balance < 1 and prediction > 0:
            return [
                {"catogory": "linear",
                 "symbol" : "CAKEUSDT", 
                 "side": "buy",
                 "price": 5.00,
                 "qty": 1 - cake_balance,
                 "type" : "FOK" }], current_ts + timedelta(seconds=1)
        
        if cake_balance > -1 and prediction < 0:
            return [
                {"catogory": "linear",
                 "symbol" : "CAKEUSDT", 
                 "side": "sell",
                 "price": 1.00,
                 "qty": cake_balance + 1,
                 "type" : "FOK" }], current_ts + timedelta(seconds=1)

        

        
        return [], current_ts + timedelta(seconds=1)



if __name__ == "__main__":

    hms = HistoricMarketSimulator()
    test_strategy = TestStrategy()


    hms.backtest_strategy2(
        test_strategy,
        datetime(2025,2,1, tzinfo=pytz.UTC),
        datetime(2025,2,2, tzinfo=pytz.UTC)
    )
