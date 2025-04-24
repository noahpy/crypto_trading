from backtesting.Strategy import Strategy
from datetime import datetime, timedelta
import pandas as pd
import json


class PeriodicDataLoader():

    def __init__(self, data_requrements, start_date):
        self.data_requrements = data_requrements
        self.start_date = start_date
        

        

    def get_data(self):
    


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

    def match_orders(self, bids, asks, orders) -> int:
        
        min_ask = min(asks.keys())
        max_bid = max(bids.keys())

        for order in orders:
            if order["side"] == "buy":
                if order["price"] >= min_ask:
                    match_quantity = min(order["quantity"], asks[min_ask])
                    match_size = match_quantity * min_ask
                    return match_quantity, -match_size

                else:
                    return 0, 0
            
            if order["side"] == "sell":
                if order["price"] <= max_bid:
                    match_quantity = min(order["quantity"], bids[max_bid])
                    match_size = match_quantity * max_bid
                    return -match_quantity, match_size
                else:
                    return 0, 0

        return 0, 0

    
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





    def backtest_strategy2(
            self,
            strategy: Strategy,
            start_date: datetime,
            end_date: datetime):
        
        
        current_ts = start_date

        data_requirements, tradable_assets, next_ts = strategy.set_up()
        
        periodic_dl = PeriodicDataLoader(data_requirements, current_ts)
        self.initialize_positions(tradable_assets)
        
        print(f"positions: {self.positions}")
        print(f"next_ts: {next_ts}")

        return None

        while current_ts <= end_date:

            data = periodic_dl.get(next_ts)

            orders = strategy.get_orders(data, next_ts)

            self.update




        
        return None




    def backtest_strategy(
            self,
            strategy: Strategy,
            start_date: datetime,
            end_date: datetime):
        

        
        category = "linear"

        curr_date = start_date
        next_ts, coins = strategy.set_up(curr_date)
        coin = coins[0]
        print(next_ts)

        coin_position = 0
        cash_position = 0

        while curr_date <= end_date:
            trade_id = 0

            print(f"{curr_date.year}-{self.zero_pad(curr_date.month)}-{self.zero_pad(curr_date.day)}")

            filepath_ob = f"{self.folder}/ob/{category}/{coin}/{curr_date.year}-{self.zero_pad(curr_date.month)}-{self.zero_pad(curr_date.day)}_{coin}_ob500.data"
            filepath_trades = f"{self.folder}/td/{coin}/{coin}{curr_date.year}-{self.zero_pad(curr_date.month)}-{self.zero_pad(curr_date.day)}.csv.gz"

            hist_trade_data = pd.read_csv(filepath_trades, compression='gzip')
            current_bids = None
            current_asks = None
            current_trades = []

            with open(filepath_ob, 'rb') as f:
                for line_number, line in enumerate(f, 1):

                    # read data from line
                    data = json.loads(line.decode('utf-8'))
                    ts = datetime.fromtimestamp(data['ts']/1000)
                    new_bids = {float(price): float(size)
                                for price, size in data['data']['b']}
                    new_asks = {float(price): float(size)
                                for price, size in data['data']['a']}

                    # update order book
                    current_bids, current_asks = self.update_order_book(
                        current_bids, current_asks,
                        new_bids, new_asks,
                        data['type'])

                    # collect trades
                    while trade_id < len(hist_trade_data) and hist_trade_data.iloc[trade_id]["timestamp"] <= data["ts"]/1000:
                        trade = hist_trade_data.iloc[trade_id]
                        current_trades.append(trade)
                        trade_id += 1

                    # skip until next_ts is reached
                    if ts < next_ts:
                        continue
                    
                    # store formatted data
                    mid_price = (max(current_bids.keys()) +
                                 min(current_asks.keys()))/2
                    # print(mid_price)

                    snapshot = {
                        'timestamp': ts,
                        'mid_price': mid_price,
                        'bids': current_bids.copy(),
                        'asks': current_asks.copy(),
                        'trades': current_trades
                    }
                    
                    
                    orders, next_ts = strategy.get_orders(snapshot, coin_position)
                    
                    coin_change, cash_change = self.match_orders(current_bids, current_asks, orders)

                    coin_position += coin_change
                    cash_position += cash_change
                    
                    if coin_change != 0:
                        print(orders)
                        print(ts)
                        print(f"coin_position: {coin_position}")
                        print(f"cash_position: {cash_position}")

                    
                    current_trades = []

                    

            curr_date += timedelta(days=1)



class TestStrategy(Strategy):

    def set_up(self):
        
        data_requirements = {"linear": {"CAKEUSDT": ["bids", "asks", "trades"]}}
        tradeable_assets = [{"category": "linear", "symbol": "CAKEUSDT", "base": "USDT"}]
        
        return data_requirements, tradeable_assets, datetime(2025,1,1)

    def get_orders(self, data, positions, unmatched_orders):
        return super().get_orders(data, positions, unmatched_orders)



if __name__ == "__main__":


    hms = HistoricMarketSimulator()
    test_strategy = TestStrategy()

    hms.backtest_strategy2(
        test_strategy,
        datetime(2025,1,1),
        datetime(2025,1,2)
    )
