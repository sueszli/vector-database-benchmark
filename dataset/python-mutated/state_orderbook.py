import numpy as np
import jesse.helpers as jh
from jesse.services import selectors
from jesse.libs import DynamicNumpyArray
from jesse.models import store_orderbook_into_db

class OrderbookState:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.storage = {}
        self.temp_storage = {}

    def init_storage(self) -> None:
        if False:
            print('Hello World!')
        for ar in selectors.get_all_routes():
            (exchange, symbol) = (ar['exchange'], ar['symbol'])
            key = jh.key(exchange, symbol)
            self.temp_storage[key] = {'last_updated_timestamp': None, 'asks': [], 'bids': []}
            self.storage[key] = DynamicNumpyArray((60, 2, 50, 2), drop_at=60)

    def format_orderbook(self, exchange: str, symbol: str) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        key = jh.key(exchange, symbol)
        asks = _trim_orderbook_list(self.temp_storage[key]['asks'], ascending=True)
        bids = _trim_orderbook_list(self.temp_storage[key]['bids'], ascending=False)
        asks = _fix_array_len(np.array(asks), 50)
        bids = _fix_array_len(np.array(bids), 50)
        return np.array([asks, bids])

    def add_orderbook(self, exchange: str, symbol: str, asks: list, bids: list) -> None:
        if False:
            for i in range(10):
                print('nop')
        key = jh.key(exchange, symbol)
        self.temp_storage[key]['asks'] = asks
        self.temp_storage[key]['bids'] = bids
        if self.temp_storage[key]['last_updated_timestamp'] is None or jh.now_to_timestamp() - self.temp_storage[key]['last_updated_timestamp'] >= 1000:
            self.temp_storage[key]['last_updated_timestamp'] = jh.now_to_timestamp()
            formatted_orderbook = self.format_orderbook(exchange, symbol)
            if jh.is_collecting_data():
                store_orderbook_into_db(exchange, symbol, formatted_orderbook)
            else:
                self.storage[key].append(formatted_orderbook)

    def get_current_orderbook(self, exchange: str, symbol: str) -> np.ndarray:
        if False:
            while True:
                i = 10
        key = jh.key(exchange, symbol)
        return self.storage[key][-1]

    def get_current_asks(self, exchange: str, symbol: str) -> np.ndarray:
        if False:
            print('Hello World!')
        key = jh.key(exchange, symbol)
        return self.storage[key][-1][0]

    def get_best_ask(self, exchange: str, symbol: str) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        key = jh.key(exchange, symbol)
        return self.storage[key][-1][0][0]

    def get_current_bids(self, exchange: str, symbol: str) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        key = jh.key(exchange, symbol)
        return self.storage[key][-1][1]

    def get_best_bid(self, exchange: str, symbol: str) -> np.ndarray:
        if False:
            return 10
        key = jh.key(exchange, symbol)
        return self.storage[key][-1][1][0]

    def get_orderbooks(self, exchange: str, symbol: str) -> np.ndarray:
        if False:
            while True:
                i = 10
        key = jh.key(exchange, symbol)
        return self.storage[key][:]

def _trim_orderbook_list(arr: list, ascending: bool, limit_len: int=50) -> list:
    if False:
        return 10
    'trims prices up to 4 digits precision'
    first_price = arr[0][0]
    if first_price < 0.1:
        unit = 1e-05
    elif first_price < 1:
        unit = 0.0001
    elif first_price < 10:
        unit = 0.001
    elif first_price < 100:
        unit = 0.01
    elif first_price < 1000:
        unit = 0.1
    elif first_price < 10000:
        unit = 1
    else:
        unit = 10
    trimmed_price = jh.orderbook_trim_price(first_price, ascending, unit)
    temp_qty = 0
    trimmed_arr = []
    for a in arr:
        if len(trimmed_arr) == limit_len:
            break
        if ascending and a[0] > trimmed_price or (not ascending and a[0] < trimmed_price):
            trimmed_arr.append([trimmed_price, temp_qty])
            temp_qty = a[1]
            trimmed_price = jh.orderbook_trim_price(a[0], ascending, unit)
        else:
            temp_qty += a[1]
    return trimmed_arr

def _fix_array_len(arr: np.ndarray, target_len: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    'make sure bids and asks have the same length'
    missing_len = target_len - len(arr)
    if missing_len < 0:
        raise ValueError(f"len cannot be smaller than array's length. {target_len} sent, while array has {len(arr)} items")
    if not missing_len:
        return arr
    return np.array([*arr, *np.full((missing_len, *arr.shape[1:]), np.nan)])