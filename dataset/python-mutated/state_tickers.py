from typing import List
import numpy as np
import jesse.helpers as jh
from jesse.services import selectors
from jesse.libs import DynamicNumpyArray
from jesse.models import store_ticker_into_db, Ticker

class TickersState:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.storage = {}

    def init_storage(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for ar in selectors.get_all_routes():
            (exchange, symbol) = (ar['exchange'], ar['symbol'])
            key = jh.key(exchange, symbol)
            self.storage[key] = DynamicNumpyArray((60, 5), drop_at=120)

    def add_ticker(self, ticker: np.ndarray, exchange: str, symbol: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        key = jh.key(exchange, symbol)
        if len(self.storage[key][:]) == 0 or jh.now_to_timestamp() - self.storage[key][-1][0] >= 1000:
            self.storage[key].append(ticker)
            if jh.is_collecting_data():
                store_ticker_into_db(exchange, symbol, ticker)
                return

    def get_tickers(self, exchange: str, symbol: str) -> List[Ticker]:
        if False:
            i = 10
            return i + 15
        key = jh.key(exchange, symbol)
        return self.storage[key][:]

    def get_current_ticker(self, exchange: str, symbol: str) -> Ticker:
        if False:
            return 10
        key = jh.key(exchange, symbol)
        return self.storage[key][-1]

    def get_past_ticker(self, exchange: str, symbol: str, number_of_tickers_ago: int) -> Ticker:
        if False:
            for i in range(10):
                print('nop')
        if number_of_tickers_ago > 120:
            raise ValueError('Max accepted value for number_of_tickers_ago is 120')
        number_of_tickers_ago = abs(number_of_tickers_ago)
        key = jh.key(exchange, symbol)
        return self.storage[key][-1 - number_of_tickers_ago]