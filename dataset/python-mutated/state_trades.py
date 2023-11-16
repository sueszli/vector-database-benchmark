from typing import List
import numpy as np
import jesse.helpers as jh
from jesse.config import config
from jesse.libs import DynamicNumpyArray
from jesse.models import store_trade_into_db
from jesse.models.Trade import Trade
from jesse.services import selectors

class TradesState:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.storage = {}
        self.temp_storage = {}

    def init_storage(self) -> None:
        if False:
            return 10
        for ar in selectors.get_all_routes():
            (exchange, symbol) = (ar['exchange'], ar['symbol'])
            key = jh.key(exchange, symbol)
            self.storage[key] = DynamicNumpyArray((60, 6), drop_at=120)
            self.temp_storage[key] = DynamicNumpyArray((100, 4))

    def add_trade(self, trade: np.ndarray, exchange: str, symbol: str) -> None:
        if False:
            i = 10
            return i + 15
        key = jh.key(exchange, symbol)
        if len(self.temp_storage[key]) and trade[0] - self.temp_storage[key][0][0] >= 1000:
            arr = self.temp_storage[key]
            buy_arr = np.array(list(filter(lambda x: x[3] == 1, arr)))
            sell_arr = np.array(list(filter(lambda x: x[3] == 0, arr)))
            generated = np.array([arr[0][0], (arr[:][:, 1] * arr[:][:, 2]).sum() / arr[:][:, 2].sum(), 0 if not len(buy_arr) else buy_arr[:, 2].sum(), 0 if not len(sell_arr) else sell_arr[:, 2].sum(), len(buy_arr), len(sell_arr)])
            if jh.is_collecting_data():
                store_trade_into_db(exchange, symbol, generated)
            else:
                self.storage[key].append(generated)
            self.temp_storage[key].flush()
        self.temp_storage[key].append(trade)

    def get_trades(self, exchange: str, symbol: str) -> List[Trade]:
        if False:
            for i in range(10):
                print('nop')
        key = jh.key(exchange, symbol)
        return self.storage[key][:]

    def get_current_trade(self, exchange: str, symbol: str) -> Trade:
        if False:
            while True:
                i = 10
        key = jh.key(exchange, symbol)
        return self.storage[key][-1]

    def get_past_trade(self, exchange: str, symbol: str, number_of_trades_ago: int) -> Trade:
        if False:
            return 10
        if number_of_trades_ago > 120:
            raise ValueError('Max accepted value for number_of_trades_ago is 120')
        number_of_trades_ago = abs(number_of_trades_ago)
        key = jh.key(exchange, symbol)
        return self.storage[key][-1 - number_of_trades_ago]