from typing import List
import pydash
from jesse.config import config
from jesse.models import Order
from jesse.services import selectors
import jesse.helpers as jh

class OrdersState:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.to_execute = []
        self.storage = {}
        for exchange in config['app']['trading_exchanges']:
            for symbol in config['app']['trading_symbols']:
                key = f'{exchange}-{symbol}'
                self.storage[key] = []

    def reset(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        used for testing\n        '
        for key in self.storage:
            self.storage[key].clear()

    def reset_trade_orders(self, exchange: str, symbol: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        used after each completed trade\n        '
        key = f'{exchange}-{symbol}'
        self.storage[key] = []

    def add_order(self, order: Order) -> None:
        if False:
            print('Hello World!')
        key = f'{order.exchange}-{order.symbol}'
        self.storage[key].append(order)

    def remove_order(self, order: Order) -> None:
        if False:
            return 10
        key = f'{order.exchange}-{order.symbol}'
        self.storage[key] = [o for o in self.storage[key] if o.id != order.id]

    def execute_pending_market_orders(self) -> None:
        if False:
            while True:
                i = 10
        if not self.to_execute:
            return
        for o in self.to_execute:
            o.execute()
        self.to_execute = []

    def get_orders(self, exchange, symbol) -> List[Order]:
        if False:
            return 10
        key = f'{exchange}-{symbol}'
        return self.storage.get(key, [])

    def get_all_orders(self, exchange: str) -> List[Order]:
        if False:
            i = 10
            return i + 15
        return [o for key in self.storage for o in self.storage[key] if o.exchange == exchange]

    def count_all_active_orders(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        c = 0
        for key in self.storage:
            if len(self.storage[key]):
                for o in self.storage[key]:
                    if o.is_active:
                        c += 1
        return c

    def count_active_orders(self, exchange: str, symbol: str) -> int:
        if False:
            return 10
        orders = self.get_orders(exchange, symbol)
        return sum((bool(o.is_active) for o in orders))

    def count(self, exchange: str, symbol: str) -> int:
        if False:
            print('Hello World!')
        return len(self.get_orders(exchange, symbol))

    def get_order_by_id(self, exchange: str, symbol: str, id: str, use_exchange_id: bool=False) -> Order:
        if False:
            for i in range(10):
                print('nop')
        key = f'{exchange}-{symbol}'
        if use_exchange_id:
            return pydash.find(self.storage[key], lambda o: o.exchange_id == id)
        return pydash.find(self.storage[key], lambda o: o.id == id)

    def get_entry_orders(self, exchange: str, symbol: str) -> List[Order]:
        if False:
            print('Hello World!')
        all_orders = self.get_orders(exchange, symbol)
        if len(all_orders) == 0:
            return []
        p = selectors.get_position(exchange, symbol)
        if p.is_close:
            entry_orders = all_orders.copy()
        else:
            entry_orders = [o for o in all_orders if o.side == jh.type_to_side(p.type)]
        entry_orders = [o for o in entry_orders if not o.is_canceled]
        return entry_orders

    def get_exit_orders(self, exchange: str, symbol: str) -> List[Order]:
        if False:
            return 10
        '\n        excludes cancel orders but includes executed orders\n        '
        all_orders = self.get_orders(exchange, symbol)
        if len(all_orders) == 0:
            return []
        p = selectors.get_position(exchange, symbol)
        if p.is_close:
            return []
        else:
            exit_orders = [o for o in all_orders if o.side != jh.type_to_side(p.type)]
        exit_orders = [o for o in exit_orders if not o.is_canceled]
        return exit_orders