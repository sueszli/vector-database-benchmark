from abc import ABC, abstractmethod
from jesse.models import Order
from jesse.services import selectors
import jesse.helpers as jh
from jesse.libs import DynamicNumpyArray

class Exchange(ABC):

    def __init__(self, name: str, starting_balance: float, fee_rate: float, exchange_type: str):
        if False:
            while True:
                i = 10
        self.assets = {}
        self.temp_reduced_amount = {}
        self.starting_assets = {}
        self.available_assets = {}
        self.fee_rate = fee_rate
        self.vars = {}
        self.buy_orders = {}
        self.sell_orders = {}
        self.name = name
        self.type = exchange_type.lower()
        self.starting_balance = starting_balance
        all_trading_routes = selectors.get_all_trading_routes()
        first_route = all_trading_routes[0]
        self.settlement_currency = jh.quote_asset(first_route.symbol)
        for r in all_trading_routes:
            base_asset = jh.base_asset(r.symbol)
            self.buy_orders[base_asset] = DynamicNumpyArray((10, 2))
            self.sell_orders[base_asset] = DynamicNumpyArray((10, 2))
            self.assets[base_asset] = 0.0
            self.assets[self.settlement_currency] = starting_balance
            self.temp_reduced_amount[base_asset] = 0.0
            self.temp_reduced_amount[self.settlement_currency] = 0.0
            self.starting_assets[base_asset] = 0.0
            self.starting_assets[self.settlement_currency] = starting_balance
            self.available_assets[base_asset] = 0.0
            self.available_assets[self.settlement_currency] = starting_balance

    @property
    @abstractmethod
    def wallet_balance(self) -> float:
        if False:
            print('Hello World!')
        pass

    @property
    @abstractmethod
    def available_margin(self) -> float:
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def on_order_submission(self, order: Order) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def on_order_execution(self, order: Order) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def on_order_cancellation(self, order: Order) -> None:
        if False:
            i = 10
            return i + 15
        pass