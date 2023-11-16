import jesse.helpers as jh
from jesse.enums import sides
from jesse.exceptions import InsufficientBalance
from jesse.models import Order
from jesse.models.Exchange import Exchange
from jesse.enums import order_types
from jesse.utils import sum_floats, subtract_floats

class SpotExchange(Exchange):

    def __init__(self, name: str, starting_balance: float, fee_rate: float):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(name, starting_balance, fee_rate, 'spot')
        self.stop_orders_sum = {}
        self.limit_orders_sum = {}
        self._started_balance = 0

    @property
    def started_balance(self) -> float:
        if False:
            i = 10
            return i + 15
        if jh.is_livetrading():
            return self._started_balance
        return self.starting_assets[jh.app_currency()]

    @property
    def wallet_balance(self) -> float:
        if False:
            i = 10
            return i + 15
        return self.assets[self.settlement_currency]

    @property
    def available_margin(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return self.wallet_balance

    def on_order_submission(self, order: Order) -> None:
        if False:
            i = 10
            return i + 15
        if jh.is_livetrading():
            return
        if order.side == sides.SELL:
            if order.type == order_types.STOP:
                self.stop_orders_sum[order.symbol] = sum_floats(self.stop_orders_sum.get(order.symbol, 0), abs(order.qty))
            elif order.type == order_types.LIMIT:
                self.limit_orders_sum[order.symbol] = sum_floats(self.limit_orders_sum.get(order.symbol, 0), abs(order.qty))
        base_asset = jh.base_asset(order.symbol)
        if order.side == sides.BUY:
            quote_balance = self.assets[self.settlement_currency]
            self.assets[self.settlement_currency] = subtract_floats(self.assets[self.settlement_currency], abs(order.qty) * order.price)
            if self.assets[self.settlement_currency] < 0:
                raise InsufficientBalance(f"Not enough balance. Available balance at {self.name} for {self.settlement_currency} is {quote_balance} but you're trying to spend {abs(order.qty * order.price)}")
        else:
            base_balance = self.assets[base_asset]
            if order.type == order_types.MARKET:
                order_qty = sum_floats(abs(order.qty), self.limit_orders_sum.get(order.symbol, 0))
            elif order.type == order_types.STOP:
                order_qty = self.stop_orders_sum[order.symbol]
            elif order.type == order_types.LIMIT:
                order_qty = self.limit_orders_sum[order.symbol]
            else:
                raise Exception(f'Unknown order type {order.type}')
            if order_qty > base_balance:
                raise InsufficientBalance(f"Not enough balance. Available balance at {self.name} for {base_asset} is {base_balance} but you're trying to sell {order_qty}")

    def on_order_execution(self, order: Order) -> None:
        if False:
            print('Hello World!')
        if jh.is_livetrading():
            return
        if order.side == sides.SELL:
            if order.type == order_types.STOP:
                self.stop_orders_sum[order.symbol] = subtract_floats(self.stop_orders_sum[order.symbol], abs(order.qty))
            elif order.type == order_types.LIMIT:
                self.limit_orders_sum[order.symbol] = subtract_floats(self.limit_orders_sum[order.symbol], abs(order.qty))
        base_asset = jh.base_asset(order.symbol)
        if order.side == sides.BUY:
            self.assets[base_asset] = sum_floats(self.assets[base_asset], abs(order.qty) * (1 - self.fee_rate))
        else:
            self.assets[self.settlement_currency] = sum_floats(self.assets[self.settlement_currency], abs(order.qty) * order.price * (1 - self.fee_rate))
            self.assets[base_asset] = subtract_floats(self.assets[base_asset], abs(order.qty))

    def on_order_cancellation(self, order: Order) -> None:
        if False:
            for i in range(10):
                print('nop')
        if jh.is_livetrading():
            return
        if order.side == sides.SELL:
            if order.type == order_types.STOP:
                self.stop_orders_sum[order.symbol] = subtract_floats(self.stop_orders_sum[order.symbol], abs(order.qty))
            elif order.type == order_types.LIMIT:
                self.limit_orders_sum[order.symbol] = subtract_floats(self.limit_orders_sum[order.symbol], abs(order.qty))
        base_asset = jh.base_asset(order.symbol)
        if order.side == sides.BUY:
            self.assets[self.settlement_currency] = sum_floats(self.assets[self.settlement_currency], abs(order.qty) * order.price)
        else:
            self.assets[base_asset] = sum_floats(self.assets[base_asset], abs(order.qty))

    def update_from_stream(self, data: dict) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Used for updating the exchange from the WS stream (only for live trading)\n        '
        if not jh.is_livetrading():
            raise Exception('This method is only for live trading')
        self.assets[self.settlement_currency] = data['balance']
        if self._started_balance == 0:
            self._started_balance = data['balance']