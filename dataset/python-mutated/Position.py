from typing import Union
import numpy as np
import jesse.helpers as jh
import jesse.services.selectors as selectors
from jesse.enums import trade_types, order_types
from jesse.exceptions import EmptyPosition, OpenPositionError
from jesse.models import Order, Exchange
from jesse.services import logger
from jesse.utils import sum_floats, subtract_floats

class Position:

    def __init__(self, exchange_name: str, symbol: str, attributes: dict=None) -> None:
        if False:
            return 10
        self.id = jh.generate_unique_id()
        self.entry_price = None
        self.exit_price = None
        self.current_price = None
        self.qty = 0
        self.previous_qty = 0
        self.opened_at = None
        self.closed_at = None
        self._mark_price = None
        self._funding_rate = None
        self._next_funding_timestamp = None
        self._liquidation_price = None
        if attributes is None:
            attributes = {}
        self.exchange_name = exchange_name
        self.exchange: Exchange = selectors.get_exchange(self.exchange_name)
        self.symbol = symbol
        self.strategy = None
        for a in attributes:
            setattr(self, a, attributes[a])

    @property
    def mark_price(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        if not jh.is_live():
            return self.current_price
        if self.exchange_type == 'spot':
            return self.current_price
        return self._mark_price

    @property
    def funding_rate(self) -> float:
        if False:
            return 10
        if not jh.is_live():
            return 0
        if self.exchange_type == 'spot':
            raise ValueError('funding rate is not applicable to spot trading')
        return self._funding_rate

    @property
    def next_funding_timestamp(self) -> Union[int, None]:
        if False:
            i = 10
            return i + 15
        if not jh.is_live():
            return None
        if self.exchange_type == 'spot':
            raise ValueError('funding rate is not applicable to spot trading')
        return self._next_funding_timestamp

    @property
    def value(self) -> float:
        if False:
            print('Hello World!')
        '\n        The value of open position in the quote currency\n\n        :return: float\n        '
        if self.is_close:
            return 0
        if self.current_price is None:
            return None
        return abs(self.current_price * self.qty)

    @property
    def type(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        The type of open position - long, short, or close\n\n        :return: str\n        '
        if self.is_long:
            return 'long'
        elif self.is_short:
            return 'short'
        return 'close'

    @property
    def pnl_percentage(self) -> float:
        if False:
            while True:
                i = 10
        '\n        Alias for self.roi\n\n        :return: float\n        '
        return self.roi

    @property
    def roi(self) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Return on Investment in percentage\n        More at: https://www.binance.com/en/support/faq/5b9ad93cb4854f5990b9fb97c03cfbeb\n        '
        if self.pnl == 0:
            return 0
        return self.pnl / self.total_cost * 100

    @property
    def total_cost(self) -> float:
        if False:
            while True:
                i = 10
        '\n        How much we paid to open this position (currently does not include fees, should we?!)\n        '
        if self.is_close:
            return np.nan
        base_cost = self.entry_price * abs(self.qty)
        if self.strategy:
            return base_cost / self.leverage
        return base_cost

    @property
    def leverage(self) -> Union[int, np.float64]:
        if False:
            for i in range(10):
                print('nop')
        if self.exchange_type == 'spot':
            return 1
        if self.strategy:
            return self.strategy.leverage
        else:
            return np.nan

    @property
    def exchange_type(self) -> str:
        if False:
            while True:
                i = 10
        return self.exchange.type

    @property
    def entry_margin(self) -> float:
        if False:
            return 10
        '\n        Alias for self.total_cost\n        '
        return self.total_cost

    @property
    def pnl(self) -> float:
        if False:
            while True:
                i = 10
        '\n        The PNL of the position\n\n        :return: float\n        '
        if abs(self.qty) < self._min_qty:
            return 0
        if self.entry_price is None:
            return 0
        if self.value is None:
            return 0
        diff = self.value - abs(self.entry_price * self.qty)
        return -diff if self.type == 'short' else diff

    @property
    def is_open(self) -> bool:
        if False:
            return 10
        '\n        Is the current position open?\n\n        :return: bool\n        '
        return self.type in ['long', 'short']

    @property
    def is_close(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Is the current position close?\n\n        :return: bool\n        '
        return self.type == 'close'

    @property
    def is_long(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Is the current position a long position?\n\n        :return: bool\n        '
        return self.qty > self._min_qty

    @property
    def is_short(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Is the current position a short position?\n\n        :return: bool\n        '
        return self.qty < -abs(self._min_qty)

    @property
    def mode(self) -> str:
        if False:
            return 10
        if self.exchange.type == 'spot':
            return 'spot'
        else:
            return self.exchange.futures_leverage_mode

    @property
    def liquidation_price(self) -> Union[float, np.float64]:
        if False:
            i = 10
            return i + 15
        '\n        The price at which the position gets liquidated. formulas are taken from:\n        https://help.bybit.com/hc/en-us/articles/900000181046-Liquidation-Price-USDT-Contract-\n        '
        if self.is_close:
            return np.nan
        if jh.is_livetrading():
            return self._liquidation_price
        if self.mode in ['cross', 'spot']:
            return np.nan
        elif self.mode == 'isolated':
            if self.type == 'long':
                return self.entry_price * (1 - self._initial_margin_rate + 0.004)
            elif self.type == 'short':
                return self.entry_price * (1 + self._initial_margin_rate - 0.004)
            else:
                return np.nan
        else:
            raise ValueError

    @property
    def _initial_margin_rate(self) -> float:
        if False:
            return 10
        return 1 / self.leverage

    @property
    def bankruptcy_price(self) -> Union[float, np.float64]:
        if False:
            for i in range(10):
                print('nop')
        if self.type == 'long':
            return self.entry_price * (1 - self._initial_margin_rate)
        elif self.type == 'short':
            return self.entry_price * (1 + self._initial_margin_rate)
        else:
            return np.nan

    @property
    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'entry_price': self.entry_price, 'qty': self.qty, 'current_price': self.current_price, 'value': self.value, 'type': self.type, 'exchange': self.exchange_name, 'pnl': self.pnl, 'pnl_percentage': self.pnl_percentage, 'leverage': self.leverage, 'liquidation_price': self.liquidation_price, 'bankruptcy_price': self.bankruptcy_price, 'mode': self.mode}

    def _mutating_close(self, close_price: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.is_close and self._can_mutate_qty:
            raise EmptyPosition('The position is already closed.')
        self.exit_price = close_price
        self.closed_at = jh.now_to_timestamp()
        if self.exchange and self.exchange.type == 'futures':
            close_qty = abs(self.qty)
            estimated_profit = jh.estimate_PNL(close_qty, self.entry_price, close_price, self.type)
            self.exchange.add_realized_pnl(estimated_profit)
            self.exchange.temp_reduced_amount[jh.base_asset(self.symbol)] += abs(close_qty * close_price)
        if self._can_mutate_qty:
            self._update_qty(0, operation='set')
        self.entry_price = None
        self._close()

    def _close(self):
        if False:
            return 10
        from jesse.store import store
        store.completed_trades.close_trade(self)

    def _mutating_reduce(self, qty: float, price: float) -> None:
        if False:
            i = 10
            return i + 15
        if not self._can_mutate_qty:
            return
        if self.is_open is False:
            raise EmptyPosition('The position is closed.')
        qty = abs(qty)
        estimated_profit = jh.estimate_PNL(qty, self.entry_price, price, self.type)
        if self.exchange and self.exchange.type == 'futures':
            self.exchange.add_realized_pnl(estimated_profit)
            self.exchange.temp_reduced_amount[jh.base_asset(self.symbol)] += abs(qty * price)
        if self.type == trade_types.LONG:
            self._update_qty(qty, operation='subtract')
        elif self.type == trade_types.SHORT:
            self._update_qty(qty, operation='add')

    def _mutating_increase(self, qty: float, price: float) -> None:
        if False:
            i = 10
            return i + 15
        if not self.is_open:
            raise OpenPositionError('position must be already open in order to increase its size')
        qty = abs(qty)
        self.entry_price = jh.estimate_average_price(qty, price, self.qty, self.entry_price)
        if self._can_mutate_qty:
            if self.type == trade_types.LONG:
                self._update_qty(qty, operation='add')
            elif self.type == trade_types.SHORT:
                self._update_qty(qty, operation='subtract')

    def _mutating_open(self, qty: float, price: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.is_open and self._can_mutate_qty:
            raise OpenPositionError('an already open position cannot be opened')
        self.entry_price = price
        self.exit_price = None
        if self._can_mutate_qty:
            self._update_qty(qty, operation='set')
        self.opened_at = jh.now_to_timestamp()
        self._open()

    def _update_qty(self, qty: float, operation='set'):
        if False:
            while True:
                i = 10
        self.previous_qty = self.qty
        if self.exchange_type == 'spot':
            if operation == 'set':
                self.qty = qty * (1 - self.exchange.fee_rate)
            elif operation == 'add':
                self.qty = sum_floats(self.qty, qty * (1 - self.exchange.fee_rate))
            elif operation == 'subtract':
                self.qty = subtract_floats(self.qty, qty)
        elif self.exchange_type == 'futures':
            if operation == 'set':
                self.qty = qty
            elif operation == 'add':
                self.qty = sum_floats(self.qty, qty)
            elif operation == 'subtract':
                self.qty = subtract_floats(self.qty, qty)
        else:
            raise NotImplementedError('exchange type not implemented')

    def _open(self):
        if False:
            while True:
                i = 10
        from jesse.store import store
        store.completed_trades.open_trade(self)

    def _on_executed_order(self, order: Order) -> None:
        if False:
            while True:
                i = 10
        if jh.is_livetrading() and self.exchange_type == 'futures':
            if order.is_partially_filled:
                before_qty = self.qty - order.filled_qty
            else:
                before_qty = self.qty - order.qty
            after_qty = self.qty
            if before_qty != 0 and after_qty == 0:
                self._close()
        elif jh.is_livetrading() and self.exchange_type == 'spot':
            before_qty = self.previous_qty
            after_qty = self.qty
            qty = order.qty
            price = order.price
            closing_position = before_qty > self._min_qty > after_qty
            if closing_position:
                self._mutating_close(price)
            opening_position = before_qty < self._min_qty < after_qty
            if opening_position:
                self._mutating_open(qty, price)
            increasing_position = after_qty > before_qty > self._min_qty
            if increasing_position:
                self._mutating_increase(qty, price)
            reducing_position = self._min_qty < after_qty < before_qty
            if reducing_position:
                self._mutating_reduce(qty, price)
        else:
            qty = order.qty
            price = order.price
            if self.exchange and self.exchange.type == 'futures':
                self.exchange.charge_fee(qty * price)
            if self.qty == 0:
                change_balance = order.type == order_types.MARKET
                self._mutating_open(qty, price)
            elif sum_floats(self.qty, qty) == 0:
                self._mutating_close(price)
            elif self.qty * qty > 0:
                if order.reduce_only:
                    logger.info('Did not increase position because order is a reduce_only order')
                else:
                    self._mutating_increase(qty, price)
            elif self.qty * qty < 0:
                if abs(qty) > abs(self.qty):
                    if order.reduce_only:
                        logger.info(f'Executed order is bigger than the current position size but it is a reduce_only order so it just closes it. Order QTY: {qty}, Position QTY: {self.qty}')
                        self._mutating_close(price)
                    else:
                        logger.info(f'Executed order is big enough to not close, but flip the position type. Order QTY: {qty}, Position QTY: {self.qty}')
                        diff_qty = sum_floats(self.qty, qty)
                        self._mutating_close(price)
                        self._mutating_open(diff_qty, price)
                else:
                    self._mutating_reduce(qty, price)
        if self.strategy:
            self.strategy._on_updated_position(order)

    def update_from_stream(self, data: dict, is_initial: bool) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Used for updating the position from the WS stream (only for live trading)\n        '
        before_qty = abs(self.qty)
        after_qty = abs(data['qty'])
        if self.exchange_type == 'futures':
            self.entry_price = data['entry_price']
            self._liquidation_price = data['liquidation_price']
        elif after_qty > self._min_qty and self.entry_price is None:
            self.entry_price = self.current_price
        if self.qty != data['qty']:
            self.previous_qty = self.qty
            self.qty = data['qty']
        opening_position = before_qty <= self._min_qty < after_qty
        closing_position = before_qty > self._min_qty >= after_qty
        if opening_position:
            if is_initial:
                from jesse.store import store
                store.completed_trades.add_order_record_only(self.exchange_name, self.symbol, jh.type_to_side(self.type), self.qty, self.entry_price)
            self.opened_at = jh.now_to_timestamp()
            self._open()
        elif closing_position:
            self.closed_at = jh.now_to_timestamp()

    @property
    def _min_notional_size(self) -> float:
        if False:
            while True:
                i = 10
        if not (jh.is_livetrading() and self.exchange_type == 'spot'):
            return 0
        return self.exchange.vars['precisions'][self.symbol]['min_notional_size']

    @property
    def _min_qty(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        if not (jh.is_livetrading() and self.exchange_type == 'spot'):
            return 0
        min_notional_size = self._min_notional_size
        if min_notional_size is None:
            return self.exchange.vars['precisions'][self.symbol]['min_qty']
        return self._min_notional_size / self.current_price

    @property
    def _can_mutate_qty(self):
        if False:
            while True:
                i = 10
        return not (self.exchange_type == 'spot' and jh.is_livetrading())