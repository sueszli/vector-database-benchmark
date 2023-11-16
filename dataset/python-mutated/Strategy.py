from abc import ABC, abstractmethod
from time import sleep
from typing import List, Dict, Union
import numpy as np
import jesse.helpers as jh
import jesse.services.logger as logger
import jesse.services.selectors as selectors
from jesse import exceptions
from jesse.enums import sides, order_submitted_via, order_types
from jesse.models import ClosedTrade, Order, Route, FuturesExchange, SpotExchange, Position
from jesse.services import metrics
from jesse.services.broker import Broker
from jesse.store import store
from jesse.services.cache import cached
from jesse.services import notifier

class Strategy(ABC):
    """
    The parent strategy class which every strategy must extend. It is the heart of the framework!
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self.id = jh.generate_unique_id()
        self.name = None
        self.symbol = None
        self.exchange = None
        self.timeframe = None
        self.hp = None
        self.index = 0
        self.vars = {}
        self.increased_count = 0
        self.reduced_count = 0
        self.buy = None
        self._buy = None
        self.sell = None
        self._sell = None
        self.stop_loss = None
        self._stop_loss = None
        self.take_profit = None
        self._take_profit = None
        self.trade: ClosedTrade = None
        self.trades_count = 0
        self._is_executing = False
        self._is_initiated = False
        self._is_handling_updated_order = False
        self.position: Position = None
        self.broker = None
        self._cached_methods = {}
        self._cached_metrics = {}

    def _init_objects(self) -> None:
        if False:
            while True:
                i = 10
        '\n        This method gets called after right creating the Strategy object. It\n        is just a workaround as a part of not being able to set them inside\n        self.__init__() for the purpose of removing __init__() methods from strategies.\n        '
        self.position = selectors.get_position(self.exchange, self.symbol)
        self.broker = Broker(self.position, self.exchange, self.symbol, self.timeframe)
        if self.hp is None and len(self.hyperparameters()) > 0:
            self.hp = {}
            for dna in self.hyperparameters():
                self.hp[dna['name']] = dna['default']

    @property
    def _price_precision(self) -> int:
        if False:
            return 10
        '\n        used when live trading because few exchanges require numbers to have a specific precision\n        '
        return selectors.get_exchange(self.exchange).vars['precisions'][self.symbol]['price_precision']

    @property
    def _qty_precision(self) -> int:
        if False:
            while True:
                i = 10
        '\n        used when live trading because few exchanges require numbers to have a specific precision\n        '
        return selectors.get_exchange(self.exchange).vars['precisions'][self.symbol]['qty_precision']

    def _broadcast(self, msg: str) -> None:
        if False:
            while True:
                i = 10
        'Broadcasts the event to all OTHER strategies\n\n        Arguments:\n            msg {str} -- [the message to broadcast]\n        '
        from jesse.routes import router
        for r in router.routes:
            if r.strategy.id == self.id:
                continue
            if msg == 'route-open-position':
                r.strategy.on_route_open_position(self)
            elif msg == 'route-close-position':
                r.strategy.on_route_close_position(self)
            elif msg == 'route-increased-position':
                r.strategy.on_route_increased_position(self)
            elif msg == 'route-reduced-position':
                r.strategy.on_route_reduced_position(self)
            elif msg == 'route-canceled':
                r.strategy.on_route_canceled(self)
            r.strategy._detect_and_handle_entry_and_exit_modifications()

    def _on_updated_position(self, order: Order) -> None:
        if False:
            return 10
        '\n        Handles the after-effect of the executed order to execute strategy\n        events. Note that it assumes that the position has already\n        been affected by the executed order.\n        '
        self._is_handling_updated_order = True
        before_qty = self.position.previous_qty
        after_qty = self.position.qty
        if abs(before_qty) <= abs(self.position._min_qty) < abs(after_qty):
            effect = 'opening_position'
        elif abs(before_qty) > abs(self.position._min_qty) >= abs(after_qty):
            effect = 'closing_position'
        elif abs(after_qty) > abs(before_qty):
            effect = 'increased_position'
        else:
            effect = 'reduced_position'
        if effect == 'opening_position':
            txt = f'OPENED {self.position.type} position for {self.symbol}: qty: {after_qty}, entry_price: {self.position.entry_price}'
            if jh.is_debuggable('position_opened'):
                logger.info(txt)
            if jh.is_live() and jh.get_config('env.notifications.events.updated_position'):
                notifier.notify(txt)
            self._on_open_position(order)
        elif effect == 'closing_position':
            txt = f'CLOSED Position for {self.symbol}'
            if jh.is_debuggable('position_closed'):
                logger.info(txt)
            if jh.is_live() and jh.get_config('env.notifications.events.updated_position'):
                notifier.notify(txt)
            self._on_close_position(order)
        elif effect == 'increased_position':
            txt = f'INCREASED Position size to {after_qty}'
            if jh.is_debuggable('position_increased'):
                logger.info(txt)
            if jh.is_live() and jh.get_config('env.notifications.events.updated_position'):
                notifier.notify(txt)
            self._on_increased_position(order)
        else:
            txt = f'REDUCED Position size to {after_qty}'
            if jh.is_debuggable('position_reduced'):
                logger.info(txt)
            if jh.is_live() and jh.get_config('env.notifications.events.updated_position'):
                notifier.notify(txt)
            self._on_reduced_position(order)
        self._is_handling_updated_order = False

    def filters(self) -> list:
        if False:
            i = 10
            return i + 15
        return []

    def hyperparameters(self) -> list:
        if False:
            print('Hello World!')
        return []

    def dna(self) -> str:
        if False:
            print('Hello World!')
        return ''

    def _execute_long(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.go_long()
        if self.buy is None:
            raise exceptions.InvalidStrategy('You forgot to set self.buy. example (qty, price)')
        elif type(self.buy) not in [tuple, list]:
            raise exceptions.InvalidStrategy(f'self.buy must be either a list or a tuple. example: (qty, price). You set: {type(self.buy)}')
        self._prepare_buy()
        if self.take_profit is not None:
            if self.exchange_type == 'spot':
                raise exceptions.InvalidStrategy("Setting self.take_profit in the go_long() method is not supported for spot trading (it's only supported in futures trading). Try setting it in self.on_open_position() instead.")
            self._validate_take_profit()
            self._prepare_take_profit()
        if self.stop_loss is not None:
            if self.exchange_type == 'spot':
                raise exceptions.InvalidStrategy("Setting self.stop_loss in the go_long() method is not supported for spot trading (it's only supported in futures trading). Try setting it in self.on_open_position() instead.")
            self._validate_stop_loss()
            self._prepare_stop_loss()
        if not self._execute_filters():
            return
        self._submit_buy_orders()

    def _submit_buy_orders(self) -> None:
        if False:
            return 10
        if jh.is_livetrading():
            price_to_compare = jh.round_price_for_live_mode(self.price, selectors.get_exchange(self.exchange).vars['precisions'][self.symbol]['price_precision'])
        else:
            price_to_compare = self.price
        for o in self._buy:
            if jh.is_price_near(o[1], price_to_compare):
                self.broker.buy_at_market(o[0])
            elif o[1] > price_to_compare:
                self.broker.start_profit_at(sides.BUY, o[0], o[1])
            elif o[1] < price_to_compare:
                self.broker.buy_at(o[0], o[1])
            else:
                raise ValueError(f'Invalid order price: o[1]:{o[1]}, self.price:{self.price}')

    def _submit_sell_orders(self) -> None:
        if False:
            print('Hello World!')
        if jh.is_livetrading():
            price_to_compare = jh.round_price_for_live_mode(self.price, selectors.get_exchange(self.exchange).vars['precisions'][self.symbol]['price_precision'])
        else:
            price_to_compare = self.price
        for o in self._sell:
            if jh.is_price_near(o[1], price_to_compare):
                self.broker.sell_at_market(o[0])
            elif o[1] < price_to_compare:
                self.broker.start_profit_at(sides.SELL, o[0], o[1])
            elif o[1] > price_to_compare:
                self.broker.sell_at(o[0], o[1])
            else:
                raise ValueError(f'Invalid order price: o[1]:{o[1]}, self.price:{self.price}')

    def _execute_short(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.go_short()
        if self.sell is None:
            raise exceptions.InvalidStrategy('You forgot to set self.sell. example (qty, price)')
        elif type(self.sell) not in [tuple, list]:
            raise exceptions.InvalidStrategy(f'self.sell must be either a list or a tuple. example: (qty, price). You set {type(self.sell)}')
        self._prepare_sell()
        if self.take_profit is not None:
            self._validate_take_profit()
            self._prepare_take_profit()
        if self.stop_loss is not None:
            self._validate_stop_loss()
            self._prepare_stop_loss()
        if not self._execute_filters():
            return
        self._submit_sell_orders()

    def _prepare_buy(self, make_copies: bool=True) -> None:
        if False:
            return 10
        try:
            self.buy = self._get_formatted_order(self.buy)
        except ValueError:
            raise exceptions.InvalidShape(f'The format of self.buy is invalid. \nIt must be either (qty, price) or [(qty, price), (qty, price)] for multiple points; but {self.buy} was given')
        if make_copies:
            self._buy = self.buy.copy()

    def _prepare_sell(self, make_copies: bool=True) -> None:
        if False:
            while True:
                i = 10
        try:
            self.sell = self._get_formatted_order(self.sell)
        except ValueError:
            raise exceptions.InvalidShape(f'The format of self.sell is invalid. \nIt must be either (qty, price) or [(qty, price), (qty, price)] for multiple points; but {self.sell} was given')
        if make_copies:
            self._sell = self.sell.copy()

    def _prepare_stop_loss(self, make_copies: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            self.stop_loss = self._get_formatted_order(self.stop_loss)
        except ValueError:
            raise exceptions.InvalidShape(f'The format of self.stop_loss is invalid. \nIt must be either (qty, price) or [(qty, price), (qty, price)] for multiple points; but {self.stop_loss} was given')
        if make_copies:
            self._stop_loss = self.stop_loss.copy()

    def _prepare_take_profit(self, make_copies: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        try:
            self.take_profit = self._get_formatted_order(self.take_profit)
        except ValueError:
            raise exceptions.InvalidShape(f'The format of self.take_profit is invalid. \nIt must be either (qty, price) or [(qty, price), (qty, price)] for multiple points; but {self.take_profit} was given')
        if make_copies:
            self._take_profit = self.take_profit.copy()

    def _validate_stop_loss(self) -> None:
        if False:
            while True:
                i = 10
        if self.stop_loss is None:
            raise exceptions.InvalidStrategy('You forgot to set self.stop_loss. example (qty, price)')
        elif type(self.stop_loss) not in [tuple, list, np.ndarray]:
            raise exceptions.InvalidStrategy(f'self.stop_loss must be either a list or a tuple. example: (qty, price). You set {type(self.stop_loss)}')

    def _validate_take_profit(self) -> None:
        if False:
            print('Hello World!')
        if self.take_profit is None:
            raise exceptions.InvalidStrategy('You forgot to set self.take_profit. example (qty, price)')
        elif type(self.take_profit) not in [tuple, list, np.ndarray]:
            raise exceptions.InvalidStrategy(f'self.take_profit must be either a list or a tuple. example: (qty, price). You set {type(self.take_profit)}')

    def _execute_filters(self) -> bool:
        if False:
            i = 10
            return i + 15
        for f in self.filters():
            try:
                passed = f()
            except TypeError:
                raise exceptions.InvalidStrategy('Invalid filter format. You need to pass filter methods WITHOUT calling them (no parentheses must be present at the end)\n\n❌ ' + 'Incorrect Example:\nreturn [\n    self.filter_1()\n]\n\n✅ ' + 'Correct Example:\nreturn [\n    self.filter_1\n]\n')
            if not passed:
                logger.info(f.__name__)
                self._reset()
                return False
        return True

    @abstractmethod
    def go_long(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def go_short(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def _execute_cancel(self) -> None:
        if False:
            while True:
                i = 10
        '\n        cancels everything so that the strategy can keep looking for new trades.\n        '
        if self.position.is_open:
            raise Exception('cannot cancel orders when position is still open. there must be a bug somewhere.')
        logger.info('cancel all remaining orders to prepare for a fresh start...')
        self.broker.cancel_all_orders()
        self._reset()
        self._broadcast('route-canceled')
        self.on_cancel()
        if not jh.is_unit_testing() and (not jh.is_live()):
            store.orders.storage[f'{self.exchange}-{self.symbol}'].clear()

    def _reset(self) -> None:
        if False:
            while True:
                i = 10
        self.buy = None
        self._buy = None
        self.sell = None
        self._sell = None
        self.stop_loss = None
        self._stop_loss = None
        self.take_profit = None
        self._take_profit = None
        store.orders.reset_trade_orders(self.exchange, self.symbol)
        self.increased_count = 0
        self.reduced_count = 0

    def on_cancel(self) -> None:
        if False:
            while True:
                i = 10
        '\n        what should happen after all active orders have been cancelled\n        '
        pass

    @abstractmethod
    def should_long(self) -> bool:
        if False:
            print('Hello World!')
        pass

    def should_short(self) -> bool:
        if False:
            print('Hello World!')
        return False

    @abstractmethod
    def should_cancel_entry(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        pass

    def before(self) -> None:
        if False:
            return 10
        "\n        Get's executed BEFORE executing the strategy's logic\n        "
        pass

    def after(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Get's executed AFTER executing the strategy's logic\n        "
        pass

    def _update_position(self) -> None:
        if False:
            return 10
        self._wait_until_executing_orders_are_fully_handled()
        if self.position.is_close:
            return
        self.update_position()
        self._detect_and_handle_entry_and_exit_modifications()

    def _detect_and_handle_entry_and_exit_modifications(self) -> None:
        if False:
            print('Hello World!')
        if self.position.is_close:
            return
        try:
            if self.is_long:
                self._prepare_buy(make_copies=False)
                if not np.array_equal(self.buy, self._buy):
                    self._buy = self.buy.copy()
                    for o in self.entry_orders:
                        if o.is_active or o.is_queued:
                            self.broker.cancel_order(o.id)
                    self._submit_buy_orders()
            elif self.is_short:
                self._prepare_sell(make_copies=False)
                if not np.array_equal(self.sell, self._sell):
                    self._sell = self.sell.copy()
                    for o in self.entry_orders:
                        if o.is_active or o.is_queued:
                            self.broker.cancel_order(o.id)
                    self._submit_sell_orders()
            if self.position.is_open and self.take_profit is not None:
                self._validate_take_profit()
                self._prepare_take_profit(False)
                if not np.array_equal(self.take_profit, self._take_profit):
                    self._take_profit = self.take_profit.copy()
                    if len(self._take_profit) == 1:
                        temp_current_price = self.price
                    else:
                        temp_current_price = None
                    for o in self.exit_orders:
                        if o.is_take_profit and (o.is_active or o.is_queued):
                            self.broker.cancel_order(o.id)
                    for o in self._take_profit:
                        if self.position.is_close:
                            logger.info('Position got closed while submitting take-profit orders. Hence, skipping further submissions')
                            break
                        if temp_current_price == o[1]:
                            order_price = self.price
                        else:
                            order_price = o[1]
                        submitted_order: Order = self.broker.reduce_position_at(o[0], order_price, self.price)
                        if submitted_order:
                            submitted_order.submitted_via = order_submitted_via.TAKE_PROFIT
            if self.position.is_open and self.stop_loss is not None:
                self._validate_stop_loss()
                self._prepare_stop_loss(False)
                if not np.array_equal(self.stop_loss, self._stop_loss):
                    self._stop_loss = self.stop_loss.copy()
                    if len(self._stop_loss) == 1:
                        temp_current_price = self.price
                    else:
                        temp_current_price = None
                    for o in self.exit_orders:
                        if o.is_stop_loss and (o.is_active or o.is_queued):
                            self.broker.cancel_order(o.id)
                    for o in self._stop_loss:
                        if self.position.is_close:
                            logger.info('Position got closed while submitting stop-loss orders. Hence, skipping further submissions')
                            break
                        if temp_current_price == o[1]:
                            order_price = self.price
                        else:
                            order_price = o[1]
                        submitted_order: Order = self.broker.reduce_position_at(o[0], order_price, self.price)
                        if submitted_order:
                            submitted_order.submitted_via = order_submitted_via.STOP_LOSS
        except TypeError:
            raise exceptions.InvalidStrategy('Something odd is going on within your strategy causing a TypeError exception. Try running it with the debug mode enabled in a backtest to see what was going on near the end, and fix it.')
        except:
            raise
        if self.position.is_open and (self.stop_loss is not None and self.take_profit is not None) and np.array_equal(self.stop_loss, self.take_profit):
            raise exceptions.InvalidStrategy('stop-loss and take-profit should not be exactly the same. Just use either one of them and it will do.')

    def update_position(self) -> None:
        if False:
            return 10
        pass

    def _wait_until_executing_orders_are_fully_handled(self):
        if False:
            return 10
        if self._is_handling_updated_order:
            logger.info("Stopped strategy execution at this time because we're still handling the result of an executed order. Trying again in 3 seconds...")
            sleep(3)

    def _check(self) -> None:
        if False:
            print('Hello World!')
        '\n        Based on the newly updated info, check if we should take action or not\n        '
        if not self._is_initiated:
            self._is_initiated = True
        self._wait_until_executing_orders_are_fully_handled()
        if jh.is_live() and jh.is_debugging():
            logger.info(f'Executing  {self.name}-{self.exchange}-{self.symbol}-{self.timeframe}')
        if len(self.entry_orders) and self.is_close and self.should_cancel_entry():
            self._execute_cancel()
            if jh.is_live():
                sleep(0.1)
                for _ in range(20):
                    if store.orders.count_active_orders(self.exchange, self.symbol) == 0:
                        break
                    logger.info('sleeping 0.2 more seconds until cancellation is over...')
                    sleep(0.2)
                if store.orders.count_active_orders(self.exchange, self.symbol) != 0:
                    raise exceptions.ExchangeNotResponding('The exchange did not respond as expected for order cancellation')
        if self.position.is_open:
            self._update_position()
            if jh.is_livetrading():
                waiting_counter = 0
                waiting_seconds = 1
                while self._have_any_pending_market_exit_orders():
                    if jh.is_debugging():
                        logger.info(f'Waiting {waiting_seconds} second for pending market exit orders to be handled...')
                    waiting_counter += 1
                    sleep(1)
                    if waiting_counter > 10:
                        raise exceptions.ExchangeNotResponding('The exchange did not respond as expected for order execution')
        self._simulate_market_order_execution()
        if self.position.is_close and self.entry_orders == []:
            self._reset()
            should_short = self.should_short()
            if self.exchange_type == 'spot' and should_short is True:
                raise exceptions.InvalidStrategy('should_short cannot be True if the exchange type is "spot".')
            should_long = self.should_long()
            if should_short and should_long:
                raise exceptions.ConflictingRules('should_short and should_long should not be true at the same time.')
            if should_long:
                self._execute_long()
            elif should_short:
                self._execute_short()

    def _have_any_pending_market_exit_orders(self) -> bool:
        if False:
            return 10
        return any((o.is_active and o.type == order_types.MARKET for o in self.exit_orders))

    @staticmethod
    def _simulate_market_order_execution() -> None:
        if False:
            while True:
                i = 10
        '\n        Simulate market order execution in backtest mode\n        '
        if jh.is_backtesting() or jh.is_unit_testing() or jh.is_paper_trading():
            store.orders.execute_pending_market_orders()

    def _on_open_position(self, order: Order) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.increased_count = 1
        self._broadcast('route-open-position')
        if self.take_profit is not None:
            for o in self._take_profit:
                if self.is_long and o[1] <= self.position.entry_price:
                    submitted_order: Order = self.broker.sell_at_market(o[0])
                    logger.info('The take-profit is below entry-price for long position, so it will be replaced with a market order instead')
                elif self.is_short and o[1] >= self.position.entry_price:
                    submitted_order: Order = self.broker.buy_at_market(o[0])
                    logger.info('The take-profit is above entry-price for a short position, so it will be replaced with a market order instead')
                else:
                    submitted_order: Order = self.broker.reduce_position_at(o[0], o[1], self.price)
                if submitted_order:
                    submitted_order.submitted_via = order_submitted_via.TAKE_PROFIT
        if self.stop_loss is not None:
            for o in self._stop_loss:
                if self.is_long and o[1] >= self.position.entry_price:
                    submitted_order: Order = self.broker.sell_at_market(o[0])
                    logger.info('The stop-loss is above entry-price for long position, so it will be replaced with a market order instead')
                elif self.is_short and o[1] <= self.position.entry_price:
                    submitted_order: Order = self.broker.buy_at_market(o[0])
                    logger.info('The stop-loss is below entry-price for a short position, so it will be replaced with a market order instead')
                else:
                    submitted_order: Order = self.broker.reduce_position_at(o[0], o[1], self.price)
                if submitted_order:
                    submitted_order.submitted_via = order_submitted_via.STOP_LOSS
        self.on_open_position(order)
        self._detect_and_handle_entry_and_exit_modifications()

    def on_open_position(self, order) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        What should happen after the open position order has been executed\n        '
        pass

    def on_close_position(self, order) -> None:
        if False:
            while True:
                i = 10
        '\n        What should happen after the open position order has been executed\n        '
        pass

    def _on_close_position(self, order: Order):
        if False:
            for i in range(10):
                print('nop')
        self._broadcast('route-close-position')
        self._execute_cancel()
        self.on_close_position(order)
        self._detect_and_handle_entry_and_exit_modifications()

    def _on_increased_position(self, order: Order) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.increased_count += 1
        self._broadcast('route-increased-position')
        self.on_increased_position(order)
        self._detect_and_handle_entry_and_exit_modifications()

    def on_increased_position(self, order) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        What should happen after the order (if any) increasing the\n        size of the position is executed. Overwrite it if needed.\n        And leave it be if your strategy doesn't require it\n        "
        pass

    def _on_reduced_position(self, order: Order) -> None:
        if False:
            return 10
        '\n        prepares for on_reduced_position() is implemented by user\n        '
        self.reduced_count += 1
        self._broadcast('route-reduced-position')
        self.on_reduced_position(order)
        self._detect_and_handle_entry_and_exit_modifications()

    def on_reduced_position(self, order) -> None:
        if False:
            print('Hello World!')
        '\n        What should happen after the order (if any) reducing the size of the position is executed.\n        '
        pass

    def on_route_open_position(self, strategy) -> None:
        if False:
            while True:
                i = 10
        'used when trading multiple routes that related\n\n        Arguments:\n            strategy {Strategy} -- the strategy that has fired (and not listening to) the event\n        '
        pass

    def on_route_close_position(self, strategy) -> None:
        if False:
            i = 10
            return i + 15
        'used when trading multiple routes that related\n\n        Arguments:\n            strategy {Strategy} -- the strategy that has fired (and not listening to) the event\n        '
        pass

    def on_route_increased_position(self, strategy) -> None:
        if False:
            for i in range(10):
                print('nop')
        'used when trading multiple routes that related\n\n        Arguments:\n            strategy {Strategy} -- the strategy that has fired (and not listening to) the event\n        '
        pass

    def on_route_reduced_position(self, strategy) -> None:
        if False:
            while True:
                i = 10
        'used when trading multiple routes that related\n\n        Arguments:\n            strategy {Strategy} -- the strategy that has fired (and not listening to) the event\n        '
        pass

    def on_route_canceled(self, strategy) -> None:
        if False:
            return 10
        'used when trading multiple routes that related\n\n        Arguments:\n            strategy {Strategy} -- the strategy that has fired (and not listening to) the event\n        '
        pass

    def _execute(self) -> None:
        if False:
            return 10
        '\n        Handles the execution permission for the strategy.\n        '
        if self._is_executing is True:
            return
        self._is_executing = True
        self.before()
        self._check()
        self.after()
        self._clear_cached_methods()
        self._is_executing = False
        self.index += 1

    def _terminate(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Optional for executing code after completion of a backTest.\n        This block will not execute in live use as a live\n        Jesse is never ending.\n        '
        if not jh.should_execute_silently() or jh.is_debugging():
            logger.info(f'Terminating {self.symbol}...')
        self.before_terminate()
        self._detect_and_handle_entry_and_exit_modifications()
        if not jh.is_live():
            store.orders.execute_pending_market_orders()
        if jh.is_live():
            self.terminate()
            return
        if self.position.is_open:
            store.app.total_open_trades += 1
            store.app.total_open_pl += self.position.pnl
            logger.info(f'Closed open {self.exchange}-{self.symbol} position at {self.position.current_price} with PNL: {round(self.position.pnl, 4)}({round(self.position.pnl_percentage, 2)}%) because we reached the end of the backtest session.')
            if self.exchange_type == 'spot':
                self.broker.cancel_all_orders()
            self.broker.reduce_position_at(self.position.qty, self.position.current_price, self.price)
            self.terminate()
            return
        if len(self.entry_orders):
            self._execute_cancel()
            logger.info('Canceled open-position orders because we reached the end of the backtest session.')
        self.terminate()

    def before_terminate(self):
        if False:
            print('Hello World!')
        pass

    def terminate(self):
        if False:
            i = 10
            return i + 15
        pass

    def watch_list(self) -> list:
        if False:
            return 10
        '\n        returns an array containing an array of key-value items that should\n        be logged when backTested, and monitored while liveTraded\n\n        Returns:\n            [array[{"key": v, "value": v}]] -- an array of dictionary objects\n        '
        return []

    def _clear_cached_methods(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for m in self._cached_methods.values():
            m.cache_clear()

    @property
    def current_candle(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Returns current trading candle\n\n        :return: np.ndarray\n        '
        return store.candles.get_current_candle(self.exchange, self.symbol, self.timeframe).copy()

    @property
    def open(self) -> float:
        if False:
            return 10
        "\n        Returns the opening price of the current candle for this strategy.\n        Just as a helper to use when writing super simple strategies.\n        Returns:\n            [float] -- the current trading candle's OPEN price\n        "
        return self.current_candle[1]

    @property
    def close(self) -> float:
        if False:
            return 10
        "\n        Returns the closing price of the current candle for this strategy.\n        Just as a helper to use when writing super simple strategies.\n        Returns:\n            [float] -- the current trading candle's CLOSE price\n        "
        return self.current_candle[2]

    @property
    def price(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        "\n        Same as self.close, except in livetrade, this is rounded as the exchanges require it.\n\n        Returns:\n            [float] -- the current trading candle's current(close) price\n        "
        return self.close

    @property
    def high(self) -> float:
        if False:
            while True:
                i = 10
        "\n        Returns the highest price of the current candle for this strategy.\n        Just as a helper to use when writing super simple strategies.\n        Returns:\n            [float] -- the current trading candle's HIGH price\n        "
        return self.current_candle[3]

    @property
    def low(self) -> float:
        if False:
            i = 10
            return i + 15
        "\n        Returns the lowest price of the current candle for this strategy.\n        Just as a helper to use when writing super simple strategies.\n        Returns:\n            [float] -- the current trading candle's LOW price\n        "
        return self.current_candle[4]

    @property
    def candles(self) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns candles for current trading route\n\n        :return: np.ndarray\n        '
        return store.candles.get_candles(self.exchange, self.symbol, self.timeframe)

    def get_candles(self, exchange: str, symbol: str, timeframe: str) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Get candles by passing exchange, symbol, and timeframe\n\n        :param exchange: str\n        :param symbol: str\n        :param timeframe: str\n\n        :return: np.ndarray\n        '
        return store.candles.get_candles(exchange, symbol, timeframe)

    @property
    def metrics(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns all the metrics of the strategy.\n        '
        if self.trades_count not in self._cached_metrics:
            self._cached_metrics[self.trades_count] = metrics.trades(store.completed_trades.trades, store.app.daily_balance, final=False)
        return self._cached_metrics[self.trades_count]

    @property
    def time(self) -> int:
        if False:
            i = 10
            return i + 15
        'returns the current time'
        return store.app.time

    @property
    def balance(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'the current capital in the trading exchange'
        return self.position.exchange.wallet_balance

    @property
    def capital(self) -> float:
        if False:
            return 10
        raise NotImplementedError('The alias "self.capital" has been removed. Please use "self.balance" instead.')

    @property
    def available_margin(self) -> float:
        if False:
            return 10
        'Current available margin considering leverage'
        return self.position.exchange.available_margin

    @property
    def leveraged_available_margin(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        'Current available margin considering leverage'
        return self.leverage * self.available_margin

    @property
    def fee_rate(self) -> float:
        if False:
            while True:
                i = 10
        return selectors.get_exchange(self.exchange).fee_rate

    @property
    def is_long(self) -> bool:
        if False:
            print('Hello World!')
        return self.position.type == 'long'

    @property
    def is_short(self) -> bool:
        if False:
            return 10
        return self.position.type == 'short'

    @property
    def is_open(self) -> bool:
        if False:
            while True:
                i = 10
        return self.position.is_open

    @property
    def is_close(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self.position.is_close

    @property
    def average_stop_loss(self) -> float:
        if False:
            return 10
        if self._stop_loss is None:
            raise exceptions.InvalidStrategy('You cannot access self.average_stop_loss before setting self.stop_loss')
        arr = self._stop_loss
        return np.abs(arr[:, 0] * arr[:, 1]).sum() / np.abs(arr[:, 0]).sum()

    @property
    def average_take_profit(self) -> float:
        if False:
            print('Hello World!')
        if self._take_profit is None:
            raise exceptions.InvalidStrategy('You cannot access self.average_take_profit before setting self.take_profit')
        arr = self._take_profit
        return np.abs(arr[:, 0] * arr[:, 1]).sum() / np.abs(arr[:, 0]).sum()

    def _get_formatted_order(self, var, round_for_live_mode=True) -> Union[list, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        if type(var) is np.ndarray:
            return var
        if var is None or var == []:
            return []
        if type(var[0]) not in [list, tuple]:
            var = [var]
        arr = np.array(var, dtype=float)
        if arr[:, 1].min() <= 0:
            raise exceptions.InvalidStrategy(f'Order price must be greater than zero: \n{var}')
        if jh.is_livetrading() and round_for_live_mode:
            current_exchange = selectors.get_exchange(self.exchange)
            if 'precisions' not in current_exchange.vars:
                return arr
            price_precision = current_exchange.vars['precisions'][self.symbol]['price_precision']
            qty_precision = current_exchange.vars['precisions'][self.symbol]['qty_precision']
            prices = jh.round_price_for_live_mode(arr[:, 1], price_precision)
            qtys = jh.round_qty_for_live_mode(arr[:, 0], qty_precision)
            arr[:, 0] = qtys
            arr[:, 1] = prices
        return arr

    @property
    def average_entry_price(self) -> float:
        if False:
            print('Hello World!')
        if self.is_long:
            arr = self._buy
        elif self.is_short:
            arr = self._sell
        elif self.has_long_entry_orders:
            arr = self._get_formatted_order(self.buy)
        elif self.has_short_entry_orders:
            arr = self._get_formatted_order(self.sell)
        else:
            return None
        if type(arr) is not np.ndarray:
            arr = None
        if arr is None and self.position.is_open:
            return self.position.entry_price
        elif arr is None:
            return None
        return np.abs(arr[:, 0] * arr[:, 1]).sum() / np.abs(arr[:, 0]).sum()

    @property
    def has_long_entry_orders(self) -> bool:
        if False:
            i = 10
            return i + 15
        if self.entry_orders == [] and self.buy is not None:
            return True
        return self.entry_orders != [] and self.entry_orders[0].side == 'buy'

    @property
    def has_short_entry_orders(self) -> bool:
        if False:
            i = 10
            return i + 15
        if self.entry_orders == [] and self.sell is not None:
            return True
        return self.entry_orders != [] and self.entry_orders[0].side == 'sell'

    def liquidate(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        closes open position with a MARKET order\n        '
        if self.position.is_close:
            return
        if self.position.pnl > 0:
            self.take_profit = (self.position.qty, self.price)
        else:
            self.stop_loss = (self.position.qty, self.price)

    @property
    def shared_vars(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return store.vars

    @property
    def routes(self) -> List[Route]:
        if False:
            for i in range(10):
                print('nop')
        from jesse.routes import router
        return router.routes

    @property
    def leverage(self) -> int:
        if False:
            return 10
        if type(self.position.exchange) is SpotExchange:
            return 1
        elif type(self.position.exchange) is FuturesExchange:
            return self.position.exchange.futures_leverage
        else:
            raise ValueError('exchange type not supported!')

    @property
    def mark_price(self) -> float:
        if False:
            print('Hello World!')
        return self.position.mark_price

    @property
    def funding_rate(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return self.position.funding_rate

    @property
    def next_funding_timestamp(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.position.next_funding_timestamp

    @property
    def liquidation_price(self) -> float:
        if False:
            while True:
                i = 10
        return self.position.liquidation_price

    @staticmethod
    def log(msg: str, log_type: str='info', send_notification: bool=False, webhook: str=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        msg = str(msg)
        if log_type == 'info':
            logger.info(msg, send_notification=jh.is_live() and send_notification, webhook=webhook)
        elif log_type == 'error':
            logger.error(msg, send_notification=jh.is_live() and send_notification)
        else:
            raise ValueError(f'log_type should be either "info" or "error". You passed {log_type}')

    @property
    def all_positions(self) -> Dict[str, Position]:
        if False:
            while True:
                i = 10
        positions_dict = {}
        for r in self.routes:
            positions_dict[r.symbol] = r.strategy.position
        return positions_dict

    @property
    def portfolio_value(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        total_position_values = 0
        if self.is_spot_trading:
            for o in self.entry_orders:
                if o.is_active:
                    total_position_values += o.value
            for (key, p) in self.all_positions.items():
                total_position_values += p.value
        elif self.is_futures_trading:
            for (key, p) in self.all_positions.items():
                total_position_values += p.pnl
        return (total_position_values + self.balance) * self.leverage

    @property
    def trades(self) -> List[ClosedTrade]:
        if False:
            print('Hello World!')
        '\n        Returns all the completed trades for this strategy.\n        '
        return store.completed_trades.trades

    @property
    def orders(self) -> List[Order]:
        if False:
            i = 10
            return i + 15
        '\n        Returns all the orders submitted by for this strategy.\n        '
        return store.orders.get_orders(self.exchange, self.symbol)

    @property
    def entry_orders(self):
        if False:
            while True:
                i = 10
        '\n        Returns all the entry orders for this position.\n        '
        return store.orders.get_entry_orders(self.exchange, self.symbol)

    @property
    def exit_orders(self):
        if False:
            print('Hello World!')
        '\n        Returns all the exit orders for this position.\n        '
        return store.orders.get_exit_orders(self.exchange, self.symbol)

    @property
    def exchange_type(self):
        if False:
            for i in range(10):
                print('nop')
        return selectors.get_exchange(self.exchange).type

    @property
    def is_spot_trading(self) -> bool:
        if False:
            while True:
                i = 10
        return self.exchange_type == 'spot'

    @property
    def is_futures_trading(self) -> bool:
        if False:
            print('Hello World!')
        return self.exchange_type == 'futures'

    @property
    def daily_balances(self):
        if False:
            for i in range(10):
                print('nop')
        return store.app.daily_balance

    @property
    def is_backtesting(self) -> bool:
        if False:
            return 10
        return jh.is_backtesting()

    @property
    def is_livetrading(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return jh.is_livetrading()

    @property
    def is_papertrading(self) -> bool:
        if False:
            return 10
        return jh.is_paper_trading()

    @property
    def is_live(self) -> bool:
        if False:
            i = 10
            return i + 15
        return jh.is_live()