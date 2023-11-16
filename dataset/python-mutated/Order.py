from playhouse.postgres_ext import *
import jesse.helpers as jh
import jesse.services.logger as logger
import jesse.services.selectors as selectors
from jesse.config import config
from jesse.services.notifier import notify
from jesse.enums import order_statuses, order_submitted_via
from jesse.services.db import database
if database.is_closed():
    database.open_connection()

class Order(Model):
    id = UUIDField(primary_key=True)
    trade_id = UUIDField(index=True, null=True)
    session_id = UUIDField(index=True)
    exchange_id = CharField(null=True)
    vars = JSONField(default={})
    symbol = CharField()
    exchange = CharField()
    side = CharField()
    type = CharField()
    reduce_only = BooleanField()
    qty = FloatField()
    filled_qty = FloatField(default=0)
    price = FloatField(null=True)
    status = CharField(default=order_statuses.ACTIVE)
    created_at = BigIntegerField()
    executed_at = BigIntegerField(null=True)
    canceled_at = BigIntegerField(null=True)
    submitted_via = None

    class Meta:
        from jesse.services.db import database
        database = database.db
        indexes = ((('trade_id', 'exchange', 'symbol', 'status', 'created_at'), False),)

    def __init__(self, attributes: dict=None, should_silent=False, **kwargs) -> None:
        if False:
            print('Hello World!')
        Model.__init__(self, attributes=attributes, **kwargs)
        if attributes is None:
            attributes = {}
        for (a, value) in attributes.items():
            setattr(self, a, value)
        if self.created_at is None:
            self.created_at = jh.now_to_timestamp()
        if not should_silent:
            if jh.is_live():
                self.notify_submission()
            if jh.is_debuggable('order_submission') and (self.is_active or self.is_queued):
                txt = f"{('QUEUED' if self.is_queued else 'SUBMITTED')} order: {self.symbol}, {self.type}, {self.side}, {self.qty}"
                if self.price:
                    txt += f', ${self.price}'
                logger.info(txt)
        e = selectors.get_exchange(self.exchange)
        e.on_order_submission(self)

    def notify_submission(self) -> None:
        if False:
            print('Hello World!')
        if config['env']['notifications']['events']['submitted_orders'] and (self.is_active or self.is_queued):
            txt = f"{('QUEUED' if self.is_queued else 'SUBMITTED')} order: {self.symbol}, {self.type}, {self.side}, {self.qty}"
            if self.price:
                txt += f', ${self.price}'
            notify(txt)

    @property
    def is_canceled(self) -> bool:
        if False:
            while True:
                i = 10
        return self.status == order_statuses.CANCELED

    @property
    def is_active(self) -> bool:
        if False:
            return 10
        return self.status == order_statuses.ACTIVE

    @property
    def is_cancellable(self):
        if False:
            while True:
                i = 10
        '\n        orders that are either active or partially filled\n        '
        return self.is_active or self.is_partially_filled or self.is_queued

    @property
    def is_queued(self) -> bool:
        if False:
            while True:
                i = 10
        "\n        Used in live mode only: it means the strategy has considered the order as submitted,\n        but the exchange does not accept it because of the distance between the current\n        price and price of the order. Hence it's been queued for later submission.\n\n        :return: bool\n        "
        return self.status == order_statuses.QUEUED

    @property
    def is_new(self) -> bool:
        if False:
            return 10
        return self.is_active

    @property
    def is_executed(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.status == order_statuses.EXECUTED

    @property
    def is_filled(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.is_executed

    @property
    def is_partially_filled(self) -> bool:
        if False:
            while True:
                i = 10
        return self.status == order_statuses.PARTIALLY_FILLED

    @property
    def is_stop_loss(self):
        if False:
            while True:
                i = 10
        return self.submitted_via == order_submitted_via.STOP_LOSS

    @property
    def is_take_profit(self):
        if False:
            return 10
        return self.submitted_via == order_submitted_via.TAKE_PROFIT

    @property
    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'id': self.id, 'session_id': self.session_id, 'exchange_id': self.exchange_id, 'symbol': self.symbol, 'side': self.side, 'type': self.type, 'qty': self.qty, 'filled_qty': self.filled_qty, 'price': self.price, 'status': self.status, 'created_at': self.created_at, 'canceled_at': self.canceled_at, 'executed_at': self.executed_at}

    @property
    def position(self):
        if False:
            i = 10
            return i + 15
        return selectors.get_position(self.exchange, self.symbol)

    @property
    def value(self) -> float:
        if False:
            print('Hello World!')
        return abs(self.qty) * self.price

    @property
    def remaining_qty(self) -> float:
        if False:
            for i in range(10):
                print('nop')
        return jh.prepare_qty(abs(self.qty) - abs(self.filled_qty), self.side)

    def queue(self):
        if False:
            print('Hello World!')
        self.status = order_statuses.QUEUED
        self.canceled_at = None
        if jh.is_debuggable('order_submission'):
            txt = f'QUEUED order: {self.symbol}, {self.type}, {self.side}, {self.qty}'
            if self.price:
                txt += f', ${round(self.price, 2)}'
                logger.info(txt)
        self.notify_submission()

    def resubmit(self):
        if False:
            i = 10
            return i + 15
        if not self.is_queued:
            raise NotSupportedError(f'Cannot resubmit an order that is not queued. Current status: {self.status}')
        self.id = jh.generate_unique_id()
        self.status = order_statuses.ACTIVE
        self.canceled_at = None
        if jh.is_debuggable('order_submission'):
            txt = f'SUBMITTED order: {self.symbol}, {self.type}, {self.side}, {self.qty}'
            if self.price:
                txt += f', ${self.price}'
                logger.info(txt)
        self.notify_submission()

    def cancel(self, silent=False, source='') -> None:
        if False:
            i = 10
            return i + 15
        if self.is_canceled or self.is_executed:
            return
        if source == 'stream' and self.is_queued:
            return
        self.canceled_at = jh.now_to_timestamp()
        self.status = order_statuses.CANCELED
        if not silent:
            txt = f'CANCELED order: {self.symbol}, {self.type}, {self.side}, {self.qty}'
            if self.price:
                txt += f', ${round(self.price, 2)}'
            if jh.is_debuggable('order_cancellation'):
                logger.info(txt)
            if jh.is_live():
                if config['env']['notifications']['events']['cancelled_orders']:
                    notify(txt)
        e = selectors.get_exchange(self.exchange)
        e.on_order_cancellation(self)

    def execute(self, silent=False) -> None:
        if False:
            print('Hello World!')
        if self.is_canceled or self.is_executed:
            return
        self.executed_at = jh.now_to_timestamp()
        self.status = order_statuses.EXECUTED
        if not silent:
            txt = f'EXECUTED order: {self.symbol}, {self.type}, {self.side}, {self.qty}'
            if self.price:
                txt += f', ${round(self.price, 2)}'
            if jh.is_debuggable('order_execution'):
                logger.info(txt)
            if jh.is_live():
                if config['env']['notifications']['events']['executed_orders']:
                    notify(txt)
        from jesse.store import store
        store.completed_trades.add_executed_order(self)
        e = selectors.get_exchange(self.exchange)
        e.on_order_execution(self)
        p = selectors.get_position(self.exchange, self.symbol)
        if p:
            p._on_executed_order(self)

    def execute_partially(self, silent=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.executed_at = jh.now_to_timestamp()
        self.status = order_statuses.PARTIALLY_FILLED
        if not silent:
            txt = f'PARTIALLY FILLED: {self.symbol}, {self.type}, {self.side}, filled qty: {self.filled_qty}, remaining qty: {self.remaining_qty}, price: {self.price}'
            if jh.is_debuggable('order_execution'):
                logger.info(txt)
            if jh.is_live():
                if config['env']['notifications']['events']['executed_orders']:
                    notify(txt)
        from jesse.store import store
        store.completed_trades.add_executed_order(self)
        p = selectors.get_position(self.exchange, self.symbol)
        if p:
            p._on_executed_order(self)
if database.is_open():
    Order.create_table()