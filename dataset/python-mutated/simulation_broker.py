from typing import List, Tuple, Dict
from rqalpha.utils.functools import lru_cache
from itertools import chain
import jsonpickle
from rqalpha.portfolio.account import Account
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.interface import AbstractBroker, Persistable
from rqalpha.utils.i18n import gettext as _
from rqalpha.core.events import EVENT, Event
from rqalpha.const import MATCHING_TYPE, ORDER_STATUS, POSITION_EFFECT, EXECUTION_PHASE, INSTRUMENT_TYPE
from rqalpha.model.order import Order
from rqalpha.environment import Environment
from .matcher import DefaultBarMatcher, AbstractMatcher, CounterPartyOfferMatcher, DefaultTickMatcher

class SimulationBroker(AbstractBroker, Persistable):

    def __init__(self, env, mod_config):
        if False:
            return 10
        self._env = env
        self._mod_config = mod_config
        self._matchers = {}
        self._match_immediately = mod_config.matching_type in [MATCHING_TYPE.CURRENT_BAR_CLOSE, MATCHING_TYPE.VWAP]
        self._open_orders = []
        self._open_auction_orders = []
        self._open_exercise_orders = []
        self._frontend_validator = {}
        if self._mod_config.matching_type == MATCHING_TYPE.COUNTERPARTY_OFFER:
            for instrument_type in INSTRUMENT_TYPE:
                self.register_matcher(instrument_type, CounterPartyOfferMatcher(self._env, self._mod_config))
        self._env.event_bus.add_listener(EVENT.BEFORE_TRADING, self.before_trading)
        self._env.event_bus.add_listener(EVENT.BAR, self.on_bar)
        self._env.event_bus.add_listener(EVENT.TICK, self.on_tick)
        self._env.event_bus.add_listener(EVENT.AFTER_TRADING, self.after_trading)
        self._env.event_bus.add_listener(EVENT.PRE_SETTLEMENT, self.pre_settlement)

    @lru_cache(1024)
    def _get_matcher(self, order_book_id):
        if False:
            return 10
        instrument_type = self._env.data_proxy.instrument(order_book_id).type
        try:
            return self._matchers[instrument_type]
        except KeyError:
            if self._env.config.base.frequency == 'tick':
                return self._matchers.setdefault(instrument_type, DefaultTickMatcher(self._env, self._mod_config))
            else:
                return self._matchers.setdefault(instrument_type, DefaultBarMatcher(self._env, self._mod_config))

    def register_matcher(self, instrument_type, matcher):
        if False:
            print('Hello World!')
        self._matchers[instrument_type] = matcher

    def get_open_orders(self, order_book_id=None):
        if False:
            while True:
                i = 10
        if order_book_id is None:
            return [order for (account, order) in chain(self._open_orders, self._open_auction_orders)]
        else:
            return [order for (account, order) in chain(self._open_orders, self._open_auction_orders) if order.order_book_id == order_book_id]

    def get_state(self):
        if False:
            return 10
        return jsonpickle.dumps({'open_orders': [o.get_state() for (account, o) in self._open_orders], 'open_auction_orders': [o.get_state() for (account, o) in self._open_auction_orders]}).encode('utf-8')

    def set_state(self, state):
        if False:
            while True:
                i = 10

        def _account_order_from_state(order_state):
            if False:
                i = 10
                return i + 15
            o = Order()
            o.set_state(order_state)
            account = self._env.get_account(o.order_book_id)
            return (account, o)
        value = jsonpickle.loads(state.decode('utf-8'))
        self._open_orders = [_account_order_from_state(v) for v in value['open_orders']]
        self._open_auction_orders = [_account_order_from_state(v) for v in value.get('open_auction_orders', [])]

    def submit_order(self, order):
        if False:
            print('Hello World!')
        self._check_subscribe(order)
        if order.position_effect == POSITION_EFFECT.MATCH:
            raise TypeError(_('unsupported position_effect {}').format(order.position_effect))
        account = self._env.get_account(order.order_book_id)
        self._env.event_bus.publish_event(Event(EVENT.ORDER_PENDING_NEW, account=account, order=order))
        if order.is_final():
            return
        if order.position_effect == POSITION_EFFECT.EXERCISE:
            return self._open_exercise_orders.append((account, order))
        if ExecutionContext.phase() == EXECUTION_PHASE.OPEN_AUCTION:
            self._open_auction_orders.append((account, order))
        else:
            self._open_orders.append((account, order))
        order.active()
        self._env.event_bus.publish_event(Event(EVENT.ORDER_CREATION_PASS, account=account, order=order))
        if self._match_immediately:
            self._match(self._env.calendar_dt)

    def cancel_order(self, order):
        if False:
            print('Hello World!')
        account = self._env.get_account(order.order_book_id)
        self._env.event_bus.publish_event(Event(EVENT.ORDER_PENDING_CANCEL, account=account, order=order))
        order.mark_cancelled(_(u'{order_id} order has been cancelled by user.').format(order_id=order.order_id))
        self._env.event_bus.publish_event(Event(EVENT.ORDER_CANCELLATION_PASS, account=account, order=order))
        try:
            self._open_orders.remove((account, order))
        except ValueError:
            pass

    def before_trading(self, _):
        if False:
            while True:
                i = 10
        for (account, order) in self._open_orders:
            order.active()
            self._env.event_bus.publish_event(Event(EVENT.ORDER_CREATION_PASS, account=account, order=order))

    def after_trading(self, __):
        if False:
            for i in range(10):
                print('nop')
        for (account, order) in self._open_orders:
            order.mark_rejected(_(u'Order Rejected: {order_book_id} can not match. Market close.').format(order_book_id=order.order_book_id))
            self._env.event_bus.publish_event(Event(EVENT.ORDER_UNSOLICITED_UPDATE, account=account, order=order))
        self._open_orders = []

    def pre_settlement(self, __):
        if False:
            for i in range(10):
                print('nop')
        for (account, order) in self._open_exercise_orders:
            self._get_matcher(order.order_book_id).match(account, order, False)
            if order.status == ORDER_STATUS.REJECTED or order.status == ORDER_STATUS.CANCELLED:
                self._env.event_bus.publish_event(Event(EVENT.ORDER_UNSOLICITED_UPDATE, account=account, order=order))
        self._open_exercise_orders.clear()

    def on_bar(self, event):
        if False:
            while True:
                i = 10
        for matcher in self._matchers.values():
            matcher.update(event)
        self._match(event.calendar_dt)

    def on_tick(self, event):
        if False:
            for i in range(10):
                print('nop')
        tick = event.tick
        self._get_matcher(tick.order_book_id).update(event)
        self._match(event.calendar_dt, tick.order_book_id)

    def _match(self, dt, order_book_id=None):
        if False:
            print('Hello World!')
        order_filter = lambda a_and_o: not a_and_o[1].is_final() and (True if order_book_id is None else a_and_o[1].order_book_id == order_book_id)
        open_order_filter = lambda a_and_o: order_filter(a_and_o) and self._env.data_proxy.instrument(a_and_o[1].order_book_id).during_continuous_auction(dt.time())
        for (account, order) in filter(open_order_filter, self._open_orders):
            self._get_matcher(order.order_book_id).match(account, order, open_auction=False)
        for (account, order) in filter(order_filter, self._open_auction_orders):
            self._get_matcher(order.order_book_id).match(account, order, open_auction=True)
        final_orders = [(a, o) for (a, o) in chain(self._open_orders, self._open_auction_orders) if o.is_final()]
        self._open_orders = [(a, o) for (a, o) in chain(self._open_orders, self._open_auction_orders) if not o.is_final()]
        self._open_auction_orders.clear()
        for (account, order) in final_orders:
            if order.status == ORDER_STATUS.REJECTED or order.status == ORDER_STATUS.CANCELLED:
                self._env.event_bus.publish_event(Event(EVENT.ORDER_UNSOLICITED_UPDATE, account=account, order=order))

    def _check_subscribe(self, order):
        if False:
            while True:
                i = 10
        if self._env.config.base.frequency == 'tick' and order.order_book_id not in self._env.get_universe():
            raise RuntimeError(_('{order_book_id} should be subscribed when frequency is tick.').format(order_book_id=order.order_book_id))