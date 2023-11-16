from copy import copy
import numpy as np
from rqalpha.interface import AbstractBroker
from rqalpha.utils.logger import user_system_log
from rqalpha.utils.i18n import gettext as _
from rqalpha.utils import is_valid_price
from rqalpha.core.events import EVENT, Event
from rqalpha.model.trade import Trade
from rqalpha.model.order import ALGO_ORDER_STYLES
from rqalpha.const import SIDE, ORDER_TYPE, POSITION_EFFECT
from .slippage import SlippageDecider

class SignalBroker(AbstractBroker):

    def __init__(self, env, mod_config):
        if False:
            for i in range(10):
                print('nop')
        self._env = env
        self._slippage_decider = SlippageDecider(mod_config.slippage_model, mod_config.slippage)
        self._price_limit = mod_config.price_limit

    def get_open_orders(self, order_book_id=None):
        if False:
            print('Hello World!')
        return []

    def submit_order(self, order):
        if False:
            while True:
                i = 10
        if order.position_effect == POSITION_EFFECT.EXERCISE:
            raise NotImplementedError('SignalBroker does not support exercise order temporarily')
        account = self._env.get_account(order.order_book_id)
        self._env.event_bus.publish_event(Event(EVENT.ORDER_PENDING_NEW, account=account, order=order))
        if order.is_final():
            return
        order.active()
        self._env.event_bus.publish_event(Event(EVENT.ORDER_CREATION_PASS, account=account, order=order))
        self._match(account, order)

    def cancel_order(self, order):
        if False:
            print('Hello World!')
        user_system_log.warn(_(u'cancel_order function is not supported in signal mode'))
        return None

    def _match(self, account, order):
        if False:
            i = 10
            return i + 15
        order_book_id = order.order_book_id
        price_board = self._env.price_board
        last_price = price_board.get_last_price(order_book_id)
        if not is_valid_price(last_price):
            instrument = self._env.get_instrument(order_book_id)
            listed_date = instrument.listed_date.date()
            if listed_date == self._env.trading_dt.date():
                reason = _('Order Cancelled: current security [{order_book_id}] can not be traded in listed date [{listed_date}]').format(order_book_id=order_book_id, listed_date=listed_date)
            else:
                reason = _(u'Order Cancelled: current bar [{order_book_id}] miss market data.').format(order_book_id=order_book_id)
            order.mark_rejected(reason)
            self._env.event_bus.publish_event(Event(EVENT.ORDER_UNSOLICITED_UPDATE, account=account, order=copy(order)))
            return
        if order.type == ORDER_TYPE.LIMIT:
            deal_price = order.frozen_price
        elif isinstance(order.style, ALGO_ORDER_STYLES):
            (deal_price, v) = self._env.data_proxy.get_algo_bar(order.order_book_id, order.style, self._env.calendar_dt)
            if np.isnan(deal_price):
                reason = _(u'Order Cancelled: {order_book_id} bar no volume').format(order_book_id=order.order_book_id)
                order.mark_rejected(reason)
                return
        else:
            deal_price = last_price
        if self._price_limit:
            if order.position_effect != POSITION_EFFECT.EXERCISE:
                if order.side == SIDE.BUY and deal_price >= price_board.get_limit_up(order_book_id):
                    order.mark_rejected(_('Order Cancelled: current bar [{order_book_id}] reach the limit_up price.').format(order_book_id=order.order_book_id))
                    self._env.event_bus.publish_event(Event(EVENT.ORDER_UNSOLICITED_UPDATE, account=account, order=copy(order)))
                    return
                if order.side == SIDE.SELL and deal_price <= price_board.get_limit_down(order_book_id):
                    order.mark_rejected(_('Order Cancelled: current bar [{order_book_id}] reach the limit_down price.').format(order_book_id=order.order_book_id))
                    self._env.event_bus.publish_event(Event(EVENT.ORDER_UNSOLICITED_UPDATE, account=account, order=copy(order)))
                    return
        ct_amount = account.calc_close_today_amount(order_book_id, order.quantity, order.position_direction)
        trade_price = self._slippage_decider.get_trade_price(order, deal_price)
        trade = Trade.__from_create__(order_id=order.order_id, price=trade_price, amount=order.quantity, side=order.side, position_effect=order.position_effect, order_book_id=order_book_id, frozen_price=order.frozen_price, close_today_amount=ct_amount)
        trade._commission = self._env.get_trade_commission(trade)
        trade._tax = self._env.get_trade_tax(trade)
        order.fill(trade)
        self._env.event_bus.publish_event(Event(EVENT.TRADE, account=account, trade=trade, order=copy(order)))