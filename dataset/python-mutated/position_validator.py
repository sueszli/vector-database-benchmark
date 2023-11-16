from typing import Optional
from rqalpha.interface import AbstractFrontendValidator, AbstractPosition
from rqalpha.const import POSITION_EFFECT
from rqalpha.utils.logger import user_system_log
from rqalpha.model.order import Order
from rqalpha.portfolio.account import Account
from rqalpha.utils.i18n import gettext as _

class PositionValidator(AbstractFrontendValidator):

    def can_cancel_order(self, order, account=None):
        if False:
            for i in range(10):
                print('nop')
        return True

    def can_submit_order(self, order, account=None):
        if False:
            while True:
                i = 10
        if account is None:
            return True
        if order.position_effect in (POSITION_EFFECT.OPEN, POSITION_EFFECT.EXERCISE):
            return True
        position = account.get_position(order.order_book_id, order.position_direction)
        if order.position_effect == POSITION_EFFECT.CLOSE_TODAY and order.quantity > position.today_closable:
            user_system_log.warn(_('Order Creation Failed: not enough today position {order_book_id} to close, target quantity is {quantity}, closable today quantity is {closable}').format(order_book_id=order.order_book_id, quantity=order.quantity, closable=position.today_closable))
            return False
        if order.position_effect == POSITION_EFFECT.CLOSE and order.quantity > position.closable:
            user_system_log.warn(_('Order Creation Failed: not enough position {order_book_id} to close or exercise, target sell quantity is {quantity}, closable quantity is {closable}').format(order_book_id=order.order_book_id, quantity=order.quantity, closable=position.closable))
            return False
        return True