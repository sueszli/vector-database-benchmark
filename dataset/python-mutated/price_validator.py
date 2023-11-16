from rqalpha.interface import AbstractFrontendValidator
from rqalpha.const import ORDER_TYPE, POSITION_EFFECT
from rqalpha.utils.logger import user_system_log
from rqalpha.utils.i18n import gettext as _

class PriceValidator(AbstractFrontendValidator):

    def __init__(self, env):
        if False:
            i = 10
            return i + 15
        self._env = env

    def can_submit_order(self, order, account=None):
        if False:
            print('Hello World!')
        if order.type != ORDER_TYPE.LIMIT or order.position_effect == POSITION_EFFECT.EXERCISE:
            return True
        limit_up = round(self._env.price_board.get_limit_up(order.order_book_id), 4)
        if order.price > limit_up:
            reason = _('Order Creation Failed: limit order price {limit_price} is higher than limit up {limit_up}, order_book_id={order_book_id}').format(order_book_id=order.order_book_id, limit_price=order.price, limit_up=limit_up)
            user_system_log.warn(reason)
            return False
        limit_down = round(self._env.price_board.get_limit_down(order.order_book_id), 4)
        if order.price < limit_down:
            reason = _('Order Creation Failed: limit order price {limit_price} is lower than limit down {limit_down}, order_book_id={order_book_id}').format(order_book_id=order.order_book_id, limit_price=order.price, limit_down=limit_down)
            user_system_log.warn(reason)
            return False
        return True

    def can_cancel_order(self, order, account=None):
        if False:
            for i in range(10):
                print('nop')
        return True