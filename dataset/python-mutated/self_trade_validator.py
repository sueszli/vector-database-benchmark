from rqalpha.interface import AbstractFrontendValidator
from rqalpha.const import ORDER_TYPE, SIDE, POSITION_EFFECT
from rqalpha.utils.i18n import gettext as _
from rqalpha.utils.logger import user_system_log

class SelfTradeValidator(AbstractFrontendValidator):

    def __init__(self, env):
        if False:
            for i in range(10):
                print('nop')
        self._env = env

    def can_submit_order(self, order, account=None):
        if False:
            return 10
        open_orders = [o for o in self._env.get_open_orders(order.order_book_id) if o.side != order.side and o.position_effect != POSITION_EFFECT.EXERCISE]
        if len(open_orders) == 0:
            return True
        reason = _('Create order failed, there are active orders leading to the risk of self-trade: [{}...]')
        if order.type == ORDER_TYPE.MARKET:
            user_system_log.warn(reason.format(open_orders[0]))
            return False
        if order.side == SIDE.BUY:
            for open_order in open_orders:
                if order.price >= open_order.price:
                    user_system_log.warn(reason.format(open_order))
                    return False
        else:
            for open_order in open_orders:
                if order.price <= open_order.price:
                    user_system_log.warn(reason.format(open_order))
                    return False

    def can_cancel_order(self, order, account=None):
        if False:
            print('Hello World!')
        return True