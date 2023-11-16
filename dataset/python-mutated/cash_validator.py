from rqalpha.interface import AbstractFrontendValidator
from rqalpha.const import POSITION_EFFECT
from rqalpha.utils.logger import user_system_log
from rqalpha.utils.i18n import gettext as _

def is_cash_enough(env, order, cash, warn=False):
    if False:
        print('Hello World!')
    instrument = env.data_proxy.instrument(order.order_book_id)
    cost_money = instrument.calc_cash_occupation(order.frozen_price, order.quantity, order.position_direction)
    cost_money += env.get_order_transaction_cost(order)
    if cost_money <= cash:
        return True
    if warn:
        user_system_log.warn(_('Order Creation Failed: not enough money to buy {order_book_id}, needs {cost_money:.2f}, cash {cash:.2f}').format(order_book_id=order.order_book_id, cost_money=cost_money, cash=cash))
    return False

class CashValidator(AbstractFrontendValidator):

    def __init__(self, env):
        if False:
            i = 10
            return i + 15
        self._env = env

    def can_submit_order(self, order, account=None):
        if False:
            print('Hello World!')
        if account is None or order.position_effect != POSITION_EFFECT.OPEN:
            return True
        return is_cash_enough(self._env, order, account.cash, warn=True)

    def can_cancel_order(self, order, account=None):
        if False:
            i = 10
            return i + 15
        return True