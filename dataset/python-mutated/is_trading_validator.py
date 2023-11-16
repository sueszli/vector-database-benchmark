from rqalpha.interface import AbstractFrontendValidator
from rqalpha.utils.logger import user_system_log
from rqalpha.utils.i18n import gettext as _
from rqalpha.const import INSTRUMENT_TYPE

class IsTradingValidator(AbstractFrontendValidator):

    def __init__(self, env):
        if False:
            print('Hello World!')
        self._env = env

    def can_submit_order(self, order, account=None):
        if False:
            i = 10
            return i + 15
        instrument = self._env.data_proxy.instrument(order.order_book_id)
        if instrument.type != INSTRUMENT_TYPE.INDX and (not instrument.listing_at(self._env.trading_dt)):
            user_system_log.warn(_(u'Order Creation Failed: {order_book_id} is not listing!').format(order_book_id=order.order_book_id))
            return False
        if instrument.type == 'CS' and self._env.data_proxy.is_suspended(order.order_book_id, self._env.trading_dt):
            user_system_log.warn(_(u'Order Creation Failed: security {order_book_id} is suspended on {date}').format(order_book_id=order.order_book_id, date=self._env.trading_dt.date()))
            return False
        return True

    def can_cancel_order(self, order, account=None):
        if False:
            print('Hello World!')
        return True