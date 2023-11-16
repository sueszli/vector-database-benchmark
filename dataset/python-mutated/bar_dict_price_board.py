import numpy as np
from rqalpha.interface import AbstractPriceBoard
from rqalpha.environment import Environment
from rqalpha.core.execution_context import ExecutionContext
from rqalpha.const import EXECUTION_PHASE

class BarDictPriceBoard(AbstractPriceBoard):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._env = Environment.get_instance()

    def _get_bar(self, order_book_id):
        if False:
            i = 10
            return i + 15
        if ExecutionContext.phase() == EXECUTION_PHASE.OPEN_AUCTION:
            return self._env.data_proxy.get_open_auction_bar(order_book_id, self._env.calendar_dt)
        return self._env.get_bar(order_book_id)

    def get_last_price(self, order_book_id):
        if False:
            return 10
        return self._get_bar(order_book_id).last

    def get_limit_up(self, order_book_id):
        if False:
            i = 10
            return i + 15
        return self._get_bar(order_book_id).limit_up

    def get_limit_down(self, order_book_id):
        if False:
            i = 10
            return i + 15
        return self._get_bar(order_book_id).limit_down

    def get_a1(self, order_book_id):
        if False:
            print('Hello World!')
        return np.nan

    def get_b1(self, order_book_id):
        if False:
            for i in range(10):
                print('nop')
        return np.nan