import datetime
import numpy as np
from rqalpha.utils.datetime_func import convert_int_to_datetime, convert_ms_int_to_datetime

class TickObject(object):

    def __init__(self, instrument, tick_dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tick 对象\n        :param instrument: Instrument\n        :param tick_dict: dict\n        '
        self._instrument = instrument
        self._tick_dict = tick_dict

    @property
    def order_book_id(self):
        if False:
            return 10
        '\n        [str] 标的代码\n        '
        return self._instrument.order_book_id

    @property
    def datetime(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        [datetime.datetime] 当前快照数据的时间戳\n        '
        try:
            dt = self._tick_dict['datetime']
        except (KeyError, ValueError):
            return datetime.datetime.min
        else:
            if not isinstance(dt, datetime.datetime):
                if dt > 10000000000000000:
                    return convert_ms_int_to_datetime(dt)
                else:
                    return convert_int_to_datetime(dt)
            return dt

    @property
    def open(self):
        if False:
            print('Hello World!')
        '\n        [float] 当日开盘价\n        '
        return self._tick_dict['open']

    @property
    def last(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        [float] 当前最新价\n        '
        try:
            return self._tick_dict['last']
        except KeyError:
            return self.prev_close

    @property
    def high(self):
        if False:
            i = 10
            return i + 15
        '\n        [float] 截止到当前的最高价\n        '
        return self._tick_dict['high']

    @property
    def low(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        [float] 截止到当前的最低价\n        '
        return self._tick_dict['low']

    @property
    def prev_close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n       [float] 昨日收盘价\n       '
        try:
            return self._tick_dict['prev_close']
        except (KeyError, ValueError):
            return 0

    @property
    def volume(self):
        if False:
            print('Hello World!')
        '\n        [float] 截止到当前的成交量\n        '
        try:
            return self._tick_dict['volume']
        except (KeyError, ValueError):
            return 0

    @property
    def total_turnover(self):
        if False:
            while True:
                i = 10
        '\n        [float] 截止到当前的成交额\n        '
        try:
            return self._tick_dict['total_turnover']
        except (KeyError, ValueError):
            return 0

    @property
    def open_interest(self):
        if False:
            while True:
                i = 10
        '\n        [float] 截止到当前的持仓量（期货专用）\n        '
        try:
            return self._tick_dict['open_interest']
        except (KeyError, ValueError):
            return 0

    @property
    def prev_settlement(self):
        if False:
            print('Hello World!')
        '\n        [float] 昨日结算价（期货专用）\n        '
        try:
            return self._tick_dict['prev_settlement']
        except (KeyError, ValueError):
            return 0

    @property
    def asks(self):
        if False:
            i = 10
            return i + 15
        '\n        [list] 卖出报盘价格，asks[0]代表盘口卖一档报盘价\n        '
        try:
            return self._tick_dict['asks']
        except (KeyError, ValueError):
            return [0] * 5

    @property
    def ask_vols(self):
        if False:
            print('Hello World!')
        '\n        [list] 卖出报盘数量，ask_vols[0]代表盘口卖一档报盘数量\n        '
        try:
            return self._tick_dict['ask_vols']
        except (KeyError, ValueError):
            return [0] * 5

    @property
    def bids(self):
        if False:
            i = 10
            return i + 15
        '\n        [list] 买入报盘价格，bids[0]代表盘口买一档报盘价\n        '
        try:
            return self._tick_dict['bids']
        except (KeyError, ValueError):
            return [0] * 5

    @property
    def bid_vols(self):
        if False:
            print('Hello World!')
        '\n        [list] 买入报盘数量，bids_vols[0]代表盘口买一档报盘数量\n        '
        try:
            return self._tick_dict['bid_vols']
        except (KeyError, ValueError):
            return [0] * 5

    @property
    def limit_up(self):
        if False:
            i = 10
            return i + 15
        '\n        [float] 涨停价\n        '
        try:
            return self._tick_dict['limit_up']
        except (KeyError, ValueError):
            return 0

    @property
    def limit_down(self):
        if False:
            i = 10
            return i + 15
        '\n        [float] 跌停价\n        '
        try:
            return self._tick_dict['limit_down']
        except (KeyError, ValueError):
            return 0

    @property
    def isnan(self):
        if False:
            for i in range(10):
                print('nop')
        return np.isnan(self.last)

    def __repr__(self):
        if False:
            while True:
                i = 10
        items = []
        for name in dir(self):
            if name.startswith('_'):
                continue
            items.append((name, getattr(self, name)))
        return 'Tick({0})'.format(', '.join(('{0}: {1}'.format(k, v) for (k, v) in items)))

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return getattr(self, key)