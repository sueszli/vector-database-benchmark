"""Manage bars for Glances output."""
from __future__ import division
from math import modf

class Bar(object):
    """Manage bar (progression or status).

    import sys
    import time
    b = Bar(10)
    for p in range(0, 100):
        b.percent = p
        print("\r%s" % b),
        time.sleep(0.1)
        sys.stdout.flush()
    """

    def __init__(self, size, percentage_char='|', empty_char=' ', pre_char='[', post_char=']', with_text=True):
        if False:
            return 10
        self.__curses_bars = [empty_char] * 5 + [percentage_char] * 5
        self.__size = size
        self.__percent = 0
        self.min_value = 0
        self.max_value = 100
        self.__pre_char = pre_char
        self.__post_char = post_char
        self.__empty_char = empty_char
        self.__with_text = with_text

    @property
    def size(self, with_decoration=False):
        if False:
            for i in range(10):
                print('nop')
        if with_decoration:
            return self.__size
        if self.__with_text:
            return self.__size - 6

    @property
    def percent(self):
        if False:
            i = 10
            return i + 15
        return self.__percent

    @percent.setter
    def percent(self, value):
        if False:
            i = 10
            return i + 15
        if value <= self.min_value:
            value = self.min_value
        if value >= self.max_value:
            value = self.max_value
        self.__percent = value

    @property
    def pre_char(self):
        if False:
            return 10
        return self.__pre_char

    @property
    def post_char(self):
        if False:
            while True:
                i = 10
        return self.__post_char

    def get(self):
        if False:
            i = 10
            return i + 15
        'Return the bars.'
        (frac, whole) = modf(self.size * self.percent / 100.0)
        ret = self.__curses_bars[8] * int(whole)
        if frac > 0:
            ret += self.__curses_bars[int(frac * 8)]
            whole += 1
        ret += self.__empty_char * int(self.size - whole)
        if self.__with_text:
            ret = '{}{:5.1f}%'.format(ret, self.percent)
        return ret

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the bars.'
        return self.get()