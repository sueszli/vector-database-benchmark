"""Manage sparklines for Glances output."""
from __future__ import unicode_literals
from __future__ import division
import sys
from glances.logger import logger
from glances.globals import nativestr
sparklines_module = True
try:
    from sparklines import sparklines
except ImportError as e:
    logger.warning('Sparklines module not found ({})'.format(e))
    sparklines_module = False
try:
    '┌┬┐╔╦╗╒╤╕╓╥╖│║─═├┼┤╠╬╣╞╪╡╟╫╢└┴┘╚╩╝╘╧╛╙╨╜'.encode(sys.stdout.encoding)
except (UnicodeEncodeError, TypeError) as e:
    logger.warning('UTF-8 is mandatory for sparklines ({})'.format(e))
    sparklines_module = False

class Sparkline(object):
    """Manage sparklines (see https://pypi.org/project/sparklines/)."""

    def __init__(self, size, pre_char='[', post_char=']', empty_char=' ', with_text=True):
        if False:
            i = 10
            return i + 15
        self.__available = sparklines_module
        self.__size = size
        self.__percent = []
        self.__pre_char = pre_char
        self.__post_char = post_char
        self.__empty_char = empty_char
        self.__with_text = with_text

    @property
    def available(self):
        if False:
            i = 10
            return i + 15
        return self.__available

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
    def percents(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__percent

    @percents.setter
    def percents(self, value):
        if False:
            print('Hello World!')
        self.__percent = value

    @property
    def pre_char(self):
        if False:
            print('Hello World!')
        return self.__pre_char

    @property
    def post_char(self):
        if False:
            while True:
                i = 10
        return self.__post_char

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the sparkline.'
        ret = sparklines(self.percents, minimum=0, maximum=100)[0]
        if self.__with_text:
            percents_without_none = [x for x in self.percents if x is not None]
            if len(percents_without_none) > 0:
                ret = '{}{:5.1f}%'.format(ret, percents_without_none[-1])
        return nativestr(ret)

    def __str__(self):
        if False:
            print('Hello World!')
        'Return the sparkline.'
        return self.get()