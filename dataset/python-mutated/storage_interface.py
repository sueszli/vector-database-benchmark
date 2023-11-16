import abc
from typing import List, Optional, Sequence, Iterable
import numpy as np
import pandas
from rqalpha.model.instrument import Instrument
from rqalpha.utils.typing import DateLike
from rqalpha.const import INSTRUMENT_TYPE

class AbstractInstrumentStore:

    @property
    @abc.abstractmethod
    def instrument_type(self):
        if False:
            return 10
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def all_id_and_syms(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def get_instruments(self, id_or_syms):
        if False:
            return 10
        raise NotImplementedError

class AbstractDayBarStore:

    @abc.abstractmethod
    def get_bars(self, order_book_id):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def get_date_range(self, order_book_id):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class AbstractCalendarStore:

    @abc.abstractmethod
    def get_trading_calendar(self):
        if False:
            return 10
        raise NotImplementedError

class AbstractDateSet:

    @abc.abstractmethod
    def contains(self, order_book_id, dates):
        if False:
            return 10
        raise NotImplementedError

class AbstractDividendStore:

    @abc.abstractmethod
    def get_dividend(self, order_book_id):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class AbstractSimpleFactorStore:

    @abc.abstractmethod
    def get_factors(self, order_book_id):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError