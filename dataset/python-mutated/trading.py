import logbook
import pandas as pd
from zipline.utils.memoize import remember_last
from zipline.utils.pandas_utils import normalize_date
log = logbook.Logger('Trading')
DEFAULT_CAPITAL_BASE = 100000.0

class SimulationParameters(object):

    def __init__(self, start_session, end_session, trading_calendar, capital_base=DEFAULT_CAPITAL_BASE, emission_rate='daily', data_frequency='daily', arena='backtest'):
        if False:
            while True:
                i = 10
        assert type(start_session) == pd.Timestamp
        assert type(end_session) == pd.Timestamp
        assert trading_calendar is not None, 'Must pass in trading calendar!'
        assert start_session <= end_session, 'Period start falls after period end.'
        assert start_session <= trading_calendar.last_trading_session, 'Period start falls after the last known trading day.'
        assert end_session >= trading_calendar.first_trading_session, 'Period end falls before the first known trading day.'
        self._start_session = normalize_date(start_session)
        self._end_session = normalize_date(end_session)
        self._capital_base = capital_base
        self._emission_rate = emission_rate
        self._data_frequency = data_frequency
        self._arena = arena
        self._trading_calendar = trading_calendar
        if not trading_calendar.is_session(self._start_session):
            self._start_session = trading_calendar.minute_to_session_label(self._start_session)
        if not trading_calendar.is_session(self._end_session):
            self._end_session = trading_calendar.minute_to_session_label(self._end_session, direction='previous')
        self._first_open = trading_calendar.open_and_close_for_session(self._start_session)[0]
        self._last_close = trading_calendar.open_and_close_for_session(self._end_session)[1]

    @property
    def capital_base(self):
        if False:
            for i in range(10):
                print('nop')
        return self._capital_base

    @property
    def emission_rate(self):
        if False:
            i = 10
            return i + 15
        return self._emission_rate

    @property
    def data_frequency(self):
        if False:
            i = 10
            return i + 15
        return self._data_frequency

    @data_frequency.setter
    def data_frequency(self, val):
        if False:
            print('Hello World!')
        self._data_frequency = val

    @property
    def arena(self):
        if False:
            while True:
                i = 10
        return self._arena

    @arena.setter
    def arena(self, val):
        if False:
            return 10
        self._arena = val

    @property
    def start_session(self):
        if False:
            print('Hello World!')
        return self._start_session

    @property
    def end_session(self):
        if False:
            for i in range(10):
                print('nop')
        return self._end_session

    @property
    def first_open(self):
        if False:
            return 10
        return self._first_open

    @property
    def last_close(self):
        if False:
            for i in range(10):
                print('nop')
        return self._last_close

    @property
    def trading_calendar(self):
        if False:
            print('Hello World!')
        return self._trading_calendar

    @property
    @remember_last
    def sessions(self):
        if False:
            for i in range(10):
                print('nop')
        return self._trading_calendar.sessions_in_range(self.start_session, self.end_session)

    def create_new(self, start_session, end_session, data_frequency=None):
        if False:
            while True:
                i = 10
        if data_frequency is None:
            data_frequency = self.data_frequency
        return SimulationParameters(start_session, end_session, self._trading_calendar, capital_base=self.capital_base, emission_rate=self.emission_rate, data_frequency=data_frequency, arena=self.arena)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '\n{class_name}(\n    start_session={start_session},\n    end_session={end_session},\n    capital_base={capital_base},\n    data_frequency={data_frequency},\n    emission_rate={emission_rate},\n    first_open={first_open},\n    last_close={last_close},\n    trading_calendar={trading_calendar}\n)'.format(class_name=self.__class__.__name__, start_session=self.start_session, end_session=self.end_session, capital_base=self.capital_base, data_frequency=self.data_frequency, emission_rate=self.emission_rate, first_open=self.first_open, last_close=self.last_close, trading_calendar=self._trading_calendar)