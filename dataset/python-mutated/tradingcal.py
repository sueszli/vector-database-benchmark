from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime, timedelta, time
from .metabase import MetaParams
from backtrader.utils.py3 import string_types, with_metaclass
from backtrader.utils import UTC
__all__ = ['TradingCalendarBase', 'TradingCalendar', 'PandasMarketCalendar']
_time_max = time(hour=23, minute=59, second=59, microsecond=999990)
(MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY) = range(7)
(ISONODAY, ISOMONDAY, ISOTUESDAY, ISOWEDNESDAY, ISOTHURSDAY, ISOFRIDAY, ISOSATURDAY, ISOSUNDAY) = range(8)
WEEKEND = [SATURDAY, SUNDAY]
ISOWEEKEND = [ISOSATURDAY, ISOSUNDAY]
ONEDAY = timedelta(days=1)

class TradingCalendarBase(with_metaclass(MetaParams, object)):

    def _nextday(self, day):
        if False:
            print('Hello World!')
        '\n        Returns the next trading day (datetime/date instance) after ``day``\n        (datetime/date instance) and the isocalendar components\n\n        The return value is a tuple with 2 components: (nextday, (y, w, d))\n        '
        raise NotImplementedError

    def schedule(self, day):
        if False:
            while True:
                i = 10
        '\n        Returns a tuple with the opening and closing times (``datetime.time``)\n        for the given ``date`` (``datetime/date`` instance)\n        '
        raise NotImplementedError

    def nextday(self, day):
        if False:
            return 10
        '\n        Returns the next trading day (datetime/date instance) after ``day``\n        (datetime/date instance)\n        '
        return self._nextday(day)[0]

    def nextday_week(self, day):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the iso week number of the next trading day, given a ``day``\n        (datetime/date) instance\n        '
        self._nextday(day)[1][1]

    def last_weekday(self, day):
        if False:
            i = 10
            return i + 15
        '\n        Returns ``True`` if the given ``day`` (datetime/date) instance is the\n        last trading day of this week\n        '
        return day.isocalendar()[1] != self._nextday(day)[1][1]

    def last_monthday(self, day):
        if False:
            print('Hello World!')
        '\n        Returns ``True`` if the given ``day`` (datetime/date) instance is the\n        last trading day of this month\n        '
        return day.month != self._nextday(day)[0].month

    def last_yearday(self, day):
        if False:
            return 10
        '\n        Returns ``True`` if the given ``day`` (datetime/date) instance is the\n        last trading day of this month\n        '
        return day.year != self._nextday(day)[0].year

class TradingCalendar(TradingCalendarBase):
    """
    Wrapper of ``pandas_market_calendars`` for a trading calendar. The package
    ``pandas_market_calendar`` must be installed

    Params:

      - ``open`` (default ``time.min``)

        Regular start of the session

      - ``close`` (default ``time.max``)

        Regular end of the session

      - ``holidays`` (default ``[]``)

        List of non-trading days (``datetime.datetime`` instances)

      - ``earlydays`` (default ``[]``)

        List of tuples determining the date and opening/closing times of days
        which do not conform to the regular trading hours where each tuple has
        (``datetime.datetime``, ``datetime.time``, ``datetime.time`` )

      - ``offdays`` (default ``ISOWEEKEND``)

        A list of weekdays in ISO format (Monday: 1 -> Sunday: 7) in which the
        market doesn't trade. This is usually Saturday and Sunday and hence the
        default

    """
    params = (('open', time.min), ('close', _time_max), ('holidays', []), ('earlydays', []), ('offdays', ISOWEEKEND))

    def __init__(self):
        if False:
            print('Hello World!')
        self._earlydays = [x[0] for x in self.p.earlydays]

    def _nextday(self, day):
        if False:
            while True:
                i = 10
        '\n        Returns the next trading day (datetime/date instance) after ``day``\n        (datetime/date instance) and the isocalendar components\n\n        The return value is a tuple with 2 components: (nextday, (y, w, d))\n        '
        while True:
            day += ONEDAY
            isocal = day.isocalendar()
            if isocal[2] in self.p.offdays or day in self.p.holidays:
                continue
            return (day, isocal)

    def schedule(self, day, tz=None):
        if False:
            while True:
                i = 10
        '\n        Returns the opening and closing times for the given ``day``. If the\n        method is called, the assumption is that ``day`` is an actual trading\n        day\n\n        The return value is a tuple with 2 components: opentime, closetime\n        '
        while True:
            dt = day.date()
            try:
                i = self._earlydays.index(dt)
                (o, c) = self.p.earlydays[i][1:]
            except ValueError:
                (o, c) = (self.p.open, self.p.close)
            closing = datetime.combine(dt, c)
            if tz is not None:
                closing = tz.localize(closing).astimezone(UTC)
                closing = closing.replace(tzinfo=None)
            if day > closing:
                day += ONEDAY
                continue
            opening = datetime.combine(dt, o)
            if tz is not None:
                opening = tz.localize(opening).astimezone(UTC)
                opening = opening.replace(tzinfo=None)
            return (opening, closing)

class PandasMarketCalendar(TradingCalendarBase):
    """
    Wrapper of ``pandas_market_calendars`` for a trading calendar. The package
    ``pandas_market_calendar`` must be installed

    Params:

      - ``calendar`` (default ``None``)

        The param ``calendar`` accepts the following:

        - string: the name of one of the calendars supported, for example
          `NYSE`. The wrapper will attempt to get a calendar instance

        - calendar instance: as returned by ``get_calendar('NYSE')``

      - ``cachesize`` (default ``365``)

        Number of days to cache in advance for lookup

    See also:

      - https://github.com/rsheftel/pandas_market_calendars

      - http://pandas-market-calendars.readthedocs.io/

    """
    params = (('calendar', None), ('cachesize', 365))

    def __init__(self):
        if False:
            while True:
                i = 10
        self._calendar = self.p.calendar
        if isinstance(self._calendar, string_types):
            import pandas_market_calendars as mcal
            self._calendar = mcal.get_calendar(self._calendar)
        import pandas as pd
        self.dcache = pd.DatetimeIndex([0.0])
        self.idcache = pd.DataFrame(index=pd.DatetimeIndex([0.0]))
        self.csize = timedelta(days=self.p.cachesize)

    def _nextday(self, day):
        if False:
            while True:
                i = 10
        '\n        Returns the next trading day (datetime/date instance) after ``day``\n        (datetime/date instance) and the isocalendar components\n\n        The return value is a tuple with 2 components: (nextday, (y, w, d))\n        '
        day += ONEDAY
        while True:
            i = self.dcache.searchsorted(day)
            if i == len(self.dcache):
                self.dcache = self._calendar.valid_days(day, day + self.csize)
                continue
            d = self.dcache[i].to_pydatetime()
            return (d, d.isocalendar())

    def schedule(self, day, tz=None):
        if False:
            i = 10
            return i + 15
        '\n        Returns the opening and closing times for the given ``day``. If the\n        method is called, the assumption is that ``day`` is an actual trading\n        day\n\n        The return value is a tuple with 2 components: opentime, closetime\n        '
        while True:
            i = self.idcache.index.searchsorted(day.date())
            if i == len(self.idcache):
                self.idcache = self._calendar.schedule(day, day + self.csize)
                continue
            st = (x.tz_localize(None) for x in self.idcache.iloc[i, 0:2])
            (opening, closing) = st
            if day > closing:
                day += ONEDAY
                continue
            return (opening.to_pydatetime(), closing.to_pydatetime())