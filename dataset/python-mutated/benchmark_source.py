import pandas as pd
from zipline.errors import InvalidBenchmarkAsset, BenchmarkAssetNotAvailableTooEarly, BenchmarkAssetNotAvailableTooLate

class BenchmarkSource(object):

    def __init__(self, benchmark_asset, trading_calendar, sessions, data_portal, emission_rate='daily', benchmark_returns=None):
        if False:
            print('Hello World!')
        self.benchmark_asset = benchmark_asset
        self.sessions = sessions
        self.emission_rate = emission_rate
        self.data_portal = data_portal
        if len(sessions) == 0:
            self._precalculated_series = pd.Series()
        elif benchmark_asset is not None:
            self._validate_benchmark(benchmark_asset)
            (self._precalculated_series, self._daily_returns) = self._initialize_precalculated_series(benchmark_asset, trading_calendar, sessions, data_portal)
        elif benchmark_returns is not None:
            self._daily_returns = daily_series = benchmark_returns.reindex(sessions).fillna(0)
            if self.emission_rate == 'minute':
                minutes = trading_calendar.minutes_for_sessions_in_range(sessions[0], sessions[-1])
                minute_series = daily_series.reindex(index=minutes, method='ffill')
                self._precalculated_series = minute_series
            else:
                self._precalculated_series = daily_series
        else:
            raise Exception('Must provide either benchmark_asset or benchmark_returns.')

    def get_value(self, dt):
        if False:
            while True:
                i = 10
        "Look up the returns for a given dt.\n\n        Parameters\n        ----------\n        dt : datetime\n            The label to look up.\n\n        Returns\n        -------\n        returns : float\n            The returns at the given dt or session.\n\n        See Also\n        --------\n        :class:`zipline.sources.benchmark_source.BenchmarkSource.daily_returns`\n\n        .. warning::\n\n           This method expects minute inputs if ``emission_rate == 'minute'``\n           and session labels when ``emission_rate == 'daily``.\n        "
        return self._precalculated_series.loc[dt]

    def get_range(self, start_dt, end_dt):
        if False:
            i = 10
            return i + 15
        "Look up the returns for a given period.\n\n        Parameters\n        ----------\n        start_dt : datetime\n            The inclusive start label.\n        end_dt : datetime\n            The inclusive end label.\n\n        Returns\n        -------\n        returns : pd.Series\n            The series of returns.\n\n        See Also\n        --------\n        :class:`zipline.sources.benchmark_source.BenchmarkSource.daily_returns`\n\n        .. warning::\n\n           This method expects minute inputs if ``emission_rate == 'minute'``\n           and session labels when ``emission_rate == 'daily``.\n        "
        return self._precalculated_series.loc[start_dt:end_dt]

    def daily_returns(self, start, end=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the daily returns for the given period.\n\n        Parameters\n        ----------\n        start : datetime\n            The inclusive starting session label.\n        end : datetime, optional\n            The inclusive ending session label. If not provided, treat\n            ``start`` as a scalar key.\n\n        Returns\n        -------\n        returns : pd.Series or float\n            The returns in the given period. The index will be the trading\n            calendar in the range [start, end]. If just ``start`` is provided,\n            return the scalar value on that day.\n        '
        if end is None:
            return self._daily_returns[start]
        return self._daily_returns[start:end]

    def _validate_benchmark(self, benchmark_asset):
        if False:
            while True:
                i = 10
        stock_dividends = self.data_portal.get_stock_dividends(self.benchmark_asset, self.sessions)
        if len(stock_dividends) > 0:
            raise InvalidBenchmarkAsset(sid=str(self.benchmark_asset), dt=stock_dividends[0]['ex_date'])
        if benchmark_asset.start_date > self.sessions[0]:
            raise BenchmarkAssetNotAvailableTooEarly(sid=str(self.benchmark_asset), dt=self.sessions[0], start_dt=benchmark_asset.start_date)
        if benchmark_asset.end_date < self.sessions[-1]:
            raise BenchmarkAssetNotAvailableTooLate(sid=str(self.benchmark_asset), dt=self.sessions[-1], end_dt=benchmark_asset.end_date)

    @staticmethod
    def _compute_daily_returns(g):
        if False:
            i = 10
            return i + 15
        return (g[-1] - g[0]) / g[0]

    @classmethod
    def downsample_minute_return_series(cls, trading_calendar, minutely_returns):
        if False:
            while True:
                i = 10
        sessions = trading_calendar.minute_index_to_session_labels(minutely_returns.index)
        closes = trading_calendar.session_closes_in_range(sessions[0], sessions[-1])
        daily_returns = minutely_returns[closes].pct_change()
        daily_returns.index = closes.index
        return daily_returns.iloc[1:]

    def _initialize_precalculated_series(self, asset, trading_calendar, trading_days, data_portal):
        if False:
            print('Hello World!')
        "\n        Internal method that pre-calculates the benchmark return series for\n        use in the simulation.\n\n        Parameters\n        ----------\n        asset:  Asset to use\n\n        trading_calendar: TradingCalendar\n\n        trading_days: pd.DateTimeIndex\n\n        data_portal: DataPortal\n\n        Notes\n        -----\n        If the benchmark asset started trading after the simulation start,\n        or finished trading before the simulation end, exceptions are raised.\n\n        If the benchmark asset started trading the same day as the simulation\n        start, the first available minute price on that day is used instead\n        of the previous close.\n\n        We use history to get an adjusted price history for each day's close,\n        as of the look-back date (the last day of the simulation).  Prices are\n        fully adjusted for dividends, splits, and mergers.\n\n        Returns\n        -------\n        returns : pd.Series\n            indexed by trading day, whose values represent the %\n            change from close to close.\n        daily_returns : pd.Series\n            the partial daily returns for each minute\n        "
        if self.emission_rate == 'minute':
            minutes = trading_calendar.minutes_for_sessions_in_range(self.sessions[0], self.sessions[-1])
            benchmark_series = data_portal.get_history_window([asset], minutes[-1], bar_count=len(minutes) + 1, frequency='1m', field='price', data_frequency=self.emission_rate, ffill=True)[asset]
            return (benchmark_series.pct_change()[1:], self.downsample_minute_return_series(trading_calendar, benchmark_series))
        start_date = asset.start_date
        if start_date < trading_days[0]:
            benchmark_series = data_portal.get_history_window([asset], trading_days[-1], bar_count=len(trading_days) + 1, frequency='1d', field='price', data_frequency=self.emission_rate, ffill=True)[asset]
            returns = benchmark_series.pct_change()[1:]
            return (returns, returns)
        elif start_date == trading_days[0]:
            benchmark_series = data_portal.get_history_window([asset], trading_days[-1], bar_count=len(trading_days), frequency='1d', field='price', data_frequency=self.emission_rate, ffill=True)[asset]
            first_open = data_portal.get_spot_value(asset, 'open', trading_days[0], 'daily')
            first_close = data_portal.get_spot_value(asset, 'close', trading_days[0], 'daily')
            first_day_return = (first_close - first_open) / first_open
            returns = benchmark_series.pct_change()[:]
            returns[0] = first_day_return
            return (returns, returns)
        else:
            raise ValueError('cannot set benchmark to asset that does not exist during the simulation period (asset start date=%r)' % start_date)