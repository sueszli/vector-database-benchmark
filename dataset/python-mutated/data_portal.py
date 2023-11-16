from operator import mul
from logbook import Logger
import numpy as np
from numpy import float64, int64, nan
import pandas as pd
from pandas import isnull
from six import iteritems
from six.moves import reduce
from zipline.assets import Asset, AssetConvertible, Equity, Future, PricingDataAssociable
from zipline.assets.continuous_futures import ContinuousFuture
from zipline.data.continuous_future_reader import ContinuousFutureSessionBarReader, ContinuousFutureMinuteBarReader
from zipline.assets.roll_finder import CalendarRollFinder, VolumeRollFinder
from zipline.data.dispatch_bar_reader import AssetDispatchMinuteBarReader, AssetDispatchSessionBarReader
from zipline.data.resample import DailyHistoryAggregator, ReindexMinuteBarReader, ReindexSessionBarReader
from zipline.data.history_loader import DailyHistoryLoader, MinuteHistoryLoader
from zipline.data.bar_reader import NoDataOnDate
from zipline.utils.math_utils import nansum, nanmean, nanstd
from zipline.utils.memoize import remember_last, weak_lru_cache
from zipline.utils.pandas_utils import normalize_date, timedelta_to_integral_minutes
from zipline.errors import HistoryWindowStartsBeforeData
log = Logger('DataPortal')
BASE_FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume', 'price', 'contract', 'sid', 'last_traded'])
OHLCV_FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume'])
OHLCVP_FIELDS = frozenset(['open', 'high', 'low', 'close', 'volume', 'price'])
HISTORY_FREQUENCIES = set(['1m', '1d'])
DEFAULT_MINUTE_HISTORY_PREFETCH = 1560
DEFAULT_DAILY_HISTORY_PREFETCH = 40
_DEF_M_HIST_PREFETCH = DEFAULT_MINUTE_HISTORY_PREFETCH
_DEF_D_HIST_PREFETCH = DEFAULT_DAILY_HISTORY_PREFETCH

class DataPortal(object):
    """Interface to all of the data that a zipline simulation needs.

    This is used by the simulation runner to answer questions about the data,
    like getting the prices of assets on a given day or to service history
    calls.

    Parameters
    ----------
    asset_finder : zipline.assets.assets.AssetFinder
        The AssetFinder instance used to resolve assets.
    trading_calendar: zipline.utils.calendar.exchange_calendar.TradingCalendar
        The calendar instance used to provide minute->session information.
    first_trading_day : pd.Timestamp
        The first trading day for the simulation.
    equity_daily_reader : BcolzDailyBarReader, optional
        The daily bar reader for equities. This will be used to service
        daily data backtests or daily history calls in a minute backetest.
        If a daily bar reader is not provided but a minute bar reader is,
        the minutes will be rolled up to serve the daily requests.
    equity_minute_reader : BcolzMinuteBarReader, optional
        The minute bar reader for equities. This will be used to service
        minute data backtests or minute history calls. This can be used
        to serve daily calls if no daily bar reader is provided.
    future_daily_reader : BcolzDailyBarReader, optional
        The daily bar ready for futures. This will be used to service
        daily data backtests or daily history calls in a minute backetest.
        If a daily bar reader is not provided but a minute bar reader is,
        the minutes will be rolled up to serve the daily requests.
    future_minute_reader : BcolzFutureMinuteBarReader, optional
        The minute bar reader for futures. This will be used to service
        minute data backtests or minute history calls. This can be used
        to serve daily calls if no daily bar reader is provided.
    adjustment_reader : SQLiteAdjustmentWriter, optional
        The adjustment reader. This is used to apply splits, dividends, and
        other adjustment data to the raw data from the readers.
    last_available_session : pd.Timestamp, optional
        The last session to make available in session-level data.
    last_available_minute : pd.Timestamp, optional
        The last minute to make available in minute-level data.
    """

    def __init__(self, asset_finder, trading_calendar, first_trading_day, equity_daily_reader=None, equity_minute_reader=None, future_daily_reader=None, future_minute_reader=None, adjustment_reader=None, last_available_session=None, last_available_minute=None, minute_history_prefetch_length=_DEF_M_HIST_PREFETCH, daily_history_prefetch_length=_DEF_D_HIST_PREFETCH):
        if False:
            i = 10
            return i + 15
        self.trading_calendar = trading_calendar
        self.asset_finder = asset_finder
        self._adjustment_reader = adjustment_reader
        self._splits_dict = {}
        self._mergers_dict = {}
        self._dividends_dict = {}
        self._augmented_sources_map = {}
        self._extra_source_df = None
        self._first_available_session = first_trading_day
        if last_available_session:
            self._last_available_session = last_available_session
        else:
            last_sessions = [reader.last_available_dt for reader in [equity_daily_reader, future_daily_reader] if reader is not None]
            if last_sessions:
                self._last_available_session = min(last_sessions)
            else:
                self._last_available_session = None
        if last_available_minute:
            self._last_available_minute = last_available_minute
        else:
            last_minutes = [reader.last_available_dt for reader in [equity_minute_reader, future_minute_reader] if reader is not None]
            if last_minutes:
                self._last_available_minute = max(last_minutes)
            else:
                self._last_available_minute = None
        aligned_equity_minute_reader = self._ensure_reader_aligned(equity_minute_reader)
        aligned_equity_session_reader = self._ensure_reader_aligned(equity_daily_reader)
        aligned_future_minute_reader = self._ensure_reader_aligned(future_minute_reader)
        aligned_future_session_reader = self._ensure_reader_aligned(future_daily_reader)
        self._roll_finders = {'calendar': CalendarRollFinder(self.trading_calendar, self.asset_finder)}
        aligned_minute_readers = {}
        aligned_session_readers = {}
        if aligned_equity_minute_reader is not None:
            aligned_minute_readers[Equity] = aligned_equity_minute_reader
        if aligned_equity_session_reader is not None:
            aligned_session_readers[Equity] = aligned_equity_session_reader
        if aligned_future_minute_reader is not None:
            aligned_minute_readers[Future] = aligned_future_minute_reader
            aligned_minute_readers[ContinuousFuture] = ContinuousFutureMinuteBarReader(aligned_future_minute_reader, self._roll_finders)
        if aligned_future_session_reader is not None:
            aligned_session_readers[Future] = aligned_future_session_reader
            self._roll_finders['volume'] = VolumeRollFinder(self.trading_calendar, self.asset_finder, aligned_future_session_reader)
            aligned_session_readers[ContinuousFuture] = ContinuousFutureSessionBarReader(aligned_future_session_reader, self._roll_finders)
        _dispatch_minute_reader = AssetDispatchMinuteBarReader(self.trading_calendar, self.asset_finder, aligned_minute_readers, self._last_available_minute)
        _dispatch_session_reader = AssetDispatchSessionBarReader(self.trading_calendar, self.asset_finder, aligned_session_readers, self._last_available_session)
        self._pricing_readers = {'minute': _dispatch_minute_reader, 'daily': _dispatch_session_reader}
        self._daily_aggregator = DailyHistoryAggregator(self.trading_calendar.schedule.market_open, _dispatch_minute_reader, self.trading_calendar)
        self._history_loader = DailyHistoryLoader(self.trading_calendar, _dispatch_session_reader, self._adjustment_reader, self.asset_finder, self._roll_finders, prefetch_length=daily_history_prefetch_length)
        self._minute_history_loader = MinuteHistoryLoader(self.trading_calendar, _dispatch_minute_reader, self._adjustment_reader, self.asset_finder, self._roll_finders, prefetch_length=minute_history_prefetch_length)
        self._first_trading_day = first_trading_day
        (self._first_trading_minute, _) = self.trading_calendar.open_and_close_for_session(self._first_trading_day) if self._first_trading_day is not None else (None, None)
        self._first_trading_day_loc = self.trading_calendar.all_sessions.get_loc(self._first_trading_day) if self._first_trading_day is not None else None

    def _ensure_reader_aligned(self, reader):
        if False:
            print('Hello World!')
        if reader is None:
            return
        if reader.trading_calendar.name == self.trading_calendar.name:
            return reader
        elif reader.data_frequency == 'minute':
            return ReindexMinuteBarReader(self.trading_calendar, reader, self._first_available_session, self._last_available_session)
        elif reader.data_frequency == 'session':
            return ReindexSessionBarReader(self.trading_calendar, reader, self._first_available_session, self._last_available_session)

    def _reindex_extra_source(self, df, source_date_index):
        if False:
            print('Hello World!')
        return df.reindex(index=source_date_index, method='ffill')

    def handle_extra_source(self, source_df, sim_params):
        if False:
            print('Hello World!')
        '\n        Extra sources always have a sid column.\n\n        We expand the given data (by forward filling) to the full range of\n        the simulation dates, so that lookup is fast during simulation.\n        '
        if source_df is None:
            return
        source_df.index = source_df.index.normalize()
        source_date_index = self.trading_calendar.sessions_in_range(sim_params.start_session, sim_params.end_session)
        grouped_by_sid = source_df.groupby(['sid'])
        group_names = grouped_by_sid.groups.keys()
        group_dict = {}
        for group_name in group_names:
            group_dict[group_name] = grouped_by_sid.get_group(group_name)
        extra_source_df = pd.DataFrame()
        for (identifier, df) in iteritems(group_dict):
            df = df.groupby(level=0).last()
            df = self._reindex_extra_source(df, source_date_index)
            for col_name in df.columns.difference(['sid']):
                if col_name not in self._augmented_sources_map:
                    self._augmented_sources_map[col_name] = {}
                self._augmented_sources_map[col_name][identifier] = df
            extra_source_df = extra_source_df.append(df)
        self._extra_source_df = extra_source_df

    def _get_pricing_reader(self, data_frequency):
        if False:
            for i in range(10):
                print('nop')
        return self._pricing_readers[data_frequency]

    def get_last_traded_dt(self, asset, dt, data_frequency):
        if False:
            i = 10
            return i + 15
        '\n        Given an asset and dt, returns the last traded dt from the viewpoint\n        of the given dt.\n\n        If there is a trade on the dt, the answer is dt provided.\n        '
        return self._get_pricing_reader(data_frequency).get_last_traded_dt(asset, dt)

    @staticmethod
    def _is_extra_source(asset, field, map):
        if False:
            while True:
                i = 10
        '\n        Internal method that determines if this asset/field combination\n        represents a fetcher value or a regular OHLCVP lookup.\n        '
        return not (field in BASE_FIELDS and isinstance(asset, (Asset, ContinuousFuture)))

    def _get_fetcher_value(self, asset, field, dt):
        if False:
            while True:
                i = 10
        day = normalize_date(dt)
        try:
            return self._augmented_sources_map[field][asset].loc[day, field]
        except KeyError:
            return np.NaN

    def _get_single_asset_value(self, session_label, asset, field, dt, data_frequency):
        if False:
            i = 10
            return i + 15
        if self._is_extra_source(asset, field, self._augmented_sources_map):
            return self._get_fetcher_value(asset, field, dt)
        if field not in BASE_FIELDS:
            raise KeyError('Invalid column: ' + str(field))
        if dt < asset.start_date or (data_frequency == 'daily' and session_label > asset.end_date) or (data_frequency == 'minute' and session_label > asset.end_date):
            if field == 'volume':
                return 0
            elif field == 'contract':
                return None
            elif field != 'last_traded':
                return np.NaN
        if data_frequency == 'daily':
            if field == 'contract':
                return self._get_current_contract(asset, session_label)
            else:
                return self._get_daily_spot_value(asset, field, session_label)
        elif field == 'last_traded':
            return self.get_last_traded_dt(asset, dt, 'minute')
        elif field == 'price':
            return self._get_minute_spot_value(asset, 'close', dt, ffill=True)
        elif field == 'contract':
            return self._get_current_contract(asset, dt)
        else:
            return self._get_minute_spot_value(asset, field, dt)

    def get_spot_value(self, assets, field, dt, data_frequency):
        if False:
            print('Hello World!')
        "\n        Public API method that returns a scalar value representing the value\n        of the desired asset's field at either the given dt.\n\n        Parameters\n        ----------\n        assets : Asset, ContinuousFuture, or iterable of same.\n            The asset or assets whose data is desired.\n        field : {'open', 'high', 'low', 'close', 'volume',\n                 'price', 'last_traded'}\n            The desired field of the asset.\n        dt : pd.Timestamp\n            The timestamp for the desired value.\n        data_frequency : str\n            The frequency of the data to query; i.e. whether the data is\n            'daily' or 'minute' bars\n\n        Returns\n        -------\n        value : float, int, or pd.Timestamp\n            The spot value of ``field`` for ``asset`` The return type is based\n            on the ``field`` requested. If the field is one of 'open', 'high',\n            'low', 'close', or 'price', the value will be a float. If the\n            ``field`` is 'volume' the value will be a int. If the ``field`` is\n            'last_traded' the value will be a Timestamp.\n        "
        assets_is_scalar = False
        if isinstance(assets, (AssetConvertible, PricingDataAssociable)):
            assets_is_scalar = True
        else:
            try:
                iter(assets)
            except TypeError:
                raise TypeError("Unexpected 'assets' value of type {}.".format(type(assets)))
        session_label = self.trading_calendar.minute_to_session_label(dt)
        if assets_is_scalar:
            return self._get_single_asset_value(session_label, assets, field, dt, data_frequency)
        else:
            get_single_asset_value = self._get_single_asset_value
            return [get_single_asset_value(session_label, asset, field, dt, data_frequency) for asset in assets]

    def get_scalar_asset_spot_value(self, asset, field, dt, data_frequency):
        if False:
            return 10
        "\n        Public API method that returns a scalar value representing the value\n        of the desired asset's field at either the given dt.\n\n        Parameters\n        ----------\n        assets : Asset\n            The asset or assets whose data is desired. This cannot be\n            an arbitrary AssetConvertible.\n        field : {'open', 'high', 'low', 'close', 'volume',\n                 'price', 'last_traded'}\n            The desired field of the asset.\n        dt : pd.Timestamp\n            The timestamp for the desired value.\n        data_frequency : str\n            The frequency of the data to query; i.e. whether the data is\n            'daily' or 'minute' bars\n\n        Returns\n        -------\n        value : float, int, or pd.Timestamp\n            The spot value of ``field`` for ``asset`` The return type is based\n            on the ``field`` requested. If the field is one of 'open', 'high',\n            'low', 'close', or 'price', the value will be a float. If the\n            ``field`` is 'volume' the value will be a int. If the ``field`` is\n            'last_traded' the value will be a Timestamp.\n        "
        return self._get_single_asset_value(self.trading_calendar.minute_to_session_label(dt), asset, field, dt, data_frequency)

    def get_adjustments(self, assets, field, dt, perspective_dt):
        if False:
            return 10
        "\n        Returns a list of adjustments between the dt and perspective_dt for the\n        given field and list of assets\n\n        Parameters\n        ----------\n        assets : list of type Asset, or Asset\n            The asset, or assets whose adjustments are desired.\n        field : {'open', 'high', 'low', 'close', 'volume',                  'price', 'last_traded'}\n            The desired field of the asset.\n        dt : pd.Timestamp\n            The timestamp for the desired value.\n        perspective_dt : pd.Timestamp\n            The timestamp from which the data is being viewed back from.\n\n        Returns\n        -------\n        adjustments : list[Adjustment]\n            The adjustments to that field.\n        "
        if isinstance(assets, Asset):
            assets = [assets]
        adjustment_ratios_per_asset = []

        def split_adj_factor(x):
            if False:
                while True:
                    i = 10
            return x if field != 'volume' else 1.0 / x
        for asset in assets:
            adjustments_for_asset = []
            split_adjustments = self._get_adjustment_list(asset, self._splits_dict, 'SPLITS')
            for (adj_dt, adj) in split_adjustments:
                if dt < adj_dt <= perspective_dt:
                    adjustments_for_asset.append(split_adj_factor(adj))
                elif adj_dt > perspective_dt:
                    break
            if field != 'volume':
                merger_adjustments = self._get_adjustment_list(asset, self._mergers_dict, 'MERGERS')
                for (adj_dt, adj) in merger_adjustments:
                    if dt < adj_dt <= perspective_dt:
                        adjustments_for_asset.append(adj)
                    elif adj_dt > perspective_dt:
                        break
                dividend_adjustments = self._get_adjustment_list(asset, self._dividends_dict, 'DIVIDENDS')
                for (adj_dt, adj) in dividend_adjustments:
                    if dt < adj_dt <= perspective_dt:
                        adjustments_for_asset.append(adj)
                    elif adj_dt > perspective_dt:
                        break
            ratio = reduce(mul, adjustments_for_asset, 1.0)
            adjustment_ratios_per_asset.append(ratio)
        return adjustment_ratios_per_asset

    def get_adjusted_value(self, asset, field, dt, perspective_dt, data_frequency, spot_value=None):
        if False:
            i = 10
            return i + 15
        "\n        Returns a scalar value representing the value\n        of the desired asset's field at the given dt with adjustments applied.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset whose data is desired.\n        field : {'open', 'high', 'low', 'close', 'volume',                  'price', 'last_traded'}\n            The desired field of the asset.\n        dt : pd.Timestamp\n            The timestamp for the desired value.\n        perspective_dt : pd.Timestamp\n            The timestamp from which the data is being viewed back from.\n        data_frequency : str\n            The frequency of the data to query; i.e. whether the data is\n            'daily' or 'minute' bars\n\n        Returns\n        -------\n        value : float, int, or pd.Timestamp\n            The value of the given ``field`` for ``asset`` at ``dt`` with any\n            adjustments known by ``perspective_dt`` applied. The return type is\n            based on the ``field`` requested. If the field is one of 'open',\n            'high', 'low', 'close', or 'price', the value will be a float. If\n            the ``field`` is 'volume' the value will be a int. If the ``field``\n            is 'last_traded' the value will be a Timestamp.\n        "
        if spot_value is None:
            if self._is_extra_source(asset, field, self._augmented_sources_map):
                spot_value = self.get_spot_value(asset, field, perspective_dt, data_frequency)
            else:
                spot_value = self.get_spot_value(asset, field, dt, data_frequency)
        if isinstance(asset, Equity):
            ratio = self.get_adjustments(asset, field, dt, perspective_dt)[0]
            spot_value *= ratio
        return spot_value

    def _get_minute_spot_value(self, asset, column, dt, ffill=False):
        if False:
            return 10
        reader = self._get_pricing_reader('minute')
        if not ffill:
            try:
                return reader.get_value(asset.sid, dt, column)
            except NoDataOnDate:
                if column != 'volume':
                    return np.nan
                else:
                    return 0
        try:
            result = reader.get_value(asset.sid, dt, column)
            if not pd.isnull(result):
                return result
        except NoDataOnDate:
            pass
        query_dt = reader.get_last_traded_dt(asset, dt)
        if pd.isnull(query_dt):
            return np.nan
        result = reader.get_value(asset.sid, query_dt, column)
        if dt == query_dt or dt.date() == query_dt.date():
            return result
        return self.get_adjusted_value(asset, column, query_dt, dt, 'minute', spot_value=result)

    def _get_daily_spot_value(self, asset, column, dt):
        if False:
            for i in range(10):
                print('nop')
        reader = self._get_pricing_reader('daily')
        if column == 'last_traded':
            last_traded_dt = reader.get_last_traded_dt(asset, dt)
            if isnull(last_traded_dt):
                return pd.NaT
            else:
                return last_traded_dt
        elif column in OHLCV_FIELDS:
            try:
                return reader.get_value(asset, dt, column)
            except NoDataOnDate:
                return np.nan
        elif column == 'price':
            found_dt = dt
            while True:
                try:
                    value = reader.get_value(asset, found_dt, 'close')
                    if not isnull(value):
                        if dt == found_dt:
                            return value
                        else:
                            return self.get_adjusted_value(asset, column, found_dt, dt, 'minute', spot_value=value)
                    else:
                        found_dt -= self.trading_calendar.day
                except NoDataOnDate:
                    return np.nan

    @remember_last
    def _get_days_for_window(self, end_date, bar_count):
        if False:
            i = 10
            return i + 15
        tds = self.trading_calendar.all_sessions
        end_loc = tds.get_loc(end_date)
        start_loc = end_loc - bar_count + 1
        if start_loc < self._first_trading_day_loc:
            raise HistoryWindowStartsBeforeData(first_trading_day=self._first_trading_day.date(), bar_count=bar_count, suggested_start_day=tds[self._first_trading_day_loc + bar_count].date())
        return tds[start_loc:end_loc + 1]

    def _get_history_daily_window(self, assets, end_dt, bar_count, field_to_use, data_frequency):
        if False:
            return 10
        '\n        Internal method that returns a dataframe containing history bars\n        of daily frequency for the given sids.\n        '
        session = self.trading_calendar.minute_to_session_label(end_dt)
        days_for_window = self._get_days_for_window(session, bar_count)
        if len(assets) == 0:
            return pd.DataFrame(None, index=days_for_window, columns=None)
        data = self._get_history_daily_window_data(assets, days_for_window, end_dt, field_to_use, data_frequency)
        return pd.DataFrame(data, index=days_for_window, columns=assets)

    def _get_history_daily_window_data(self, assets, days_for_window, end_dt, field_to_use, data_frequency):
        if False:
            print('Hello World!')
        if data_frequency == 'daily':
            return self._get_daily_window_data(assets, field_to_use, days_for_window, extra_slot=False)
        else:
            daily_data = self._get_daily_window_data(assets, field_to_use, days_for_window[0:-1])
            if field_to_use == 'open':
                minute_value = self._daily_aggregator.opens(assets, end_dt)
            elif field_to_use == 'high':
                minute_value = self._daily_aggregator.highs(assets, end_dt)
            elif field_to_use == 'low':
                minute_value = self._daily_aggregator.lows(assets, end_dt)
            elif field_to_use == 'close':
                minute_value = self._daily_aggregator.closes(assets, end_dt)
            elif field_to_use == 'volume':
                minute_value = self._daily_aggregator.volumes(assets, end_dt)
            elif field_to_use == 'sid':
                minute_value = [int(self._get_current_contract(asset, end_dt)) for asset in assets]
            daily_data[-1] = minute_value
            return daily_data

    def _handle_minute_history_out_of_bounds(self, bar_count):
        if False:
            i = 10
            return i + 15
        cal = self.trading_calendar
        first_trading_minute_loc = cal.all_minutes.get_loc(self._first_trading_minute) if self._first_trading_minute is not None else None
        suggested_start_day = cal.minute_to_session_label(cal.all_minutes[first_trading_minute_loc + bar_count] + cal.day)
        raise HistoryWindowStartsBeforeData(first_trading_day=self._first_trading_day.date(), bar_count=bar_count, suggested_start_day=suggested_start_day.date())

    def _get_history_minute_window(self, assets, end_dt, bar_count, field_to_use):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal method that returns a dataframe containing history bars\n        of minute frequency for the given sids.\n        '
        try:
            minutes_for_window = self.trading_calendar.minutes_window(end_dt, -bar_count)
        except KeyError:
            self._handle_minute_history_out_of_bounds(bar_count)
        if minutes_for_window[0] < self._first_trading_minute:
            self._handle_minute_history_out_of_bounds(bar_count)
        asset_minute_data = self._get_minute_window_data(assets, field_to_use, minutes_for_window)
        return pd.DataFrame(asset_minute_data, index=minutes_for_window, columns=assets)

    def get_history_window(self, assets, end_dt, bar_count, frequency, field, data_frequency, ffill=True):
        if False:
            i = 10
            return i + 15
        '\n        Public API method that returns a dataframe containing the requested\n        history window.  Data is fully adjusted.\n\n        Parameters\n        ----------\n        assets : list of zipline.data.Asset objects\n            The assets whose data is desired.\n\n        bar_count: int\n            The number of bars desired.\n\n        frequency: string\n            "1d" or "1m"\n\n        field: string\n            The desired field of the asset.\n\n        data_frequency: string\n            The frequency of the data to query; i.e. whether the data is\n            \'daily\' or \'minute\' bars.\n\n        ffill: boolean\n            Forward-fill missing values. Only has effect if field\n            is \'price\'.\n\n        Returns\n        -------\n        A dataframe containing the requested data.\n        '
        if field not in OHLCVP_FIELDS and field != 'sid':
            raise ValueError('Invalid field: {0}'.format(field))
        if bar_count < 1:
            raise ValueError('bar_count must be >= 1, but got {}'.format(bar_count))
        if frequency == '1d':
            if field == 'price':
                df = self._get_history_daily_window(assets, end_dt, bar_count, 'close', data_frequency)
            else:
                df = self._get_history_daily_window(assets, end_dt, bar_count, field, data_frequency)
        elif frequency == '1m':
            if field == 'price':
                df = self._get_history_minute_window(assets, end_dt, bar_count, 'close')
            else:
                df = self._get_history_minute_window(assets, end_dt, bar_count, field)
        else:
            raise ValueError('Invalid frequency: {0}'.format(frequency))
        if field == 'price':
            if frequency == '1m':
                ffill_data_frequency = 'minute'
            elif frequency == '1d':
                ffill_data_frequency = 'daily'
            else:
                raise Exception('Only 1d and 1m are supported for forward-filling.')
            assets_with_leading_nan = np.where(isnull(df.iloc[0]))[0]
            (history_start, history_end) = df.index[[0, -1]]
            if ffill_data_frequency == 'daily' and data_frequency == 'minute':
                history_start -= self.trading_calendar.day
            initial_values = []
            for asset in df.columns[assets_with_leading_nan]:
                last_traded = self.get_last_traded_dt(asset, history_start, ffill_data_frequency)
                if isnull(last_traded):
                    initial_values.append(nan)
                else:
                    initial_values.append(self.get_adjusted_value(asset, field, dt=last_traded, perspective_dt=history_end, data_frequency=ffill_data_frequency))
            df.iloc[0, assets_with_leading_nan] = np.array(initial_values, dtype=np.float64)
            df.fillna(method='ffill', inplace=True)
            normed_index = df.index.normalize()
            for asset in df.columns:
                if history_end >= asset.end_date:
                    df.loc[normed_index > asset.end_date, asset] = nan
        return df

    def _get_minute_window_data(self, assets, field, minutes_for_window):
        if False:
            i = 10
            return i + 15
        '\n        Internal method that gets a window of adjusted minute data for an asset\n        and specified date range.  Used to support the history API method for\n        minute bars.\n\n        Missing bars are filled with NaN.\n\n        Parameters\n        ----------\n        assets : iterable[Asset]\n            The assets whose data is desired.\n\n        field: string\n            The specific field to return.  "open", "high", "close_price", etc.\n\n        minutes_for_window: pd.DateTimeIndex\n            The list of minutes representing the desired window.  Each minute\n            is a pd.Timestamp.\n\n        Returns\n        -------\n        A numpy array with requested values.\n        '
        return self._minute_history_loader.history(assets, minutes_for_window, field, False)

    def _get_daily_window_data(self, assets, field, days_in_window, extra_slot=True):
        if False:
            i = 10
            return i + 15
        '\n        Internal method that gets a window of adjusted daily data for a sid\n        and specified date range.  Used to support the history API method for\n        daily bars.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset whose data is desired.\n\n        start_dt: pandas.Timestamp\n            The start of the desired window of data.\n\n        bar_count: int\n            The number of days of data to return.\n\n        field: string\n            The specific field to return.  "open", "high", "close_price", etc.\n\n        extra_slot: boolean\n            Whether to allocate an extra slot in the returned numpy array.\n            This extra slot will hold the data for the last partial day.  It\'s\n            much better to create it here than to create a copy of the array\n            later just to add a slot.\n\n        Returns\n        -------\n        A numpy array with requested values.  Any missing slots filled with\n        nan.\n\n        '
        bar_count = len(days_in_window)
        dtype = float64 if field != 'sid' else int64
        if extra_slot:
            return_array = np.zeros((bar_count + 1, len(assets)), dtype=dtype)
        else:
            return_array = np.zeros((bar_count, len(assets)), dtype=dtype)
        if field != 'volume':
            return_array[:] = np.NAN
        if bar_count != 0:
            data = self._history_loader.history(assets, days_in_window, field, extra_slot)
            if extra_slot:
                return_array[:len(return_array) - 1, :] = data
            else:
                return_array[:len(data)] = data
        return return_array

    def _get_adjustment_list(self, asset, adjustments_dict, table_name):
        if False:
            i = 10
            return i + 15
        '\n        Internal method that returns a list of adjustments for the given sid.\n\n        Parameters\n        ----------\n        asset : Asset\n            The asset for which to return adjustments.\n\n        adjustments_dict: dict\n            A dictionary of sid -> list that is used as a cache.\n\n        table_name: string\n            The table that contains this data in the adjustments db.\n\n        Returns\n        -------\n        adjustments: list\n            A list of [multiplier, pd.Timestamp], earliest first\n\n        '
        if self._adjustment_reader is None:
            return []
        sid = int(asset)
        try:
            adjustments = adjustments_dict[sid]
        except KeyError:
            adjustments = adjustments_dict[sid] = self._adjustment_reader.get_adjustments_for_sid(table_name, sid)
        return adjustments

    def get_splits(self, assets, dt):
        if False:
            i = 10
            return i + 15
        '\n        Returns any splits for the given sids and the given dt.\n\n        Parameters\n        ----------\n        assets : container\n            Assets for which we want splits.\n        dt : pd.Timestamp\n            The date for which we are checking for splits. Note: this is\n            expected to be midnight UTC.\n\n        Returns\n        -------\n        splits : list[(asset, float)]\n            List of splits, where each split is a (asset, ratio) tuple.\n        '
        if self._adjustment_reader is None or not assets:
            return []
        seconds = int(dt.value / 1000000000.0)
        splits = self._adjustment_reader.conn.execute('SELECT sid, ratio FROM SPLITS WHERE effective_date = ?', (seconds,)).fetchall()
        splits = [split for split in splits if split[0] in assets]
        splits = [(self.asset_finder.retrieve_asset(split[0]), split[1]) for split in splits]
        return splits

    def get_stock_dividends(self, sid, trading_days):
        if False:
            while True:
                i = 10
        '\n        Returns all the stock dividends for a specific sid that occur\n        in the given trading range.\n\n        Parameters\n        ----------\n        sid: int\n            The asset whose stock dividends should be returned.\n\n        trading_days: pd.DatetimeIndex\n            The trading range.\n\n        Returns\n        -------\n        list: A list of objects with all relevant attributes populated.\n        All timestamp fields are converted to pd.Timestamps.\n        '
        if self._adjustment_reader is None:
            return []
        if len(trading_days) == 0:
            return []
        start_dt = trading_days[0].value / 1000000000.0
        end_dt = trading_days[-1].value / 1000000000.0
        dividends = self._adjustment_reader.conn.execute('SELECT * FROM stock_dividend_payouts WHERE sid = ? AND ex_date > ? AND pay_date < ?', (int(sid), start_dt, end_dt)).fetchall()
        dividend_info = []
        for dividend_tuple in dividends:
            dividend_info.append({'declared_date': dividend_tuple[1], 'ex_date': pd.Timestamp(dividend_tuple[2], unit='s'), 'pay_date': pd.Timestamp(dividend_tuple[3], unit='s'), 'payment_sid': dividend_tuple[4], 'ratio': dividend_tuple[5], 'record_date': pd.Timestamp(dividend_tuple[6], unit='s'), 'sid': dividend_tuple[7]})
        return dividend_info

    def contains(self, asset, field):
        if False:
            i = 10
            return i + 15
        return field in BASE_FIELDS or (field in self._augmented_sources_map and asset in self._augmented_sources_map[field])

    def get_fetcher_assets(self, dt):
        if False:
            print('Hello World!')
        '\n        Returns a list of assets for the current date, as defined by the\n        fetcher data.\n\n        Returns\n        -------\n        list: a list of Asset objects.\n        '
        if self._extra_source_df is None:
            return []
        day = normalize_date(dt)
        if day in self._extra_source_df.index:
            assets = self._extra_source_df.loc[day]['sid']
        else:
            return []
        if isinstance(assets, pd.Series):
            return [x for x in assets if isinstance(x, Asset)]
        else:
            return [assets] if isinstance(assets, Asset) else []

    @weak_lru_cache(20)
    def _get_minute_count_for_transform(self, ending_minute, days_count):
        if False:
            return 10
        cal = self.trading_calendar
        ending_session = cal.minute_to_session_label(ending_minute, direction='none')
        ending_session_minute_count = timedelta_to_integral_minutes(ending_minute - cal.open_and_close_for_session(ending_session)[0]) + 1
        if days_count == 1:
            return ending_session_minute_count
        completed_sessions = cal.sessions_window(cal.previous_session_label(ending_session), 2 - days_count)
        completed_sessions_minute_count = self.trading_calendar.minutes_count_for_sessions_in_range(completed_sessions[0], completed_sessions[-1])
        return ending_session_minute_count + completed_sessions_minute_count

    def get_simple_transform(self, asset, transform_name, dt, data_frequency, bars=None):
        if False:
            while True:
                i = 10
        if transform_name == 'returns':
            hst = self.get_history_window([asset], dt, 2, '1d', 'price', data_frequency, ffill=True)[asset]
            return (hst.iloc[-1] - hst.iloc[0]) / hst.iloc[0]
        if bars is None:
            raise ValueError('bars cannot be None!')
        if data_frequency == 'minute':
            freq_str = '1m'
            calculated_bar_count = int(self._get_minute_count_for_transform(dt, bars))
        else:
            freq_str = '1d'
            calculated_bar_count = bars
        price_arr = self.get_history_window([asset], dt, calculated_bar_count, freq_str, 'price', data_frequency, ffill=True)[asset]
        if transform_name == 'mavg':
            return nanmean(price_arr)
        elif transform_name == 'stddev':
            return nanstd(price_arr, ddof=1)
        elif transform_name == 'vwap':
            volume_arr = self.get_history_window([asset], dt, calculated_bar_count, freq_str, 'volume', data_frequency, ffill=True)[asset]
            vol_sum = nansum(volume_arr)
            try:
                ret = nansum(price_arr * volume_arr) / vol_sum
            except ZeroDivisionError:
                ret = np.nan
            return ret

    def get_current_future_chain(self, continuous_future, dt):
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the future chain for the contract at the given `dt` according\n        the `continuous_future` specification.\n\n        Returns\n        -------\n\n        future_chain : list[Future]\n            A list of active futures, where the first index is the current\n            contract specified by the continuous future definition, the second\n            is the next upcoming contract and so on.\n        '
        rf = self._roll_finders[continuous_future.roll_style]
        session = self.trading_calendar.minute_to_session_label(dt)
        contract_center = rf.get_contract_center(continuous_future.root_symbol, session, continuous_future.offset)
        oc = self.asset_finder.get_ordered_contracts(continuous_future.root_symbol)
        chain = oc.active_chain(contract_center, session.value)
        return self.asset_finder.retrieve_all(chain)

    def _get_current_contract(self, continuous_future, dt):
        if False:
            for i in range(10):
                print('nop')
        rf = self._roll_finders[continuous_future.roll_style]
        contract_sid = rf.get_contract_center(continuous_future.root_symbol, dt, continuous_future.offset)
        if contract_sid is None:
            return None
        return self.asset_finder.retrieve_asset(contract_sid)

    @property
    def adjustment_reader(self):
        if False:
            i = 10
            return i + 15
        return self._adjustment_reader