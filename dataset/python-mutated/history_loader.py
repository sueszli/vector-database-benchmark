from abc import ABCMeta, abstractmethod, abstractproperty
from numpy import concatenate
from lru import LRU
from pandas import isnull
from toolz import sliding_window
from six import with_metaclass
from zipline.assets import Equity, Future
from zipline.assets.continuous_futures import ContinuousFuture
from zipline.lib._int64window import AdjustedArrayWindow as Int64Window
from zipline.lib._float64window import AdjustedArrayWindow as Float64Window
from zipline.lib.adjustment import Float64Multiply, Float64Add
from zipline.utils.cache import ExpiringCache
from zipline.utils.math_utils import number_of_decimal_places
from zipline.utils.memoize import lazyval
from zipline.utils.numpy_utils import float64_dtype
from zipline.utils.pandas_utils import find_in_sorted_index, normalize_date
DEFAULT_ASSET_PRICE_DECIMALS = 3

class HistoryCompatibleUSEquityAdjustmentReader(object):

    def __init__(self, adjustment_reader):
        if False:
            return 10
        self._adjustments_reader = adjustment_reader

    def load_pricing_adjustments(self, columns, dts, assets):
        if False:
            return 10
        '\n        Returns\n        -------\n        adjustments : list[dict[int -> Adjustment]]\n            A list, where each element corresponds to the `columns`, of\n            mappings from index to adjustment objects to apply at that index.\n        '
        out = [None] * len(columns)
        for (i, column) in enumerate(columns):
            adjs = {}
            for asset in assets:
                adjs.update(self._get_adjustments_in_range(asset, dts, column))
            out[i] = adjs
        return out

    def _get_adjustments_in_range(self, asset, dts, field):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the Float64Multiply objects to pass to an AdjustedArrayWindow.\n\n        For the use of AdjustedArrayWindow in the loader, which looks back\n        from current simulation time back to a window of data the dictionary is\n        structured with:\n        - the key into the dictionary for adjustments is the location of the\n        day from which the window is being viewed.\n        - the start of all multiply objects is always 0 (in each window all\n          adjustments are overlapping)\n        - the end of the multiply object is the location before the calendar\n          location of the adjustment action, making all days before the event\n          adjusted.\n\n        Parameters\n        ----------\n        asset : Asset\n            The assets for which to get adjustments.\n        dts : iterable of datetime64-like\n            The dts for which adjustment data is needed.\n        field : str\n            OHLCV field for which to get the adjustments.\n\n        Returns\n        -------\n        out : dict[loc -> Float64Multiply]\n            The adjustments as a dict of loc -> Float64Multiply\n        '
        sid = int(asset)
        start = normalize_date(dts[0])
        end = normalize_date(dts[-1])
        adjs = {}
        if field != 'volume':
            mergers = self._adjustments_reader.get_adjustments_for_sid('mergers', sid)
            for m in mergers:
                dt = m[0]
                if start < dt <= end:
                    end_loc = dts.searchsorted(dt)
                    adj_loc = end_loc
                    mult = Float64Multiply(0, end_loc - 1, 0, 0, m[1])
                    try:
                        adjs[adj_loc].append(mult)
                    except KeyError:
                        adjs[adj_loc] = [mult]
            divs = self._adjustments_reader.get_adjustments_for_sid('dividends', sid)
            for d in divs:
                dt = d[0]
                if start < dt <= end:
                    end_loc = dts.searchsorted(dt)
                    adj_loc = end_loc
                    mult = Float64Multiply(0, end_loc - 1, 0, 0, d[1])
                    try:
                        adjs[adj_loc].append(mult)
                    except KeyError:
                        adjs[adj_loc] = [mult]
        splits = self._adjustments_reader.get_adjustments_for_sid('splits', sid)
        for s in splits:
            dt = s[0]
            if start < dt <= end:
                if field == 'volume':
                    ratio = 1.0 / s[1]
                else:
                    ratio = s[1]
                end_loc = dts.searchsorted(dt)
                adj_loc = end_loc
                mult = Float64Multiply(0, end_loc - 1, 0, 0, ratio)
                try:
                    adjs[adj_loc].append(mult)
                except KeyError:
                    adjs[adj_loc] = [mult]
        return adjs

class ContinuousFutureAdjustmentReader(object):
    """
    Calculates adjustments for continuous futures, based on the
    close and open of the contracts on the either side of each roll.
    """

    def __init__(self, trading_calendar, asset_finder, bar_reader, roll_finders, frequency):
        if False:
            return 10
        self._trading_calendar = trading_calendar
        self._asset_finder = asset_finder
        self._bar_reader = bar_reader
        self._roll_finders = roll_finders
        self._frequency = frequency

    def load_pricing_adjustments(self, columns, dts, assets):
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        adjustments : list[dict[int -> Adjustment]]\n            A list, where each element corresponds to the `columns`, of\n            mappings from index to adjustment objects to apply at that index.\n        '
        out = [None] * len(columns)
        for (i, column) in enumerate(columns):
            adjs = {}
            for asset in assets:
                adjs.update(self._get_adjustments_in_range(asset, dts, column))
            out[i] = adjs
        return out

    def _make_adjustment(self, adjustment_type, front_close, back_close, end_loc):
        if False:
            return 10
        adj_base = back_close - front_close
        if adjustment_type == 'mul':
            adj_value = 1.0 + adj_base / front_close
            adj_class = Float64Multiply
        elif adjustment_type == 'add':
            adj_value = adj_base
            adj_class = Float64Add
        return adj_class(0, end_loc, 0, 0, adj_value)

    def _get_adjustments_in_range(self, cf, dts, field):
        if False:
            for i in range(10):
                print('nop')
        if field == 'volume' or field == 'sid':
            return {}
        if cf.adjustment is None:
            return {}
        rf = self._roll_finders[cf.roll_style]
        partitions = []
        rolls = rf.get_rolls(cf.root_symbol, dts[0], dts[-1], cf.offset)
        tc = self._trading_calendar
        adjs = {}
        for (front, back) in sliding_window(2, rolls):
            (front_sid, roll_dt) = front
            back_sid = back[0]
            dt = tc.previous_session_label(roll_dt)
            if self._frequency == 'minute':
                dt = tc.open_and_close_for_session(dt)[1]
                roll_dt = tc.open_and_close_for_session(roll_dt)[0]
            partitions.append((front_sid, back_sid, dt, roll_dt))
        for partition in partitions:
            (front_sid, back_sid, dt, roll_dt) = partition
            last_front_dt = self._bar_reader.get_last_traded_dt(self._asset_finder.retrieve_asset(front_sid), dt)
            last_back_dt = self._bar_reader.get_last_traded_dt(self._asset_finder.retrieve_asset(back_sid), dt)
            if isnull(last_front_dt) or isnull(last_back_dt):
                continue
            front_close = self._bar_reader.get_value(front_sid, last_front_dt, 'close')
            back_close = self._bar_reader.get_value(back_sid, last_back_dt, 'close')
            adj_loc = dts.searchsorted(roll_dt)
            end_loc = adj_loc - 1
            adj = self._make_adjustment(cf.adjustment, front_close, back_close, end_loc)
            try:
                adjs[adj_loc].append(adj)
            except KeyError:
                adjs[adj_loc] = [adj]
        return adjs

class SlidingWindow(object):
    """
    Wrapper around an AdjustedArrayWindow which supports monotonically
    increasing (by datetime) requests for a sized window of data.

    Parameters
    ----------
    window : AdjustedArrayWindow
       Window of pricing data with prefetched values beyond the current
       simulation dt.
    cal_start : int
       Index in the overall calendar at which the window starts.
    """

    def __init__(self, window, size, cal_start, offset):
        if False:
            while True:
                i = 10
        self.window = window
        self.cal_start = cal_start
        self.current = next(window)
        self.offset = offset
        self.most_recent_ix = self.cal_start + size

    def get(self, end_ix):
        if False:
            return 10
        '\n        Returns\n        -------\n        out : A np.ndarray of the equity pricing up to end_ix after adjustments\n              and rounding have been applied.\n        '
        if self.most_recent_ix == end_ix:
            return self.current
        target = end_ix - self.cal_start - self.offset + 1
        self.current = self.window.seek(target)
        self.most_recent_ix = end_ix
        return self.current

class HistoryLoader(with_metaclass(ABCMeta)):
    """
    Loader for sliding history windows, with support for adjustments.

    Parameters
    ----------
    trading_calendar: TradingCalendar
        Contains the grouping logic needed to assign minutes to periods.
    reader : DailyBarReader, MinuteBarReader
        Reader for pricing bars.
    adjustment_reader : SQLiteAdjustmentReader
        Reader for adjustment data.
    """
    FIELDS = ('open', 'high', 'low', 'close', 'volume', 'sid')

    def __init__(self, trading_calendar, reader, equity_adjustment_reader, asset_finder, roll_finders=None, sid_cache_size=1000, prefetch_length=0):
        if False:
            print('Hello World!')
        self.trading_calendar = trading_calendar
        self._asset_finder = asset_finder
        self._reader = reader
        self._adjustment_readers = {}
        if equity_adjustment_reader is not None:
            self._adjustment_readers[Equity] = HistoryCompatibleUSEquityAdjustmentReader(equity_adjustment_reader)
        if roll_finders:
            self._adjustment_readers[ContinuousFuture] = ContinuousFutureAdjustmentReader(trading_calendar, asset_finder, reader, roll_finders, self._frequency)
        self._window_blocks = {field: ExpiringCache(LRU(sid_cache_size)) for field in self.FIELDS}
        self._prefetch_length = prefetch_length

    @abstractproperty
    def _frequency(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @abstractproperty
    def _calendar(self):
        if False:
            i = 10
            return i + 15
        pass

    @abstractmethod
    def _array(self, start, end, assets, field):
        if False:
            print('Hello World!')
        pass

    def _decimal_places_for_asset(self, asset, reference_date):
        if False:
            while True:
                i = 10
        if isinstance(asset, Future) and asset.tick_size:
            return number_of_decimal_places(asset.tick_size)
        elif isinstance(asset, ContinuousFuture):
            oc = self._asset_finder.get_ordered_contracts(asset.root_symbol)
            contract_sid = oc.contract_before_auto_close(reference_date.value)
            if contract_sid is not None:
                contract = self._asset_finder.retrieve_asset(contract_sid)
                if contract.tick_size:
                    return number_of_decimal_places(contract.tick_size)
        return DEFAULT_ASSET_PRICE_DECIMALS

    def _ensure_sliding_windows(self, assets, dts, field, is_perspective_after):
        if False:
            print('Hello World!')
        "\n        Ensure that there is a Float64Multiply window for each asset that can\n        provide data for the given parameters.\n        If the corresponding window for the (assets, len(dts), field) does not\n        exist, then create a new one.\n        If a corresponding window does exist for (assets, len(dts), field), but\n        can not provide data for the current dts range, then create a new\n        one and replace the expired window.\n\n        Parameters\n        ----------\n        assets : iterable of Assets\n            The assets in the window\n        dts : iterable of datetime64-like\n            The datetimes for which to fetch data.\n            Makes an assumption that all dts are present and contiguous,\n            in the calendar.\n        field : str\n            The OHLCV field for which to retrieve data.\n        is_perspective_after : bool\n            see: `PricingHistoryLoader.history`\n\n        Returns\n        -------\n        out : list of Float64Window with sufficient data so that each asset's\n        window can provide `get` for the index corresponding with the last\n        value in `dts`\n        "
        end = dts[-1]
        size = len(dts)
        asset_windows = {}
        needed_assets = []
        cal = self._calendar
        assets = self._asset_finder.retrieve_all(assets)
        end_ix = find_in_sorted_index(cal, end)
        for asset in assets:
            try:
                window = self._window_blocks[field].get((asset, size, is_perspective_after), end)
            except KeyError:
                needed_assets.append(asset)
            else:
                if end_ix < window.most_recent_ix:
                    needed_assets.append(asset)
                else:
                    asset_windows[asset] = window
        if needed_assets:
            offset = 0
            start_ix = find_in_sorted_index(cal, dts[0])
            prefetch_end_ix = min(end_ix + self._prefetch_length, len(cal) - 1)
            prefetch_end = cal[prefetch_end_ix]
            prefetch_dts = cal[start_ix:prefetch_end_ix + 1]
            if is_perspective_after:
                adj_end_ix = min(prefetch_end_ix + 1, len(cal) - 1)
                adj_dts = cal[start_ix:adj_end_ix + 1]
            else:
                adj_dts = prefetch_dts
            prefetch_len = len(prefetch_dts)
            array = self._array(prefetch_dts, needed_assets, field)
            if field == 'sid':
                window_type = Int64Window
            else:
                window_type = Float64Window
            view_kwargs = {}
            if field == 'volume':
                array = array.astype(float64_dtype)
            for (i, asset) in enumerate(needed_assets):
                adj_reader = None
                try:
                    adj_reader = self._adjustment_readers[type(asset)]
                except KeyError:
                    adj_reader = None
                if adj_reader is not None:
                    adjs = adj_reader.load_pricing_adjustments([field], adj_dts, [asset])[0]
                else:
                    adjs = {}
                window = window_type(array[:, i].reshape(prefetch_len, 1), view_kwargs, adjs, offset, size, int(is_perspective_after), self._decimal_places_for_asset(asset, dts[-1]))
                sliding_window = SlidingWindow(window, size, start_ix, offset)
                asset_windows[asset] = sliding_window
                self._window_blocks[field].set((asset, size, is_perspective_after), sliding_window, prefetch_end)
        return [asset_windows[asset] for asset in assets]

    def history(self, assets, dts, field, is_perspective_after):
        if False:
            for i in range(10):
                print('nop')
        "\n        A window of pricing data with adjustments applied assuming that the\n        end of the window is the day before the current simulation time.\n\n        Parameters\n        ----------\n        assets : iterable of Assets\n            The assets in the window.\n        dts : iterable of datetime64-like\n            The datetimes for which to fetch data.\n            Makes an assumption that all dts are present and contiguous,\n            in the calendar.\n        field : str\n            The OHLCV field for which to retrieve data.\n        is_perspective_after : bool\n            True, if the window is being viewed immediately after the last dt\n            in the sliding window.\n            False, if the window is viewed on the last dt.\n\n            This flag is used for handling the case where the last dt in the\n            requested window immediately precedes a corporate action, e.g.:\n\n            - is_perspective_after is True\n\n            When the viewpoint is after the last dt in the window, as when a\n            daily history window is accessed from a simulation that uses a\n            minute data frequency, the history call to this loader will not\n            include the current simulation dt. At that point in time, the raw\n            data for the last day in the window will require adjustment, so the\n            most recent adjustment with respect to the simulation time is\n            applied to the last dt in the requested window.\n\n            An example equity which has a 0.5 split ratio dated for 05-27,\n            with the dts for a history call of 5 bars with a '1d' frequency at\n            05-27 9:31. Simulation frequency is 'minute'.\n\n            (In this case this function is called with 4 daily dts, and the\n             calling function is responsible for stitching back on the\n             'current' dt)\n\n            |       |       |       |       | last dt | <-- viewer is here |\n            |       | 05-23 | 05-24 | 05-25 | 05-26   | 05-27 9:31         |\n            | raw   | 10.10 | 10.20 | 10.30 | 10.40   |                    |\n            | adj   |  5.05 |  5.10 |  5.15 |  5.25   |                    |\n\n            The adjustment is applied to the last dt, 05-26, and all previous\n            dts.\n\n            - is_perspective_after is False, daily\n\n            When the viewpoint is the same point in time as the last dt in the\n            window, as when a daily history window is accessed from a\n            simulation that uses a daily data frequency, the history call will\n            include the current dt. At that point in time, the raw data for the\n            last day in the window will be post-adjustment, so no adjustment\n            is applied to the last dt.\n\n            An example equity which has a 0.5 split ratio dated for 05-27,\n            with the dts for a history call of 5 bars with a '1d' frequency at\n            05-27 0:00. Simulation frequency is 'daily'.\n\n            |       |       |       |       |       | <-- viewer is here |\n            |       |       |       |       |       | last dt            |\n            |       | 05-23 | 05-24 | 05-25 | 05-26 | 05-27              |\n            | raw   | 10.10 | 10.20 | 10.30 | 10.40 | 5.25               |\n            | adj   |  5.05 |  5.10 |  5.15 |  5.20 | 5.25               |\n\n            Adjustments are applied 05-23 through 05-26 but not to the last dt,\n            05-27\n\n        Returns\n        -------\n        out : np.ndarray with shape(len(days between start, end), len(assets))\n        "
        block = self._ensure_sliding_windows(assets, dts, field, is_perspective_after)
        end_ix = self._calendar.searchsorted(dts[-1])
        return concatenate([window.get(end_ix) for window in block], axis=1)

class DailyHistoryLoader(HistoryLoader):

    @property
    def _frequency(self):
        if False:
            return 10
        return 'daily'

    @property
    def _calendar(self):
        if False:
            i = 10
            return i + 15
        return self._reader.sessions

    def _array(self, dts, assets, field):
        if False:
            while True:
                i = 10
        return self._reader.load_raw_arrays([field], dts[0], dts[-1], assets)[0]

class MinuteHistoryLoader(HistoryLoader):

    @property
    def _frequency(self):
        if False:
            return 10
        return 'minute'

    @lazyval
    def _calendar(self):
        if False:
            while True:
                i = 10
        mm = self.trading_calendar.all_minutes
        start = mm.searchsorted(self._reader.first_trading_day)
        end = mm.searchsorted(self._reader.last_available_dt, side='right')
        return mm[start:end]

    def _array(self, dts, assets, field):
        if False:
            while True:
                i = 10
        return self._reader.load_raw_arrays([field], dts[0], dts[-1], assets)[0]