from abc import ABCMeta, abstractmethod
import json
import os
from glob import glob
from os.path import join
from textwrap import dedent
from lru import LRU
import bcolz
from bcolz import ctable
import h5py
from intervaltree import IntervalTree
import logbook
import numpy as np
import pandas as pd
from pandas import HDFStore
import tables
from six import with_metaclass
from toolz import keymap, valmap
from trading_calendars import get_calendar
from zipline.data._minute_bar_internal import minute_value, find_position_of_minute, find_last_traded_position_internal
from zipline.gens.sim_engine import NANOS_IN_MINUTE
from zipline.data.bar_reader import BarReader, NoDataForSid, NoDataOnDate
from zipline.data.bcolz_daily_bars import check_uint32_safe
from zipline.utils.cli import maybe_show_progress
from zipline.utils.compat import mappingproxy
from zipline.utils.memoize import lazyval
logger = logbook.Logger('MinuteBars')
US_EQUITIES_MINUTES_PER_DAY = 390
FUTURES_MINUTES_PER_DAY = 1440
DEFAULT_EXPECTEDLEN = US_EQUITIES_MINUTES_PER_DAY * 252 * 15
OHLC_RATIO = 1000

class BcolzMinuteOverlappingData(Exception):
    pass

class BcolzMinuteWriterColumnMismatch(Exception):
    pass

class MinuteBarReader(BarReader):

    @property
    def data_frequency(self):
        if False:
            while True:
                i = 10
        return 'minute'

def _calc_minute_index(market_opens, minutes_per_day):
    if False:
        print('Hello World!')
    minutes = np.zeros(len(market_opens) * minutes_per_day, dtype='datetime64[ns]')
    deltas = np.arange(0, minutes_per_day, dtype='timedelta64[m]')
    for (i, market_open) in enumerate(market_opens):
        start = market_open.asm8
        minute_values = start + deltas
        start_ix = minutes_per_day * i
        end_ix = start_ix + minutes_per_day
        minutes[start_ix:end_ix] = minute_values
    return pd.to_datetime(minutes, utc=True, box=True)

def _sid_subdir_path(sid):
    if False:
        return 10
    '\n    Format subdir path to limit the number directories in any given\n    subdirectory to 100.\n\n    The number in each directory is designed to support at least 100000\n    equities.\n\n    Parameters\n    ----------\n    sid : int\n        Asset identifier.\n\n    Returns\n    -------\n    out : string\n        A path for the bcolz rootdir, including subdirectory prefixes based on\n        the padded string representation of the given sid.\n\n        e.g. 1 is formatted as 00/00/000001.bcolz\n    '
    padded_sid = format(sid, '06')
    return os.path.join(padded_sid[0:2], padded_sid[2:4], '{0}.bcolz'.format(str(padded_sid)))

def convert_cols(cols, scale_factor, sid, invalid_data_behavior):
    if False:
        i = 10
        return i + 15
    "Adapt OHLCV columns into uint32 columns.\n\n    Parameters\n    ----------\n    cols : dict\n        A dict mapping each column name (open, high, low, close, volume)\n        to a float column to convert to uint32.\n    scale_factor : int\n        Factor to use to scale float values before converting to uint32.\n    sid : int\n        Sid of the relevant asset, for logging.\n    invalid_data_behavior : str\n        Specifies behavior when data cannot be converted to uint32.\n        If 'raise', raises an exception.\n        If 'warn', logs a warning and filters out incompatible values.\n        If 'ignore', silently filters out incompatible values.\n    "
    scaled_opens = (np.nan_to_num(cols['open']) * scale_factor).round()
    scaled_highs = (np.nan_to_num(cols['high']) * scale_factor).round()
    scaled_lows = (np.nan_to_num(cols['low']) * scale_factor).round()
    scaled_closes = (np.nan_to_num(cols['close']) * scale_factor).round()
    exclude_mask = np.zeros_like(scaled_opens, dtype=bool)
    for (col_name, scaled_col) in [('open', scaled_opens), ('high', scaled_highs), ('low', scaled_lows), ('close', scaled_closes)]:
        max_val = scaled_col.max()
        try:
            check_uint32_safe(max_val, col_name)
        except ValueError:
            if invalid_data_behavior == 'raise':
                raise
            if invalid_data_behavior == 'warn':
                logger.warn('Values for sid={}, col={} contain some too large for uint32 (max={}), filtering them out', sid, col_name, max_val)
            exclude_mask &= scaled_col >= np.iinfo(np.uint32).max
    opens = scaled_opens.astype(np.uint32)
    highs = scaled_highs.astype(np.uint32)
    lows = scaled_lows.astype(np.uint32)
    closes = scaled_closes.astype(np.uint32)
    volumes = cols['volume'].astype(np.uint32)
    opens[exclude_mask] = 0
    highs[exclude_mask] = 0
    lows[exclude_mask] = 0
    closes[exclude_mask] = 0
    volumes[exclude_mask] = 0
    return (opens, highs, lows, closes, volumes)

class BcolzMinuteBarMetadata(object):
    """
    Parameters
    ----------
    ohlc_ratio : int
         The factor by which the pricing data is multiplied so that the
         float data can be stored as an integer.
    calendar :  trading_calendars.trading_calendar.TradingCalendar
        The TradingCalendar on which the minute bars are based.
    start_session : datetime
        The first trading session in the data set.
    end_session : datetime
        The last trading session in the data set.
    minutes_per_day : int
        The number of minutes per each period.
    """
    FORMAT_VERSION = 3
    METADATA_FILENAME = 'metadata.json'

    @classmethod
    def metadata_path(cls, rootdir):
        if False:
            while True:
                i = 10
        return os.path.join(rootdir, cls.METADATA_FILENAME)

    @classmethod
    def read(cls, rootdir):
        if False:
            print('Hello World!')
        path = cls.metadata_path(rootdir)
        with open(path) as fp:
            raw_data = json.load(fp)
            try:
                version = raw_data['version']
            except KeyError:
                version = 0
            default_ohlc_ratio = raw_data['ohlc_ratio']
            if version >= 1:
                minutes_per_day = raw_data['minutes_per_day']
            else:
                minutes_per_day = US_EQUITIES_MINUTES_PER_DAY
            if version >= 2:
                calendar = get_calendar(raw_data['calendar_name'])
                start_session = pd.Timestamp(raw_data['start_session'], tz='UTC')
                end_session = pd.Timestamp(raw_data['end_session'], tz='UTC')
            else:
                calendar = get_calendar('XNYS')
                start_session = pd.Timestamp(raw_data['first_trading_day'], tz='UTC')
                end_session = calendar.minute_to_session_label(pd.Timestamp(raw_data['market_closes'][-1], unit='m', tz='UTC'))
            if version >= 3:
                ohlc_ratios_per_sid = raw_data['ohlc_ratios_per_sid']
                if ohlc_ratios_per_sid is not None:
                    ohlc_ratios_per_sid = keymap(int, ohlc_ratios_per_sid)
            else:
                ohlc_ratios_per_sid = None
            return cls(default_ohlc_ratio, ohlc_ratios_per_sid, calendar, start_session, end_session, minutes_per_day, version=version)

    def __init__(self, default_ohlc_ratio, ohlc_ratios_per_sid, calendar, start_session, end_session, minutes_per_day, version=FORMAT_VERSION):
        if False:
            for i in range(10):
                print('nop')
        self.calendar = calendar
        self.start_session = start_session
        self.end_session = end_session
        self.default_ohlc_ratio = default_ohlc_ratio
        self.ohlc_ratios_per_sid = ohlc_ratios_per_sid
        self.minutes_per_day = minutes_per_day
        self.version = version

    def write(self, rootdir):
        if False:
            i = 10
            return i + 15
        "\n        Write the metadata to a JSON file in the rootdir.\n\n        Values contained in the metadata are:\n\n        version : int\n            The value of FORMAT_VERSION of this class.\n        ohlc_ratio : int\n            The default ratio by which to multiply the pricing data to\n            convert the floats from floats to an integer to fit within\n            the np.uint32. If ohlc_ratios_per_sid is None or does not\n            contain a mapping for a given sid, this ratio is used.\n        ohlc_ratios_per_sid : dict\n             A dict mapping each sid in the output to the factor by\n             which the pricing data is multiplied so that the float data\n             can be stored as an integer.\n        minutes_per_day : int\n            The number of minutes per each period.\n        calendar_name : str\n            The name of the TradingCalendar on which the minute bars are\n            based.\n        start_session : datetime\n            'YYYY-MM-DD' formatted representation of the first trading\n            session in the data set.\n        end_session : datetime\n            'YYYY-MM-DD' formatted representation of the last trading\n            session in the data set.\n\n        Deprecated, but included for backwards compatibility:\n\n        first_trading_day : string\n            'YYYY-MM-DD' formatted representation of the first trading day\n             available in the dataset.\n        market_opens : list\n            List of int64 values representing UTC market opens as\n            minutes since epoch.\n        market_closes : list\n            List of int64 values representing UTC market closes as\n            minutes since epoch.\n        "
        calendar = self.calendar
        slicer = calendar.schedule.index.slice_indexer(self.start_session, self.end_session)
        schedule = calendar.schedule[slicer]
        market_opens = schedule.market_open
        market_closes = schedule.market_close
        metadata = {'version': self.version, 'ohlc_ratio': self.default_ohlc_ratio, 'ohlc_ratios_per_sid': self.ohlc_ratios_per_sid, 'minutes_per_day': self.minutes_per_day, 'calendar_name': self.calendar.name, 'start_session': str(self.start_session.date()), 'end_session': str(self.end_session.date()), 'first_trading_day': str(self.start_session.date()), 'market_opens': market_opens.values.astype('datetime64[m]').astype(np.int64).tolist(), 'market_closes': market_closes.values.astype('datetime64[m]').astype(np.int64).tolist()}
        with open(self.metadata_path(rootdir), 'w+') as fp:
            json.dump(metadata, fp)

class BcolzMinuteBarWriter(object):
    """
    Class capable of writing minute OHLCV data to disk into bcolz format.

    Parameters
    ----------
    rootdir : string
        Path to the root directory into which to write the metadata and
        bcolz subdirectories.
    calendar : trading_calendars.trading_calendar.TradingCalendar
        The trading calendar on which to base the minute bars. Used to
        get the market opens used as a starting point for each periodic
        span of minutes in the index, and the market closes that
        correspond with the market opens.
    minutes_per_day : int
        The number of minutes per each period. Defaults to 390, the mode
        of minutes in NYSE trading days.
    start_session : datetime
        The first trading session in the data set.
    end_session : datetime
        The last trading session in the data set.
    default_ohlc_ratio : int, optional
        The default ratio by which to multiply the pricing data to
        convert from floats to integers that fit within np.uint32. If
        ohlc_ratios_per_sid is None or does not contain a mapping for a
        given sid, this ratio is used. Default is OHLC_RATIO (1000).
    ohlc_ratios_per_sid : dict, optional
        A dict mapping each sid in the output to the ratio by which to
        multiply the pricing data to convert the floats from floats to
        an integer to fit within the np.uint32.
    expectedlen : int, optional
        The expected length of the dataset, used when creating the initial
        bcolz ctable.

        If the expectedlen is not used, the chunksize and corresponding
        compression ratios are not ideal.

        Defaults to supporting 15 years of NYSE equity market data.
        see: http://bcolz.blosc.org/opt-tips.html#informing-about-the-length-of-your-carrays # noqa
    write_metadata : bool, optional
        If True, writes the minute bar metadata (on init of the writer).
        If False, no metadata is written (existing metadata is
        retained). Default is True.

    Notes
    -----
    Writes a bcolz directory for each individual sid, all contained within
    a root directory which also contains metadata about the entire dataset.

    Each individual asset's data is stored as a bcolz table with a column for
    each pricing field: (open, high, low, close, volume)

    The open, high, low, and close columns are integers which are 1000 times
    the quoted price, so that the data can represented and stored as an
    np.uint32, supporting market prices quoted up to the thousands place.

    volume is a np.uint32 with no mutation of the tens place.

    The 'index' for each individual asset are a repeating period of minutes of
    length `minutes_per_day` starting from each market open.
    The file format does not account for half-days.
    e.g.:
    2016-01-19 14:31
    2016-01-19 14:32
    ...
    2016-01-19 20:59
    2016-01-19 21:00
    2016-01-20 14:31
    2016-01-20 14:32
    ...
    2016-01-20 20:59
    2016-01-20 21:00

    All assets are written with a common 'index', sharing a common first
    trading day. Assets that do not begin trading until after the first trading
    day will have zeros for all pricing data up and until data is traded.

    'index' is in quotations, because bcolz does not provide an index. The
    format allows index-like behavior by writing each minute's data into the
    corresponding position of the enumeration of the aforementioned datetime
    index.

    The datetimes which correspond to each position are written in the metadata
    as integer nanoseconds since the epoch into the `minute_index` key.

    See Also
    --------
    zipline.data.minute_bars.BcolzMinuteBarReader
    """
    COL_NAMES = ('open', 'high', 'low', 'close', 'volume')

    def __init__(self, rootdir, calendar, start_session, end_session, minutes_per_day, default_ohlc_ratio=OHLC_RATIO, ohlc_ratios_per_sid=None, expectedlen=DEFAULT_EXPECTEDLEN, write_metadata=True):
        if False:
            i = 10
            return i + 15
        self._rootdir = rootdir
        self._start_session = start_session
        self._end_session = end_session
        self._calendar = calendar
        slicer = calendar.schedule.index.slice_indexer(start_session, end_session)
        self._schedule = calendar.schedule[slicer]
        self._session_labels = self._schedule.index
        self._minutes_per_day = minutes_per_day
        self._expectedlen = expectedlen
        self._default_ohlc_ratio = default_ohlc_ratio
        self._ohlc_ratios_per_sid = ohlc_ratios_per_sid
        self._minute_index = _calc_minute_index(self._schedule.market_open, self._minutes_per_day)
        if write_metadata:
            metadata = BcolzMinuteBarMetadata(self._default_ohlc_ratio, self._ohlc_ratios_per_sid, self._calendar, self._start_session, self._end_session, self._minutes_per_day)
            metadata.write(self._rootdir)

    @classmethod
    def open(cls, rootdir, end_session=None):
        if False:
            i = 10
            return i + 15
        '\n        Open an existing ``rootdir`` for writing.\n\n        Parameters\n        ----------\n        end_session : Timestamp (optional)\n            When appending, the intended new ``end_session``.\n        '
        metadata = BcolzMinuteBarMetadata.read(rootdir)
        return BcolzMinuteBarWriter(rootdir, metadata.calendar, metadata.start_session, end_session if end_session is not None else metadata.end_session, metadata.minutes_per_day, metadata.default_ohlc_ratio, metadata.ohlc_ratios_per_sid, write_metadata=end_session is not None)

    @property
    def first_trading_day(self):
        if False:
            i = 10
            return i + 15
        return self._start_session

    def ohlc_ratio_for_sid(self, sid):
        if False:
            print('Hello World!')
        if self._ohlc_ratios_per_sid is not None:
            try:
                return self._ohlc_ratios_per_sid[sid]
            except KeyError:
                pass
        return self._default_ohlc_ratio

    def sidpath(self, sid):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        sid : int\n            Asset identifier.\n\n        Returns\n        -------\n        out : string\n            Full path to the bcolz rootdir for the given sid.\n        '
        sid_subdir = _sid_subdir_path(sid)
        return join(self._rootdir, sid_subdir)

    def last_date_in_output_for_sid(self, sid):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        sid : int\n            Asset identifier.\n\n        Returns\n        -------\n        out : pd.Timestamp\n            The midnight of the last date written in to the output for the\n            given sid.\n        '
        sizes_path = '{0}/close/meta/sizes'.format(self.sidpath(sid))
        if not os.path.exists(sizes_path):
            return pd.NaT
        with open(sizes_path, mode='r') as f:
            sizes = f.read()
        data = json.loads(sizes)
        num_days = data['shape'][0] // self._minutes_per_day
        if num_days == 0:
            return pd.NaT
        return self._session_labels[num_days - 1]

    def _init_ctable(self, path):
        if False:
            print('Hello World!')
        '\n        Create empty ctable for given path.\n\n        Parameters\n        ----------\n        path : string\n            The path to rootdir of the new ctable.\n        '
        sid_containing_dirname = os.path.dirname(path)
        if not os.path.exists(sid_containing_dirname):
            os.makedirs(sid_containing_dirname)
        initial_array = np.empty(0, np.uint32)
        table = ctable(rootdir=path, columns=[initial_array, initial_array, initial_array, initial_array, initial_array], names=['open', 'high', 'low', 'close', 'volume'], expectedlen=self._expectedlen, mode='w')
        table.flush()
        return table

    def _ensure_ctable(self, sid):
        if False:
            while True:
                i = 10
        'Ensure that a ctable exists for ``sid``, then return it.'
        sidpath = self.sidpath(sid)
        if not os.path.exists(sidpath):
            return self._init_ctable(sidpath)
        return bcolz.ctable(rootdir=sidpath, mode='a')

    def _zerofill(self, table, numdays):
        if False:
            i = 10
            return i + 15
        minute_offset = len(table) % self._minutes_per_day
        num_to_prepend = numdays * self._minutes_per_day - minute_offset
        prepend_array = np.zeros(num_to_prepend, np.uint32)
        table.append([prepend_array] * 5)
        table.flush()

    def pad(self, sid, date):
        if False:
            print('Hello World!')
        '\n        Fill sid container with empty data through the specified date.\n\n        If the last recorded trade is not at the close, then that day will be\n        padded with zeros until its close. Any day after that (up to and\n        including the specified date) will be padded with `minute_per_day`\n        worth of zeros\n\n        Parameters\n        ----------\n        sid : int\n            The asset identifier for the data being written.\n        date : datetime-like\n            The date used to calculate how many slots to be pad.\n            The padding is done through the date, i.e. after the padding is\n            done the `last_date_in_output_for_sid` will be equal to `date`\n        '
        table = self._ensure_ctable(sid)
        last_date = self.last_date_in_output_for_sid(sid)
        tds = self._session_labels
        if date <= last_date or date < tds[0]:
            return
        if pd.isnull(last_date):
            days_to_zerofill = tds[tds.slice_indexer(end=date)]
        else:
            days_to_zerofill = tds[tds.slice_indexer(start=last_date + tds.freq, end=date)]
        self._zerofill(table, len(days_to_zerofill))
        new_last_date = self.last_date_in_output_for_sid(sid)
        assert new_last_date == date, 'new_last_date={0} != date={1}'.format(new_last_date, date)

    def set_sid_attrs(self, sid, **kwargs):
        if False:
            while True:
                i = 10
        "Write all the supplied kwargs as attributes of the sid's file.\n        "
        table = self._ensure_ctable(sid)
        for (k, v) in kwargs.items():
            table.attrs[k] = v

    def write(self, data, show_progress=False, invalid_data_behavior='warn'):
        if False:
            return 10
        "Write a stream of minute data.\n\n        Parameters\n        ----------\n        data : iterable[(int, pd.DataFrame)]\n            The data to write. Each element should be a tuple of sid, data\n            where data has the following format:\n              columns : ('open', 'high', 'low', 'close', 'volume')\n                  open : float64\n                  high : float64\n                  low  : float64\n                  close : float64\n                  volume : float64|int64\n              index : DatetimeIndex of market minutes.\n            A given sid may appear more than once in ``data``; however,\n            the dates must be strictly increasing.\n        show_progress : bool, optional\n            Whether or not to show a progress bar while writing.\n        "
        ctx = maybe_show_progress(data, show_progress=show_progress, item_show_func=lambda e: e if e is None else str(e[0]), label='Merging minute equity files:')
        write_sid = self.write_sid
        with ctx as it:
            for e in it:
                write_sid(*e, invalid_data_behavior=invalid_data_behavior)

    def write_sid(self, sid, df, invalid_data_behavior='warn'):
        if False:
            return 10
        "\n        Write the OHLCV data for the given sid.\n        If there is no bcolz ctable yet created for the sid, create it.\n        If the length of the bcolz ctable is not exactly to the date before\n        the first day provided, fill the ctable with 0s up to that date.\n\n        Parameters\n        ----------\n        sid : int\n            The asset identifer for the data being written.\n        df : pd.DataFrame\n            DataFrame of market data with the following characteristics.\n            columns : ('open', 'high', 'low', 'close', 'volume')\n                open : float64\n                high : float64\n                low  : float64\n                close : float64\n                volume : float64|int64\n            index : DatetimeIndex of market minutes.\n        "
        cols = {'open': df.open.values, 'high': df.high.values, 'low': df.low.values, 'close': df.close.values, 'volume': df.volume.values}
        dts = df.index.values
        self._write_cols(sid, dts, cols, invalid_data_behavior)

    def write_cols(self, sid, dts, cols, invalid_data_behavior='warn'):
        if False:
            while True:
                i = 10
        "\n        Write the OHLCV data for the given sid.\n        If there is no bcolz ctable yet created for the sid, create it.\n        If the length of the bcolz ctable is not exactly to the date before\n        the first day provided, fill the ctable with 0s up to that date.\n\n        Parameters\n        ----------\n        sid : int\n            The asset identifier for the data being written.\n        dts : datetime64 array\n            The dts corresponding to values in cols.\n        cols : dict of str -> np.array\n            dict of market data with the following characteristics.\n            keys are ('open', 'high', 'low', 'close', 'volume')\n            open : float64\n            high : float64\n            low  : float64\n            close : float64\n            volume : float64|int64\n        "
        if not all((len(dts) == len(cols[name]) for name in self.COL_NAMES)):
            raise BcolzMinuteWriterColumnMismatch('Length of dts={0} should match cols: {1}'.format(len(dts), ' '.join(('{0}={1}'.format(name, len(cols[name])) for name in self.COL_NAMES))))
        self._write_cols(sid, dts, cols, invalid_data_behavior)

    def _write_cols(self, sid, dts, cols, invalid_data_behavior):
        if False:
            i = 10
            return i + 15
        "\n        Internal method for `write_cols` and `write`.\n\n        Parameters\n        ----------\n        sid : int\n            The asset identifier for the data being written.\n        dts : datetime64 array\n            The dts corresponding to values in cols.\n        cols : dict of str -> np.array\n            dict of market data with the following characteristics.\n            keys are ('open', 'high', 'low', 'close', 'volume')\n            open : float64\n            high : float64\n            low  : float64\n            close : float64\n            volume : float64|int64\n        "
        table = self._ensure_ctable(sid)
        tds = self._session_labels
        input_first_day = self._calendar.minute_to_session_label(pd.Timestamp(dts[0]), direction='previous')
        last_date = self.last_date_in_output_for_sid(sid)
        day_before_input = input_first_day - tds.freq
        self.pad(sid, day_before_input)
        table = self._ensure_ctable(sid)
        num_rec_mins = table.size
        all_minutes = self._minute_index
        last_minute_to_write = pd.Timestamp(dts[-1], tz='UTC')
        if num_rec_mins > 0:
            last_recorded_minute = all_minutes[num_rec_mins - 1]
            if last_minute_to_write <= last_recorded_minute:
                raise BcolzMinuteOverlappingData(dedent('\n                Data with last_date={0} already includes input start={1} for\n                sid={2}'.strip()).format(last_date, input_first_day, sid))
        latest_min_count = all_minutes.get_loc(last_minute_to_write)
        all_minutes_in_window = all_minutes[num_rec_mins:latest_min_count + 1]
        minutes_count = all_minutes_in_window.size
        open_col = np.zeros(minutes_count, dtype=np.uint32)
        high_col = np.zeros(minutes_count, dtype=np.uint32)
        low_col = np.zeros(minutes_count, dtype=np.uint32)
        close_col = np.zeros(minutes_count, dtype=np.uint32)
        vol_col = np.zeros(minutes_count, dtype=np.uint32)
        dt_ixs = np.searchsorted(all_minutes_in_window.values, dts.astype('datetime64[ns]'))
        ohlc_ratio = self.ohlc_ratio_for_sid(sid)
        (open_col[dt_ixs], high_col[dt_ixs], low_col[dt_ixs], close_col[dt_ixs], vol_col[dt_ixs]) = convert_cols(cols, ohlc_ratio, sid, invalid_data_behavior)
        table.append([open_col, high_col, low_col, close_col, vol_col])
        table.flush()

    def data_len_for_day(self, day):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the number of data points up to and including the\n        provided day.\n        '
        day_ix = self._session_labels.get_loc(day)
        num_days = day_ix + 1
        return num_days * self._minutes_per_day

    def truncate(self, date):
        if False:
            while True:
                i = 10
        'Truncate data beyond this date in all ctables.'
        truncate_slice_end = self.data_len_for_day(date)
        glob_path = os.path.join(self._rootdir, '*', '*', '*.bcolz')
        sid_paths = sorted(glob(glob_path))
        for sid_path in sid_paths:
            file_name = os.path.basename(sid_path)
            try:
                table = bcolz.open(rootdir=sid_path)
            except IOError:
                continue
            if table.len <= truncate_slice_end:
                logger.info('{0} not past truncate date={1}.', file_name, date)
                continue
            logger.info('Truncating {0} at end_date={1}', file_name, date.date())
            table.resize(truncate_slice_end)
        metadata = BcolzMinuteBarMetadata.read(self._rootdir)
        metadata.end_session = date
        metadata.write(self._rootdir)

class BcolzMinuteBarReader(MinuteBarReader):
    """
    Reader for data written by BcolzMinuteBarWriter

    Parameters
    ----------
    rootdir : string
        The root directory containing the metadata and asset bcolz
        directories.

    See Also
    --------
    zipline.data.minute_bars.BcolzMinuteBarWriter
    """
    FIELDS = ('open', 'high', 'low', 'close', 'volume')
    DEFAULT_MINUTELY_SID_CACHE_SIZES = {'close': 3000, 'open': 1550, 'high': 1550, 'low': 1550, 'volume': 1550}
    assert set(FIELDS) == set(DEFAULT_MINUTELY_SID_CACHE_SIZES), 'FIELDS should match DEFAULT_MINUTELY_SID_CACHE_SIZES keys'
    _default_proxy = mappingproxy(DEFAULT_MINUTELY_SID_CACHE_SIZES)

    def __init__(self, rootdir, sid_cache_sizes=_default_proxy):
        if False:
            return 10
        self._rootdir = rootdir
        metadata = self._get_metadata()
        self._start_session = metadata.start_session
        self._end_session = metadata.end_session
        self.calendar = metadata.calendar
        slicer = self.calendar.schedule.index.slice_indexer(self._start_session, self._end_session)
        self._schedule = self.calendar.schedule[slicer]
        self._market_opens = self._schedule.market_open
        self._market_open_values = self._market_opens.values.astype('datetime64[m]').astype(np.int64)
        self._market_closes = self._schedule.market_close
        self._market_close_values = self._market_closes.values.astype('datetime64[m]').astype(np.int64)
        self._default_ohlc_inverse = 1.0 / metadata.default_ohlc_ratio
        ohlc_ratios = metadata.ohlc_ratios_per_sid
        if ohlc_ratios:
            self._ohlc_inverses_per_sid = valmap(lambda x: 1.0 / x, ohlc_ratios)
        else:
            self._ohlc_inverses_per_sid = None
        self._minutes_per_day = metadata.minutes_per_day
        self._carrays = {field: LRU(sid_cache_sizes[field]) for field in self.FIELDS}
        self._last_get_value_dt_position = None
        self._last_get_value_dt_value = None
        self._known_zero_volume_dict = {}

    def _get_metadata(self):
        if False:
            print('Hello World!')
        return BcolzMinuteBarMetadata.read(self._rootdir)

    @property
    def trading_calendar(self):
        if False:
            for i in range(10):
                print('nop')
        return self.calendar

    @lazyval
    def last_available_dt(self):
        if False:
            i = 10
            return i + 15
        (_, close) = self.calendar.open_and_close_for_session(self._end_session)
        return close

    @property
    def first_trading_day(self):
        if False:
            while True:
                i = 10
        return self._start_session

    def _ohlc_ratio_inverse_for_sid(self, sid):
        if False:
            print('Hello World!')
        if self._ohlc_inverses_per_sid is not None:
            try:
                return self._ohlc_inverses_per_sid[sid]
            except KeyError:
                pass
        return self._default_ohlc_inverse

    def _minutes_to_exclude(self):
        if False:
            print('Hello World!')
        '\n        Calculate the minutes which should be excluded when a window\n        occurs on days which had an early close, i.e. days where the close\n        based on the regular period of minutes per day and the market close\n        do not match.\n\n        Returns\n        -------\n        List of DatetimeIndex representing the minutes to exclude because\n        of early closes.\n        '
        market_opens = self._market_opens.values.astype('datetime64[m]')
        market_closes = self._market_closes.values.astype('datetime64[m]')
        minutes_per_day = (market_closes - market_opens).astype(np.int64)
        early_indices = np.where(minutes_per_day != self._minutes_per_day - 1)[0]
        early_opens = self._market_opens[early_indices]
        early_closes = self._market_closes[early_indices]
        minutes = [(market_open, early_close) for (market_open, early_close) in zip(early_opens, early_closes)]
        return minutes

    @lazyval
    def _minute_exclusion_tree(self):
        if False:
            while True:
                i = 10
        '\n        Build an interval tree keyed by the start and end of each range\n        of positions should be dropped from windows. (These are the minutes\n        between an early close and the minute which would be the close based\n        on the regular period if there were no early close.)\n        The value of each node is the same start and end position stored as\n        a tuple.\n\n        The data is stored as such in support of a fast answer to the question,\n        does a given start and end position overlap any of the exclusion spans?\n\n        Returns\n        -------\n        IntervalTree containing nodes which represent the minutes to exclude\n        because of early closes.\n        '
        itree = IntervalTree()
        for (market_open, early_close) in self._minutes_to_exclude():
            start_pos = self._find_position_of_minute(early_close) + 1
            end_pos = self._find_position_of_minute(market_open) + self._minutes_per_day - 1
            data = (start_pos, end_pos)
            itree[start_pos:end_pos + 1] = data
        return itree

    def _exclusion_indices_for_range(self, start_idx, end_idx):
        if False:
            return 10
        '\n        Returns\n        -------\n        List of tuples of (start, stop) which represent the ranges of minutes\n        which should be excluded when a market minute window is requested.\n        '
        itree = self._minute_exclusion_tree
        if itree.overlaps(start_idx, end_idx):
            ranges = []
            intervals = itree[start_idx:end_idx]
            for interval in intervals:
                ranges.append(interval.data)
            return sorted(ranges)
        else:
            return None

    def _get_carray_path(self, sid, field):
        if False:
            i = 10
            return i + 15
        sid_subdir = _sid_subdir_path(sid)
        return os.path.join(self._rootdir, sid_subdir, field)

    def _open_minute_file(self, field, sid):
        if False:
            i = 10
            return i + 15
        sid = int(sid)
        try:
            carray = self._carrays[field][sid]
        except KeyError:
            try:
                carray = self._carrays[field][sid] = bcolz.carray(rootdir=self._get_carray_path(sid, field), mode='r')
            except IOError:
                raise NoDataForSid('No minute data for sid {}.'.format(sid))
        return carray

    def table_len(self, sid):
        if False:
            for i in range(10):
                print('nop')
        'Returns the length of the underlying table for this sid.'
        return len(self._open_minute_file('close', sid))

    def get_sid_attr(self, sid, name):
        if False:
            print('Hello World!')
        sid_subdir = _sid_subdir_path(sid)
        sid_path = os.path.join(self._rootdir, sid_subdir)
        attrs = bcolz.attrs.attrs(sid_path, 'r')
        try:
            return attrs[name]
        except KeyError:
            return None

    def get_value(self, sid, dt, field):
        if False:
            while True:
                i = 10
        "\n        Retrieve the pricing info for the given sid, dt, and field.\n\n        Parameters\n        ----------\n        sid : int\n            Asset identifier.\n        dt : datetime-like\n            The datetime at which the trade occurred.\n        field : string\n            The type of pricing data to retrieve.\n            ('open', 'high', 'low', 'close', 'volume')\n\n        Returns\n        -------\n        out : float|int\n\n        The market data for the given sid, dt, and field coordinates.\n\n        For OHLC:\n            Returns a float if a trade occurred at the given dt.\n            If no trade occurred, a np.nan is returned.\n\n        For volume:\n            Returns the integer value of the volume.\n            (A volume of 0 signifies no trades for the given dt.)\n        "
        if self._last_get_value_dt_value == dt.value:
            minute_pos = self._last_get_value_dt_position
        else:
            try:
                minute_pos = self._find_position_of_minute(dt)
            except ValueError:
                raise NoDataOnDate()
            self._last_get_value_dt_value = dt.value
            self._last_get_value_dt_position = minute_pos
        try:
            value = self._open_minute_file(field, sid)[minute_pos]
        except IndexError:
            value = 0
        if value == 0:
            if field == 'volume':
                return 0
            else:
                return np.nan
        if field != 'volume':
            value *= self._ohlc_ratio_inverse_for_sid(sid)
        return value

    def get_last_traded_dt(self, asset, dt):
        if False:
            print('Hello World!')
        minute_pos = self._find_last_traded_position(asset, dt)
        if minute_pos == -1:
            return pd.NaT
        return self._pos_to_minute(minute_pos)

    def _find_last_traded_position(self, asset, dt):
        if False:
            return 10
        volumes = self._open_minute_file('volume', asset)
        start_date_minute = asset.start_date.value / NANOS_IN_MINUTE
        dt_minute = dt.value / NANOS_IN_MINUTE
        try:
            earliest_dt_to_search = self._known_zero_volume_dict[asset.sid]
        except KeyError:
            earliest_dt_to_search = start_date_minute
        if dt_minute < earliest_dt_to_search:
            return -1
        pos = find_last_traded_position_internal(self._market_open_values, self._market_close_values, dt_minute, earliest_dt_to_search, volumes, self._minutes_per_day)
        if pos == -1:
            try:
                self._known_zero_volume_dict[asset.sid] = max(dt_minute, self._known_zero_volume_dict[asset.sid])
            except KeyError:
                self._known_zero_volume_dict[asset.sid] = dt_minute
        return pos

    def _pos_to_minute(self, pos):
        if False:
            print('Hello World!')
        minute_epoch = minute_value(self._market_open_values, pos, self._minutes_per_day)
        return pd.Timestamp(minute_epoch, tz='UTC', unit='m')

    def _find_position_of_minute(self, minute_dt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal method that returns the position of the given minute in the\n        list of every trading minute since market open of the first trading\n        day. Adjusts non market minutes to the last close.\n\n        ex. this method would return 1 for 2002-01-02 9:32 AM Eastern, if\n        2002-01-02 is the first trading day of the dataset.\n\n        Parameters\n        ----------\n        minute_dt: pd.Timestamp\n            The minute whose position should be calculated.\n\n        Returns\n        -------\n        int: The position of the given minute in the list of all trading\n        minutes since market open on the first trading day.\n        '
        return find_position_of_minute(self._market_open_values, self._market_close_values, minute_dt.value / NANOS_IN_MINUTE, self._minutes_per_day, False)

    def load_raw_arrays(self, fields, start_dt, end_dt, sids):
        if False:
            print('Hello World!')
        "\n        Parameters\n        ----------\n        fields : list of str\n           'open', 'high', 'low', 'close', or 'volume'\n        start_dt: Timestamp\n           Beginning of the window range.\n        end_dt: Timestamp\n           End of the window range.\n        sids : list of int\n           The asset identifiers in the window.\n\n        Returns\n        -------\n        list of np.ndarray\n            A list with an entry per field of ndarrays with shape\n            (minutes in range, sids) with a dtype of float64, containing the\n            values for the respective field over start and end dt range.\n        "
        start_idx = self._find_position_of_minute(start_dt)
        end_idx = self._find_position_of_minute(end_dt)
        num_minutes = end_idx - start_idx + 1
        results = []
        indices_to_exclude = self._exclusion_indices_for_range(start_idx, end_idx)
        if indices_to_exclude is not None:
            for (excl_start, excl_stop) in indices_to_exclude:
                length = excl_stop - excl_start + 1
                num_minutes -= length
        shape = (num_minutes, len(sids))
        for field in fields:
            if field != 'volume':
                out = np.full(shape, np.nan)
            else:
                out = np.zeros(shape, dtype=np.uint32)
            for (i, sid) in enumerate(sids):
                carray = self._open_minute_file(field, sid)
                values = carray[start_idx:end_idx + 1]
                if indices_to_exclude is not None:
                    for (excl_start, excl_stop) in indices_to_exclude[::-1]:
                        excl_slice = np.s_[excl_start - start_idx:excl_stop - start_idx + 1]
                        values = np.delete(values, excl_slice)
                where = values != 0
                if field != 'volume':
                    out[:len(where), i][where] = values[where] * self._ohlc_ratio_inverse_for_sid(sid)
                else:
                    out[:len(where), i][where] = values[where]
            results.append(out)
        return results

class MinuteBarUpdateReader(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for minute update readers.
    """

    @abstractmethod
    def read(self, dts, sids):
        if False:
            for i in range(10):
                print('nop')
        '\n        Read and return pricing update data.\n\n        Parameters\n        ----------\n        dts : DatetimeIndex\n            The minutes for which to read the pricing updates.\n        sids : iter[int]\n            The sids for which to read the pricing updates.\n\n        Returns\n        -------\n        data : iter[(int, DataFrame)]\n            Returns an iterable of ``sid`` to the corresponding OHLCV data.\n        '
        raise NotImplementedError()

class H5MinuteBarUpdateWriter(object):
    """
    Writer for files containing minute bar updates for consumption by a writer
    for a ``MinuteBarReader`` format.

    Parameters
    ----------
    path : str
        The destination path.
    complevel : int, optional
        The HDF5 complevel, defaults to ``5``.
    complib : str, optional
        The HDF5 complib, defaults to ``zlib``.
    """
    FORMAT_VERSION = 0
    _COMPLEVEL = 5
    _COMPLIB = 'zlib'

    def __init__(self, path, complevel=None, complib=None):
        if False:
            print('Hello World!')
        self._complevel = complevel if complevel is not None else self._COMPLEVEL
        self._complib = complib if complib is not None else self._COMPLIB
        self._path = path

    def write(self, frames):
        if False:
            i = 10
            return i + 15
        '\n        Write the frames to the target HDF5 file, using the format used by\n        ``pd.Panel.to_hdf``\n\n        Parameters\n        ----------\n        frames : iter[(int, DataFrame)] or dict[int -> DataFrame]\n            An iterable or other mapping of sid to the corresponding OHLCV\n            pricing data.\n        '
        with HDFStore(self._path, 'w', complevel=self._complevel, complib=self._complib) as store:
            panel = pd.Panel.from_dict(dict(frames))
            panel.to_hdf(store, 'updates')
        with tables.open_file(self._path, mode='r+') as h5file:
            h5file.set_node_attr('/', 'version', 0)

class H5MinuteBarUpdateReader(MinuteBarUpdateReader):
    """
    Reader for minute bar updates stored in HDF5 files.

    Parameters
    ----------
    path : str
        The path of the HDF5 file from which to source data.
    """

    def __init__(self, path):
        if False:
            i = 10
            return i + 15
        try:
            self._panel = pd.read_hdf(path)
            return
        except TypeError:
            pass
        with h5py.File(path, 'r') as f:
            updates = f['updates']
            values = updates['block0_values']
            items = updates['axis0']
            major = updates['axis1']
            minor = updates['axis2']
            try:
                tz = major.attrs['tz'].decode()
            except OSError:
                tz = 'UTC'
            self._panel = pd.Panel(data=np.array(values).T, items=np.array(items), major_axis=pd.DatetimeIndex(major, tz=tz, freq='T'), minor_axis=np.array(minor).astype('U'))

    def read(self, dts, sids):
        if False:
            return 10
        panel = self._panel[sids, dts, :]
        return panel.iteritems()