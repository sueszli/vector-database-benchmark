from functools import partial
import warnings
from bcolz import carray, ctable
import logbook
import numpy as np
from numpy import array, full, iinfo, nan
from pandas import DatetimeIndex, NaT, read_csv, to_datetime, Timestamp
from six import iteritems, viewkeys
from toolz import compose
from trading_calendars import get_calendar
from zipline.data.session_bars import CurrencyAwareSessionBarReader
from zipline.data.bar_reader import NoDataAfterDate, NoDataBeforeDate, NoDataOnDate
from zipline.utils.functional import apply
from zipline.utils.input_validation import expect_element
from zipline.utils.numpy_utils import iNaT, float64_dtype, uint32_dtype
from zipline.utils.memoize import lazyval
from zipline.utils.cli import maybe_show_progress
from ._equities import _compute_row_slices, _read_bcolz_data
logger = logbook.Logger('UsEquityPricing')
OHLC = frozenset(['open', 'high', 'low', 'close'])
US_EQUITY_PRICING_BCOLZ_COLUMNS = ('open', 'high', 'low', 'close', 'volume', 'day', 'id')
UINT32_MAX = iinfo(np.uint32).max

def check_uint32_safe(value, colname):
    if False:
        i = 10
        return i + 15
    if value >= UINT32_MAX:
        raise ValueError("Value %s from column '%s' is too large" % (value, colname))

@expect_element(invalid_data_behavior={'warn', 'raise', 'ignore'})
def winsorise_uint32(df, invalid_data_behavior, column, *columns):
    if False:
        while True:
            i = 10
    "Drops any record where a value would not fit into a uint32.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        The dataframe to winsorise.\n    invalid_data_behavior : {'warn', 'raise', 'ignore'}\n        What to do when data is outside the bounds of a uint32.\n    *columns : iterable[str]\n        The names of the columns to check.\n\n    Returns\n    -------\n    truncated : pd.DataFrame\n        ``df`` with values that do not fit into a uint32 zeroed out.\n    "
    columns = list((column,) + columns)
    mask = df[columns] > UINT32_MAX
    if invalid_data_behavior != 'ignore':
        mask |= df[columns].isnull()
    else:
        df[columns] = np.nan_to_num(df[columns])
    mv = mask.values
    if mv.any():
        if invalid_data_behavior == 'raise':
            raise ValueError('%d values out of bounds for uint32: %r' % (mv.sum(), df[mask.any(axis=1)]))
        if invalid_data_behavior == 'warn':
            warnings.warn('Ignoring %d values because they are out of bounds for uint32: %r' % (mv.sum(), df[mask.any(axis=1)]), stacklevel=3)
    df[mask] = 0
    return df

class BcolzDailyBarWriter(object):
    """
    Class capable of writing daily OHLCV data to disk in a format that can
    be read efficiently by BcolzDailyOHLCVReader.

    Parameters
    ----------
    filename : str
        The location at which we should write our output.
    calendar : zipline.utils.calendar.trading_calendar
        Calendar to use to compute asset calendar offsets.
    start_session: pd.Timestamp
        Midnight UTC session label.
    end_session: pd.Timestamp
        Midnight UTC session label.

    See Also
    --------
    zipline.data.bcolz_daily_bars.BcolzDailyBarReader
    """
    _csv_dtypes = {'open': float64_dtype, 'high': float64_dtype, 'low': float64_dtype, 'close': float64_dtype, 'volume': float64_dtype}

    def __init__(self, filename, calendar, start_session, end_session):
        if False:
            return 10
        self._filename = filename
        if start_session != end_session:
            if not calendar.is_session(start_session):
                raise ValueError('Start session %s is invalid!' % start_session)
            if not calendar.is_session(end_session):
                raise ValueError('End session %s is invalid!' % end_session)
        self._start_session = start_session
        self._end_session = end_session
        self._calendar = calendar

    @property
    def progress_bar_message(self):
        if False:
            i = 10
            return i + 15
        return 'Merging daily equity files:'

    def progress_bar_item_show_func(self, value):
        if False:
            return 10
        return value if value is None else str(value[0])

    def write(self, data, assets=None, show_progress=False, invalid_data_behavior='warn'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parameters\n        ----------\n        data : iterable[tuple[int, pandas.DataFrame or bcolz.ctable]]\n            The data chunks to write. Each chunk should be a tuple of sid\n            and the data for that asset.\n        assets : set[int], optional\n            The assets that should be in ``data``. If this is provided\n            we will check ``data`` against the assets and provide better\n            progress information.\n        show_progress : bool, optional\n            Whether or not to show a progress bar while writing.\n        invalid_data_behavior : {'warn', 'raise', 'ignore'}, optional\n            What to do when data is encountered that is outside the range of\n            a uint32.\n\n        Returns\n        -------\n        table : bcolz.ctable\n            The newly-written table.\n        "
        ctx = maybe_show_progress(((sid, self.to_ctable(df, invalid_data_behavior)) for (sid, df) in data), show_progress=show_progress, item_show_func=self.progress_bar_item_show_func, label=self.progress_bar_message, length=len(assets) if assets is not None else None)
        with ctx as it:
            return self._write_internal(it, assets)

    def write_csvs(self, asset_map, show_progress=False, invalid_data_behavior='warn'):
        if False:
            while True:
                i = 10
        "Read CSVs as DataFrames from our asset map.\n\n        Parameters\n        ----------\n        asset_map : dict[int -> str]\n            A mapping from asset id to file path with the CSV data for that\n            asset\n        show_progress : bool\n            Whether or not to show a progress bar while writing.\n        invalid_data_behavior : {'warn', 'raise', 'ignore'}\n            What to do when data is encountered that is outside the range of\n            a uint32.\n        "
        read = partial(read_csv, parse_dates=['day'], index_col='day', dtype=self._csv_dtypes)
        return self.write(((asset, read(path)) for (asset, path) in iteritems(asset_map)), assets=viewkeys(asset_map), show_progress=show_progress, invalid_data_behavior=invalid_data_behavior)

    def _write_internal(self, iterator, assets):
        if False:
            i = 10
            return i + 15
        '\n        Internal implementation of write.\n\n        `iterator` should be an iterator yielding pairs of (asset, ctable).\n        '
        total_rows = 0
        first_row = {}
        last_row = {}
        calendar_offset = {}
        columns = {k: carray(array([], dtype=uint32_dtype)) for k in US_EQUITY_PRICING_BCOLZ_COLUMNS}
        earliest_date = None
        sessions = self._calendar.sessions_in_range(self._start_session, self._end_session)
        if assets is not None:

            @apply
            def iterator(iterator=iterator, assets=set(assets)):
                if False:
                    for i in range(10):
                        print('nop')
                for (asset_id, table) in iterator:
                    if asset_id not in assets:
                        raise ValueError('unknown asset id %r' % asset_id)
                    yield (asset_id, table)
        for (asset_id, table) in iterator:
            nrows = len(table)
            for column_name in columns:
                if column_name == 'id':
                    columns['id'].append(full((nrows,), asset_id, dtype='uint32'))
                    continue
                columns[column_name].append(table[column_name])
            if earliest_date is None:
                earliest_date = table['day'][0]
            else:
                earliest_date = min(earliest_date, table['day'][0])
            asset_key = str(asset_id)
            first_row[asset_key] = total_rows
            last_row[asset_key] = total_rows + nrows - 1
            total_rows += nrows
            table_day_to_session = compose(self._calendar.minute_to_session_label, partial(Timestamp, unit='s', tz='UTC'))
            asset_first_day = table_day_to_session(table['day'][0])
            asset_last_day = table_day_to_session(table['day'][-1])
            asset_sessions = sessions[sessions.slice_indexer(asset_first_day, asset_last_day)]
            assert len(table) == len(asset_sessions), 'Got {} rows for daily bars table with first day={}, last day={}, expected {} rows.\nMissing sessions: {}\nExtra sessions: {}'.format(len(table), asset_first_day.date(), asset_last_day.date(), len(asset_sessions), asset_sessions.difference(to_datetime(np.array(table['day']), unit='s', utc=True)).tolist(), to_datetime(np.array(table['day']), unit='s', utc=True).difference(asset_sessions).tolist())
            calendar_offset[asset_key] = sessions.get_loc(asset_first_day)
        full_table = ctable(columns=[columns[colname] for colname in US_EQUITY_PRICING_BCOLZ_COLUMNS], names=US_EQUITY_PRICING_BCOLZ_COLUMNS, rootdir=self._filename, mode='w')
        full_table.attrs['first_trading_day'] = earliest_date if earliest_date is not None else iNaT
        full_table.attrs['first_row'] = first_row
        full_table.attrs['last_row'] = last_row
        full_table.attrs['calendar_offset'] = calendar_offset
        full_table.attrs['calendar_name'] = self._calendar.name
        full_table.attrs['start_session_ns'] = self._start_session.value
        full_table.attrs['end_session_ns'] = self._end_session.value
        full_table.flush()
        return full_table

    @expect_element(invalid_data_behavior={'warn', 'raise', 'ignore'})
    def to_ctable(self, raw_data, invalid_data_behavior):
        if False:
            i = 10
            return i + 15
        if isinstance(raw_data, ctable):
            return raw_data
        winsorise_uint32(raw_data, invalid_data_behavior, 'volume', *OHLC)
        processed = (raw_data[list(OHLC)] * 1000).round().astype('uint32')
        dates = raw_data.index.values.astype('datetime64[s]')
        check_uint32_safe(dates.max().view(np.int64), 'day')
        processed['day'] = dates.astype('uint32')
        processed['volume'] = raw_data.volume.astype('uint32')
        return ctable.fromdataframe(processed)

class BcolzDailyBarReader(CurrencyAwareSessionBarReader):
    """
    Reader for raw pricing data written by BcolzDailyOHLCVWriter.

    Parameters
    ----------
    table : bcolz.ctable
        The ctable contaning the pricing data, with attrs corresponding to the
        Attributes list below.
    read_all_threshold : int
        The number of equities at which; below, the data is read by reading a
        slice from the carray per asset.  above, the data is read by pulling
        all of the data for all assets into memory and then indexing into that
        array for each day and asset pair.  Used to tune performance of reads
        when using a small or large number of equities.

    Attributes
    ----------
    The table with which this loader interacts contains the following
    attributes:

    first_row : dict
        Map from asset_id -> index of first row in the dataset with that id.
    last_row : dict
        Map from asset_id -> index of last row in the dataset with that id.
    calendar_offset : dict
        Map from asset_id -> calendar index of first row.
    start_session_ns: int
        Epoch ns of the first session used in this dataset.
    end_session_ns: int
        Epoch ns of the last session used in this dataset.
    calendar_name: str
        String identifier of trading calendar used (ie, "NYSE").

    We use first_row and last_row together to quickly find ranges of rows to
    load when reading an asset's data into memory.

    We use calendar_offset and calendar to orient loaded blocks within a
    range of queried dates.

    Notes
    ------
    A Bcolz CTable is comprised of Columns and Attributes.
    The table with which this loader interacts contains the following columns:

    ['open', 'high', 'low', 'close', 'volume', 'day', 'id'].

    The data in these columns is interpreted as follows:

    - Price columns ('open', 'high', 'low', 'close') are interpreted as 1000 *
      as-traded dollar value.
    - Volume is interpreted as as-traded volume.
    - Day is interpreted as seconds since midnight UTC, Jan 1, 1970.
    - Id is the asset id of the row.

    The data in each column is grouped by asset and then sorted by day within
    each asset block.

    The table is built to represent a long time range of data, e.g. ten years
    of equity data, so the lengths of each asset block is not equal to each
    other. The blocks are clipped to the known start and end date of each asset
    to cut down on the number of empty values that would need to be included to
    make a regular/cubic dataset.

    When read across the open, high, low, close, and volume with the same
    index should represent the same asset and day.

    See Also
    --------
    zipline.data.bcolz_daily_bars.BcolzDailyBarWriter
    """

    def __init__(self, table, read_all_threshold=3000):
        if False:
            while True:
                i = 10
        self._maybe_table_rootdir = table
        self._spot_cols = {}
        self.PRICE_ADJUSTMENT_FACTOR = 0.001
        self._read_all_threshold = read_all_threshold

    @lazyval
    def _table(self):
        if False:
            while True:
                i = 10
        maybe_table_rootdir = self._maybe_table_rootdir
        if isinstance(maybe_table_rootdir, ctable):
            return maybe_table_rootdir
        return ctable(rootdir=maybe_table_rootdir, mode='r')

    @lazyval
    def sessions(self):
        if False:
            print('Hello World!')
        if 'calendar' in self._table.attrs.attrs:
            return DatetimeIndex(self._table.attrs['calendar'], tz='UTC')
        else:
            cal = get_calendar(self._table.attrs['calendar_name'])
            start_session_ns = self._table.attrs['start_session_ns']
            start_session = Timestamp(start_session_ns, tz='UTC')
            end_session_ns = self._table.attrs['end_session_ns']
            end_session = Timestamp(end_session_ns, tz='UTC')
            sessions = cal.sessions_in_range(start_session, end_session)
            return sessions

    @lazyval
    def _first_rows(self):
        if False:
            print('Hello World!')
        return {int(asset_id): start_index for (asset_id, start_index) in iteritems(self._table.attrs['first_row'])}

    @lazyval
    def _last_rows(self):
        if False:
            print('Hello World!')
        return {int(asset_id): end_index for (asset_id, end_index) in iteritems(self._table.attrs['last_row'])}

    @lazyval
    def _calendar_offsets(self):
        if False:
            while True:
                i = 10
        return {int(id_): offset for (id_, offset) in iteritems(self._table.attrs['calendar_offset'])}

    @lazyval
    def first_trading_day(self):
        if False:
            print('Hello World!')
        try:
            return Timestamp(self._table.attrs['first_trading_day'], unit='s', tz='UTC')
        except KeyError:
            return None

    @lazyval
    def trading_calendar(self):
        if False:
            while True:
                i = 10
        if 'calendar_name' in self._table.attrs.attrs:
            return get_calendar(self._table.attrs['calendar_name'])
        else:
            return None

    @property
    def last_available_dt(self):
        if False:
            print('Hello World!')
        return self.sessions[-1]

    def _compute_slices(self, start_idx, end_idx, assets):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the raw row indices to load for each asset on a query for the\n        given dates after applying a shift.\n\n        Parameters\n        ----------\n        start_idx : int\n            Index of first date for which we want data.\n        end_idx : int\n            Index of last date for which we want data.\n        assets : pandas.Int64Index\n            Assets for which we want to compute row indices\n\n        Returns\n        -------\n        A 3-tuple of (first_rows, last_rows, offsets):\n        first_rows : np.array[intp]\n            Array with length == len(assets) containing the index of the first\n            row to load for each asset in `assets`.\n        last_rows : np.array[intp]\n            Array with length == len(assets) containing the index of the last\n            row to load for each asset in `assets`.\n        offset : np.array[intp]\n            Array with length == (len(asset) containing the index in a buffer\n            of length `dates` corresponding to the first row of each asset.\n\n            The value of offset[i] will be 0 if asset[i] existed at the start\n            of a query.  Otherwise, offset[i] will be equal to the number of\n            entries in `dates` for which the asset did not yet exist.\n        '
        return _compute_row_slices(self._first_rows, self._last_rows, self._calendar_offsets, start_idx, end_idx, assets)

    def load_raw_arrays(self, columns, start_date, end_date, assets):
        if False:
            return 10
        start_idx = self._load_raw_arrays_date_to_index(start_date)
        end_idx = self._load_raw_arrays_date_to_index(end_date)
        (first_rows, last_rows, offsets) = self._compute_slices(start_idx, end_idx, assets)
        read_all = len(assets) > self._read_all_threshold
        return _read_bcolz_data(self._table, (end_idx - start_idx + 1, len(assets)), list(columns), first_rows, last_rows, offsets, read_all)

    def _load_raw_arrays_date_to_index(self, date):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.sessions.get_loc(date)
        except KeyError:
            raise NoDataOnDate(date)

    def _spot_col(self, colname):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the colname from daily_bar_table and read all of it into memory,\n        caching the result.\n\n        Parameters\n        ----------\n        colname : string\n            A name of a OHLCV carray in the daily_bar_table\n\n        Returns\n        -------\n        array (uint32)\n            Full read array of the carray in the daily_bar_table with the\n            given colname.\n        '
        try:
            col = self._spot_cols[colname]
        except KeyError:
            col = self._spot_cols[colname] = self._table[colname]
        return col

    def get_last_traded_dt(self, asset, day):
        if False:
            while True:
                i = 10
        volumes = self._spot_col('volume')
        search_day = day
        while True:
            try:
                ix = self.sid_day_index(asset, search_day)
            except NoDataBeforeDate:
                return NaT
            except NoDataAfterDate:
                prev_day_ix = self.sessions.get_loc(search_day) - 1
                if prev_day_ix > -1:
                    search_day = self.sessions[prev_day_ix]
                continue
            except NoDataOnDate:
                return NaT
            if volumes[ix] != 0:
                return search_day
            prev_day_ix = self.sessions.get_loc(search_day) - 1
            if prev_day_ix > -1:
                search_day = self.sessions[prev_day_ix]
            else:
                return NaT

    def sid_day_index(self, sid, day):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        sid : int\n            The asset identifier.\n        day : datetime64-like\n            Midnight of the day for which data is requested.\n\n        Returns\n        -------\n        int\n            Index into the data tape for the given sid and day.\n            Raises a NoDataOnDate exception if the given day and sid is before\n            or after the date range of the equity.\n        '
        try:
            day_loc = self.sessions.get_loc(day)
        except Exception:
            raise NoDataOnDate('day={0} is outside of calendar={1}'.format(day, self.sessions))
        offset = day_loc - self._calendar_offsets[sid]
        if offset < 0:
            raise NoDataBeforeDate('No data on or before day={0} for sid={1}'.format(day, sid))
        ix = self._first_rows[sid] + offset
        if ix > self._last_rows[sid]:
            raise NoDataAfterDate('No data on or after day={0} for sid={1}'.format(day, sid))
        return ix

    def get_value(self, sid, dt, field):
        if False:
            print('Hello World!')
        "\n        Parameters\n        ----------\n        sid : int\n            The asset identifier.\n        day : datetime64-like\n            Midnight of the day for which data is requested.\n        colname : string\n            The price field. e.g. ('open', 'high', 'low', 'close', 'volume')\n\n        Returns\n        -------\n        float\n            The spot price for colname of the given sid on the given day.\n            Raises a NoDataOnDate exception if the given day and sid is before\n            or after the date range of the equity.\n            Returns -1 if the day is within the date range, but the price is\n            0.\n        "
        ix = self.sid_day_index(sid, dt)
        price = self._spot_col(field)[ix]
        if field != 'volume':
            if price == 0:
                return nan
            else:
                return price * 0.001
        else:
            return price

    def currency_codes(self, sids):
        if False:
            print('Hello World!')
        first_rows = self._first_rows
        out = []
        for sid in sids:
            if sid in first_rows:
                out.append('USD')
            else:
                out.append(None)
        return np.array(out, dtype=object)