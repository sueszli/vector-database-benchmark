from abc import ABCMeta, abstractmethod
from collections import namedtuple
import hashlib
from textwrap import dedent
import warnings
from logbook import Logger
import numpy
import pandas as pd
from pandas import read_csv
import pytz
import requests
from six import StringIO, iteritems, with_metaclass
from zipline.errors import MultipleSymbolsFound, SymbolNotFound, ZiplineError
from zipline.protocol import DATASOURCE_TYPE, Event
from zipline.assets import Equity
logger = Logger('Requests Source Logger')

def roll_dts_to_midnight(dts, trading_day):
    if False:
        for i in range(10):
            print('nop')
    if len(dts) == 0:
        return dts
    return pd.DatetimeIndex((dts.tz_convert('US/Eastern') - pd.Timedelta(hours=16)).date, tz='UTC') + trading_day

class FetcherEvent(Event):
    pass

class FetcherCSVRedirectError(ZiplineError):
    msg = dedent('        Attempt to fetch_csv from a redirected url. {url}\n        must be changed to {new_url}\n        ')

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.url = kwargs['url']
        self.new_url = kwargs['new_url']
        self.extra = kwargs['extra']
        super(FetcherCSVRedirectError, self).__init__(*args, **kwargs)
ALLOWED_REQUESTS_KWARGS = {'params', 'headers', 'auth', 'cert'}
ALLOWED_READ_CSV_KWARGS = {'sep', 'dialect', 'doublequote', 'escapechar', 'quotechar', 'quoting', 'skipinitialspace', 'lineterminator', 'header', 'index_col', 'names', 'prefix', 'skiprows', 'skipfooter', 'skip_footer', 'na_values', 'true_values', 'false_values', 'delimiter', 'converters', 'dtype', 'delim_whitespace', 'as_recarray', 'na_filter', 'compact_ints', 'use_unsigned', 'buffer_lines', 'warn_bad_lines', 'error_bad_lines', 'keep_default_na', 'thousands', 'comment', 'decimal', 'keep_date_col', 'nrows', 'chunksize', 'encoding', 'usecols'}
SHARED_REQUESTS_KWARGS = {'stream': True, 'allow_redirects': False}

def mask_requests_args(url, validating=False, params_checker=None, **kwargs):
    if False:
        return 10
    requests_kwargs = {key: val for (key, val) in iteritems(kwargs) if key in ALLOWED_REQUESTS_KWARGS}
    if params_checker is not None:
        (url, s_params) = params_checker(url)
        if s_params:
            if 'params' in requests_kwargs:
                requests_kwargs['params'].update(s_params)
            else:
                requests_kwargs['params'] = s_params
    requests_kwargs['timeout'] = 1.0 if validating else 30.0
    requests_kwargs.update(SHARED_REQUESTS_KWARGS)
    request_pair = namedtuple('RequestPair', ('requests_kwargs', 'url'))
    return request_pair(requests_kwargs, url)

class PandasCSV(with_metaclass(ABCMeta, object)):

    def __init__(self, pre_func, post_func, asset_finder, trading_day, start_date, end_date, date_column, date_format, timezone, symbol, mask, symbol_column, data_frequency, country_code, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.start_date = start_date
        self.end_date = end_date
        self.date_column = date_column
        self.date_format = date_format
        self.timezone = timezone
        self.mask = mask
        self.symbol_column = symbol_column or 'symbol'
        self.data_frequency = data_frequency
        self.country_code = country_code
        invalid_kwargs = set(kwargs) - ALLOWED_READ_CSV_KWARGS
        if invalid_kwargs:
            raise TypeError('Unexpected keyword arguments: %s' % invalid_kwargs)
        self.pandas_kwargs = self.mask_pandas_args(kwargs)
        self.symbol = symbol
        self.finder = asset_finder
        self.trading_day = trading_day
        self.pre_func = pre_func
        self.post_func = post_func

    @property
    def fields(self):
        if False:
            print('Hello World!')
        return self.df.columns.tolist()

    def get_hash(self):
        if False:
            return 10
        return self.namestring

    @abstractmethod
    def fetch_data(self):
        if False:
            for i in range(10):
                print('nop')
        return

    @staticmethod
    def parse_date_str_series(format_str, tz, date_str_series, data_frequency, trading_day):
        if False:
            for i in range(10):
                print('nop')
        '\n        Efficient parsing for a 1d Pandas/numpy object containing string\n        representations of dates.\n\n        Note: pd.to_datetime is significantly faster when no format string is\n        passed, and in pandas 0.12.0 the %p strptime directive is not correctly\n        handled if a format string is explicitly passed, but AM/PM is handled\n        properly if format=None.\n\n        Moreover, we were previously ignoring this parameter unintentionally\n        because we were incorrectly passing it as a positional.  For all these\n        reasons, we ignore the format_str parameter when parsing datetimes.\n        '
        if format_str is not None:
            logger.warn("The 'format_str' parameter to fetch_csv is deprecated. Ignoring and defaulting to pandas default date parsing.")
            format_str = None
        tz_str = str(tz)
        if tz_str == pytz.utc.zone:
            parsed = pd.to_datetime(date_str_series.values, format=format_str, utc=True, errors='coerce')
        else:
            parsed = pd.to_datetime(date_str_series.values, format=format_str, errors='coerce').tz_localize(tz_str).tz_convert('UTC')
        if data_frequency == 'daily':
            parsed = roll_dts_to_midnight(parsed, trading_day)
        return parsed

    def mask_pandas_args(self, kwargs):
        if False:
            while True:
                i = 10
        pandas_kwargs = {key: val for (key, val) in iteritems(kwargs) if key in ALLOWED_READ_CSV_KWARGS}
        if 'usecols' in pandas_kwargs:
            usecols = pandas_kwargs['usecols']
            if usecols and self.date_column not in usecols:
                with_date = list(usecols)
                with_date.append(self.date_column)
                pandas_kwargs['usecols'] = with_date
        pandas_kwargs.setdefault('keep_default_na', False)
        pandas_kwargs.setdefault('na_values', {'symbol': []})
        return pandas_kwargs

    def _lookup_unconflicted_symbol(self, symbol):
        if False:
            print('Hello World!')
        '\n        Attempt to find a unique asset whose symbol is the given string.\n\n        If multiple assets have held the given symbol, return a 0.\n\n        If no asset has held the given symbol, return a  NaN.\n        '
        try:
            uppered = symbol.upper()
        except AttributeError:
            return numpy.nan
        try:
            return self.finder.lookup_symbol(uppered, as_of_date=None, country_code=self.country_code)
        except MultipleSymbolsFound:
            return 0
        except SymbolNotFound:
            return numpy.nan

    def load_df(self):
        if False:
            while True:
                i = 10
        df = self.fetch_data()
        if self.pre_func:
            df = self.pre_func(df)
        df['dt'] = self.parse_date_str_series(self.date_format, self.timezone, df[self.date_column], self.data_frequency, self.trading_day).values
        df = df[df['dt'].notnull()]
        if self.symbol is not None:
            df['sid'] = self.symbol
        elif self.finder:
            df.sort_values(by=self.symbol_column, inplace=True)
            try:
                df.pop('sid')
                warnings.warn("Assignment of the 'sid' column of a DataFrame is not supported by Fetcher. The 'sid' column has been overwritten.", category=UserWarning, stacklevel=2)
            except KeyError:
                pass
            unique_symbols = df[self.symbol_column].unique()
            sid_series = pd.Series(data=map(self._lookup_unconflicted_symbol, unique_symbols), index=unique_symbols, name='sid')
            df = df.join(sid_series, on=self.symbol_column)
            conflict_rows = df[df['sid'] == 0]
            for (row_idx, row) in conflict_rows.iterrows():
                try:
                    asset = self.finder.lookup_symbol(row[self.symbol_column], row['dt'].replace(tzinfo=pytz.utc), country_code=self.country_code) or numpy.nan
                except SymbolNotFound:
                    asset = numpy.nan
                df.ix[row_idx, 'sid'] = asset
            length_before_drop = len(df)
            df = df[df['sid'].notnull()]
            no_sid_count = length_before_drop - len(df)
            if no_sid_count:
                logger.warn('Dropped {} rows from fetched csv.'.format(no_sid_count), no_sid_count, extra={'syslog': True})
        else:
            df['sid'] = df['symbol']
        df.drop_duplicates(['sid', 'dt'])
        df.set_index(['dt'], inplace=True)
        df = df.tz_localize('UTC')
        df.sort_index(inplace=True)
        cols_to_drop = [self.date_column]
        if self.symbol is None:
            cols_to_drop.append(self.symbol_column)
        df = df[df.columns.drop(cols_to_drop)]
        if self.post_func:
            df = self.post_func(df)
        return df

    def __iter__(self):
        if False:
            while True:
                i = 10
        asset_cache = {}
        for (dt, series) in self.df.iterrows():
            if dt < self.start_date:
                continue
            if dt > self.end_date:
                return
            event = FetcherEvent()
            event.dt = dt
            for (k, v) in series.iteritems():
                if isinstance(v, numpy.integer):
                    v = int(v)
                setattr(event, k, v)
            if event.sid in asset_cache:
                event.sid = asset_cache[event.sid]
            elif hasattr(event.sid, 'start_date'):
                asset_cache[event.sid] = event.sid
            elif self.finder and isinstance(event.sid, int):
                asset = self.finder.retrieve_asset(event.sid, default_none=True)
                if asset:
                    event.sid = asset_cache[asset] = asset
                elif self.mask:
                    continue
                elif self.symbol is None:
                    event.sid = asset_cache[event.sid] = Equity(event.sid)
            event.type = DATASOURCE_TYPE.CUSTOM
            event.source_id = self.namestring
            yield event

class PandasRequestsCSV(PandasCSV):
    MAX_DOCUMENT_SIZE = 1024 * 1024 * 100
    CONTENT_CHUNK_SIZE = 4096

    def __init__(self, url, pre_func, post_func, asset_finder, trading_day, start_date, end_date, date_column, date_format, timezone, symbol, mask, symbol_column, data_frequency, country_code, special_params_checker=None, **kwargs):
        if False:
            return 10
        (self._requests_kwargs, self.url) = mask_requests_args(url, params_checker=special_params_checker, **kwargs)
        remaining_kwargs = {k: v for (k, v) in iteritems(kwargs) if k not in self.requests_kwargs}
        self.namestring = type(self).__name__
        super(PandasRequestsCSV, self).__init__(pre_func, post_func, asset_finder, trading_day, start_date, end_date, date_column, date_format, timezone, symbol, mask, symbol_column, data_frequency, country_code=country_code, **remaining_kwargs)
        self.fetch_size = None
        self.fetch_hash = None
        self.df = self.load_df()
        self.special_params_checker = special_params_checker

    @property
    def requests_kwargs(self):
        if False:
            i = 10
            return i + 15
        return self._requests_kwargs

    def fetch_url(self, url):
        if False:
            i = 10
            return i + 15
        info = 'checking {url} with {params}'
        logger.info(info.format(url=url, params=self.requests_kwargs))
        try:
            response = requests.get(url, **self.requests_kwargs)
        except requests.exceptions.ConnectionError:
            raise Exception('Could not connect to %s' % url)
        if not response.ok:
            raise Exception('Problem reaching %s' % url)
        elif response.is_redirect:
            new_url = response.headers['location']
            raise FetcherCSVRedirectError(url=url, new_url=new_url, extra={'old_url': url, 'new_url': new_url})
        content_length = 0
        logger.info('{} connection established in {:.1f} seconds'.format(url, response.elapsed.total_seconds()))
        for chunk in response.iter_content(self.CONTENT_CHUNK_SIZE, decode_unicode=True):
            if content_length > self.MAX_DOCUMENT_SIZE:
                raise Exception('Document size too big.')
            if chunk:
                content_length += len(chunk)
                yield chunk
        return

    def fetch_data(self):
        if False:
            return 10
        data = self.fetch_url(self.url)
        fd = StringIO()
        if isinstance(data, str):
            fd.write(data)
        else:
            for chunk in data:
                fd.write(chunk)
        self.fetch_size = fd.tell()
        fd.seek(0)
        try:
            frames = read_csv(fd, **self.pandas_kwargs)
            frames_hash = hashlib.md5(str(fd.getvalue()).encode('utf-8'))
            self.fetch_hash = frames_hash.hexdigest()
        except pd.parser.CParserError:
            raise Exception('Error parsing remote CSV data.')
        finally:
            fd.close()
        return frames