"""
Module for building a complete dataset from local directory with csv files.
"""
import os
import sys
from logbook import Logger, StreamHandler
from numpy import empty
from pandas import DataFrame, read_csv, Index, Timedelta, NaT
from trading_calendars import register_calendar_alias
from zipline.utils.cli import maybe_show_progress
from . import core as bundles
handler = StreamHandler(sys.stdout, format_string=' | {record.message}')
logger = Logger(__name__)
logger.handlers.append(handler)

def csvdir_equities(tframes=None, csvdir=None):
    if False:
        print('Hello World!')
    '\n    Generate an ingest function for custom data bundle\n    This function can be used in ~/.zipline/extension.py\n    to register bundle with custom parameters, e.g. with\n    a custom trading calendar.\n\n    Parameters\n    ----------\n    tframes: tuple, optional\n        The data time frames, supported timeframes: \'daily\' and \'minute\'\n    csvdir : string, optional, default: CSVDIR environment variable\n        The path to the directory of this structure:\n        <directory>/<timeframe1>/<symbol1>.csv\n        <directory>/<timeframe1>/<symbol2>.csv\n        <directory>/<timeframe1>/<symbol3>.csv\n        <directory>/<timeframe2>/<symbol1>.csv\n        <directory>/<timeframe2>/<symbol2>.csv\n        <directory>/<timeframe2>/<symbol3>.csv\n\n    Returns\n    -------\n    ingest : callable\n        The bundle ingest function\n\n    Examples\n    --------\n    This code should be added to ~/.zipline/extension.py\n    .. code-block:: python\n       from zipline.data.bundles import csvdir_equities, register\n       register(\'custom-csvdir-bundle\',\n                csvdir_equities(["daily", "minute"],\n                \'/full/path/to/the/csvdir/directory\'))\n    '
    return CSVDIRBundle(tframes, csvdir).ingest

class CSVDIRBundle:
    """
    Wrapper class to call csvdir_bundle with provided
    list of time frames and a path to the csvdir directory
    """

    def __init__(self, tframes=None, csvdir=None):
        if False:
            i = 10
            return i + 15
        self.tframes = tframes
        self.csvdir = csvdir

    def ingest(self, environ, asset_db_writer, minute_bar_writer, daily_bar_writer, adjustment_writer, calendar, start_session, end_session, cache, show_progress, output_dir):
        if False:
            print('Hello World!')
        csvdir_bundle(environ, asset_db_writer, minute_bar_writer, daily_bar_writer, adjustment_writer, calendar, start_session, end_session, cache, show_progress, output_dir, self.tframes, self.csvdir)

@bundles.register('csvdir')
def csvdir_bundle(environ, asset_db_writer, minute_bar_writer, daily_bar_writer, adjustment_writer, calendar, start_session, end_session, cache, show_progress, output_dir, tframes=None, csvdir=None):
    if False:
        return 10
    '\n    Build a zipline data bundle from the directory with csv files.\n    '
    if not csvdir:
        csvdir = environ.get('CSVDIR')
        if not csvdir:
            raise ValueError('CSVDIR environment variable is not set')
    if not os.path.isdir(csvdir):
        raise ValueError('%s is not a directory' % csvdir)
    if not tframes:
        tframes = set(['daily', 'minute']).intersection(os.listdir(csvdir))
        if not tframes:
            raise ValueError("'daily' and 'minute' directories not found in '%s'" % csvdir)
    divs_splits = {'divs': DataFrame(columns=['sid', 'amount', 'ex_date', 'record_date', 'declared_date', 'pay_date']), 'splits': DataFrame(columns=['sid', 'ratio', 'effective_date'])}
    for tframe in tframes:
        ddir = os.path.join(csvdir, tframe)
        symbols = sorted((item.split('.csv')[0] for item in os.listdir(ddir) if '.csv' in item))
        if not symbols:
            raise ValueError('no <symbol>.csv* files found in %s' % ddir)
        dtype = [('start_date', 'datetime64[ns]'), ('end_date', 'datetime64[ns]'), ('auto_close_date', 'datetime64[ns]'), ('symbol', 'object')]
        metadata = DataFrame(empty(len(symbols), dtype=dtype))
        if tframe == 'minute':
            writer = minute_bar_writer
        else:
            writer = daily_bar_writer
        writer.write(_pricing_iter(ddir, symbols, metadata, divs_splits, show_progress), show_progress=show_progress)
        metadata['exchange'] = 'CSVDIR'
        asset_db_writer.write(equities=metadata)
        divs_splits['divs']['sid'] = divs_splits['divs']['sid'].astype(int)
        divs_splits['splits']['sid'] = divs_splits['splits']['sid'].astype(int)
        adjustment_writer.write(splits=divs_splits['splits'], dividends=divs_splits['divs'])

def _pricing_iter(csvdir, symbols, metadata, divs_splits, show_progress):
    if False:
        for i in range(10):
            print('nop')
    with maybe_show_progress(symbols, show_progress, label='Loading custom pricing data: ') as it:
        files = os.listdir(csvdir)
        for (sid, symbol) in enumerate(it):
            logger.debug('%s: sid %s' % (symbol, sid))
            try:
                fname = [fname for fname in files if '%s.csv' % symbol in fname][0]
            except IndexError:
                raise ValueError('%s.csv file is not in %s' % (symbol, csvdir))
            dfr = read_csv(os.path.join(csvdir, fname), parse_dates=[0], infer_datetime_format=True, index_col=0).sort_index()
            start_date = dfr.index[0]
            end_date = dfr.index[-1]
            ac_date = end_date + Timedelta(days=1)
            metadata.iloc[sid] = (start_date, end_date, ac_date, symbol)
            if 'split' in dfr.columns:
                tmp = 1.0 / dfr[dfr['split'] != 1.0]['split']
                split = DataFrame(data=tmp.index.tolist(), columns=['effective_date'])
                split['ratio'] = tmp.tolist()
                split['sid'] = sid
                splits = divs_splits['splits']
                index = Index(range(splits.shape[0], splits.shape[0] + split.shape[0]))
                split.set_index(index, inplace=True)
                divs_splits['splits'] = splits.append(split)
            if 'dividend' in dfr.columns:
                tmp = dfr[dfr['dividend'] != 0.0]['dividend']
                div = DataFrame(data=tmp.index.tolist(), columns=['ex_date'])
                div['record_date'] = NaT
                div['declared_date'] = NaT
                div['pay_date'] = NaT
                div['amount'] = tmp.tolist()
                div['sid'] = sid
                divs = divs_splits['divs']
                ind = Index(range(divs.shape[0], divs.shape[0] + div.shape[0]))
                div.set_index(ind, inplace=True)
                divs_splits['divs'] = divs.append(div)
            yield (sid, dfr)
register_calendar_alias('CSVDIR', 'NYSE')