from collections import namedtuple
import re
import numpy as np
import pandas as pd
import sqlalchemy as sa
from toolz import first
from zipline.errors import AssetDBVersionError
from zipline.assets.asset_db_schema import ASSET_DB_VERSION, asset_db_table_names, asset_router, equities as equities_table, equity_symbol_mappings, equity_supplementary_mappings as equity_supplementary_mappings_table, futures_contracts as futures_contracts_table, exchanges as exchanges_table, futures_root_symbols, metadata, version_info
from zipline.utils.compat import ExitStack
from zipline.utils.preprocess import preprocess
from zipline.utils.range import from_tuple, intersecting_ranges
from zipline.utils.sqlite_utils import coerce_string_to_eng
AssetData = namedtuple('AssetData', ('equities', 'equities_mappings', 'futures', 'exchanges', 'root_symbols', 'equity_supplementary_mappings'))
SQLITE_MAX_VARIABLE_NUMBER = 999
symbol_columns = frozenset({'symbol', 'company_symbol', 'share_class_symbol'})
mapping_columns = symbol_columns | {'start_date', 'end_date'}
_index_columns = {'equities': 'sid', 'equity_supplementary_mappings': 'sid', 'futures': 'sid', 'exchanges': 'exchange', 'root_symbols': 'root_symbol'}

def _normalize_index_columns_in_place(equities, equity_supplementary_mappings, futures, exchanges, root_symbols):
    if False:
        for i in range(10):
            print('nop')
    "\n    Update dataframes in place to set indentifier columns as indices.\n\n    For each input frame, if the frame has a column with the same name as its\n    associated index column, set that column as the index.\n\n    Otherwise, assume the index already contains identifiers.\n\n    If frames are passed as None, they're ignored.\n    "
    for (frame, column_name) in ((equities, 'sid'), (equity_supplementary_mappings, 'sid'), (futures, 'sid'), (exchanges, 'exchange'), (root_symbols, 'root_symbol')):
        if frame is not None and column_name in frame:
            frame.set_index(column_name, inplace=True)

def _default_none(df, column):
    if False:
        i = 10
        return i + 15
    return None

def _no_default(df, column):
    if False:
        for i in range(10):
            print('nop')
    if not df.empty:
        raise ValueError('no default value for column %r' % column)
_equities_defaults = {'symbol': _default_none, 'asset_name': _default_none, 'start_date': lambda df, col: 0, 'end_date': lambda df, col: np.iinfo(np.int64).max, 'first_traded': _default_none, 'auto_close_date': _default_none, 'exchange': _no_default}
_direct_equities_defaults = _equities_defaults.copy()
del _direct_equities_defaults['symbol']
_futures_defaults = {'symbol': _default_none, 'root_symbol': _default_none, 'asset_name': _default_none, 'start_date': lambda df, col: 0, 'end_date': lambda df, col: np.iinfo(np.int64).max, 'first_traded': _default_none, 'exchange': _default_none, 'notice_date': _default_none, 'expiration_date': _default_none, 'auto_close_date': _default_none, 'tick_size': _default_none, 'multiplier': lambda df, col: 1}
_exchanges_defaults = {'canonical_name': lambda df, col: df.index, 'country_code': lambda df, col: '??'}
_root_symbols_defaults = {'sector': _default_none, 'description': _default_none, 'exchange': _default_none}
_equity_supplementary_mappings_defaults = {'value': _default_none, 'field': _default_none, 'start_date': lambda df, col: 0, 'end_date': lambda df, col: np.iinfo(np.int64).max}
_equity_symbol_mappings_defaults = {'sid': _no_default, 'company_symbol': _default_none, 'share_class_symbol': _default_none, 'symbol': _default_none, 'start_date': lambda df, col: 0, 'end_date': lambda df, col: np.iinfo(np.int64).max}
_delimited_symbol_delimiters_regex = re.compile('[./\\-_]')
_delimited_symbol_default_triggers = frozenset({np.nan, None, ''})

def split_delimited_symbol(symbol):
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes in a symbol that may be delimited and splits it in to a company\n    symbol and share class symbol. Also returns the fuzzy symbol, which is the\n    symbol without any fuzzy characters at all.\n\n    Parameters\n    ----------\n    symbol : str\n        The possibly-delimited symbol to be split\n\n    Returns\n    -------\n    company_symbol : str\n        The company part of the symbol.\n    share_class_symbol : str\n        The share class part of a symbol.\n    '
    if symbol in _delimited_symbol_default_triggers:
        return ('', '')
    symbol = symbol.upper()
    split_list = re.split(pattern=_delimited_symbol_delimiters_regex, string=symbol, maxsplit=1)
    company_symbol = split_list[0]
    if len(split_list) > 1:
        share_class_symbol = split_list[1]
    else:
        share_class_symbol = ''
    return (company_symbol, share_class_symbol)

def _generate_output_dataframe(data_subset, defaults):
    if False:
        for i in range(10):
            print('nop')
    "\n    Generates an output dataframe from the given subset of user-provided\n    data, the given column names, and the given default values.\n\n    Parameters\n    ----------\n    data_subset : DataFrame\n        A DataFrame, usually from an AssetData object,\n        that contains the user's input metadata for the asset type being\n        processed\n    defaults : dict\n        A dict where the keys are the names of the columns of the desired\n        output DataFrame and the values are a function from dataframe and\n        column name to the default values to insert in the DataFrame if no user\n        data is provided\n\n    Returns\n    -------\n    DataFrame\n        A DataFrame containing all user-provided metadata, and default values\n        wherever user-provided metadata was missing\n    "
    cols = set(data_subset.columns)
    desired_cols = set(defaults)
    data_subset.drop(cols - desired_cols, axis=1, inplace=True)
    for col in desired_cols - cols:
        data_subset[col] = defaults[col](data_subset, col)
    return data_subset

def _check_asset_group(group):
    if False:
        print('Hello World!')
    row = group.sort_values('end_date').iloc[-1]
    row.start_date = group.start_date.min()
    row.end_date = group.end_date.max()
    row.drop(list(symbol_columns), inplace=True)
    return row

def _format_range(r):
    if False:
        for i in range(10):
            print('nop')
    return (str(pd.Timestamp(r.start, unit='ns')), str(pd.Timestamp(r.stop, unit='ns')))

def _check_symbol_mappings(df, exchanges, asset_exchange):
    if False:
        i = 10
        return i + 15
    'Check that there are no cases where multiple symbols resolve to the same\n    asset at the same time in the same country.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        The equity symbol mappings table.\n    exchanges : pd.DataFrame\n        The exchanges table.\n    asset_exchange : pd.Series\n        A series that maps sids to the exchange the asset is in.\n\n    Raises\n    ------\n    ValueError\n        Raised when there are ambiguous symbol mappings.\n    '
    mappings = df.set_index('sid')[list(mapping_columns)].copy()
    mappings['country_code'] = exchanges['country_code'][asset_exchange.loc[df['sid']]].values
    ambigious = {}

    def check_intersections(persymbol):
        if False:
            while True:
                i = 10
        intersections = list(intersecting_ranges(map(from_tuple, zip(persymbol.start_date, persymbol.end_date))))
        if intersections:
            data = persymbol[['start_date', 'end_date']].astype('datetime64[ns]')
            msg_component = '\n  '.join(str(data).splitlines())
            ambigious[persymbol.name] = (intersections, msg_component)
    mappings.groupby(['symbol', 'country_code']).apply(check_intersections)
    if ambigious:
        raise ValueError('Ambiguous ownership for %d symbol%s, multiple assets held the following symbols:\n%s' % (len(ambigious), '' if len(ambigious) == 1 else 's', '\n'.join(('%s (%s):\n  intersections: %s\n  %s' % (symbol, country_code, tuple(map(_format_range, intersections)), cs) for ((symbol, country_code), (intersections, cs)) in sorted(ambigious.items(), key=first)))))

def _split_symbol_mappings(df, exchanges):
    if False:
        while True:
            i = 10
    'Split out the symbol: sid mappings from the raw data.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        The dataframe with multiple rows for each symbol: sid pair.\n    exchanges : pd.DataFrame\n        The exchanges table.\n\n    Returns\n    -------\n    asset_info : pd.DataFrame\n        The asset info with one row per asset.\n    symbol_mappings : pd.DataFrame\n        The dataframe of just symbol: sid mappings. The index will be\n        the sid, then there will be three columns: symbol, start_date, and\n        end_date.\n    '
    mappings = df[list(mapping_columns)]
    with pd.option_context('mode.chained_assignment', None):
        mappings['sid'] = mappings.index
    mappings.reset_index(drop=True, inplace=True)
    asset_exchange = df[['exchange', 'end_date']].sort_values('end_date').groupby(level=0)['exchange'].nth(-1)
    _check_symbol_mappings(mappings, exchanges, asset_exchange)
    return (df.groupby(level=0).apply(_check_asset_group), mappings)

def _dt_to_epoch_ns(dt_series):
    if False:
        i = 10
        return i + 15
    'Convert a timeseries into an Int64Index of nanoseconds since the epoch.\n\n    Parameters\n    ----------\n    dt_series : pd.Series\n        The timeseries to convert.\n\n    Returns\n    -------\n    idx : pd.Int64Index\n        The index converted to nanoseconds since the epoch.\n    '
    index = pd.to_datetime(dt_series.values)
    if index.tzinfo is None:
        index = index.tz_localize('UTC')
    else:
        index = index.tz_convert('UTC')
    return index.view(np.int64)

def check_version_info(conn, version_table, expected_version):
    if False:
        while True:
            i = 10
    '\n    Checks for a version value in the version table.\n\n    Parameters\n    ----------\n    conn : sa.Connection\n        The connection to use to perform the check.\n    version_table : sa.Table\n        The version table of the asset database\n    expected_version : int\n        The expected version of the asset database\n\n    Raises\n    ------\n    AssetDBVersionError\n        If the version is in the table and not equal to ASSET_DB_VERSION.\n    '
    version_from_table = conn.execute(sa.select((version_table.c.version,))).scalar()
    if version_from_table is None:
        version_from_table = 0
    if version_from_table != expected_version:
        raise AssetDBVersionError(db_version=version_from_table, expected_version=expected_version)

def write_version_info(conn, version_table, version_value):
    if False:
        for i in range(10):
            print('nop')
    '\n    Inserts the version value in to the version table.\n\n    Parameters\n    ----------\n    conn : sa.Connection\n        The connection to use to execute the insert.\n    version_table : sa.Table\n        The version table of the asset database\n    version_value : int\n        The version to write in to the database\n\n    '
    conn.execute(sa.insert(version_table, values={'version': version_value}))

class _empty(object):
    columns = ()

class AssetDBWriter(object):
    """Class used to write data to an assets db.

    Parameters
    ----------
    engine : Engine or str
        An SQLAlchemy engine or path to a SQL database.
    """
    DEFAULT_CHUNK_SIZE = SQLITE_MAX_VARIABLE_NUMBER

    @preprocess(engine=coerce_string_to_eng(require_exists=False))
    def __init__(self, engine):
        if False:
            while True:
                i = 10
        self.engine = engine

    def _real_write(self, equities, equity_symbol_mappings, equity_supplementary_mappings, futures, exchanges, root_symbols, chunk_size):
        if False:
            while True:
                i = 10
        with self.engine.begin() as conn:
            self.init_db(conn)
            if exchanges is not None:
                self._write_df_to_table(exchanges_table, exchanges, conn, chunk_size)
            if root_symbols is not None:
                self._write_df_to_table(futures_root_symbols, root_symbols, conn, chunk_size)
            if equity_supplementary_mappings is not None:
                self._write_df_to_table(equity_supplementary_mappings_table, equity_supplementary_mappings, conn, chunk_size)
            if futures is not None:
                self._write_assets('future', futures, conn, chunk_size)
            if equities is not None:
                self._write_assets('equity', equities, conn, chunk_size, mapping_data=equity_symbol_mappings)

    def write_direct(self, equities=None, equity_symbol_mappings=None, equity_supplementary_mappings=None, futures=None, exchanges=None, root_symbols=None, chunk_size=DEFAULT_CHUNK_SIZE):
        if False:
            for i in range(10):
                print('nop')
        "Write asset metadata to a sqlite database in the format that it is\n        stored in the assets db.\n\n        Parameters\n        ----------\n        equities : pd.DataFrame, optional\n            The equity metadata. The columns for this dataframe are:\n\n              symbol : str\n                  The ticker symbol for this equity.\n              asset_name : str\n                  The full name for this asset.\n              start_date : datetime\n                  The date when this asset was created.\n              end_date : datetime, optional\n                  The last date we have trade data for this asset.\n              first_traded : datetime, optional\n                  The first date we have trade data for this asset.\n              auto_close_date : datetime, optional\n                  The date on which to close any positions in this asset.\n              exchange : str\n                  The exchange where this asset is traded.\n\n            The index of this dataframe should contain the sids.\n        futures : pd.DataFrame, optional\n            The future contract metadata. The columns for this dataframe are:\n\n              symbol : str\n                  The ticker symbol for this futures contract.\n              root_symbol : str\n                  The root symbol, or the symbol with the expiration stripped\n                  out.\n              asset_name : str\n                  The full name for this asset.\n              start_date : datetime, optional\n                  The date when this asset was created.\n              end_date : datetime, optional\n                  The last date we have trade data for this asset.\n              first_traded : datetime, optional\n                  The first date we have trade data for this asset.\n              exchange : str\n                  The exchange where this asset is traded.\n              notice_date : datetime\n                  The date when the owner of the contract may be forced\n                  to take physical delivery of the contract's asset.\n              expiration_date : datetime\n                  The date when the contract expires.\n              auto_close_date : datetime\n                  The date when the broker will automatically close any\n                  positions in this contract.\n              tick_size : float\n                  The minimum price movement of the contract.\n              multiplier: float\n                  The amount of the underlying asset represented by this\n                  contract.\n        exchanges : pd.DataFrame, optional\n            The exchanges where assets can be traded. The columns of this\n            dataframe are:\n\n              exchange : str\n                  The full name of the exchange.\n              canonical_name : str\n                  The canonical name of the exchange.\n              country_code : str\n                  The ISO 3166 alpha-2 country code of the exchange.\n        root_symbols : pd.DataFrame, optional\n            The root symbols for the futures contracts. The columns for this\n            dataframe are:\n\n              root_symbol : str\n                  The root symbol name.\n              root_symbol_id : int\n                  The unique id for this root symbol.\n              sector : string, optional\n                  The sector of this root symbol.\n              description : string, optional\n                  A short description of this root symbol.\n              exchange : str\n                  The exchange where this root symbol is traded.\n        equity_supplementary_mappings : pd.DataFrame, optional\n            Additional mappings from values of abitrary type to assets.\n        chunk_size : int, optional\n            The amount of rows to write to the SQLite table at once.\n            This defaults to the default number of bind params in sqlite.\n            If you have compiled sqlite3 with more bind or less params you may\n            want to pass that value here.\n\n        "
        if equities is not None:
            equities = _generate_output_dataframe(equities, _direct_equities_defaults)
            if equity_symbol_mappings is None:
                raise ValueError('equities provided with no symbol mapping data')
            equity_symbol_mappings = _generate_output_dataframe(equity_symbol_mappings, _equity_symbol_mappings_defaults)
            _check_symbol_mappings(equity_symbol_mappings, exchanges, equities['exchange'])
        if equity_supplementary_mappings is not None:
            equity_supplementary_mappings = _generate_output_dataframe(equity_supplementary_mappings, _equity_supplementary_mappings_defaults)
        if futures is not None:
            futures = _generate_output_dataframe(_futures_defaults, futures)
        if exchanges is not None:
            exchanges = _generate_output_dataframe(exchanges.set_index('exchange'), _exchanges_defaults)
        if root_symbols is not None:
            root_symbols = _generate_output_dataframe(root_symbols, _root_symbols_defaults)
        _normalize_index_columns_in_place(equities=equities, equity_supplementary_mappings=equity_supplementary_mappings, futures=futures, exchanges=exchanges, root_symbols=root_symbols)
        self._real_write(equities=equities, equity_symbol_mappings=equity_symbol_mappings, equity_supplementary_mappings=equity_supplementary_mappings, futures=futures, exchanges=exchanges, root_symbols=root_symbols, chunk_size=chunk_size)

    def write(self, equities=None, futures=None, exchanges=None, root_symbols=None, equity_supplementary_mappings=None, chunk_size=DEFAULT_CHUNK_SIZE):
        if False:
            while True:
                i = 10
        "Write asset metadata to a sqlite database.\n\n        Parameters\n        ----------\n        equities : pd.DataFrame, optional\n            The equity metadata. The columns for this dataframe are:\n\n              symbol : str\n                  The ticker symbol for this equity.\n              asset_name : str\n                  The full name for this asset.\n              start_date : datetime\n                  The date when this asset was created.\n              end_date : datetime, optional\n                  The last date we have trade data for this asset.\n              first_traded : datetime, optional\n                  The first date we have trade data for this asset.\n              auto_close_date : datetime, optional\n                  The date on which to close any positions in this asset.\n              exchange : str\n                  The exchange where this asset is traded.\n\n            The index of this dataframe should contain the sids.\n        futures : pd.DataFrame, optional\n            The future contract metadata. The columns for this dataframe are:\n\n              symbol : str\n                  The ticker symbol for this futures contract.\n              root_symbol : str\n                  The root symbol, or the symbol with the expiration stripped\n                  out.\n              asset_name : str\n                  The full name for this asset.\n              start_date : datetime, optional\n                  The date when this asset was created.\n              end_date : datetime, optional\n                  The last date we have trade data for this asset.\n              first_traded : datetime, optional\n                  The first date we have trade data for this asset.\n              exchange : str\n                  The exchange where this asset is traded.\n              notice_date : datetime\n                  The date when the owner of the contract may be forced\n                  to take physical delivery of the contract's asset.\n              expiration_date : datetime\n                  The date when the contract expires.\n              auto_close_date : datetime\n                  The date when the broker will automatically close any\n                  positions in this contract.\n              tick_size : float\n                  The minimum price movement of the contract.\n              multiplier: float\n                  The amount of the underlying asset represented by this\n                  contract.\n        exchanges : pd.DataFrame, optional\n            The exchanges where assets can be traded. The columns of this\n            dataframe are:\n\n              exchange : str\n                  The full name of the exchange.\n              canonical_name : str\n                  The canonical name of the exchange.\n              country_code : str\n                  The ISO 3166 alpha-2 country code of the exchange.\n        root_symbols : pd.DataFrame, optional\n            The root symbols for the futures contracts. The columns for this\n            dataframe are:\n\n              root_symbol : str\n                  The root symbol name.\n              root_symbol_id : int\n                  The unique id for this root symbol.\n              sector : string, optional\n                  The sector of this root symbol.\n              description : string, optional\n                  A short description of this root symbol.\n              exchange : str\n                  The exchange where this root symbol is traded.\n        equity_supplementary_mappings : pd.DataFrame, optional\n            Additional mappings from values of abitrary type to assets.\n        chunk_size : int, optional\n            The amount of rows to write to the SQLite table at once.\n            This defaults to the default number of bind params in sqlite.\n            If you have compiled sqlite3 with more bind or less params you may\n            want to pass that value here.\n\n        See Also\n        --------\n        zipline.assets.asset_finder\n        "
        if exchanges is None:
            exchange_names = [df['exchange'] for df in (equities, futures, root_symbols) if df is not None]
            if exchange_names:
                exchanges = pd.DataFrame({'exchange': pd.concat(exchange_names).unique()})
        data = self._load_data(equities if equities is not None else pd.DataFrame(), futures if futures is not None else pd.DataFrame(), exchanges if exchanges is not None else pd.DataFrame(), root_symbols if root_symbols is not None else pd.DataFrame(), equity_supplementary_mappings if equity_supplementary_mappings is not None else pd.DataFrame())
        self._real_write(equities=data.equities, equity_symbol_mappings=data.equities_mappings, equity_supplementary_mappings=data.equity_supplementary_mappings, futures=data.futures, root_symbols=data.root_symbols, exchanges=data.exchanges, chunk_size=chunk_size)

    def _write_df_to_table(self, tbl, df, txn, chunk_size):
        if False:
            for i in range(10):
                print('nop')
        df = df.copy()
        for (column, dtype) in df.dtypes.iteritems():
            if dtype.kind == 'M':
                df[column] = _dt_to_epoch_ns(df[column])
        df.to_sql(tbl.name, txn.connection, index=True, index_label=first(tbl.primary_key.columns).name, if_exists='append', chunksize=chunk_size)

    def _write_assets(self, asset_type, assets, txn, chunk_size, mapping_data=None):
        if False:
            for i in range(10):
                print('nop')
        if asset_type == 'future':
            tbl = futures_contracts_table
            if mapping_data is not None:
                raise TypeError('no mapping data expected for futures')
        elif asset_type == 'equity':
            tbl = equities_table
            if mapping_data is None:
                raise TypeError('mapping data required for equities')
            self._write_df_to_table(equity_symbol_mappings, mapping_data, txn, chunk_size)
        else:
            raise ValueError("asset_type must be in {'future', 'equity'}, got: %s" % asset_type)
        self._write_df_to_table(tbl, assets, txn, chunk_size)
        pd.DataFrame({asset_router.c.sid.name: assets.index.values, asset_router.c.asset_type.name: asset_type}).to_sql(asset_router.name, txn.connection, if_exists='append', index=False, chunksize=chunk_size)

    def _all_tables_present(self, txn):
        if False:
            return 10
        '\n        Checks if any tables are present in the current assets database.\n\n        Parameters\n        ----------\n        txn : Transaction\n            The open transaction to check in.\n\n        Returns\n        -------\n        has_tables : bool\n            True if any tables are present, otherwise False.\n        '
        conn = txn.connect()
        for table_name in asset_db_table_names:
            if txn.dialect.has_table(conn, table_name):
                return True
        return False

    def init_db(self, txn=None):
        if False:
            for i in range(10):
                print('nop')
        'Connect to database and create tables.\n\n        Parameters\n        ----------\n        txn : sa.engine.Connection, optional\n            The transaction to execute in. If this is not provided, a new\n            transaction will be started with the engine provided.\n\n        Returns\n        -------\n        metadata : sa.MetaData\n            The metadata that describes the new assets db.\n        '
        with ExitStack() as stack:
            if txn is None:
                txn = stack.enter_context(self.engine.begin())
            tables_already_exist = self._all_tables_present(txn)
            metadata.create_all(txn, checkfirst=True)
            if tables_already_exist:
                check_version_info(txn, version_info, ASSET_DB_VERSION)
            else:
                write_version_info(txn, version_info, ASSET_DB_VERSION)

    def _normalize_equities(self, equities, exchanges):
        if False:
            print('Hello World!')
        if 'company_name' in equities.columns and 'asset_name' not in equities.columns:
            equities['asset_name'] = equities['company_name']
        if 'file_name' in equities.columns:
            equities['symbol'] = equities['file_name']
        equities_output = _generate_output_dataframe(data_subset=equities, defaults=_equities_defaults)
        tuple_series = equities_output['symbol'].apply(split_delimited_symbol)
        split_symbols = pd.DataFrame(tuple_series.tolist(), columns=['company_symbol', 'share_class_symbol'], index=tuple_series.index)
        equities_output = pd.concat((equities_output, split_symbols), axis=1)
        for col in symbol_columns:
            equities_output[col] = equities_output[col].str.upper()
        for col in ('start_date', 'end_date', 'first_traded', 'auto_close_date'):
            equities_output[col] = _dt_to_epoch_ns(equities_output[col])
        return _split_symbol_mappings(equities_output, exchanges)

    def _normalize_futures(self, futures):
        if False:
            return 10
        futures_output = _generate_output_dataframe(data_subset=futures, defaults=_futures_defaults)
        for col in ('symbol', 'root_symbol'):
            futures_output[col] = futures_output[col].str.upper()
        for col in ('start_date', 'end_date', 'first_traded', 'notice_date', 'expiration_date', 'auto_close_date'):
            futures_output[col] = _dt_to_epoch_ns(futures_output[col])
        return futures_output

    def _normalize_equity_supplementary_mappings(self, mappings):
        if False:
            while True:
                i = 10
        mappings_output = _generate_output_dataframe(data_subset=mappings, defaults=_equity_supplementary_mappings_defaults)
        for col in ('start_date', 'end_date'):
            mappings_output[col] = _dt_to_epoch_ns(mappings_output[col])
        return mappings_output

    def _load_data(self, equities, futures, exchanges, root_symbols, equity_supplementary_mappings):
        if False:
            print('Hello World!')
        '\n        Returns a standard set of pandas.DataFrames:\n        equities, futures, exchanges, root_symbols\n        '
        _normalize_index_columns_in_place(equities=equities, equity_supplementary_mappings=equity_supplementary_mappings, futures=futures, exchanges=exchanges, root_symbols=root_symbols)
        futures_output = self._normalize_futures(futures)
        equity_supplementary_mappings_output = self._normalize_equity_supplementary_mappings(equity_supplementary_mappings)
        exchanges_output = _generate_output_dataframe(data_subset=exchanges, defaults=_exchanges_defaults)
        (equities_output, equities_mappings) = self._normalize_equities(equities, exchanges_output)
        root_symbols_output = _generate_output_dataframe(data_subset=root_symbols, defaults=_root_symbols_defaults)
        return AssetData(equities=equities_output, equities_mappings=equities_mappings, futures=futures_output, exchanges=exchanges_output, root_symbols=root_symbols_output, equity_supplementary_mappings=equity_supplementary_mappings_output)