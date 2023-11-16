import logging
from typing import Optional
import numpy as np
import pandas as pd
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS
from freqtrade.enums import CandleType
from .idatahandler import IDataHandler
logger = logging.getLogger(__name__)

class HDF5DataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(self, pair: str, timeframe: str, data: pd.DataFrame, candle_type: CandleType) -> None:
        if False:
            print('Hello World!')
        '\n        Store data in hdf5 file.\n        :param pair: Pair - used to generate filename\n        :param timeframe: Timeframe - used to generate filename\n        :param data: Dataframe containing OHLCV data\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: None\n        '
        key = self._pair_ohlcv_key(pair, timeframe)
        _data = data.copy()
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)
        _data.loc[:, self._columns].to_hdf(filename, key, mode='a', complevel=9, complib='blosc', format='table', data_columns=['date'])

    def _ohlcv_load(self, pair: str, timeframe: str, timerange: Optional[TimeRange], candle_type: CandleType) -> pd.DataFrame:
        if False:
            return 10
        '\n        Internal method used to load data for one pair from disk.\n        Implements the loading and conversion to a Pandas dataframe.\n        Timerange trimming and dataframe validation happens outside of this method.\n        :param pair: Pair to load data\n        :param timeframe: Timeframe (e.g. "5m")\n        :param timerange: Limit data to be loaded to this timerange.\n                        Optionally implemented by subclasses to avoid loading\n                        all data where possible.\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: DataFrame with ohlcv data, or empty DataFrame\n        '
        key = self._pair_ohlcv_key(pair, timeframe)
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        if not filename.exists():
            filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True)
            if not filename.exists():
                return pd.DataFrame(columns=self._columns)
        where = []
        if timerange:
            if timerange.starttype == 'date':
                where.append(f'date >= Timestamp({timerange.startts * 1000000000.0})')
            if timerange.stoptype == 'date':
                where.append(f'date <= Timestamp({timerange.stopts * 1000000000.0})')
        pairdata = pd.read_hdf(filename, key=key, mode='r', where=where)
        if list(pairdata.columns) != self._columns:
            raise ValueError('Wrong dataframe format')
        pairdata = pairdata.astype(dtype={'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
        pairdata = pairdata.reset_index(drop=True)
        return pairdata

    def ohlcv_append(self, pair: str, timeframe: str, data: pd.DataFrame, candle_type: CandleType) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Append data to existing data structures\n        :param pair: Pair\n        :param timeframe: Timeframe this ohlcv data is for\n        :param data: Data to append.\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        '
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: pd.DataFrame) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Store trades data (list of Dicts) to file\n        :param pair: Pair - used for filename\n        :param data: Dataframe containing trades\n                     column sequence as in DEFAULT_TRADES_COLUMNS\n        '
        key = self._pair_trades_key(pair)
        data.to_hdf(self._pair_trades_filename(self._datadir, pair), key, mode='a', complevel=9, complib='blosc', format='table', data_columns=['timestamp'])

    def trades_append(self, pair: str, data: pd.DataFrame):
        if False:
            return 10
        '\n        Append data to existing files\n        :param pair: Pair - used for filename\n        :param data: Dataframe containing trades\n                     column sequence as in DEFAULT_TRADES_COLUMNS\n        '
        raise NotImplementedError()

    def _trades_load(self, pair: str, timerange: Optional[TimeRange]=None) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Load a pair from h5 file.\n        :param pair: Load trades for this pair\n        :param timerange: Timerange to load trades for - currently not implemented\n        :return: Dataframe containing trades\n        '
        key = self._pair_trades_key(pair)
        filename = self._pair_trades_filename(self._datadir, pair)
        if not filename.exists():
            return pd.DataFrame(columns=DEFAULT_TRADES_COLUMNS)
        where = []
        if timerange:
            if timerange.starttype == 'date':
                where.append(f'timestamp >= {timerange.startts * 1000.0}')
            if timerange.stoptype == 'date':
                where.append(f'timestamp < {timerange.stopts * 1000.0}')
        trades: pd.DataFrame = pd.read_hdf(filename, key=key, mode='r', where=where)
        trades[['id', 'type']] = trades[['id', 'type']].replace({np.nan: None})
        return trades

    @classmethod
    def _get_file_extension(cls):
        if False:
            print('Hello World!')
        return 'h5'

    @classmethod
    def _pair_ohlcv_key(cls, pair: str, timeframe: str) -> str:
        if False:
            i = 10
            return i + 15
        pair_esc = pair.replace(':', '_')
        return f'{pair_esc}/ohlcv/tf_{timeframe}'

    @classmethod
    def _pair_trades_key(cls, pair: str) -> str:
        if False:
            while True:
                i = 10
        return f'{pair}/trades'