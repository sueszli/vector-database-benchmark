"""
Abstract datahandler interface.
It's subclasses handle and storing data from disk.

"""
import logging
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Type
from pandas import DataFrame
from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_TRADES_COLUMNS, ListPairsWithTimeframes
from freqtrade.data.converter import clean_ohlcv_dataframe, trades_convert_types, trades_df_remove_duplicates, trim_dataframe
from freqtrade.enums import CandleType, TradingMode
from freqtrade.exchange import timeframe_to_seconds
logger = logging.getLogger(__name__)

class IDataHandler(ABC):
    _OHLCV_REGEX = '^([a-zA-Z_\\d-]+)\\-(\\d+[a-zA-Z]{1,2})\\-?([a-zA-Z_]*)?(?=\\.)'

    def __init__(self, datadir: Path) -> None:
        if False:
            print('Hello World!')
        self._datadir = datadir

    @classmethod
    def _get_file_extension(cls) -> str:
        if False:
            while True:
                i = 10
        '\n        Get file extension for this particular datahandler\n        '
        raise NotImplementedError()

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path, trading_mode: TradingMode) -> ListPairsWithTimeframes:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of all pairs with ohlcv data available in this datadir\n        :param datadir: Directory to search for ohlcv files\n        :param trading_mode: trading-mode to be used\n        :return: List of Tuples of (pair, timeframe, CandleType)\n        '
        if trading_mode == TradingMode.FUTURES:
            datadir = datadir.joinpath('futures')
        _tmp = [re.search(cls._OHLCV_REGEX, p.name) for p in datadir.glob(f'*.{cls._get_file_extension()}')]
        return [(cls.rebuild_pair_from_filename(match[1]), cls.rebuild_timeframe_from_filename(match[2]), CandleType.from_string(match[3])) for match in _tmp if match and len(match.groups()) > 1]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str, candle_type: CandleType) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Returns a list of all pairs with ohlcv data available in this datadir\n        for the specified timeframe\n        :param datadir: Directory to search for ohlcv files\n        :param timeframe: Timeframe to search pairs for\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: List of Pairs\n        '
        candle = ''
        if candle_type != CandleType.SPOT:
            datadir = datadir.joinpath('futures')
            candle = f'-{candle_type}'
        ext = cls._get_file_extension()
        _tmp = [re.search('^(\\S+)(?=\\-' + timeframe + candle + f'.{ext})', p.name) for p in datadir.glob(f'*{timeframe}{candle}.{ext}')]
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    @abstractmethod
    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        if False:
            print('Hello World!')
        '\n        Store ohlcv data.\n        :param pair: Pair - used to generate filename\n        :param timeframe: Timeframe - used to generate filename\n        :param data: Dataframe containing OHLCV data\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: None\n        '

    def ohlcv_data_min_max(self, pair: str, timeframe: str, candle_type: CandleType) -> Tuple[datetime, datetime]:
        if False:
            while True:
                i = 10
        '\n        Returns the min and max timestamp for the given pair and timeframe.\n        :param pair: Pair to get min/max for\n        :param timeframe: Timeframe to get min/max for\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: (min, max)\n        '
        data = self._ohlcv_load(pair, timeframe, None, candle_type)
        if data.empty:
            return (datetime.fromtimestamp(0, tz=timezone.utc), datetime.fromtimestamp(0, tz=timezone.utc))
        return (data.iloc[0]['date'].to_pydatetime(), data.iloc[-1]['date'].to_pydatetime())

    @abstractmethod
    def _ohlcv_load(self, pair: str, timeframe: str, timerange: Optional[TimeRange], candle_type: CandleType) -> DataFrame:
        if False:
            print('Hello World!')
        '\n        Internal method used to load data for one pair from disk.\n        Implements the loading and conversion to a Pandas dataframe.\n        Timerange trimming and dataframe validation happens outside of this method.\n        :param pair: Pair to load data\n        :param timeframe: Timeframe (e.g. "5m")\n        :param timerange: Limit data to be loaded to this timerange.\n                        Optionally implemented by subclasses to avoid loading\n                        all data where possible.\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: DataFrame with ohlcv data, or empty DataFrame\n        '

    def ohlcv_purge(self, pair: str, timeframe: str, candle_type: CandleType) -> bool:
        if False:
            print('Hello World!')
        '\n        Remove data for this pair\n        :param pair: Delete data for this pair.\n        :param timeframe: Timeframe (e.g. "5m")\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: True when deleted, false if file did not exist.\n        '
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        if filename.exists():
            filename.unlink()
            return True
        return False

    @abstractmethod
    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        if False:
            return 10
        '\n        Append data to existing data structures\n        :param pair: Pair\n        :param timeframe: Timeframe this ohlcv data is for\n        :param data: Data to append.\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        '

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Returns a list of all pairs for which trade data is available in this\n        :param datadir: Directory to search for ohlcv files\n        :return: List of Pairs\n        '
        _ext = cls._get_file_extension()
        _tmp = [re.search('^(\\S+)(?=\\-trades.' + _ext + ')', p.name) for p in datadir.glob(f'*trades.{_ext}')]
        return [cls.rebuild_pair_from_filename(match[0]) for match in _tmp if match]

    @abstractmethod
    def _trades_store(self, pair: str, data: DataFrame) -> None:
        if False:
            while True:
                i = 10
        '\n        Store trades data (list of Dicts) to file\n        :param pair: Pair - used for filename\n        :param data: Dataframe containing trades\n                     column sequence as in DEFAULT_TRADES_COLUMNS\n        '

    @abstractmethod
    def trades_append(self, pair: str, data: DataFrame):
        if False:
            while True:
                i = 10
        '\n        Append data to existing files\n        :param pair: Pair - used for filename\n        :param data: Dataframe containing trades\n                     column sequence as in DEFAULT_TRADES_COLUMNS\n        '

    @abstractmethod
    def _trades_load(self, pair: str, timerange: Optional[TimeRange]=None) -> DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Load a pair from file, either .json.gz or .json\n        :param pair: Load trades for this pair\n        :param timerange: Timerange to load trades for - currently not implemented\n        :return: Dataframe containing trades\n        '

    def trades_store(self, pair: str, data: DataFrame) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Store trades data (list of Dicts) to file\n        :param pair: Pair - used for filename\n        :param data: Dataframe containing trades\n                     column sequence as in DEFAULT_TRADES_COLUMNS\n        '
        self._trades_store(pair, data[DEFAULT_TRADES_COLUMNS])

    def trades_purge(self, pair: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Remove data for this pair\n        :param pair: Delete data for this pair.\n        :return: True when deleted, false if file did not exist.\n        '
        filename = self._pair_trades_filename(self._datadir, pair)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def trades_load(self, pair: str, timerange: Optional[TimeRange]=None) -> DataFrame:
        if False:
            return 10
        '\n        Load a pair from file, either .json.gz or .json\n        Removes duplicates in the process.\n        :param pair: Load trades for this pair\n        :param timerange: Timerange to load trades for - currently not implemented\n        :return: List of trades\n        '
        trades = trades_df_remove_duplicates(self._trades_load(pair, timerange=timerange))
        trades = trades_convert_types(trades)
        return trades

    @classmethod
    def create_dir_if_needed(cls, datadir: Path):
        if False:
            i = 10
            return i + 15
        '\n        Creates datadir if necessary\n        should only create directories for "futures" mode at the moment.\n        '
        if not datadir.parent.is_dir():
            datadir.parent.mkdir()

    @classmethod
    def _pair_data_filename(cls, datadir: Path, pair: str, timeframe: str, candle_type: CandleType, no_timeframe_modify: bool=False) -> Path:
        if False:
            print('Hello World!')
        pair_s = misc.pair_to_filename(pair)
        candle = ''
        if not no_timeframe_modify:
            timeframe = cls.timeframe_to_file(timeframe)
        if candle_type != CandleType.SPOT:
            datadir = datadir.joinpath('futures')
            candle = f'-{candle_type}'
        filename = datadir.joinpath(f'{pair_s}-{timeframe}{candle}.{cls._get_file_extension()}')
        return filename

    @classmethod
    def _pair_trades_filename(cls, datadir: Path, pair: str) -> Path:
        if False:
            return 10
        pair_s = misc.pair_to_filename(pair)
        filename = datadir.joinpath(f'{pair_s}-trades.{cls._get_file_extension()}')
        return filename

    @staticmethod
    def timeframe_to_file(timeframe: str):
        if False:
            print('Hello World!')
        return timeframe.replace('M', 'Mo')

    @staticmethod
    def rebuild_timeframe_from_filename(timeframe: str) -> str:
        if False:
            while True:
                i = 10
        '\n        converts timeframe from disk to file\n        Replaces mo with M (to avoid problems on case-insensitive filesystems)\n        '
        return re.sub('1mo', '1M', timeframe, flags=re.IGNORECASE)

    @staticmethod
    def rebuild_pair_from_filename(pair: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Rebuild pair name from filename\n        Assumes a asset name of max. 7 length to also support BTC-PERP and BTC-PERP:USD names.\n        '
        res = re.sub('^(([A-Za-z\\d]{1,10})|^([A-Za-z\\-]{1,6}))(_)', '\\g<1>/', pair, 1)
        res = re.sub('_', ':', res, 1)
        return res

    def ohlcv_load(self, pair, timeframe: str, candle_type: CandleType, *, timerange: Optional[TimeRange]=None, fill_missing: bool=True, drop_incomplete: bool=False, startup_candles: int=0, warn_no_data: bool=True) -> DataFrame:
        if False:
            return 10
        '\n        Load cached candle (OHLCV) data for the given pair.\n\n        :param pair: Pair to load data for\n        :param timeframe: Timeframe (e.g. "5m")\n        :param timerange: Limit data to be loaded to this timerange\n        :param fill_missing: Fill missing values with "No action"-candles\n        :param drop_incomplete: Drop last candle assuming it may be incomplete.\n        :param startup_candles: Additional candles to load at the start of the period\n        :param warn_no_data: Log a warning message when no data is found\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: DataFrame with ohlcv data, or empty DataFrame\n        '
        timerange_startup = deepcopy(timerange)
        if startup_candles > 0 and timerange_startup:
            timerange_startup.subtract_start(timeframe_to_seconds(timeframe) * startup_candles)
        pairdf = self._ohlcv_load(pair, timeframe, timerange=timerange_startup, candle_type=candle_type)
        if self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data):
            return pairdf
        else:
            enddate = pairdf.iloc[-1]['date']
            if timerange_startup:
                self._validate_pairdata(pair, pairdf, timeframe, candle_type, timerange_startup)
                pairdf = trim_dataframe(pairdf, timerange_startup)
                if self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data, True):
                    return pairdf
            pairdf = clean_ohlcv_dataframe(pairdf, timeframe, pair=pair, fill_missing=fill_missing, drop_incomplete=drop_incomplete and enddate == pairdf.iloc[-1]['date'])
            self._check_empty_df(pairdf, pair, timeframe, candle_type, warn_no_data)
            return pairdf

    def _check_empty_df(self, pairdf: DataFrame, pair: str, timeframe: str, candle_type: CandleType, warn_no_data: bool, warn_price: bool=False) -> bool:
        if False:
            while True:
                i = 10
        '\n        Warn on empty dataframe\n        '
        if pairdf.empty:
            if warn_no_data:
                logger.warning(f'No history for {pair}, {candle_type}, {timeframe} found. Use `freqtrade download-data` to download the data')
            return True
        elif warn_price:
            candle_price_gap = 0
            if candle_type in (CandleType.SPOT, CandleType.FUTURES) and (not pairdf.empty) and ('close' in pairdf.columns) and ('open' in pairdf.columns):
                gaps = (pairdf['open'] - pairdf['close'].shift(1)) / pairdf['close'].shift(1)
                gaps = gaps.dropna()
                if len(gaps):
                    candle_price_gap = max(abs(gaps))
            if candle_price_gap > 0.1:
                logger.info(f'Price jump in {pair}, {timeframe}, {candle_type} between two candles of {candle_price_gap:.2%} detected.')
        return False

    def _validate_pairdata(self, pair, pairdata: DataFrame, timeframe: str, candle_type: CandleType, timerange: TimeRange):
        if False:
            return 10
        '\n        Validates pairdata for missing data at start end end and logs warnings.\n        :param pairdata: Dataframe to validate\n        :param timerange: Timerange specified for start and end dates\n        '
        if timerange.starttype == 'date':
            if pairdata.iloc[0]['date'] > timerange.startdt:
                logger.warning(f"{pair}, {candle_type}, {timeframe}, data starts at {pairdata.iloc[0]['date']:%Y-%m-%d %H:%M:%S}")
        if timerange.stoptype == 'date':
            if pairdata.iloc[-1]['date'] < timerange.stopdt:
                logger.warning(f"{pair}, {candle_type}, {timeframe}, data ends at {pairdata.iloc[-1]['date']:%Y-%m-%d %H:%M:%S}")

    def rename_futures_data(self, pair: str, new_pair: str, timeframe: str, candle_type: CandleType):
        if False:
            print('Hello World!')
        '\n        Temporary method to migrate data from old naming to new naming (BTC/USDT -> BTC/USDT:USDT)\n        Only used for binance to support the binance futures naming unification.\n        '
        file_old = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        file_new = self._pair_data_filename(self._datadir, new_pair, timeframe, candle_type)
        if file_new.exists():
            logger.warning(f"{file_new} exists already, can't migrate {pair}.")
            return
        file_old.rename(file_new)

def get_datahandlerclass(datatype: str) -> Type[IDataHandler]:
    if False:
        return 10
    '\n    Get datahandler class.\n    Could be done using Resolvers, but since this may be called often and resolvers\n    are rather expensive, doing this directly should improve performance.\n    :param datatype: datatype to use.\n    :return: Datahandler class\n    '
    if datatype == 'json':
        from .jsondatahandler import JsonDataHandler
        return JsonDataHandler
    elif datatype == 'jsongz':
        from .jsondatahandler import JsonGzDataHandler
        return JsonGzDataHandler
    elif datatype == 'hdf5':
        from .hdf5datahandler import HDF5DataHandler
        return HDF5DataHandler
    elif datatype == 'feather':
        from .featherdatahandler import FeatherDataHandler
        return FeatherDataHandler
    elif datatype == 'parquet':
        from .parquetdatahandler import ParquetDataHandler
        return ParquetDataHandler
    else:
        raise ValueError(f'No datahandler for datatype {datatype} available.')

def get_datahandler(datadir: Path, data_format: Optional[str]=None, data_handler: Optional[IDataHandler]=None) -> IDataHandler:
    if False:
        print('Hello World!')
    '\n    :param datadir: Folder to save data\n    :param data_format: dataformat to use\n    :param data_handler: returns this datahandler if it exists or initializes a new one\n    '
    if not data_handler:
        HandlerClass = get_datahandlerclass(data_format or 'feather')
        data_handler = HandlerClass(datadir)
    return data_handler