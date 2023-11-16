import logging
from typing import Optional
from pandas import DataFrame, read_parquet, to_datetime
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, DEFAULT_TRADES_COLUMNS, TradeList
from freqtrade.enums import CandleType
from .idatahandler import IDataHandler
logger = logging.getLogger(__name__)

class ParquetDataHandler(IDataHandler):
    _columns = DEFAULT_DATAFRAME_COLUMNS

    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Store data in json format "values".\n            format looks as follows:\n            [[<date>,<open>,<high>,<low>,<close>]]\n        :param pair: Pair - used to generate filename\n        :param timeframe: Timeframe - used to generate filename\n        :param data: Dataframe containing OHLCV data\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: None\n        '
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type)
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).loc[:, self._columns].to_parquet(filename)

    def _ohlcv_load(self, pair: str, timeframe: str, timerange: Optional[TimeRange], candle_type: CandleType) -> DataFrame:
        if False:
            print('Hello World!')
        '\n        Internal method used to load data for one pair from disk.\n        Implements the loading and conversion to a Pandas dataframe.\n        Timerange trimming and dataframe validation happens outside of this method.\n        :param pair: Pair to load data\n        :param timeframe: Timeframe (e.g. "5m")\n        :param timerange: Limit data to be loaded to this timerange.\n                        Optionally implemented by subclasses to avoid loading\n                        all data where possible.\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        :return: DataFrame with ohlcv data, or empty DataFrame\n        '
        filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type)
        if not filename.exists():
            filename = self._pair_data_filename(self._datadir, pair, timeframe, candle_type=candle_type, no_timeframe_modify=True)
            if not filename.exists():
                return DataFrame(columns=self._columns)
        pairdata = read_parquet(filename)
        pairdata.columns = self._columns
        pairdata = pairdata.astype(dtype={'open': 'float', 'high': 'float', 'low': 'float', 'close': 'float', 'volume': 'float'})
        pairdata['date'] = to_datetime(pairdata['date'], unit='ms', utc=True)
        return pairdata

    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame, candle_type: CandleType) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Append data to existing data structures\n        :param pair: Pair\n        :param timeframe: Timeframe this ohlcv data is for\n        :param data: Data to append.\n        :param candle_type: Any of the enum CandleType (must match trading mode!)\n        '
        raise NotImplementedError()

    def _trades_store(self, pair: str, data: DataFrame) -> None:
        if False:
            return 10
        '\n        Store trades data (list of Dicts) to file\n        :param pair: Pair - used for filename\n        :param data: Dataframe containing trades\n                     column sequence as in DEFAULT_TRADES_COLUMNS\n        '
        filename = self._pair_trades_filename(self._datadir, pair)
        self.create_dir_if_needed(filename)
        data.reset_index(drop=True).to_parquet(filename)

    def trades_append(self, pair: str, data: DataFrame):
        if False:
            for i in range(10):
                print('nop')
        '\n        Append data to existing files\n        :param pair: Pair - used for filename\n        :param data: Dataframe containing trades\n                     column sequence as in DEFAULT_TRADES_COLUMNS\n        '
        raise NotImplementedError()

    def _trades_load(self, pair: str, timerange: Optional[TimeRange]=None) -> TradeList:
        if False:
            return 10
        '\n        Load a pair from file, either .json.gz or .json\n        # TODO: respect timerange ...\n        :param pair: Load trades for this pair\n        :param timerange: Timerange to load trades for - currently not implemented\n        :return: List of trades\n        '
        filename = self._pair_trades_filename(self._datadir, pair)
        if not filename.exists():
            return DataFrame(columns=DEFAULT_TRADES_COLUMNS)
        tradesdata = read_parquet(filename)
        return tradesdata

    @classmethod
    def _get_file_extension(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'parquet'