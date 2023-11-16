import logging
from pathlib import Path
from typing import Dict
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.misc import file_dump_joblib, file_dump_json
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename
from freqtrade.types import BacktestResultType
logger = logging.getLogger(__name__)

def store_backtest_stats(recordfilename: Path, stats: BacktestResultType, dtappendix: str) -> Path:
    if False:
        return 10
    '\n    Stores backtest results\n    :param recordfilename: Path object, which can either be a filename or a directory.\n        Filenames will be appended with a timestamp right before the suffix\n        while for directories, <directory>/backtest-result-<datetime>.json will be used as filename\n    :param stats: Dataframe containing the backtesting statistics\n    :param dtappendix: Datetime to use for the filename\n    '
    if recordfilename.is_dir():
        filename = recordfilename / f'backtest-result-{dtappendix}.json'
    else:
        filename = Path.joinpath(recordfilename.parent, f'{recordfilename.stem}-{dtappendix}').with_suffix(recordfilename.suffix)
    file_dump_json(get_backtest_metadata_filename(filename), stats['metadata'])
    stats_copy = {'strategy': stats['strategy'], 'strategy_comparison': stats['strategy_comparison']}
    file_dump_json(filename, stats_copy)
    latest_filename = Path.joinpath(filename.parent, LAST_BT_RESULT_FN)
    file_dump_json(latest_filename, {'latest_backtest': str(filename.name)})
    return filename

def _store_backtest_analysis_data(recordfilename: Path, data: Dict[str, Dict], dtappendix: str, name: str) -> Path:
    if False:
        i = 10
        return i + 15
    '\n    Stores backtest trade candles for analysis\n    :param recordfilename: Path object, which can either be a filename or a directory.\n        Filenames will be appended with a timestamp right before the suffix\n        while for directories, <directory>/backtest-result-<datetime>_<name>.pkl will be used\n        as filename\n    :param candles: Dict containing the backtesting data for analysis\n    :param dtappendix: Datetime to use for the filename\n    :param name: Name to use for the file, e.g. signals, rejected\n    '
    if recordfilename.is_dir():
        filename = recordfilename / f'backtest-result-{dtappendix}_{name}.pkl'
    else:
        filename = Path.joinpath(recordfilename.parent, f'{recordfilename.stem}-{dtappendix}_{name}.pkl')
    file_dump_joblib(filename, data)
    return filename

def store_backtest_analysis_results(recordfilename: Path, candles: Dict[str, Dict], trades: Dict[str, Dict], dtappendix: str) -> None:
    if False:
        while True:
            i = 10
    _store_backtest_analysis_data(recordfilename, candles, dtappendix, 'signals')
    _store_backtest_analysis_data(recordfilename, trades, dtappendix, 'rejected')