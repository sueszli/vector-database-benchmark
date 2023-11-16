import logging
from typing import Any, Dict
from freqtrade import constants
from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.misc import round_coin_value
logger = logging.getLogger(__name__)

def setup_optimize_configuration(args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    '\n    Prepare the configuration for the Hyperopt module\n    :param args: Cli args from Arguments()\n    :param method: Bot running mode\n    :return: Configuration\n    '
    config = setup_utils_configuration(args, method)
    no_unlimited_runmodes = {RunMode.BACKTEST: 'backtesting', RunMode.HYPEROPT: 'hyperoptimization'}
    if method in no_unlimited_runmodes.keys():
        wallet_size = config['dry_run_wallet'] * config['tradable_balance_ratio']
        if config['stake_amount'] != constants.UNLIMITED_STAKE_AMOUNT and config['stake_amount'] > wallet_size:
            wallet = round_coin_value(wallet_size, config['stake_currency'])
            stake = round_coin_value(config['stake_amount'], config['stake_currency'])
            raise OperationalException(f'Starting balance ({wallet}) is smaller than stake_amount {stake}. Wallet is calculated as `dry_run_wallet * tradable_balance_ratio`.')
    return config

def start_backtesting(args: Dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    '\n    Start Backtesting script\n    :param args: Cli args from Arguments()\n    :return: None\n    '
    from freqtrade.optimize.backtesting import Backtesting
    config = setup_optimize_configuration(args, RunMode.BACKTEST)
    logger.info('Starting freqtrade in Backtesting mode')
    backtesting = Backtesting(config)
    backtesting.start()

def start_backtesting_show(args: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Show previous backtest result\n    '
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    from freqtrade.data.btanalysis import load_backtest_stats
    from freqtrade.optimize.optimize_reports import show_backtest_results, show_sorted_pairlist
    results = load_backtest_stats(config['exportfilename'])
    show_backtest_results(config, results)
    show_sorted_pairlist(config, results)

def start_hyperopt(args: Dict[str, Any]) -> None:
    if False:
        return 10
    '\n    Start hyperopt script\n    :param args: Cli args from Arguments()\n    :return: None\n    '
    try:
        from filelock import FileLock, Timeout
        from freqtrade.optimize.hyperopt import Hyperopt
    except ImportError as e:
        raise OperationalException(f'{e}. Please ensure that the hyperopt dependencies are installed.') from e
    config = setup_optimize_configuration(args, RunMode.HYPEROPT)
    logger.info('Starting freqtrade in Hyperopt mode')
    lock = FileLock(Hyperopt.get_lock_filename(config))
    try:
        with lock.acquire(timeout=1):
            logging.getLogger('hyperopt.tpe').setLevel(logging.WARNING)
            logging.getLogger('filelock').setLevel(logging.WARNING)
            hyperopt = Hyperopt(config)
            hyperopt.start()
    except Timeout:
        logger.info('Another running instance of freqtrade Hyperopt detected.')
        logger.info('Simultaneous execution of multiple Hyperopt commands is not supported. Hyperopt module is resource hungry. Please run your Hyperopt sequentially or on separate machines.')
        logger.info('Quitting now.')

def start_edge(args: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Start Edge script\n    :param args: Cli args from Arguments()\n    :return: None\n    '
    from freqtrade.optimize.edge_cli import EdgeCli
    config = setup_optimize_configuration(args, RunMode.EDGE)
    logger.info('Starting freqtrade in Edge mode')
    edge_cli = EdgeCli(config)
    edge_cli.start()

def start_lookahead_analysis(args: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Start the backtest bias tester script\n    :param args: Cli args from Arguments()\n    :return: None\n    '
    from freqtrade.optimize.analysis.lookahead_helpers import LookaheadAnalysisSubFunctions
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    LookaheadAnalysisSubFunctions.start(config)

def start_recursive_analysis(args: Dict[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Start the backtest recursive tester script\n    :param args: Cli args from Arguments()\n    :return: None\n    '
    from freqtrade.optimize.analysis.recursive_helpers import RecursiveAnalysisSubFunctions
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    RecursiveAnalysisSubFunctions.start(config)