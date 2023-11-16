from typing import Any, Dict
from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException

def validate_plot_args(args: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    if not args.get('datadir') and (not args.get('config')):
        raise OperationalException('You need to specify either `--datadir` or `--config` for plot-profit and plot-dataframe.')

def start_plot_dataframe(args: Dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Entrypoint for dataframe plotting\n    '
    from freqtrade.plot.plotting import load_and_plot_trades
    validate_plot_args(args)
    config = setup_utils_configuration(args, RunMode.PLOT)
    load_and_plot_trades(config)

def start_plot_profit(args: Dict[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Entrypoint for plot_profit\n    '
    from freqtrade.plot.plotting import plot_profit
    validate_plot_args(args)
    config = setup_utils_configuration(args, RunMode.PLOT)
    plot_profit(config)