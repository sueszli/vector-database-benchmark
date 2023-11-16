"""
CalmarHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime
from pandas import DataFrame
from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_calmar
from freqtrade.optimize.hyperopt import IHyperOptLoss

class CalmarHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Calmar Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int, min_date: datetime, max_date: datetime, config: Config, *args, **kwargs) -> float:
        if False:
            print('Hello World!')
        '\n        Objective function, returns smaller number for more optimal results.\n\n        Uses Calmar Ratio calculation.\n        '
        starting_balance = config['dry_run_wallet']
        calmar_ratio = calculate_calmar(results, min_date, max_date, starting_balance)
        return -calmar_ratio