"""
SortinoHyperOptLoss

This module defines the alternative HyperOptLoss class which can be used for
Hyperoptimization.
"""
from datetime import datetime
from pandas import DataFrame
from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_sortino
from freqtrade.optimize.hyperopt import IHyperOptLoss

class SortinoHyperOptLoss(IHyperOptLoss):
    """
    Defines the loss function for hyperopt.

    This implementation uses the Sortino Ratio calculation.
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int, min_date: datetime, max_date: datetime, config: Config, *args, **kwargs) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Objective function, returns smaller number for more optimal results.\n\n        Uses Sortino Ratio calculation.\n        '
        starting_balance = config['dry_run_wallet']
        sortino_ratio = calculate_sortino(results, min_date, max_date, starting_balance)
        return -sortino_ratio