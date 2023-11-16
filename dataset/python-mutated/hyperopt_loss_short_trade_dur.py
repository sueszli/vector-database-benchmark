"""
ShortTradeDurHyperOptLoss
This module defines the default HyperoptLoss class which is being used for
Hyperoptimization.
"""
from math import exp
from pandas import DataFrame
from freqtrade.optimize.hyperopt import IHyperOptLoss
TARGET_TRADES = 600
EXPECTED_MAX_PROFIT = 3.0
MAX_ACCEPTED_TRADE_DURATION = 300

class ShortTradeDurHyperOptLoss(IHyperOptLoss):
    """
    Defines the default loss function for hyperopt
    """

    @staticmethod
    def hyperopt_loss_function(results: DataFrame, trade_count: int, *args, **kwargs) -> float:
        if False:
            return 10
        '\n        Objective function, returns smaller number for better results\n        This is the Default algorithm\n        Weights are distributed as follows:\n        * 0.4 to trade duration\n        * 0.25: Avoiding trade loss\n        * 1.0 to total profit, compared to the expected value (`EXPECTED_MAX_PROFIT`) defined above\n        '
        total_profit = results['profit_ratio'].sum()
        trade_duration = results['trade_duration'].mean()
        trade_loss = 1 - 0.25 * exp(-(trade_count - TARGET_TRADES) ** 2 / 10 ** 5.8)
        profit_loss = max(0, 1 - total_profit / EXPECTED_MAX_PROFIT)
        duration_loss = 0.4 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1)
        result = trade_loss + profit_loss + duration_loss
        return result
DefaultHyperOptLoss = ShortTradeDurHyperOptLoss