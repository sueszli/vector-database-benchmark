"""
Performance pair list filter
"""
import logging
from typing import Any, Dict, List
import pandas as pd
from freqtrade.constants import Config
from freqtrade.exchange.types import Tickers
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter
logger = logging.getLogger(__name__)

class PerformanceFilter(IPairList):

    def __init__(self, exchange, pairlistmanager, config: Config, pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)
        self._minutes = pairlistconfig.get('minutes', 0)
        self._min_profit = pairlistconfig.get('min_profit')

    @property
    def needstickers(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Boolean property defining if tickers are necessary.\n        If no Pairlist requires tickers, an empty List is passed\n        as tickers argument to filter_pairlist\n        '
        return False

    def short_desc(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Short allowlist method description - used for startup-messages\n        '
        return f'{self.name} - Sorting pairs by performance.'

    @staticmethod
    def description() -> str:
        if False:
            while True:
                i = 10
        return 'Filter pairs by performance.'

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        if False:
            for i in range(10):
                print('nop')
        return {'minutes': {'type': 'number', 'default': 0, 'description': 'Minutes', 'help': 'Consider trades from the last X minutes. 0 means all trades.'}, 'min_profit': {'type': 'number', 'default': None, 'description': 'Minimum profit', 'help': 'Minimum profit in percent. Pairs with less profit are removed.'}}

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Filters and sorts pairlist and returns the allowlist again.\n        Called on each bot iteration - please use internal caching if necessary\n        :param pairlist: pairlist to filter or sort\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: new allowlist\n        '
        try:
            performance = pd.DataFrame(Trade.get_overall_performance(self._minutes))
        except AttributeError:
            self.log_once('PerformanceFilter is not available in this mode.', logger.warning)
            return pairlist
        if len(performance) == 0:
            return pairlist
        list_df = pd.DataFrame({'pair': pairlist})
        list_df['prior_idx'] = list_df.index
        sorted_df = list_df.merge(performance, on='pair', how='left').fillna(0).sort_values(by=['profit_ratio', 'count', 'prior_idx'], ascending=[False, True, True])
        if self._min_profit is not None:
            removed = sorted_df[sorted_df['profit_ratio'] < self._min_profit]
            for (_, row) in removed.iterrows():
                self.log_once(f"Removing pair {row['pair']} since {row['profit_ratio']} is below {self._min_profit}", logger.info)
            sorted_df = sorted_df[sorted_df['profit_ratio'] >= self._min_profit]
        pairlist = sorted_df['pair'].tolist()
        return pairlist