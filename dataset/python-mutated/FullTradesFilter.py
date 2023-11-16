"""
Full trade slots pair list filter
"""
import logging
from typing import Any, Dict, List
from freqtrade.constants import Config
from freqtrade.exchange.types import Tickers
from freqtrade.persistence import Trade
from freqtrade.plugins.pairlist.IPairList import IPairList
logger = logging.getLogger(__name__)

class FullTradesFilter(IPairList):

    def __init__(self, exchange, pairlistmanager, config: Config, pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        if False:
            while True:
                i = 10
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)

    @property
    def needstickers(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Boolean property defining if tickers are necessary.\n        If no Pairlist requires tickers, an empty List is passed\n        as tickers argument to filter_pairlist\n        '
        return False

    def short_desc(self) -> str:
        if False:
            return 10
        '\n        Short allowlist method description - used for startup-messages\n        '
        return f'{self.name} - Shrink whitelist when trade slots are full.'

    @staticmethod
    def description() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'Shrink whitelist when trade slots are full.'

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Filters and sorts pairlist and returns the allowlist again.\n        Called on each bot iteration - please use internal caching if necessary\n        :param pairlist: pairlist to filter or sort\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: new allowlist\n        '
        num_open = Trade.get_open_trade_count()
        max_trades = self._config['max_open_trades']
        if num_open >= max_trades and max_trades > 0:
            return []
        return pairlist