"""
Static Pair List provider

Provides pair white list as it configured in config
"""
import logging
from copy import deepcopy
from typing import Any, Dict, List
from freqtrade.constants import Config
from freqtrade.exchange.types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter
logger = logging.getLogger(__name__)

class StaticPairList(IPairList):
    is_pairlist_generator = True

    def __init__(self, exchange, pairlistmanager, config: Config, pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)
        self._allow_inactive = self._pairlistconfig.get('allow_inactive', False)

    @property
    def needstickers(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Boolean property defining if tickers are necessary.\n        If no Pairlist requires tickers, an empty Dict is passed\n        as tickers argument to filter_pairlist\n        '
        return False

    def short_desc(self) -> str:
        if False:
            return 10
        '\n        Short whitelist method description - used for startup-messages\n        -> Please overwrite in subclasses\n        '
        return f'{self.name}'

    @staticmethod
    def description() -> str:
        if False:
            print('Hello World!')
        return 'Use pairlist as configured in config.'

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        if False:
            for i in range(10):
                print('nop')
        return {'allow_inactive': {'type': 'boolean', 'default': False, 'description': 'Allow inactive pairs', 'help': 'Allow inactive pairs to be in the whitelist.'}}

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate the pairlist\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: List of pairs\n        '
        if self._allow_inactive:
            return self.verify_whitelist(self._config['exchange']['pair_whitelist'], logger.info, keep_invalid=True)
        else:
            return self._whitelist_for_active_markets(self.verify_whitelist(self._config['exchange']['pair_whitelist'], logger.info))

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Filters and sorts pairlist and returns the whitelist again.\n        Called on each bot iteration - please use internal caching if necessary\n        :param pairlist: pairlist to filter or sort\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: new whitelist\n        '
        pairlist_ = deepcopy(pairlist)
        for pair in self._config['exchange']['pair_whitelist']:
            if pair not in pairlist_:
                pairlist_.append(pair)
        return pairlist_