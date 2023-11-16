"""
Shuffle pair list filter
"""
import logging
import random
from typing import Any, Dict, List, Literal
from freqtrade.constants import Config
from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.exchange.types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter
from freqtrade.util.periodic_cache import PeriodicCache
logger = logging.getLogger(__name__)
ShuffleValues = Literal['candle', 'iteration']

class ShuffleFilter(IPairList):

    def __init__(self, exchange, pairlistmanager, config: Config, pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        if False:
            return 10
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)
        if config.get('runmode') in (RunMode.LIVE, RunMode.DRY_RUN):
            self._seed = None
            logger.info('Live mode detected, not applying seed.')
        else:
            self._seed = pairlistconfig.get('seed')
            logger.info(f'Backtesting mode detected, applying seed value: {self._seed}')
        self._random = random.Random(self._seed)
        self._shuffle_freq: ShuffleValues = pairlistconfig.get('shuffle_frequency', 'candle')
        self.__pairlist_cache = PeriodicCache(maxsize=1000, ttl=timeframe_to_seconds(self._config['timeframe']))

    @property
    def needstickers(self) -> bool:
        if False:
            return 10
        '\n        Boolean property defining if tickers are necessary.\n        If no Pairlist requires tickers, an empty Dict is passed\n        as tickers argument to filter_pairlist\n        '
        return False

    def short_desc(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Short whitelist method description - used for startup-messages\n        '
        return f'{self.name} - Shuffling pairs every {self._shuffle_freq}' + (f', seed = {self._seed}.' if self._seed is not None else '.')

    @staticmethod
    def description() -> str:
        if False:
            return 10
        return 'Randomize pairlist order.'

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        if False:
            while True:
                i = 10
        return {'shuffle_frequency': {'type': 'option', 'default': 'candle', 'options': ['candle', 'iteration'], 'description': 'Shuffle frequency', 'help': "Shuffle frequency. Can be either 'candle' or 'iteration'."}, 'seed': {'type': 'number', 'default': None, 'description': 'Random Seed', 'help': 'Seed for random number generator. Not used in live mode.'}}

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if False:
            return 10
        '\n        Filters and sorts pairlist and returns the whitelist again.\n        Called on each bot iteration - please use internal caching if necessary\n        :param pairlist: pairlist to filter or sort\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: new whitelist\n        '
        pairlist_bef = tuple(pairlist)
        pairlist_new = self.__pairlist_cache.get(pairlist_bef)
        if pairlist_new and self._shuffle_freq == 'candle':
            return pairlist_new
        self._random.shuffle(pairlist)
        self.__pairlist_cache[pairlist_bef] = pairlist
        return pairlist