"""
Offset pair list filter
"""
import logging
from typing import Any, Dict, List
from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter
logger = logging.getLogger(__name__)

class OffsetFilter(IPairList):

    def __init__(self, exchange, pairlistmanager, config: Config, pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        if False:
            return 10
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)
        self._offset = pairlistconfig.get('offset', 0)
        self._number_pairs = pairlistconfig.get('number_assets', 0)
        if self._offset < 0:
            raise OperationalException('OffsetFilter requires offset to be >= 0')

    @property
    def needstickers(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Boolean property defining if tickers are necessary.\n        If no Pairlist requires tickers, an empty Dict is passed\n        as tickers argument to filter_pairlist\n        '
        return False

    def short_desc(self) -> str:
        if False:
            return 10
        '\n        Short whitelist method description - used for startup-messages\n        '
        if self._number_pairs:
            return f'{self.name} - Taking {self._number_pairs} Pairs, starting from {self._offset}.'
        return f'{self.name} - Offsetting pairs by {self._offset}.'

    @staticmethod
    def description() -> str:
        if False:
            i = 10
            return i + 15
        return 'Offset pair list filter.'

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        if False:
            while True:
                i = 10
        return {'offset': {'type': 'number', 'default': 0, 'description': 'Offset', 'help': 'Offset of the pairlist.'}, 'number_assets': {'type': 'number', 'default': 0, 'description': 'Number of assets', 'help': 'Number of assets to use from the pairlist, starting from offset.'}}

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Filters and sorts pairlist and returns the whitelist again.\n        Called on each bot iteration - please use internal caching if necessary\n        :param pairlist: pairlist to filter or sort\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: new whitelist\n        '
        if self._offset > len(pairlist):
            self.log_once(f'Offset of {self._offset} is larger than ' + f'pair count of {len(pairlist)}', logger.warning)
        pairs = pairlist[self._offset:]
        if self._number_pairs:
            pairs = pairs[:self._number_pairs]
        self.log_once(f'Searching {len(pairs)} pairs: {pairs}', logger.info)
        return pairs