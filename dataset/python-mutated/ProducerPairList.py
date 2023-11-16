"""
External Pair List provider

Provides pair list from Leader data
"""
import logging
from typing import Any, Dict, List, Optional
from freqtrade.exceptions import OperationalException
from freqtrade.exchange.types import Tickers
from freqtrade.plugins.pairlist.IPairList import IPairList, PairlistParameter
logger = logging.getLogger(__name__)

class ProducerPairList(IPairList):
    """
    PairList plugin for use with external_message_consumer.
    Will use pairs given from leader data.

    Usage:
        "pairlists": [
            {
                "method": "ProducerPairList",
                "number_assets": 5,
                "producer_name": "default",
            }
        ],
    """
    is_pairlist_generator = True

    def __init__(self, exchange, pairlistmanager, config: Dict[str, Any], pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(exchange, pairlistmanager, config, pairlistconfig, pairlist_pos)
        self._num_assets: int = self._pairlistconfig.get('number_assets', 0)
        self._producer_name = self._pairlistconfig.get('producer_name', 'default')
        if not config.get('external_message_consumer', {}).get('enabled'):
            raise OperationalException('ProducerPairList requires external_message_consumer to be enabled.')

    @property
    def needstickers(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Boolean property defining if tickers are necessary.\n        If no Pairlist requires tickers, an empty Dict is passed\n        as tickers argument to filter_pairlist\n        '
        return False

    def short_desc(self) -> str:
        if False:
            print('Hello World!')
        '\n        Short whitelist method description - used for startup-messages\n        -> Please overwrite in subclasses\n        '
        return f'{self.name} - {self._producer_name}'

    @staticmethod
    def description() -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'Get a pairlist from an upstream bot.'

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        if False:
            for i in range(10):
                print('nop')
        return {'number_assets': {'type': 'number', 'default': 0, 'description': 'Number of assets', 'help': 'Number of assets to use from the pairlist'}, 'producer_name': {'type': 'string', 'default': 'default', 'description': 'Producer name', 'help': 'Name of the producer to use. Requires additional external_message_consumer configuration.'}}

    def _filter_pairlist(self, pairlist: Optional[List[str]]):
        if False:
            return 10
        upstream_pairlist = self._pairlistmanager._dataprovider.get_producer_pairs(self._producer_name)
        if pairlist is None:
            pairlist = self._pairlistmanager._dataprovider.get_producer_pairs(self._producer_name)
        pairs = list(dict.fromkeys(pairlist + upstream_pairlist))
        if self._num_assets:
            pairs = pairs[:self._num_assets]
        return pairs

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        if False:
            return 10
        '\n        Generate the pairlist\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: List of pairs\n        '
        pairs = self._filter_pairlist(None)
        self.log_once(f'Received pairs: {pairs}', logger.debug)
        pairs = self._whitelist_for_active_markets(self.verify_whitelist(pairs, logger.info))
        return pairs

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Filters and sorts pairlist and returns the whitelist again.\n        Called on each bot iteration - please use internal caching if necessary\n        :param pairlist: pairlist to filter or sort\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: new whitelist\n        '
        return self._filter_pairlist(pairlist)