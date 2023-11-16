"""
This module load custom pairlists
"""
import logging
from pathlib import Path
from freqtrade.constants import Config
from freqtrade.plugins.pairlist.IPairList import IPairList
from freqtrade.resolvers import IResolver
logger = logging.getLogger(__name__)

class PairListResolver(IResolver):
    """
    This class contains all the logic to load custom PairList class
    """
    object_type = IPairList
    object_type_str = 'Pairlist'
    user_subdir = None
    initial_search_path = Path(__file__).parent.parent.joinpath('plugins/pairlist').resolve()

    @staticmethod
    def load_pairlist(pairlist_name: str, exchange, pairlistmanager, config: Config, pairlistconfig: dict, pairlist_pos: int) -> IPairList:
        if False:
            print('Hello World!')
        '\n        Load the pairlist with pairlist_name\n        :param pairlist_name: Classname of the pairlist\n        :param exchange: Initialized exchange class\n        :param pairlistmanager: Initialized pairlist manager\n        :param config: configuration dictionary\n        :param pairlistconfig: Configuration dedicated to this pairlist\n        :param pairlist_pos: Position of the pairlist in the list of pairlists\n        :return: initialized Pairlist class\n        '
        return PairListResolver.load_object(pairlist_name, config, kwargs={'exchange': exchange, 'pairlistmanager': pairlistmanager, 'config': config, 'pairlistconfig': pairlistconfig, 'pairlist_pos': pairlist_pos})