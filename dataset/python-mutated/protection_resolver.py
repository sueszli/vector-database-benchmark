"""
This module load custom pairlists
"""
import logging
from pathlib import Path
from typing import Dict
from freqtrade.constants import Config
from freqtrade.plugins.protections import IProtection
from freqtrade.resolvers import IResolver
logger = logging.getLogger(__name__)

class ProtectionResolver(IResolver):
    """
    This class contains all the logic to load custom PairList class
    """
    object_type = IProtection
    object_type_str = 'Protection'
    user_subdir = None
    initial_search_path = Path(__file__).parent.parent.joinpath('plugins/protections').resolve()

    @staticmethod
    def load_protection(protection_name: str, config: Config, protection_config: Dict) -> IProtection:
        if False:
            return 10
        '\n        Load the protection with protection_name\n        :param protection_name: Classname of the pairlist\n        :param config: configuration dictionary\n        :param protection_config: Configuration dedicated to this pairlist\n        :return: initialized Protection class\n        '
        return ProtectionResolver.load_object(protection_name, config, kwargs={'config': config, 'protection_config': protection_config})