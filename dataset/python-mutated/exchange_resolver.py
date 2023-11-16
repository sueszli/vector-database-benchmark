"""
This module loads custom exchanges
"""
import logging
from inspect import isclass
from typing import Any, Dict, List, Optional
import freqtrade.exchange as exchanges
from freqtrade.constants import Config, ExchangeConfig
from freqtrade.exchange import MAP_EXCHANGE_CHILDCLASS, Exchange
from freqtrade.resolvers import IResolver
logger = logging.getLogger(__name__)

class ExchangeResolver(IResolver):
    """
    This class contains all the logic to load a custom exchange class
    """
    object_type = Exchange

    @staticmethod
    def load_exchange(config: Config, *, exchange_config: Optional[ExchangeConfig]=None, validate: bool=True, load_leverage_tiers: bool=False) -> Exchange:
        if False:
            while True:
                i = 10
        '\n        Load the custom class from config parameter\n        :param exchange_name: name of the Exchange to load\n        :param config: configuration dictionary\n        '
        exchange_name: str = config['exchange']['name']
        exchange_name = MAP_EXCHANGE_CHILDCLASS.get(exchange_name, exchange_name)
        exchange_name = exchange_name.title()
        exchange = None
        try:
            exchange = ExchangeResolver._load_exchange(exchange_name, kwargs={'config': config, 'validate': validate, 'exchange_config': exchange_config, 'load_leverage_tiers': load_leverage_tiers})
        except ImportError:
            logger.info(f'No {exchange_name} specific subclass found. Using the generic class instead.')
        if not exchange:
            exchange = Exchange(config, validate=validate, exchange_config=exchange_config)
        return exchange

    @staticmethod
    def _load_exchange(exchange_name: str, kwargs: dict) -> Exchange:
        if False:
            print('Hello World!')
        '\n        Loads the specified exchange.\n        Only checks for exchanges exported in freqtrade.exchanges\n        :param exchange_name: name of the module to import\n        :return: Exchange instance or None\n        '
        try:
            ex_class = getattr(exchanges, exchange_name)
            exchange = ex_class(**kwargs)
            if exchange:
                logger.info(f"Using resolved exchange '{exchange_name}'...")
                return exchange
        except AttributeError:
            pass
        raise ImportError(f"Impossible to load Exchange '{exchange_name}'. This class does not exist or contains Python code errors.")

    @classmethod
    def search_all_objects(cls, config: Config, enum_failed: bool, recursive: bool=False) -> List[Dict[str, Any]]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Searches for valid objects\n        :param config: Config object\n        :param enum_failed: If True, will return None for modules which fail.\n            Otherwise, failing modules are skipped.\n        :param recursive: Recursively walk directory tree searching for strategies\n        :return: List of dicts containing 'name', 'class' and 'location' entries\n        "
        result = []
        for exchange_name in dir(exchanges):
            exchange = getattr(exchanges, exchange_name)
            if isclass(exchange) and issubclass(exchange, Exchange):
                result.append({'name': exchange_name, 'class': exchange, 'location': exchange.__module__, 'location_rel: ': exchange.__module__.replace('freqtrade.', '')})
        return result