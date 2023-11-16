"""
PairList Handler base class
"""
import logging
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange, market_is_active
from freqtrade.exchange.types import Ticker, Tickers
from freqtrade.mixins import LoggingMixin
logger = logging.getLogger(__name__)

class __PairlistParameterBase(TypedDict):
    description: str
    help: str

class __NumberPairlistParameter(__PairlistParameterBase):
    type: Literal['number']
    default: Union[int, float, None]

class __StringPairlistParameter(__PairlistParameterBase):
    type: Literal['string']
    default: Union[str, None]

class __OptionPairlistParameter(__PairlistParameterBase):
    type: Literal['option']
    default: Union[str, None]
    options: List[str]

class __BoolPairlistParameter(__PairlistParameterBase):
    type: Literal['boolean']
    default: Union[bool, None]
PairlistParameter = Union[__NumberPairlistParameter, __StringPairlistParameter, __OptionPairlistParameter, __BoolPairlistParameter]

class IPairList(LoggingMixin, ABC):
    is_pairlist_generator = False

    def __init__(self, exchange: Exchange, pairlistmanager, config: Config, pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        :param exchange: Exchange instance\n        :param pairlistmanager: Instantiated Pairlist manager\n        :param config: Global bot configuration\n        :param pairlistconfig: Configuration for this Pairlist Handler - can be empty.\n        :param pairlist_pos: Position of the Pairlist Handler in the chain\n        '
        self._enabled = True
        self._exchange: Exchange = exchange
        self._pairlistmanager = pairlistmanager
        self._config = config
        self._pairlistconfig = pairlistconfig
        self._pairlist_pos = pairlist_pos
        self.refresh_period = self._pairlistconfig.get('refresh_period', 1800)
        LoggingMixin.__init__(self, logger, self.refresh_period)

    @property
    def name(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Gets name of the class\n        -> no need to overwrite in subclasses\n        '
        return self.__class__.__name__

    @abstractproperty
    def needstickers(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Boolean property defining if tickers are necessary.\n        If no Pairlist requires tickers, an empty Dict is passed\n        as tickers argument to filter_pairlist\n        '
        return False

    @staticmethod
    @abstractmethod
    def description() -> str:
        if False:
            while True:
                i = 10
        '\n        Return description of this Pairlist Handler\n        -> Please overwrite in subclasses\n        '
        return ''

    @staticmethod
    def available_parameters() -> Dict[str, PairlistParameter]:
        if False:
            return 10
        '\n        Return parameters used by this Pairlist Handler, and their type\n        contains a dictionary with the parameter name as key, and a dictionary\n        with the type and default value.\n        -> Please overwrite in subclasses\n        '
        return {}

    @staticmethod
    def refresh_period_parameter() -> Dict[str, PairlistParameter]:
        if False:
            print('Hello World!')
        return {'refresh_period': {'type': 'number', 'default': 1800, 'description': 'Refresh period', 'help': 'Refresh period in seconds'}}

    @abstractmethod
    def short_desc(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Short whitelist method description - used for startup-messages\n        -> Please overwrite in subclasses\n        '

    def _validate_pair(self, pair: str, ticker: Optional[Ticker]) -> bool:
        if False:
            print('Hello World!')
        "\n        Check one pair against Pairlist Handler's specific conditions.\n\n        Either implement it in the Pairlist Handler or override the generic\n        filter_pairlist() method.\n\n        :param pair: Pair that's currently validated\n        :param ticker: ticker dict as returned from ccxt.fetch_ticker\n        :return: True if the pair can stay, false if it should be removed\n        "
        raise NotImplementedError()

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Generate the pairlist.\n\n        This method is called once by the pairlistmanager in the refresh_pairlist()\n        method to supply the starting pairlist for the chain of the Pairlist Handlers.\n        Pairlist Filters (those Pairlist Handlers that cannot be used at the first\n        position in the chain) shall not override this base implementation --\n        it will raise the exception if a Pairlist Handler is used at the first\n        position in the chain.\n\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: List of pairs\n        '
        raise OperationalException('This Pairlist Handler should not be used at the first position in the list of Pairlist Handlers.')

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if False:
            print('Hello World!')
        '\n        Filters and sorts pairlist and returns the whitelist again.\n\n        Called on each bot iteration - please use internal caching if necessary\n        This generic implementation calls self._validate_pair() for each pair\n        in the pairlist.\n\n        Some Pairlist Handlers override this generic implementation and employ\n        own filtration.\n\n        :param pairlist: pairlist to filter or sort\n        :param tickers: Tickers (from exchange.get_tickers). May be cached.\n        :return: new whitelist\n        '
        if self._enabled:
            for p in deepcopy(pairlist):
                if not self._validate_pair(p, tickers[p] if p in tickers else None):
                    pairlist.remove(p)
        return pairlist

    def verify_blacklist(self, pairlist: List[str], logmethod) -> List[str]:
        if False:
            while True:
                i = 10
        "\n        Proxy method to verify_blacklist for easy access for child classes.\n        :param pairlist: Pairlist to validate\n        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`.\n        :return: pairlist - blacklisted pairs\n        "
        return self._pairlistmanager.verify_blacklist(pairlist, logmethod)

    def verify_whitelist(self, pairlist: List[str], logmethod, keep_invalid: bool=False) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Proxy method to verify_whitelist for easy access for child classes.\n        :param pairlist: Pairlist to validate\n        :param logmethod: Function that'll be called, `logger.info` or `logger.warning`\n        :param keep_invalid: If sets to True, drops invalid pairs silently while expanding regexes.\n        :return: pairlist - whitelisted pairs\n        "
        return self._pairlistmanager.verify_whitelist(pairlist, logmethod, keep_invalid)

    def _whitelist_for_active_markets(self, pairlist: List[str]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check available markets and remove pair from whitelist if necessary\n        :param pairlist: the sorted list of pairs the user might want to trade\n        :return: the list of pairs the user wants to trade without those unavailable or\n        black_listed\n        '
        markets = self._exchange.markets
        if not markets:
            raise OperationalException('Markets not loaded. Make sure that exchange is initialized correctly.')
        sanitized_whitelist: List[str] = []
        for pair in pairlist:
            if pair not in markets:
                self.log_once(f'Pair {pair} is not compatible with exchange {self._exchange.name}. Removing it from whitelist..', logger.warning)
                continue
            if not self._exchange.market_is_tradable(markets[pair]):
                self.log_once(f'Pair {pair} is not tradable with Freqtrade.Removing it from whitelist..', logger.warning)
                continue
            if self._exchange.get_pair_quote_currency(pair) != self._config['stake_currency']:
                self.log_once(f"Pair {pair} is not compatible with your stake currency {self._config['stake_currency']}. Removing it from whitelist..", logger.warning)
                continue
            market = markets[pair]
            if not market_is_active(market):
                self.log_once(f'Ignoring {pair} from whitelist. Market is not active.', logger.info)
                continue
            if pair not in sanitized_whitelist:
                sanitized_whitelist.append(pair)
        return sanitized_whitelist