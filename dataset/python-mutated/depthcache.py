import logging
from operator import itemgetter
import asyncio
import time
from typing import Optional, Dict, Callable
from .helpers import get_loop
from .streams import BinanceSocketManager
from .threaded_stream import ThreadedApiManager

class DepthCache(object):

    def __init__(self, symbol, conv_type: Callable=float):
        if False:
            while True:
                i = 10
        'Initialise the DepthCache\n\n        :param symbol: Symbol to create depth cache for\n        :type symbol: string\n        :param conv_type: Optional type to represent price, and amount, default is float.\n        :type conv_type: function.\n\n        '
        self.symbol = symbol
        self._bids = {}
        self._asks = {}
        self.update_time = None
        self.conv_type: Callable = conv_type
        self._log = logging.getLogger(__name__)

    def add_bid(self, bid):
        if False:
            while True:
                i = 10
        'Add a bid to the cache\n\n        :param bid:\n        :return:\n\n        '
        self._bids[bid[0]] = self.conv_type(bid[1])
        if bid[1] == '0.00000000':
            del self._bids[bid[0]]

    def add_ask(self, ask):
        if False:
            print('Hello World!')
        'Add an ask to the cache\n\n        :param ask:\n        :return:\n\n        '
        self._asks[ask[0]] = self.conv_type(ask[1])
        if ask[1] == '0.00000000':
            del self._asks[ask[0]]

    def get_bids(self):
        if False:
            return 10
        'Get the current bids\n\n        :return: list of bids with price and quantity as conv_type\n\n        .. code-block:: python\n\n            [\n                [\n                    0.0001946,  # Price\n                    45.0        # Quantity\n                ],\n                [\n                    0.00019459,\n                    2384.0\n                ],\n                [\n                    0.00019158,\n                    5219.0\n                ],\n                [\n                    0.00019157,\n                    1180.0\n                ],\n                [\n                    0.00019082,\n                    287.0\n                ]\n            ]\n\n        '
        return DepthCache.sort_depth(self._bids, reverse=True, conv_type=self.conv_type)

    def get_asks(self):
        if False:
            return 10
        "Get the current asks\n\n        :return: list of asks with price and quantity as conv_type.\n\n        .. code-block:: python\n\n            [\n                [\n                    0.0001955,  # Price\n                    57.0'       # Quantity\n                ],\n                [\n                    0.00019699,\n                    778.0\n                ],\n                [\n                    0.000197,\n                    64.0\n                ],\n                [\n                    0.00019709,\n                    1130.0\n                ],\n                [\n                    0.0001971,\n                    385.0\n                ]\n            ]\n\n        "
        return DepthCache.sort_depth(self._asks, reverse=False, conv_type=self.conv_type)

    @staticmethod
    def sort_depth(vals, reverse=False, conv_type: Callable=float):
        if False:
            for i in range(10):
                print('nop')
        'Sort bids or asks by price\n        '
        if isinstance(vals, dict):
            lst = [[conv_type(price), conv_type(quantity)] for (price, quantity) in vals.items()]
        elif isinstance(vals, list):
            lst = [[conv_type(price), conv_type(quantity)] for (price, quantity) in vals]
        else:
            raise ValueError(f'Unknown order book depth data type: {type(vals)}')
        lst = sorted(lst, key=itemgetter(0), reverse=reverse)
        return lst

class BaseDepthCacheManager:
    DEFAULT_REFRESH = 60 * 30
    TIMEOUT = 60

    def __init__(self, client, symbol, loop=None, refresh_interval=None, bm=None, limit=10, conv_type=float):
        if False:
            for i in range(10):
                print('nop')
        'Create a DepthCacheManager instance\n\n        :param client: Binance API client\n        :type client: binance.Client\n        :param loop:\n        :type loop:\n        :param symbol: Symbol to create depth cache for\n        :type symbol: string\n        :param refresh_interval: Optional number of seconds between cache refresh, use 0 or None to disable\n        :type refresh_interval: int\n        :param bm: Optional BinanceSocketManager\n        :type bm: BinanceSocketManager\n        :param limit: Optional number of orders to get from orderbook\n        :type limit: int\n        :param conv_type: Optional type to represent price, and amount, default is float.\n        :type conv_type: function.\n\n        '
        self._client = client
        self._depth_cache = None
        self._loop = loop or get_loop()
        self._symbol = symbol
        self._limit = limit
        self._last_update_id = None
        self._bm = bm or BinanceSocketManager(self._client)
        self._refresh_interval = refresh_interval or self.DEFAULT_REFRESH
        self._conn_key = None
        self._conv_type = conv_type
        self._log = logging.getLogger(__name__)

    async def __aenter__(self):
        await asyncio.gather(self._init_cache(), self._start_socket())
        await self._socket.__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self._socket.__aexit__(*args, **kwargs)

    async def recv(self):
        dc = None
        while not dc:
            try:
                res = await asyncio.wait_for(self._socket.recv(), timeout=self.TIMEOUT)
            except Exception as e:
                self._log.warning(e)
            else:
                dc = await self._depth_event(res)
        return dc

    async def _init_cache(self):
        """Initialise the depth cache calling REST endpoint

        :return:
        """
        self._depth_cache = DepthCache(self._symbol, conv_type=self._conv_type)
        if self._refresh_interval:
            self._refresh_time = int(time.time()) + self._refresh_interval

    async def _start_socket(self):
        """Start the depth cache socket

        :return:
        """
        self._socket = self._get_socket()

    def _get_socket(self):
        if False:
            return 10
        raise NotImplementedError

    async def _depth_event(self, msg):
        """Handle a depth event

        :param msg:
        :return:

        """
        if not msg:
            return None
        if 'e' in msg and msg['e'] == 'error':
            await self.close()
            return None
        return await self._process_depth_message(msg)

    async def _process_depth_message(self, msg):
        """Process a depth event message.

        :param msg: Depth event message.
        :return:

        """
        self._apply_orders(msg)
        res = self._depth_cache
        if self._refresh_interval and int(time.time()) > self._refresh_time:
            await self._init_cache()
        return res

    def _apply_orders(self, msg):
        if False:
            print('Hello World!')
        assert self._depth_cache
        for bid in msg.get('b', []) + msg.get('bids', []):
            self._depth_cache.add_bid(bid)
        for ask in msg.get('a', []) + msg.get('asks', []):
            self._depth_cache.add_ask(ask)
        self._depth_cache.update_time = msg.get('E') or msg.get('lastUpdateId')

    def get_depth_cache(self):
        if False:
            while True:
                i = 10
        'Get the current depth cache\n\n        :return: DepthCache object\n\n        '
        return self._depth_cache

    async def close(self):
        """Close the open socket for this manager

        :return:
        """
        self._depth_cache = None

    def get_symbol(self):
        if False:
            i = 10
            return i + 15
        'Get the symbol\n\n        :return: symbol\n        '
        return self._symbol

class DepthCacheManager(BaseDepthCacheManager):

    def __init__(self, client, symbol, loop=None, refresh_interval=None, bm=None, limit=500, conv_type=float, ws_interval=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialise the DepthCacheManager\n\n        :param client: Binance API client\n        :type client: binance.Client\n        :param loop: asyncio loop\n        :param symbol: Symbol to create depth cache for\n        :type symbol: string\n        :param refresh_interval: Optional number of seconds between cache refresh, use 0 or None to disable\n        :type refresh_interval: int\n        :param limit: Optional number of orders to get from orderbook\n        :type limit: int\n        :param conv_type: Optional type to represent price, and amount, default is float.\n        :type conv_type: function.\n        :param ws_interval: Optional interval for updates on websocket, default None. If not set, updates happen every second. Must be 0, None (1s) or 100 (100ms).\n        :type ws_interval: int\n\n        '
        super().__init__(client, symbol, loop, refresh_interval, bm, limit, conv_type)
        self._ws_interval = ws_interval

    async def _init_cache(self):
        """Initialise the depth cache calling REST endpoint

        :return:
        """
        self._last_update_id = None
        self._depth_message_buffer = []
        res = await self._client.get_order_book(symbol=self._symbol, limit=self._limit)
        await super()._init_cache()
        self._apply_orders(res)
        assert self._depth_cache
        for bid in res['bids']:
            self._depth_cache.add_bid(bid)
        for ask in res['asks']:
            self._depth_cache.add_ask(ask)
        self._last_update_id = res['lastUpdateId']
        for msg in self._depth_message_buffer:
            await self._process_depth_message(msg)
        self._depth_message_buffer = []

    async def _start_socket(self):
        """Start the depth cache socket

        :return:
        """
        if not getattr(self, '_depth_message_buffer', None):
            self._depth_message_buffer = []
        await super()._start_socket()

    def _get_socket(self):
        if False:
            for i in range(10):
                print('nop')
        return self._bm.depth_socket(self._symbol, interval=self._ws_interval)

    async def _process_depth_message(self, msg):
        """Process a depth event message.

        :param msg: Depth event message.
        :return:

        """
        if self._last_update_id is None:
            self._depth_message_buffer.append(msg)
            return
        if msg['u'] <= self._last_update_id:
            return
        elif msg['U'] != self._last_update_id + 1:
            await self._init_cache()
        self._apply_orders(msg)
        res = self._depth_cache
        self._last_update_id = msg['u']
        if self._refresh_interval and int(time.time()) > self._refresh_time:
            await self._init_cache()
        return res

class FuturesDepthCacheManager(BaseDepthCacheManager):

    async def _process_depth_message(self, msg):
        """Process a depth event message.

        :param msg: Depth event message.
        :return:

        """
        msg = msg.get('data')
        return await super()._process_depth_message(msg)

    def _apply_orders(self, msg):
        if False:
            return 10
        assert self._depth_cache
        self._depth_cache._bids = msg.get('b', [])
        self._depth_cache._asks = msg.get('a', [])
        self._depth_cache.update_time = msg.get('E') or msg.get('lastUpdateId')

    def _get_socket(self):
        if False:
            i = 10
            return i + 15
        sock = self._bm.futures_depth_socket(self._symbol)
        return sock

class OptionsDepthCacheManager(BaseDepthCacheManager):

    def _get_socket(self):
        if False:
            while True:
                i = 10
        return self._bm.options_depth_socket(self._symbol)

class ThreadedDepthCacheManager(ThreadedApiManager):

    def __init__(self, api_key: Optional[str]=None, api_secret: Optional[str]=None, requests_params: Optional[Dict[str, str]]=None, tld: str='com', testnet: bool=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(api_key, api_secret, requests_params, tld, testnet)

    def _start_depth_cache(self, dcm_class, callback: Callable, symbol: str, refresh_interval=None, bm=None, limit=10, conv_type=float, **kwargs) -> str:
        if False:
            while True:
                i = 10
        while not self._client:
            time.sleep(0.01)
        dcm = dcm_class(client=self._client, symbol=symbol, loop=self._loop, refresh_interval=refresh_interval, bm=bm, limit=limit, conv_type=conv_type, **kwargs)
        path = symbol.lower() + '@depth' + str(limit)
        self._socket_running[path] = True
        self._loop.call_soon(asyncio.create_task, self.start_listener(dcm, path, callback))
        return path

    def start_depth_cache(self, callback: Callable, symbol: str, refresh_interval=None, bm=None, limit=10, conv_type=float, ws_interval=0) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._start_depth_cache(dcm_class=DepthCacheManager, callback=callback, symbol=symbol, refresh_interval=refresh_interval, bm=bm, limit=limit, conv_type=conv_type, ws_interval=ws_interval)

    def start_futures_depth_socket(self, callback: Callable, symbol: str, refresh_interval=None, bm=None, limit=10, conv_type=float) -> str:
        if False:
            return 10
        return self._start_depth_cache(dcm_class=FuturesDepthCacheManager, callback=callback, symbol=symbol, refresh_interval=refresh_interval, bm=bm, limit=limit, conv_type=conv_type)

    def start_options_depth_socket(self, callback: Callable, symbol: str, refresh_interval=None, bm=None, limit=10, conv_type=float) -> str:
        if False:
            print('Hello World!')
        return self._start_depth_cache(dcm_class=OptionsDepthCacheManager, callback=callback, symbol=symbol, refresh_interval=refresh_interval, bm=bm, limit=limit, conv_type=conv_type)