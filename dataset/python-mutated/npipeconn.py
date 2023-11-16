import queue
import requests.adapters
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
from .npipesocket import NpipeSocket
import urllib3
import urllib3.connection
RecentlyUsedContainer = urllib3._collections.RecentlyUsedContainer

class NpipeHTTPConnection(urllib3.connection.HTTPConnection):

    def __init__(self, npipe_path, timeout=60):
        if False:
            while True:
                i = 10
        super().__init__('localhost', timeout=timeout)
        self.npipe_path = npipe_path
        self.timeout = timeout

    def connect(self):
        if False:
            while True:
                i = 10
        sock = NpipeSocket()
        sock.settimeout(self.timeout)
        sock.connect(self.npipe_path)
        self.sock = sock

class NpipeHTTPConnectionPool(urllib3.connectionpool.HTTPConnectionPool):

    def __init__(self, npipe_path, timeout=60, maxsize=10):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('localhost', timeout=timeout, maxsize=maxsize)
        self.npipe_path = npipe_path
        self.timeout = timeout

    def _new_conn(self):
        if False:
            return 10
        return NpipeHTTPConnection(self.npipe_path, self.timeout)

    def _get_conn(self, timeout):
        if False:
            i = 10
            return i + 15
        conn = None
        try:
            conn = self.pool.get(block=self.block, timeout=timeout)
        except AttributeError as ae:
            raise urllib3.exceptions.ClosedPoolError(self, 'Pool is closed.') from ae
        except queue.Empty:
            if self.block:
                raise urllib3.exceptions.EmptyPoolError(self, 'Pool reached maximum size and no more connections are allowed.') from None
        return conn or self._new_conn()

class NpipeHTTPAdapter(BaseHTTPAdapter):
    __attrs__ = requests.adapters.HTTPAdapter.__attrs__ + ['npipe_path', 'pools', 'timeout', 'max_pool_size']

    def __init__(self, base_url, timeout=60, pool_connections=constants.DEFAULT_NUM_POOLS, max_pool_size=constants.DEFAULT_MAX_POOL_SIZE):
        if False:
            i = 10
            return i + 15
        self.npipe_path = base_url.replace('npipe://', '')
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.pools = RecentlyUsedContainer(pool_connections, dispose_func=lambda p: p.close())
        super().__init__()

    def get_connection(self, url, proxies=None):
        if False:
            i = 10
            return i + 15
        with self.pools.lock:
            pool = self.pools.get(url)
            if pool:
                return pool
            pool = NpipeHTTPConnectionPool(self.npipe_path, self.timeout, maxsize=self.max_pool_size)
            self.pools[url] = pool
        return pool

    def request_url(self, request, proxies):
        if False:
            i = 10
            return i + 15
        return request.path_url