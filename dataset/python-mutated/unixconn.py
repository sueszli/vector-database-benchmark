import requests.adapters
import socket
from docker.transport.basehttpadapter import BaseHTTPAdapter
from .. import constants
import urllib3
import urllib3.connection
RecentlyUsedContainer = urllib3._collections.RecentlyUsedContainer

class UnixHTTPConnection(urllib3.connection.HTTPConnection):

    def __init__(self, base_url, unix_socket, timeout=60):
        if False:
            print('Hello World!')
        super().__init__('localhost', timeout=timeout)
        self.base_url = base_url
        self.unix_socket = unix_socket
        self.timeout = timeout

    def connect(self):
        if False:
            while True:
                i = 10
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.unix_socket)
        self.sock = sock

class UnixHTTPConnectionPool(urllib3.connectionpool.HTTPConnectionPool):

    def __init__(self, base_url, socket_path, timeout=60, maxsize=10):
        if False:
            print('Hello World!')
        super().__init__('localhost', timeout=timeout, maxsize=maxsize)
        self.base_url = base_url
        self.socket_path = socket_path
        self.timeout = timeout

    def _new_conn(self):
        if False:
            for i in range(10):
                print('nop')
        return UnixHTTPConnection(self.base_url, self.socket_path, self.timeout)

class UnixHTTPAdapter(BaseHTTPAdapter):
    __attrs__ = requests.adapters.HTTPAdapter.__attrs__ + ['pools', 'socket_path', 'timeout', 'max_pool_size']

    def __init__(self, socket_url, timeout=60, pool_connections=constants.DEFAULT_NUM_POOLS, max_pool_size=constants.DEFAULT_MAX_POOL_SIZE):
        if False:
            for i in range(10):
                print('nop')
        socket_path = socket_url.replace('http+unix://', '')
        if not socket_path.startswith('/'):
            socket_path = f'/{socket_path}'
        self.socket_path = socket_path
        self.timeout = timeout
        self.max_pool_size = max_pool_size
        self.pools = RecentlyUsedContainer(pool_connections, dispose_func=lambda p: p.close())
        super().__init__()

    def get_connection(self, url, proxies=None):
        if False:
            for i in range(10):
                print('nop')
        with self.pools.lock:
            pool = self.pools.get(url)
            if pool:
                return pool
            pool = UnixHTTPConnectionPool(url, self.socket_path, self.timeout, maxsize=self.max_pool_size)
            self.pools[url] = pool
        return pool

    def request_url(self, request, proxies):
        if False:
            for i in range(10):
                print('nop')
        return request.path_url