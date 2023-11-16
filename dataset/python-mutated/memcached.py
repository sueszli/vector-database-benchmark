from pylibmc import Client, ClientPool, NotFound
from ..backend import RateLimiterBackend

class MemcachedBackend(RateLimiterBackend):
    """A rate limiter backend for Memcached_.

    Examples:

      >>> from dramatiq.rate_limits.backends import MemcachedBackend
      >>> backend = MemcachedBackend(servers=["127.0.0.1"], binary=True)

    Parameters:
      pool(ClientPool): An optional pylibmc client pool to use.  If
        this is passed, all other connection params are ignored.
      pool_size(int): The size of the connection pool to use.
      **parameters: Connection parameters are passed directly
        to :class:`pylibmc.Client`.

    .. _memcached: https://memcached.org
    """

    def __init__(self, *, pool=None, pool_size=8, **parameters):
        if False:
            return 10
        behaviors = parameters.setdefault('behaviors', {})
        behaviors['cas'] = True
        self.pool = pool or ClientPool(Client(**parameters), pool_size)

    def add(self, key, value, ttl):
        if False:
            for i in range(10):
                print('nop')
        with self.pool.reserve(block=True) as client:
            return client.add(key, value, time=int(ttl / 1000))

    def incr(self, key, amount, maximum, ttl):
        if False:
            i = 10
            return i + 15
        with self.pool.reserve(block=True) as client:
            return client.incr(key, amount) <= maximum

    def decr(self, key, amount, minimum, ttl):
        if False:
            print('Hello World!')
        with self.pool.reserve(block=True) as client:
            return client.decr(key, amount) >= minimum

    def incr_and_sum(self, key, keys, amount, maximum, ttl):
        if False:
            return 10
        ttl = int(ttl / 1000)
        with self.pool.reserve(block=True) as client:
            client.add(key, 0, time=ttl)
            while True:
                (value, cid) = client.gets(key)
                if cid is None:
                    return False
                value += amount
                if value > maximum:
                    return False
                key_list = keys() if callable(keys) else keys
                mapping = client.get_multi(key_list)
                total = amount + sum(mapping.values())
                if total > maximum:
                    return False
                try:
                    swapped = client.cas(key, value, cid, ttl)
                    if swapped:
                        return True
                except NotFound:
                    continue