import json
import redis
from warehouse.legacy.api.xmlrpc.cache.interfaces import CacheError
DEFAULT_EXPIRES = 86400

class StubMetricReporter:

    def increment(self, metric_name):
        if False:
            print('Hello World!')
        return

class RedisLru:
    """
    Redis backed LRU cache for functions which return an object which
    can survive json.dumps() and json.loads() intact
    """

    def __init__(self, conn, name='lru', expires=None, metric_reporter=None):
        if False:
            print('Hello World!')
        '\n        conn:            Redis Connection Object\n        name:            Prefix for all keys in the cache\n        expires:         Default expiration\n        metric_reporter: Object implementing an `increment(<string>)` method\n        '
        self.conn = conn
        self.name = name
        self.expires = expires if expires else DEFAULT_EXPIRES
        if callable(getattr(metric_reporter, 'increment', None)):
            self.metric_reporter = metric_reporter
        else:
            self.metric_reporter = StubMetricReporter()

    def format_key(self, func_name, tag):
        if False:
            while True:
                i = 10
        if tag is not None and tag != 'None':
            return ':'.join([self.name, tag, func_name])
        return ':'.join([self.name, 'tag', func_name])

    def get(self, func_name, key, tag):
        if False:
            return 10
        try:
            value = self.conn.hget(self.format_key(func_name, tag), str(key))
        except (redis.exceptions.RedisError, redis.exceptions.ConnectionError):
            self.metric_reporter.increment(f'{self.name}.cache.error')
            return None
        if value:
            self.metric_reporter.increment(f'{self.name}.cache.hit')
            value = json.loads(value)
        return value

    def add(self, func_name, key, value, tag, expires):
        if False:
            while True:
                i = 10
        try:
            self.metric_reporter.increment(f'{self.name}.cache.miss')
            pipeline = self.conn.pipeline()
            pipeline.hset(self.format_key(func_name, tag), str(key), json.dumps(value))
            ttl = expires if expires else self.expires
            pipeline.expire(self.format_key(func_name, tag), ttl)
            pipeline.execute()
            return value
        except (redis.exceptions.RedisError, redis.exceptions.ConnectionError):
            self.metric_reporter.increment(f'{self.name}.cache.error')
            return value

    def purge(self, tag):
        if False:
            print('Hello World!')
        try:
            keys = self.conn.scan_iter(':'.join([self.name, tag, '*']), count=1000)
            pipeline = self.conn.pipeline()
            for key in keys:
                pipeline.delete(key)
            pipeline.execute()
            self.metric_reporter.increment(f'{self.name}.cache.purge')
        except (redis.exceptions.RedisError, redis.exceptions.ConnectionError):
            self.metric_reporter.increment(f'{self.name}.cache.error')
            raise CacheError()

    def fetch(self, func, args, kwargs, key, tag, expires):
        if False:
            return 10
        return self.get(func.__name__, str(key), str(tag)) or self.add(func.__name__, str(key), func(*args, **kwargs), str(tag), expires)