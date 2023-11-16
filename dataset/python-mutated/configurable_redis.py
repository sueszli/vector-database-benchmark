from __future__ import annotations
DOCUMENTATION = '\n    cache: configurable_redis\n    short_description: Use Redis DB for cache\n    description:\n        - This cache uses JSON formatted, per host records saved in Redis.\n    version_added: "1.9"\n    requirements:\n      - redis>=2.4.5 (python lib)\n    options:\n      _uri:\n        description:\n          - A colon separated string of connection information for Redis.\n        required: True\n        env:\n          - name: ANSIBLE_CACHE_PLUGIN_CONNECTION\n        ini:\n          - key: fact_caching_connection\n            section: defaults\n      _prefix:\n        description: User defined prefix to use when creating the DB entries\n        default: ansible_facts\n        env:\n          - name: ANSIBLE_CACHE_PLUGIN_PREFIX\n        ini:\n          - key: fact_caching_prefix\n            section: defaults\n      _timeout:\n        default: 86400\n        description: Expiration timeout for the cache plugin data\n        env:\n          - name: ANSIBLE_CACHE_PLUGIN_TIMEOUT\n        ini:\n          - key: fact_caching_timeout\n            section: defaults\n        type: integer\n'
import time
import json
from ansible.errors import AnsibleError
from ansible.parsing.ajson import AnsibleJSONEncoder, AnsibleJSONDecoder
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
try:
    from redis import StrictRedis, VERSION
except ImportError:
    raise AnsibleError("The 'redis' python module (version 2.4.5 or newer) is required for the redis fact cache, 'pip install redis'")
display = Display()

class CacheModule(BaseCacheModule):
    """
    A caching module backed by redis.
    Keys are maintained in a zset with their score being the timestamp
    when they are inserted. This allows for the usage of 'zremrangebyscore'
    to expire keys. This mechanism is used or a pattern matched 'scan' for
    performance.
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        connection = []
        super(CacheModule, self).__init__(*args, **kwargs)
        if self.get_option('_uri'):
            connection = self.get_option('_uri').split(':')
        self._timeout = float(self.get_option('_timeout'))
        self._prefix = self.get_option('_prefix')
        self._cache = {}
        self._db = StrictRedis(*connection)
        self._keys_set = 'ansible_cache_keys'

    def _make_key(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self._prefix + key

    def get(self, key):
        if False:
            print('Hello World!')
        if key not in self._cache:
            value = self._db.get(self._make_key(key))
            if value is None:
                self.delete(key)
                raise KeyError
            self._cache[key] = json.loads(value, cls=AnsibleJSONDecoder)
        return self._cache.get(key)

    def set(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        value2 = json.dumps(value, cls=AnsibleJSONEncoder, sort_keys=True, indent=4)
        if self._timeout > 0:
            self._db.setex(self._make_key(key), int(self._timeout), value2)
        else:
            self._db.set(self._make_key(key), value2)
        if VERSION[0] == 2:
            self._db.zadd(self._keys_set, time.time(), key)
        else:
            self._db.zadd(self._keys_set, {key: time.time()})
        self._cache[key] = value

    def _expire_keys(self):
        if False:
            print('Hello World!')
        if self._timeout > 0:
            expiry_age = time.time() - self._timeout
            self._db.zremrangebyscore(self._keys_set, 0, expiry_age)

    def keys(self):
        if False:
            return 10
        self._expire_keys()
        return self._db.zrange(self._keys_set, 0, -1)

    def contains(self, key):
        if False:
            while True:
                i = 10
        self._expire_keys()
        return self._db.zrank(self._keys_set, key) is not None

    def delete(self, key):
        if False:
            return 10
        if key in self._cache:
            del self._cache[key]
        self._db.delete(self._make_key(key))
        self._db.zrem(self._keys_set, key)

    def flush(self):
        if False:
            i = 10
            return i + 15
        for key in self.keys():
            self.delete(key)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        ret = dict()
        for key in self.keys():
            ret[key] = self.get(key)
        return ret

    def __getstate__(self):
        if False:
            return 10
        return dict()

    def __setstate__(self, data):
        if False:
            while True:
                i = 10
        self.__init__()