import json
from wechatpy.session import SessionStorage
from wechatpy.utils import to_text

class RedisStorage(SessionStorage):

    def __init__(self, redis, prefix='wechatpy'):
        if False:
            i = 10
            return i + 15
        for method_name in ('get', 'set', 'delete'):
            assert hasattr(redis, method_name)
        self.redis = redis
        self.prefix = prefix

    def key_name(self, key):
        if False:
            i = 10
            return i + 15
        return f'{self.prefix}:{key}'

    def get(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        key = self.key_name(key)
        value = self.redis.get(key)
        if value is None:
            return default
        return json.loads(to_text(value))

    def set(self, key, value, ttl=None):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            return
        key = self.key_name(key)
        value = json.dumps(value)
        self.redis.set(key, value, ex=ttl)

    def delete(self, key):
        if False:
            i = 10
            return i + 15
        key = self.key_name(key)
        self.redis.delete(key)