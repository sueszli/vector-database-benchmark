import pickle
import re
import redis
from redis.commands.search import Search
import frappe
from frappe.utils import cstr

class RedisearchWrapper(Search):

    def sugadd(self, key, *suggestions, **kwargs):
        if False:
            print('Hello World!')
        return super().sugadd(self.client.make_key(key), *suggestions, **kwargs)

    def suglen(self, key):
        if False:
            while True:
                i = 10
        return super().suglen(self.client.make_key(key))

    def sugdel(self, key, string):
        if False:
            for i in range(10):
                print('nop')
        return super().sugdel(self.client.make_key(key), string)

    def sugget(self, key, *args, **kwargs):
        if False:
            while True:
                i = 10
        return super().sugget(self.client.make_key(key), *args, **kwargs)

class RedisWrapper(redis.Redis):
    """Redis client that will automatically prefix conf.db_name"""

    def connected(self):
        if False:
            print('Hello World!')
        try:
            self.ping()
            return True
        except redis.exceptions.ConnectionError:
            return False

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        'WARNING: Added for backward compatibility to support frappe.cache().method(...)'
        return self

    def make_key(self, key, user=None, shared=False):
        if False:
            print('Hello World!')
        if shared:
            return key
        if user:
            if user is True:
                user = frappe.session.user
            key = f'user:{user}:{key}'
        return f'{frappe.conf.db_name}|{key}'.encode()

    def set_value(self, key, val, user=None, expires_in_sec=None, shared=False):
        if False:
            for i in range(10):
                print('nop')
        'Sets cache value.\n\n\t\t:param key: Cache key\n\t\t:param val: Value to be cached\n\t\t:param user: Prepends key with User\n\t\t:param expires_in_sec: Expire value of this key in X seconds\n\t\t'
        key = self.make_key(key, user, shared)
        if not expires_in_sec:
            frappe.local.cache[key] = val
        try:
            if expires_in_sec:
                self.setex(name=key, time=expires_in_sec, value=pickle.dumps(val))
            else:
                self.set(key, pickle.dumps(val))
        except redis.exceptions.ConnectionError:
            return None

    def get_value(self, key, generator=None, user=None, expires=False, shared=False):
        if False:
            return 10
        "Returns cache value. If not found and generator function is\n\t\t        given, it will call the generator.\n\n\t\t:param key: Cache key.\n\t\t:param generator: Function to be called to generate a value if `None` is returned.\n\t\t:param expires: If the key is supposed to be with an expiry, don't store it in frappe.local\n\t\t"
        original_key = key
        key = self.make_key(key, user, shared)
        if key in frappe.local.cache:
            val = frappe.local.cache[key]
        else:
            val = None
            try:
                val = self.get(key)
            except redis.exceptions.ConnectionError:
                pass
            if val is not None:
                val = pickle.loads(val)
            if not expires:
                if val is None and generator:
                    val = generator()
                    self.set_value(original_key, val, user=user)
                else:
                    frappe.local.cache[key] = val
        return val

    def get_all(self, key):
        if False:
            print('Hello World!')
        ret = {}
        for k in self.get_keys(key):
            ret[key] = self.get_value(k)
        return ret

    def get_keys(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Return keys starting with `key`.'
        try:
            key = self.make_key(key + '*')
            return self.keys(key)
        except redis.exceptions.ConnectionError:
            regex = re.compile(cstr(key).replace('|', '\\|').replace('*', '[\\w]*'))
            return [k for k in list(frappe.local.cache) if regex.match(cstr(k))]

    def delete_keys(self, key):
        if False:
            return 10
        'Delete keys with wildcard `*`.'
        self.delete_value(self.get_keys(key), make_keys=False)

    def delete_key(self, *args, **kwargs):
        if False:
            return 10
        self.delete_value(*args, **kwargs)

    def delete_value(self, keys, user=None, make_keys=True, shared=False):
        if False:
            for i in range(10):
                print('nop')
        'Delete value, list of values.'
        if not keys:
            return
        if not isinstance(keys, (list, tuple)):
            keys = (keys,)
        if make_keys:
            keys = [self.make_key(k, shared=shared, user=user) for k in keys]
        for key in keys:
            frappe.local.cache.pop(key, None)
        try:
            self.delete(*keys)
        except redis.exceptions.ConnectionError:
            pass

    def lpush(self, key, value):
        if False:
            return 10
        super().lpush(self.make_key(key), value)

    def rpush(self, key, value):
        if False:
            i = 10
            return i + 15
        super().rpush(self.make_key(key), value)

    def lpop(self, key):
        if False:
            while True:
                i = 10
        return super().lpop(self.make_key(key))

    def rpop(self, key):
        if False:
            print('Hello World!')
        return super().rpop(self.make_key(key))

    def llen(self, key):
        if False:
            i = 10
            return i + 15
        return super().llen(self.make_key(key))

    def lrange(self, key, start, stop):
        if False:
            return 10
        return super().lrange(self.make_key(key), start, stop)

    def ltrim(self, key, start, stop):
        if False:
            i = 10
            return i + 15
        return super().ltrim(self.make_key(key), start, stop)

    def hset(self, name: str, key: str, value, shared: bool=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if key is None:
            return
        _name = self.make_key(name, shared=shared)
        frappe.local.cache.setdefault(_name, {})[key] = value
        try:
            super().hset(_name, key, pickle.dumps(value), *args, **kwargs)
        except redis.exceptions.ConnectionError:
            pass

    def hexists(self, name: str, key: str, shared: bool=False) -> bool:
        if False:
            i = 10
            return i + 15
        if key is None:
            return False
        _name = self.make_key(name, shared=shared)
        try:
            return super().hexists(_name, key)
        except redis.exceptions.ConnectionError:
            return False

    def exists(self, *names: str, user=None, shared=None) -> int:
        if False:
            return 10
        names = [self.make_key(n, user=user, shared=shared) for n in names]
        try:
            return super().exists(*names)
        except redis.exceptions.ConnectionError:
            return False

    def hgetall(self, name):
        if False:
            i = 10
            return i + 15
        value = super().hgetall(self.make_key(name))
        return {key: pickle.loads(value) for (key, value) in value.items()}

    def hget(self, name, key, generator=None, shared=False):
        if False:
            while True:
                i = 10
        _name = self.make_key(name, shared=shared)
        if _name not in frappe.local.cache:
            frappe.local.cache[_name] = {}
        if not key:
            return None
        if key in frappe.local.cache[_name]:
            return frappe.local.cache[_name][key]
        value = None
        try:
            value = super().hget(_name, key)
        except redis.exceptions.ConnectionError:
            pass
        if value is not None:
            value = pickle.loads(value)
            frappe.local.cache[_name][key] = value
        elif generator:
            value = generator()
            self.hset(name, key, value, shared=shared)
        return value

    def hdel(self, name, key, shared=False):
        if False:
            while True:
                i = 10
        _name = self.make_key(name, shared=shared)
        if _name in frappe.local.cache:
            if key in frappe.local.cache[_name]:
                del frappe.local.cache[_name][key]
        try:
            super().hdel(_name, key)
        except redis.exceptions.ConnectionError:
            pass

    def hdel_keys(self, name_starts_with, key):
        if False:
            return 10
        'Delete hash names with wildcard `*` and key'
        for name in self.get_keys(name_starts_with):
            name = name.split('|', 1)[1]
            self.hdel(name, key)

    def hkeys(self, name):
        if False:
            i = 10
            return i + 15
        try:
            return super().hkeys(self.make_key(name))
        except redis.exceptions.ConnectionError:
            return []

    def sadd(self, name, *values):
        if False:
            i = 10
            return i + 15
        'Add a member/members to a given set'
        super().sadd(self.make_key(name), *values)

    def srem(self, name, *values):
        if False:
            i = 10
            return i + 15
        'Remove a specific member/list of members from the set'
        super().srem(self.make_key(name), *values)

    def sismember(self, name, value):
        if False:
            i = 10
            return i + 15
        'Returns True or False based on if a given value is present in the set'
        return super().sismember(self.make_key(name), value)

    def spop(self, name):
        if False:
            i = 10
            return i + 15
        'Removes and returns a random member from the set'
        return super().spop(self.make_key(name))

    def srandmember(self, name, count=None):
        if False:
            i = 10
            return i + 15
        'Returns a random member from the set'
        return super().srandmember(self.make_key(name))

    def smembers(self, name):
        if False:
            return 10
        'Return all members of the set'
        return super().smembers(self.make_key(name))

    def ft(self, index_name='idx'):
        if False:
            while True:
                i = 10
        return RedisearchWrapper(client=self, index_name=self.make_key(index_name))