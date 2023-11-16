"""A collection of convenient functions and redis/lua scripts.

This code was partial inspired by the `Bullet-Proofing Lua Scripts in RedisPy`_
article.

.. _Bullet-Proofing Lua Scripts in RedisPy:
   https://redis.com/blog/bullet-proofing-lua-scripts-in-redispy/

"""
import hmac
from searx import get_setting
LUA_SCRIPT_STORAGE = {}
"A global dictionary to cache client's ``Script`` objects, used by\n:py:obj:`lua_script_storage`"

def lua_script_storage(client, script):
    if False:
        return 10
    'Returns a redis :py:obj:`Script\n    <redis.commands.core.CoreCommands.register_script>` instance.\n\n    Due to performance reason the ``Script`` object is instantiated only once\n    for a client (``client.register_script(..)``) and is cached in\n    :py:obj:`LUA_SCRIPT_STORAGE`.\n\n    '
    client_id = id(client)
    if LUA_SCRIPT_STORAGE.get(client_id) is None:
        LUA_SCRIPT_STORAGE[client_id] = {}
    if LUA_SCRIPT_STORAGE[client_id].get(script) is None:
        LUA_SCRIPT_STORAGE[client_id][script] = client.register_script(script)
    return LUA_SCRIPT_STORAGE[client_id][script]
PURGE_BY_PREFIX = "\nlocal prefix = tostring(ARGV[1])\nfor i, name in ipairs(redis.call('KEYS', prefix .. '*')) do\n    redis.call('EXPIRE', name, 0)\nend\n"

def purge_by_prefix(client, prefix: str='SearXNG_'):
    if False:
        print('Hello World!')
    'Purge all keys with ``prefix`` from database.\n\n    Queries all keys in the database by the given prefix and set expire time to\n    zero.  The default prefix will drop all keys which has been set by SearXNG\n    (drops SearXNG schema entirely from database).\n\n    The implementation is the lua script from string :py:obj:`PURGE_BY_PREFIX`.\n    The lua script uses EXPIRE_ instead of DEL_: if there are a lot keys to\n    delete and/or their values are big, `DEL` could take more time and blocks\n    the command loop while `EXPIRE` turns back immediate.\n\n    :param prefix: prefix of the key to delete (default: ``SearXNG_``)\n    :type name: str\n\n    .. _EXPIRE: https://redis.io/commands/expire/\n    .. _DEL: https://redis.io/commands/del/\n\n    '
    script = lua_script_storage(client, PURGE_BY_PREFIX)
    script(args=[prefix])

def secret_hash(name: str):
    if False:
        for i in range(10):
            print('nop')
    'Creates a hash of the ``name``.\n\n    Combines argument ``name`` with the ``secret_key`` from :ref:`settings\n    server`.  This function can be used to get a more anonymized name of a Redis\n    KEY.\n\n    :param name: the name to create a secret hash for\n    :type name: str\n    '
    m = hmac.new(bytes(name, encoding='utf-8'), digestmod='sha256')
    m.update(bytes(get_setting('server.secret_key'), encoding='utf-8'))
    return m.hexdigest()
INCR_COUNTER = "\nlocal limit = tonumber(ARGV[1])\nlocal expire = tonumber(ARGV[2])\nlocal c_name = KEYS[1]\n\nlocal c = redis.call('GET', c_name)\n\nif not c then\n    c = redis.call('INCR', c_name)\n    if expire > 0 then\n        redis.call('EXPIRE', c_name, expire)\n    end\nelse\n    c = tonumber(c)\n    if limit == 0 or c < limit then\n       c = redis.call('INCR', c_name)\n    end\nend\nreturn c\n"

def incr_counter(client, name: str, limit: int=0, expire: int=0):
    if False:
        return 10
    'Increment a counter and return the new value.\n\n    If counter with redis key ``SearXNG_counter_<name>`` does not exists it is\n    created with initial value 1 returned.  The replacement ``<name>`` is a\n    *secret hash* of the value from argument ``name`` (see\n    :py:func:`secret_hash`).\n\n    The implementation of the redis counter is the lua script from string\n    :py:obj:`INCR_COUNTER`.\n\n    :param name: name of the counter\n    :type name: str\n\n    :param expire: live-time of the counter in seconds (default ``None`` means\n      infinite).\n    :type expire: int / see EXPIRE_\n\n    :param limit: limit where the counter stops to increment (default ``None``)\n    :type limit: int / limit is 2^64 see INCR_\n\n    :return: value of the incremented counter\n    :type return: int\n\n    .. _EXPIRE: https://redis.io/commands/expire/\n    .. _INCR: https://redis.io/commands/incr/\n\n    A simple demo of a counter with expire time and limit::\n\n      >>> for i in range(6):\n      ...   i, incr_counter(client, "foo", 3, 5) # max 3, duration 5 sec\n      ...   time.sleep(1) # from the third call on max has been reached\n      ...\n      (0, 1)\n      (1, 2)\n      (2, 3)\n      (3, 3)\n      (4, 3)\n      (5, 1)\n\n    '
    script = lua_script_storage(client, INCR_COUNTER)
    name = 'SearXNG_counter_' + secret_hash(name)
    c = script(args=[limit, expire], keys=[name])
    return c

def drop_counter(client, name):
    if False:
        print('Hello World!')
    'Drop counter with redis key ``SearXNG_counter_<name>``\n\n    The replacement ``<name>`` is a *secret hash* of the value from argument\n    ``name`` (see :py:func:`incr_counter` and :py:func:`incr_sliding_window`).\n    '
    name = 'SearXNG_counter_' + secret_hash(name)
    client.delete(name)
INCR_SLIDING_WINDOW = "\nlocal expire = tonumber(ARGV[1])\nlocal name = KEYS[1]\nlocal current_time = redis.call('TIME')\n\nredis.call('ZREMRANGEBYSCORE', name, 0, current_time[1] - expire)\nredis.call('ZADD', name, current_time[1], current_time[1] .. current_time[2])\nlocal result = redis.call('ZCOUNT', name, 0, current_time[1] + 1)\nredis.call('EXPIRE', name, expire)\nreturn result\n"

def incr_sliding_window(client, name: str, duration: int):
    if False:
        for i in range(10):
            print('nop')
    'Increment a sliding-window counter and return the new value.\n\n    If counter with redis key ``SearXNG_counter_<name>`` does not exists it is\n    created with initial value 1 returned.  The replacement ``<name>`` is a\n    *secret hash* of the value from argument ``name`` (see\n    :py:func:`secret_hash`).\n\n    :param name: name of the counter\n    :type name: str\n\n    :param duration: live-time of the sliding window in seconds\n    :typeduration: int\n\n    :return: value of the incremented counter\n    :type return: int\n\n    The implementation of the redis counter is the lua script from string\n    :py:obj:`INCR_SLIDING_WINDOW`.  The lua script uses `sorted sets in Redis`_\n    to implement a sliding window for the redis key ``SearXNG_counter_<name>``\n    (ZADD_).  The current TIME_ is used to score the items in the sorted set and\n    the time window is moved by removing items with a score lower current time\n    minus *duration* time (ZREMRANGEBYSCORE_).\n\n    The EXPIRE_ time (the duration of the sliding window) is refreshed on each\n    call (increment) and if there is no call in this duration, the sorted\n    set expires from the redis DB.\n\n    The return value is the amount of items in the sorted set (ZCOUNT_), what\n    means the number of calls in the sliding window.\n\n    .. _Sorted sets in Redis:\n       https://redis.com/ebook/part-1-getting-started/chapter-1-getting-to-know-redis/1-2-what-redis-data-structures-look-like/1-2-5-sorted-sets-in-redis/\n    .. _TIME: https://redis.io/commands/time/\n    .. _ZADD: https://redis.io/commands/zadd/\n    .. _EXPIRE: https://redis.io/commands/expire/\n    .. _ZREMRANGEBYSCORE: https://redis.io/commands/zremrangebyscore/\n    .. _ZCOUNT: https://redis.io/commands/zcount/\n\n    A simple demo of the sliding window::\n\n      >>> for i in range(5):\n      ...   incr_sliding_window(client, "foo", 3) # duration 3 sec\n      ...   time.sleep(1) # from the third call (second) on the window is moved\n      ...\n      1\n      2\n      3\n      3\n      3\n      >>> time.sleep(3)  # wait until expire\n      >>> incr_sliding_window(client, "foo", 3)\n      1\n\n    '
    script = lua_script_storage(client, INCR_SLIDING_WINDOW)
    name = 'SearXNG_counter_' + secret_hash(name)
    c = script(args=[duration], keys=[name])
    return c