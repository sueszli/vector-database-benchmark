"""
Module to provide redis functionality to Salt

.. versionadded:: 2014.7.0

:configuration: This module requires the redis python module and uses the
    following defaults which may be overridden in the minion configuration:

.. code-block:: yaml

    redis.host: 'salt'
    redis.port: 6379
    redis.db: 0
    redis.password: None
"""
import salt.utils.args
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
__virtualname__ = 'redis'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load this module if redis python module is installed\n    '
    if HAS_REDIS:
        return __virtualname__
    else:
        return (False, 'The redis execution module failed to load: the redis python library is not available.')

def _connect(host=None, port=None, db=None, password=None):
    if False:
        print('Hello World!')
    '\n    Returns an instance of the redis client\n    '
    if not host:
        host = __salt__['config.option']('redis.host')
    if not port:
        port = __salt__['config.option']('redis.port')
    if not db:
        db = __salt__['config.option']('redis.db')
    if not password:
        password = __salt__['config.option']('redis.password')
    return redis.StrictRedis(host, port, db, password, decode_responses=True)

def _sconnect(host=None, port=None, password=None):
    if False:
        print('Hello World!')
    '\n    Returns an instance of the redis client\n    '
    if host is None:
        host = __salt__['config.option']('redis_sentinel.host', 'localhost')
    if port is None:
        port = __salt__['config.option']('redis_sentinel.port', 26379)
    if password is None:
        password = __salt__['config.option']('redis_sentinel.password')
    return redis.StrictRedis(host, port, password=password, decode_responses=True)

def bgrewriteaof(host=None, port=None, db=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Asynchronously rewrite the append-only file\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.bgrewriteaof\n    "
    server = _connect(host, port, db, password)
    return server.bgrewriteaof()

def bgsave(host=None, port=None, db=None, password=None):
    if False:
        return 10
    "\n    Asynchronously save the dataset to disk\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.bgsave\n    "
    server = _connect(host, port, db, password)
    return server.bgsave()

def config_get(pattern='*', host=None, port=None, db=None, password=None):
    if False:
        print('Hello World!')
    "\n    Get redis server configuration values\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.config_get\n        salt '*' redis.config_get port\n    "
    server = _connect(host, port, db, password)
    return server.config_get(pattern)

def config_set(name, value, host=None, port=None, db=None, password=None):
    if False:
        print('Hello World!')
    "\n    Set redis server configuration values\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.config_set masterauth luv_kittens\n    "
    server = _connect(host, port, db, password)
    return server.config_set(name, value)

def dbsize(host=None, port=None, db=None, password=None):
    if False:
        while True:
            i = 10
    "\n    Return the number of keys in the selected database\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.dbsize\n    "
    server = _connect(host, port, db, password)
    return server.dbsize()

def delete(*keys, **connection_args):
    if False:
        return 10
    "\n    Deletes the keys from redis, returns number of keys deleted\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.delete foo\n    "
    conn_args = {}
    for arg in ['host', 'port', 'db', 'password']:
        if arg in connection_args:
            conn_args[arg] = connection_args[arg]
    server = _connect(**conn_args)
    return server.delete(*keys)

def exists(key, host=None, port=None, db=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return true if the key exists in redis\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.exists foo\n    "
    server = _connect(host, port, db, password)
    return server.exists(key)

def expire(key, seconds, host=None, port=None, db=None, password=None):
    if False:
        print('Hello World!')
    "\n    Set a keys time to live in seconds\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.expire foo 300\n    "
    server = _connect(host, port, db, password)
    return server.expire(key, seconds)

def expireat(key, timestamp, host=None, port=None, db=None, password=None):
    if False:
        return 10
    "\n    Set a keys expire at given UNIX time\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.expireat foo 1400000000\n    "
    server = _connect(host, port, db, password)
    return server.expireat(key, timestamp)

def flushall(host=None, port=None, db=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove all keys from all databases\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.flushall\n    "
    server = _connect(host, port, db, password)
    return server.flushall()

def flushdb(host=None, port=None, db=None, password=None):
    if False:
        i = 10
        return i + 15
    "\n    Remove all keys from the selected database\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.flushdb\n    "
    server = _connect(host, port, db, password)
    return server.flushdb()

def get_key(key, host=None, port=None, db=None, password=None):
    if False:
        i = 10
        return i + 15
    "\n    Get redis key value\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.get_key foo\n    "
    server = _connect(host, port, db, password)
    return server.get(key)

def hdel(key, *fields, **options):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete one of more hash fields.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hdel foo_hash bar_field1 bar_field2\n    "
    host = options.get('host', None)
    port = options.get('port', None)
    database = options.get('db', None)
    password = options.get('password', None)
    server = _connect(host, port, database, password)
    return server.hdel(key, *fields)

def hexists(key, field, host=None, port=None, db=None, password=None):
    if False:
        return 10
    "\n    Determine if a hash fields exists.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hexists foo_hash bar_field\n    "
    server = _connect(host, port, db, password)
    return server.hexists(key, field)

def hget(key, field, host=None, port=None, db=None, password=None):
    if False:
        while True:
            i = 10
    "\n    Get specific field value from a redis hash, returns dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hget foo_hash bar_field\n    "
    server = _connect(host, port, db, password)
    return server.hget(key, field)

def hgetall(key, host=None, port=None, db=None, password=None):
    if False:
        return 10
    "\n    Get all fields and values from a redis hash, returns dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hgetall foo_hash\n    "
    server = _connect(host, port, db, password)
    return server.hgetall(key)

def hincrby(key, field, increment=1, host=None, port=None, db=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Increment the integer value of a hash field by the given number.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hincrby foo_hash bar_field 5\n    "
    server = _connect(host, port, db, password)
    return server.hincrby(key, field, amount=increment)

def hincrbyfloat(key, field, increment=1.0, host=None, port=None, db=None, password=None):
    if False:
        print('Hello World!')
    "\n    Increment the float value of a hash field by the given number.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hincrbyfloat foo_hash bar_field 5.17\n    "
    server = _connect(host, port, db, password)
    return server.hincrbyfloat(key, field, amount=increment)

def hlen(key, host=None, port=None, db=None, password=None):
    if False:
        while True:
            i = 10
    "\n    Returns number of fields of a hash.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hlen foo_hash\n    "
    server = _connect(host, port, db, password)
    return server.hlen(key)

def hmget(key, *fields, **options):
    if False:
        return 10
    "\n    Returns the values of all the given hash fields.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hmget foo_hash bar_field1 bar_field2\n    "
    host = options.get('host', None)
    port = options.get('port', None)
    database = options.get('db', None)
    password = options.get('password', None)
    server = _connect(host, port, database, password)
    return server.hmget(key, *fields)

def hmset(key, **fieldsvals):
    if False:
        for i in range(10):
            print('nop')
    "\n    Sets multiple hash fields to multiple values.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hmset foo_hash bar_field1=bar_value1 bar_field2=bar_value2\n    "
    host = fieldsvals.pop('host', None)
    port = fieldsvals.pop('port', None)
    database = fieldsvals.pop('db', None)
    password = fieldsvals.pop('password', None)
    server = _connect(host, port, database, password)
    return server.hmset(key, salt.utils.args.clean_kwargs(**fieldsvals))

def hset(key, field, value, host=None, port=None, db=None, password=None):
    if False:
        while True:
            i = 10
    "\n    Set the value of a hash field.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hset foo_hash bar_field bar_value\n    "
    server = _connect(host, port, db, password)
    return server.hset(key, field, value)

def hsetnx(key, field, value, host=None, port=None, db=None, password=None):
    if False:
        print('Hello World!')
    "\n    Set the value of a hash field only if the field does not exist.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hsetnx foo_hash bar_field bar_value\n    "
    server = _connect(host, port, db, password)
    return server.hsetnx(key, field, value)

def hvals(key, host=None, port=None, db=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return all the values in a hash.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hvals foo_hash bar_field1 bar_value1\n    "
    server = _connect(host, port, db, password)
    return server.hvals(key)

def hscan(key, cursor=0, match=None, count=None, host=None, port=None, db=None, password=None):
    if False:
        while True:
            i = 10
    "\n    Incrementally iterate hash fields and associated values.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.hscan foo_hash match='field_prefix_*' count=1\n    "
    server = _connect(host, port, db, password)
    return server.hscan(key, cursor=cursor, match=match, count=count)

def info(host=None, port=None, db=None, password=None):
    if False:
        return 10
    "\n    Get information and statistics about the server\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.info\n    "
    server = _connect(host, port, db, password)
    return server.info()

def keys(pattern='*', host=None, port=None, db=None, password=None):
    if False:
        return 10
    "\n    Get redis keys, supports glob style patterns\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.keys\n        salt '*' redis.keys test*\n    "
    server = _connect(host, port, db, password)
    return server.keys(pattern)

def key_type(key, host=None, port=None, db=None, password=None):
    if False:
        i = 10
        return i + 15
    "\n    Get redis key type\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.type foo\n    "
    server = _connect(host, port, db, password)
    return server.type(key)

def lastsave(host=None, port=None, db=None, password=None):
    if False:
        while True:
            i = 10
    "\n    Get the UNIX time in seconds of the last successful save to disk\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.lastsave\n    "
    server = _connect(host, port, db, password)
    return int(server.lastsave().timestamp())

def llen(key, host=None, port=None, db=None, password=None):
    if False:
        i = 10
        return i + 15
    "\n    Get the length of a list in Redis\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.llen foo_list\n    "
    server = _connect(host, port, db, password)
    return server.llen(key)

def lrange(key, start, stop, host=None, port=None, db=None, password=None):
    if False:
        while True:
            i = 10
    "\n    Get a range of values from a list in Redis\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.lrange foo_list 0 10\n    "
    server = _connect(host, port, db, password)
    return server.lrange(key, start, stop)

def ping(host=None, port=None, db=None, password=None):
    if False:
        while True:
            i = 10
    "\n    Ping the server, returns False on connection errors\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.ping\n    "
    server = _connect(host, port, db, password)
    try:
        return server.ping()
    except redis.ConnectionError:
        return False

def save(host=None, port=None, db=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Synchronously save the dataset to disk\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.save\n    "
    server = _connect(host, port, db, password)
    return server.save()

def set_key(key, value, host=None, port=None, db=None, password=None):
    if False:
        return 10
    "\n    Set redis key value\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.set_key foo bar\n    "
    server = _connect(host, port, db, password)
    return server.set(key, value)

def shutdown(host=None, port=None, db=None, password=None):
    if False:
        return 10
    "\n    Synchronously save the dataset to disk and then shut down the server\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.shutdown\n    "
    server = _connect(host, port, db, password)
    try:
        server.ping()
    except redis.ConnectionError:
        return False
    server.shutdown()
    try:
        server.ping()
    except redis.ConnectionError:
        return True
    return False

def slaveof(master_host=None, master_port=None, host=None, port=None, db=None, password=None):
    if False:
        i = 10
        return i + 15
    "\n    Make the server a slave of another instance, or promote it as master\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Become slave of redis-n01.example.com:6379\n        salt '*' redis.slaveof redis-n01.example.com 6379\n        salt '*' redis.slaveof redis-n01.example.com\n        # Become master\n        salt '*' redis.slaveof\n    "
    if master_host and (not master_port):
        master_port = 6379
    server = _connect(host, port, db, password)
    return server.slaveof(master_host, master_port)

def smembers(key, host=None, port=None, db=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get members in a Redis set\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.smembers foo_set\n    "
    server = _connect(host, port, db, password)
    return list(server.smembers(key))

def time(host=None, port=None, db=None, password=None):
    if False:
        print('Hello World!')
    "\n    Return the current server UNIX time in seconds\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.time\n    "
    server = _connect(host, port, db, password)
    return server.time()[0]

def zcard(key, host=None, port=None, db=None, password=None):
    if False:
        i = 10
        return i + 15
    "\n    Get the length of a sorted set in Redis\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.zcard foo_sorted\n    "
    server = _connect(host, port, db, password)
    return server.zcard(key)

def zrange(key, start, stop, host=None, port=None, db=None, password=None):
    if False:
        i = 10
        return i + 15
    "\n    Get a range of values from a sorted set in Redis by index\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.zrange foo_sorted 0 10\n    "
    server = _connect(host, port, db, password)
    return server.zrange(key, start, stop)

def sentinel_get_master_ip(master, host=None, port=None, password=None):
    if False:
        i = 10
        return i + 15
    "\n    Get ip for sentinel master\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.sentinel_get_master_ip 'mymaster'\n    "
    server = _sconnect(host, port, password)
    ret = server.sentinel_get_master_addr_by_name(master)
    return dict(list(zip(('master_host', 'master_port'), ret)))

def get_master_ip(host=None, port=None, password=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get host information about slave\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' redis.get_master_ip\n    "
    server = _connect(host, port, password)
    srv_info = server.info()
    ret = (srv_info.get('master_host', ''), srv_info.get('master_port', ''))
    return dict(list(zip(('master_host', 'master_port'), ret)))