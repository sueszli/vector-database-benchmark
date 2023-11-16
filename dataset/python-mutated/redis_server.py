"""Redis is an open source (BSD licensed), in-memory data structure (key value
based) store.  Before configuring the ``redis_server`` engine, you must install
the dependency redis_.

Configuration
=============

Select a database to search in and set its index in the option ``db``.  You can
either look for exact matches or use partial keywords to find what you are
looking for by configuring ``exact_match_only``.

Example
=======

Below is an example configuration:

.. code:: yaml

  # Required dependency: redis

  - name: myredis
    shortcut : rds
    engine: redis_server
    exact_match_only: false
    host: '127.0.0.1'
    port: 6379
    enable_http: true
    password: ''
    db: 0

Implementations
===============

"""
import redis
engine_type = 'offline'
host = '127.0.0.1'
port = 6379
password = ''
db = 0
paging = False
result_template = 'key-value.html'
exact_match_only = True
_redis_client = None

def init(_engine_settings):
    if False:
        print('Hello World!')
    global _redis_client
    _redis_client = redis.StrictRedis(host=host, port=port, db=db, password=password or None, decode_responses=True)

def search(query, _params):
    if False:
        while True:
            i = 10
    if not exact_match_only:
        return search_keys(query)
    ret = _redis_client.hgetall(query)
    if ret:
        ret['template'] = result_template
        return [ret]
    if ' ' in query:
        (qset, rest) = query.split(' ', 1)
        ret = []
        for res in _redis_client.hscan_iter(qset, match='*{}*'.format(rest)):
            ret.append({res[0]: res[1], 'template': result_template})
        return ret
    return []

def search_keys(query):
    if False:
        for i in range(10):
            print('nop')
    ret = []
    for key in _redis_client.scan_iter(match='*{}*'.format(query)):
        key_type = _redis_client.type(key)
        res = None
        if key_type == 'hash':
            res = _redis_client.hgetall(key)
        elif key_type == 'list':
            res = dict(enumerate(_redis_client.lrange(key, 0, -1)))
        if res:
            res['template'] = result_template
            res['redis_key'] = key
            ret.append(res)
    return ret