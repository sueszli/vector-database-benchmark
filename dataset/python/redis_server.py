# SPDX-License-Identifier: AGPL-3.0-or-later
# lint: pylint
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

import redis  # pylint: disable=import-error

engine_type = 'offline'

# redis connection variables
host = '127.0.0.1'
port = 6379
password = ''
db = 0

# engine specific variables
paging = False
result_template = 'key-value.html'
exact_match_only = True

_redis_client = None


def init(_engine_settings):
    global _redis_client  # pylint: disable=global-statement
    _redis_client = redis.StrictRedis(
        host=host,
        port=port,
        db=db,
        password=password or None,
        decode_responses=True,
    )


def search(query, _params):
    if not exact_match_only:
        return search_keys(query)

    ret = _redis_client.hgetall(query)
    if ret:
        ret['template'] = result_template
        return [ret]

    if ' ' in query:
        qset, rest = query.split(' ', 1)
        ret = []
        for res in _redis_client.hscan_iter(qset, match='*{}*'.format(rest)):
            ret.append(
                {
                    res[0]: res[1],
                    'template': result_template,
                }
            )
        return ret
    return []


def search_keys(query):
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
