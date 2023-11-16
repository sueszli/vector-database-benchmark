"""Implementation of the redis client (redis-py_).

.. _redis-py: https://github.com/redis/redis-py

This implementation uses the :ref:`settings redis` setup from ``settings.yml``.
A redis DB connect can be tested by::

  >>> from searx import redisdb
  >>> redisdb.initialize()
  True
  >>> db = redisdb.client()
  >>> db.set("foo", "bar")
  True
  >>> db.get("foo")
  b'bar'
  >>>

"""
import os
import pwd
import logging
import redis
from searx import get_setting
OLD_REDIS_URL_DEFAULT_URL = 'unix:///usr/local/searxng-redis/run/redis.sock?db=0'
'This was the default Redis URL in settings.yml.'
_CLIENT = None
logger = logging.getLogger(__name__)

def client() -> redis.Redis:
    if False:
        print('Hello World!')
    return _CLIENT

def initialize():
    if False:
        for i in range(10):
            print('nop')
    global _CLIENT
    redis_url = get_setting('redis.url')
    if not redis_url:
        return False
    try:
        _CLIENT = redis.Redis.from_url(redis_url)
        kwargs = _CLIENT.get_connection_kwargs().copy()
        kwargs.pop('password', None)
        kwargs = ' '.join([f'{k}={v!r}' for (k, v) in kwargs.items()])
        logger.info('connecting to Redis %s', kwargs)
        _CLIENT.ping()
        logger.info('connected to Redis')
        return True
    except redis.exceptions.RedisError as e:
        _CLIENT = None
        _pw = pwd.getpwuid(os.getuid())
        logger.exception("[%s (%s)] can't connect redis DB ...", _pw.pw_name, _pw.pw_uid)
        if redis_url == OLD_REDIS_URL_DEFAULT_URL and isinstance(e, redis.exceptions.ConnectionError):
            logger.info("You can safely ignore the above Redis error if you don't use Redis. You can remove this error by setting redis.url to false in your settings.yml.")
    return False