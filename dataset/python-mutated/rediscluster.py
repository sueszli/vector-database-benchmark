"""
Provide token storage in Redis cluster.

To get started simply start a redis cluster and assign all hashslots to the connected nodes.
Add the redis hostname and port to master configs as eauth_redis_host and eauth_redis_port.
Default values for these configs are as follow:

.. code-block:: yaml

    eauth_redis_host: localhost
    eauth_redis_port: 6379

:depends:   - redis-py-cluster Python package
"""
import hashlib
import logging
import os
import salt.payload
try:
    import rediscluster
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
log = logging.getLogger(__name__)
__virtualname__ = 'rediscluster'

def __virtual__():
    if False:
        return 10
    if not HAS_REDIS:
        return (False, 'Could not use redis for tokens; rediscluster python client is not installed.')
    return __virtualname__

def _redis_client(opts):
    if False:
        return 10
    '\n    Connect to the redis host and return a StrictRedisCluster client object.\n    If connection fails then return None.\n    '
    redis_host = opts.get('eauth_redis_host', 'localhost')
    redis_port = opts.get('eauth_redis_port', 6379)
    try:
        return rediscluster.StrictRedisCluster(host=redis_host, port=redis_port, decode_responses=True)
    except rediscluster.exceptions.RedisClusterException as err:
        log.warning('Failed to connect to redis at %s:%s - %s', redis_host, redis_port, err)
        return None

def mk_token(opts, tdata):
    if False:
        return 10
    "\n    Mint a new token using the config option hash_type and store tdata with 'token' attribute set\n    to the token.\n    This module uses the hash of random 512 bytes as a token.\n\n    :param opts: Salt master config options\n    :param tdata: Token data to be stored with 'token' attribute of this dict set to the token.\n    :returns: tdata with token if successful. Empty dict if failed.\n    "
    redis_client = _redis_client(opts)
    if not redis_client:
        return {}
    hash_type = getattr(hashlib, opts.get('hash_type', 'md5'))
    tok = str(hash_type(os.urandom(512)).hexdigest())
    try:
        while redis_client.get(tok) is not None:
            tok = str(hash_type(os.urandom(512)).hexdigest())
    except Exception as err:
        log.warning('Authentication failure: cannot get token %s from redis: %s', tok, err)
        return {}
    tdata['token'] = tok
    try:
        redis_client.set(tok, salt.payload.dumps(tdata))
    except Exception as err:
        log.warning('Authentication failure: cannot save token %s to redis: %s', tok, err)
        return {}
    return tdata

def get_token(opts, tok):
    if False:
        i = 10
        return i + 15
    '\n    Fetch the token data from the store.\n\n    :param opts: Salt master config options\n    :param tok: Token value to get\n    :returns: Token data if successful. Empty dict if failed.\n    '
    redis_client = _redis_client(opts)
    if not redis_client:
        return {}
    try:
        tdata = salt.payload.loads(redis_client.get(tok))
        return tdata
    except Exception as err:
        log.warning('Authentication failure: cannot get token %s from redis: %s', tok, err)
        return {}

def rm_token(opts, tok):
    if False:
        i = 10
        return i + 15
    '\n    Remove token from the store.\n\n    :param opts: Salt master config options\n    :param tok: Token to remove\n    :returns: Empty dict if successful. None if failed.\n    '
    redis_client = _redis_client(opts)
    if not redis_client:
        return
    try:
        redis_client.delete(tok)
        return {}
    except Exception as err:
        log.warning('Could not remove token %s: %s', tok, err)

def list_tokens(opts):
    if False:
        while True:
            i = 10
    '\n    List all tokens in the store.\n\n    :param opts: Salt master config options\n    :returns: List of dicts (token_data)\n    '
    ret = []
    redis_client = _redis_client(opts)
    if not redis_client:
        return []
    try:
        return [k.decode('utf8') for k in redis_client.keys()]
    except Exception as err:
        log.warning('Failed to list keys: %s', err)
        return []