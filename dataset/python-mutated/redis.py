"""Redis result store backend."""
import time
from contextlib import contextmanager
from functools import partial
from ssl import CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED
from urllib.parse import unquote
from kombu.utils.functional import retry_over_time
from kombu.utils.objects import cached_property
from kombu.utils.url import _parse_url, maybe_sanitize_url
from celery import states
from celery._state import task_join_will_block
from celery.canvas import maybe_signature
from celery.exceptions import BackendStoreError, ChordError, ImproperlyConfigured
from celery.result import GroupResult, allow_join_result
from celery.utils.functional import _regen, dictfilter
from celery.utils.log import get_logger
from celery.utils.time import humanize_seconds
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
from .base import BaseKeyValueStoreBackend
try:
    import redis.connection
    from kombu.transport.redis import get_redis_error_classes
except ImportError:
    redis = None
    get_redis_error_classes = None
try:
    import redis.sentinel
except ImportError:
    pass
__all__ = ('RedisBackend', 'SentinelBackend')
E_REDIS_MISSING = '\nYou need to install the redis library in order to use the Redis result store backend.\n'
E_REDIS_SENTINEL_MISSING = '\nYou need to install the redis library with support of sentinel in order to use the Redis result store backend.\n'
W_REDIS_SSL_CERT_OPTIONAL = '\nSetting ssl_cert_reqs=CERT_OPTIONAL when connecting to redis means that celery might not validate the identity of the redis broker when connecting. This leaves you vulnerable to man in the middle attacks.\n'
W_REDIS_SSL_CERT_NONE = '\nSetting ssl_cert_reqs=CERT_NONE when connecting to redis means that celery will not validate the identity of the redis broker when connecting. This leaves you vulnerable to man in the middle attacks.\n'
E_REDIS_SSL_PARAMS_AND_SCHEME_MISMATCH = '\nSSL connection parameters have been provided but the specified URL scheme is redis://. A Redis SSL connection URL should use the scheme rediss://.\n'
E_REDIS_SSL_CERT_REQS_MISSING_INVALID = '\nA rediss:// URL must have parameter ssl_cert_reqs and this must be set to CERT_REQUIRED, CERT_OPTIONAL, or CERT_NONE\n'
E_LOST = 'Connection to Redis lost: Retry (%s/%s) %s.'
E_RETRY_LIMIT_EXCEEDED = '\nRetry limit exceeded while trying to reconnect to the Celery redis result store backend. The Celery application must be restarted.\n'
logger = get_logger(__name__)

class ResultConsumer(BaseResultConsumer):
    _pubsub = None

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._get_key_for_task = self.backend.get_key_for_task
        self._decode_result = self.backend.decode_result
        self._ensure = self.backend.ensure
        self._connection_errors = self.backend.connection_errors
        self.subscribed_to = set()

    def on_after_fork(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.backend.client.connection_pool.reset()
            if self._pubsub is not None:
                self._pubsub.close()
        except KeyError as e:
            logger.warning(str(e))
        super().on_after_fork()

    def _reconnect_pubsub(self):
        if False:
            for i in range(10):
                print('nop')
        self._pubsub = None
        self.backend.client.connection_pool.reset()
        if self.subscribed_to:
            metas = self.backend.client.mget(self.subscribed_to)
            metas = [meta for meta in metas if meta]
            for meta in metas:
                self.on_state_change(self._decode_result(meta), None)
        self._pubsub = self.backend.client.pubsub(ignore_subscribe_messages=True)
        if self.subscribed_to:
            self._pubsub.subscribe(*self.subscribed_to)
        else:
            self._pubsub.connection = self._pubsub.connection_pool.get_connection('pubsub', self._pubsub.shard_hint)
            self._pubsub.connection.register_connect_callback(self._pubsub.on_connect)

    @contextmanager
    def reconnect_on_error(self):
        if False:
            return 10
        try:
            yield
        except self._connection_errors:
            try:
                self._ensure(self._reconnect_pubsub, ())
            except self._connection_errors:
                logger.critical(E_RETRY_LIMIT_EXCEEDED)
                raise

    def _maybe_cancel_ready_task(self, meta):
        if False:
            while True:
                i = 10
        if meta['status'] in states.READY_STATES:
            self.cancel_for(meta['task_id'])

    def on_state_change(self, meta, message):
        if False:
            return 10
        super().on_state_change(meta, message)
        self._maybe_cancel_ready_task(meta)

    def start(self, initial_task_id, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._pubsub = self.backend.client.pubsub(ignore_subscribe_messages=True)
        self._consume_from(initial_task_id)

    def on_wait_for_pending(self, result, **kwargs):
        if False:
            while True:
                i = 10
        for meta in result._iter_meta(**kwargs):
            if meta is not None:
                self.on_state_change(meta, None)

    def stop(self):
        if False:
            return 10
        if self._pubsub is not None:
            self._pubsub.close()

    def drain_events(self, timeout=None):
        if False:
            i = 10
            return i + 15
        if self._pubsub:
            with self.reconnect_on_error():
                message = self._pubsub.get_message(timeout=timeout)
                if message and message['type'] == 'message':
                    self.on_state_change(self._decode_result(message['data']), message)
        elif timeout:
            time.sleep(timeout)

    def consume_from(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        if self._pubsub is None:
            return self.start(task_id)
        self._consume_from(task_id)

    def _consume_from(self, task_id):
        if False:
            return 10
        key = self._get_key_for_task(task_id)
        if key not in self.subscribed_to:
            self.subscribed_to.add(key)
            with self.reconnect_on_error():
                self._pubsub.subscribe(key)

    def cancel_for(self, task_id):
        if False:
            while True:
                i = 10
        key = self._get_key_for_task(task_id)
        self.subscribed_to.discard(key)
        if self._pubsub:
            with self.reconnect_on_error():
                self._pubsub.unsubscribe(key)

class RedisBackend(BaseKeyValueStoreBackend, AsyncBackendMixin):
    """Redis task result store.

    It makes use of the following commands:
    GET, MGET, DEL, INCRBY, EXPIRE, SET, SETEX
    """
    ResultConsumer = ResultConsumer
    redis = redis
    connection_class_ssl = redis.SSLConnection if redis else None
    max_connections = None
    supports_autoexpire = True
    supports_native_join = True
    _MAX_STR_VALUE_SIZE = 536870912

    def __init__(self, host=None, port=None, db=None, password=None, max_connections=None, url=None, connection_pool=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(expires_type=int, **kwargs)
        _get = self.app.conf.get
        if self.redis is None:
            raise ImproperlyConfigured(E_REDIS_MISSING.strip())
        if host and '://' in host:
            (url, host) = (host, None)
        self.max_connections = max_connections or _get('redis_max_connections') or self.max_connections
        self._ConnectionPool = connection_pool
        socket_timeout = _get('redis_socket_timeout')
        socket_connect_timeout = _get('redis_socket_connect_timeout')
        retry_on_timeout = _get('redis_retry_on_timeout')
        socket_keepalive = _get('redis_socket_keepalive')
        health_check_interval = _get('redis_backend_health_check_interval')
        self.connparams = {'host': _get('redis_host') or 'localhost', 'port': _get('redis_port') or 6379, 'db': _get('redis_db') or 0, 'password': _get('redis_password'), 'max_connections': self.max_connections, 'socket_timeout': socket_timeout and float(socket_timeout), 'retry_on_timeout': retry_on_timeout or False, 'socket_connect_timeout': socket_connect_timeout and float(socket_connect_timeout)}
        username = _get('redis_username')
        if username:
            self.connparams['username'] = username
        if health_check_interval:
            self.connparams['health_check_interval'] = health_check_interval
        if socket_keepalive:
            self.connparams['socket_keepalive'] = socket_keepalive
        ssl = _get('redis_backend_use_ssl')
        if ssl:
            self.connparams.update(ssl)
            self.connparams['connection_class'] = self.connection_class_ssl
        if url:
            self.connparams = self._params_from_url(url, self.connparams)
        if 'connection_class' in self.connparams and issubclass(self.connparams['connection_class'], redis.SSLConnection):
            ssl_cert_reqs_missing = 'MISSING'
            ssl_string_to_constant = {'CERT_REQUIRED': CERT_REQUIRED, 'CERT_OPTIONAL': CERT_OPTIONAL, 'CERT_NONE': CERT_NONE, 'required': CERT_REQUIRED, 'optional': CERT_OPTIONAL, 'none': CERT_NONE}
            ssl_cert_reqs = self.connparams.get('ssl_cert_reqs', ssl_cert_reqs_missing)
            ssl_cert_reqs = ssl_string_to_constant.get(ssl_cert_reqs, ssl_cert_reqs)
            if ssl_cert_reqs not in ssl_string_to_constant.values():
                raise ValueError(E_REDIS_SSL_CERT_REQS_MISSING_INVALID)
            if ssl_cert_reqs == CERT_OPTIONAL:
                logger.warning(W_REDIS_SSL_CERT_OPTIONAL)
            elif ssl_cert_reqs == CERT_NONE:
                logger.warning(W_REDIS_SSL_CERT_NONE)
            self.connparams['ssl_cert_reqs'] = ssl_cert_reqs
        self.url = url
        (self.connection_errors, self.channel_errors) = get_redis_error_classes() if get_redis_error_classes else ((), ())
        self.result_consumer = self.ResultConsumer(self, self.app, self.accept, self._pending_results, self._pending_messages)

    def _params_from_url(self, url, defaults):
        if False:
            for i in range(10):
                print('nop')
        (scheme, host, port, username, password, path, query) = _parse_url(url)
        connparams = dict(defaults, **dictfilter({'host': host, 'port': port, 'username': username, 'password': password, 'db': query.pop('virtual_host', None)}))
        if scheme == 'socket':
            connparams.update({'connection_class': self.redis.UnixDomainSocketConnection, 'path': '/' + path})
            connparams.pop('host', None)
            connparams.pop('port', None)
            connparams.pop('socket_connect_timeout')
        else:
            connparams['db'] = path
        ssl_param_keys = ['ssl_ca_certs', 'ssl_certfile', 'ssl_keyfile', 'ssl_cert_reqs']
        if scheme == 'redis':
            if any((key in connparams for key in ssl_param_keys)) or any((key in query for key in ssl_param_keys)):
                raise ValueError(E_REDIS_SSL_PARAMS_AND_SCHEME_MISMATCH)
        if scheme == 'rediss':
            connparams['connection_class'] = redis.SSLConnection
            for ssl_setting in ssl_param_keys:
                ssl_val = query.pop(ssl_setting, None)
                if ssl_val:
                    connparams[ssl_setting] = unquote(ssl_val)
        db = connparams.get('db') or 0
        db = db.strip('/') if isinstance(db, str) else db
        connparams['db'] = int(db)
        for (key, value) in query.items():
            if key in redis.connection.URL_QUERY_ARGUMENT_PARSERS:
                query[key] = redis.connection.URL_QUERY_ARGUMENT_PARSERS[key](value)
        connparams.update(query)
        return connparams

    @cached_property
    def retry_policy(self):
        if False:
            while True:
                i = 10
        retry_policy = super().retry_policy
        if 'retry_policy' in self._transport_options:
            retry_policy = retry_policy.copy()
            retry_policy.update(self._transport_options['retry_policy'])
        return retry_policy

    def on_task_call(self, producer, task_id):
        if False:
            print('Hello World!')
        if not task_join_will_block():
            self.result_consumer.consume_from(task_id)

    def get(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self.client.get(key)

    def mget(self, keys):
        if False:
            return 10
        return self.client.mget(keys)

    def ensure(self, fun, args, **policy):
        if False:
            print('Hello World!')
        retry_policy = dict(self.retry_policy, **policy)
        max_retries = retry_policy.get('max_retries')
        return retry_over_time(fun, self.connection_errors, args, {}, partial(self.on_connection_error, max_retries), **retry_policy)

    def on_connection_error(self, max_retries, exc, intervals, retries):
        if False:
            i = 10
            return i + 15
        tts = next(intervals)
        logger.error(E_LOST.strip(), retries, max_retries or 'Inf', humanize_seconds(tts, 'in '))
        return tts

    def set(self, key, value, **retry_policy):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, str) and len(value) > self._MAX_STR_VALUE_SIZE:
            raise BackendStoreError('value too large for Redis backend')
        return self.ensure(self._set, (key, value), **retry_policy)

    def _set(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        with self.client.pipeline() as pipe:
            if self.expires:
                pipe.setex(key, self.expires, value)
            else:
                pipe.set(key, value)
            pipe.publish(key, value)
            pipe.execute()

    def forget(self, task_id):
        if False:
            for i in range(10):
                print('nop')
        super().forget(task_id)
        self.result_consumer.cancel_for(task_id)

    def delete(self, key):
        if False:
            return 10
        self.client.delete(key)

    def incr(self, key):
        if False:
            while True:
                i = 10
        return self.client.incr(key)

    def expire(self, key, value):
        if False:
            return 10
        return self.client.expire(key, value)

    def add_to_chord(self, group_id, result):
        if False:
            return 10
        self.client.incr(self.get_key_for_group(group_id, '.t'), 1)

    def _unpack_chord_result(self, tup, decode, EXCEPTION_STATES=states.EXCEPTION_STATES, PROPAGATE_STATES=states.PROPAGATE_STATES):
        if False:
            for i in range(10):
                print('nop')
        (_, tid, state, retval) = decode(tup)
        if state in EXCEPTION_STATES:
            retval = self.exception_to_python(retval)
        if state in PROPAGATE_STATES:
            raise ChordError(f'Dependency {tid} raised {retval!r}')
        return retval

    def set_chord_size(self, group_id, chord_size):
        if False:
            while True:
                i = 10
        self.set(self.get_key_for_group(group_id, '.s'), chord_size)

    def apply_chord(self, header_result_args, body, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(header_result_args[1], _regen):
            header_result = self.app.GroupResult(*header_result_args)
            if any((isinstance(nr, GroupResult) for nr in header_result.results)):
                header_result.save(backend=self)

    @cached_property
    def _chord_zset(self):
        if False:
            return 10
        return self._transport_options.get('result_chord_ordered', True)

    @cached_property
    def _transport_options(self):
        if False:
            return 10
        return self.app.conf.get('result_backend_transport_options', {})

    def on_chord_part_return(self, request, state, result, propagate=None, **kwargs):
        if False:
            while True:
                i = 10
        app = self.app
        (tid, gid, group_index) = (request.id, request.group, request.group_index)
        if not gid or not tid:
            return
        if group_index is None:
            group_index = '+inf'
        client = self.client
        jkey = self.get_key_for_group(gid, '.j')
        tkey = self.get_key_for_group(gid, '.t')
        skey = self.get_key_for_group(gid, '.s')
        result = self.encode_result(result, state)
        encoded = self.encode([1, tid, state, result])
        with client.pipeline() as pipe:
            pipeline = (pipe.zadd(jkey, {encoded: group_index}).zcount(jkey, '-inf', '+inf') if self._chord_zset else pipe.rpush(jkey, encoded).llen(jkey)).get(tkey).get(skey)
            if self.expires:
                pipeline = pipeline.expire(jkey, self.expires).expire(tkey, self.expires).expire(skey, self.expires)
            (_, readycount, totaldiff, chord_size_bytes) = pipeline.execute()[:4]
        totaldiff = int(totaldiff or 0)
        if chord_size_bytes:
            try:
                callback = maybe_signature(request.chord, app=app)
                total = int(chord_size_bytes) + totaldiff
                if readycount == total:
                    header_result = GroupResult.restore(gid)
                    if header_result is not None:
                        header_result.on_ready()
                        join_func = header_result.join_native if header_result.supports_native_join else header_result.join
                        with allow_join_result():
                            resl = join_func(timeout=app.conf.result_chord_join_timeout, propagate=True)
                    else:
                        (decode, unpack) = (self.decode, self._unpack_chord_result)
                        with client.pipeline() as pipe:
                            if self._chord_zset:
                                pipeline = pipe.zrange(jkey, 0, -1)
                            else:
                                pipeline = pipe.lrange(jkey, 0, total)
                            (resl,) = pipeline.execute()
                        resl = [unpack(tup, decode) for tup in resl]
                    try:
                        callback.delay(resl)
                    except Exception as exc:
                        logger.exception('Chord callback for %r raised: %r', request.group, exc)
                        return self.chord_error_from_stack(callback, ChordError(f'Callback error: {exc!r}'))
                    finally:
                        with client.pipeline() as pipe:
                            pipe.delete(jkey).delete(tkey).delete(skey).execute()
            except ChordError as exc:
                logger.exception('Chord %r raised: %r', request.group, exc)
                return self.chord_error_from_stack(callback, exc)
            except Exception as exc:
                logger.exception('Chord %r raised: %r', request.group, exc)
                return self.chord_error_from_stack(callback, ChordError(f'Join error: {exc!r}'))

    def _create_client(self, **params):
        if False:
            return 10
        return self._get_client()(connection_pool=self._get_pool(**params))

    def _get_client(self):
        if False:
            i = 10
            return i + 15
        return self.redis.StrictRedis

    def _get_pool(self, **params):
        if False:
            i = 10
            return i + 15
        return self.ConnectionPool(**params)

    @property
    def ConnectionPool(self):
        if False:
            while True:
                i = 10
        if self._ConnectionPool is None:
            self._ConnectionPool = self.redis.ConnectionPool
        return self._ConnectionPool

    @cached_property
    def client(self):
        if False:
            return 10
        return self._create_client(**self.connparams)

    def __reduce__(self, args=(), kwargs=None):
        if False:
            return 10
        kwargs = {} if not kwargs else kwargs
        return super().__reduce__(args, dict(kwargs, expires=self.expires, url=self.url))
if getattr(redis, 'sentinel', None):

    class SentinelManagedSSLConnection(redis.sentinel.SentinelManagedConnection, redis.SSLConnection):
        """Connect to a Redis server using Sentinel + TLS.

        Use Sentinel to identify which Redis server is the current master
        to connect to and when connecting to the Master server, use an
        SSL Connection.
        """

class SentinelBackend(RedisBackend):
    """Redis sentinel task result store."""
    _SERVER_URI_SEPARATOR = ';'
    sentinel = getattr(redis, 'sentinel', None)
    connection_class_ssl = SentinelManagedSSLConnection if sentinel else None

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if self.sentinel is None:
            raise ImproperlyConfigured(E_REDIS_SENTINEL_MISSING.strip())
        super().__init__(*args, **kwargs)

    def as_uri(self, include_password=False):
        if False:
            while True:
                i = 10
        'Return the server addresses as URIs, sanitizing the password or not.'
        if include_password:
            return super().as_uri(include_password=include_password)
        uri_chunks = (maybe_sanitize_url(chunk) for chunk in (self.url or '').split(self._SERVER_URI_SEPARATOR))
        return self._SERVER_URI_SEPARATOR.join((uri[:-1] if uri.endswith(':///') else uri for uri in uri_chunks))

    def _params_from_url(self, url, defaults):
        if False:
            print('Hello World!')
        chunks = url.split(self._SERVER_URI_SEPARATOR)
        connparams = dict(defaults, hosts=[])
        for chunk in chunks:
            data = super()._params_from_url(url=chunk, defaults=defaults)
            connparams['hosts'].append(data)
        for param in ('host', 'port', 'db', 'password'):
            connparams.pop(param)
        for param in ('db', 'password'):
            if connparams['hosts'] and param in connparams['hosts'][0]:
                connparams[param] = connparams['hosts'][0].get(param)
        return connparams

    def _get_sentinel_instance(self, **params):
        if False:
            return 10
        connparams = params.copy()
        hosts = connparams.pop('hosts')
        min_other_sentinels = self._transport_options.get('min_other_sentinels', 0)
        sentinel_kwargs = self._transport_options.get('sentinel_kwargs', {})
        sentinel_instance = self.sentinel.Sentinel([(cp['host'], cp['port']) for cp in hosts], min_other_sentinels=min_other_sentinels, sentinel_kwargs=sentinel_kwargs, **connparams)
        return sentinel_instance

    def _get_pool(self, **params):
        if False:
            print('Hello World!')
        sentinel_instance = self._get_sentinel_instance(**params)
        master_name = self._transport_options.get('master_name', None)
        return sentinel_instance.master_for(service_name=master_name, redis_class=self._get_client()).connection_pool