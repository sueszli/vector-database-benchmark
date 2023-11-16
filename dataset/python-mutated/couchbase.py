"""Couchbase result store backend."""
from kombu.utils.url import _parse_url
from celery.exceptions import ImproperlyConfigured
from .base import KeyValueStoreBackend
try:
    from couchbase.auth import PasswordAuthenticator
    from couchbase.cluster import Cluster
except ImportError:
    Cluster = PasswordAuthenticator = None
try:
    from couchbase_core._libcouchbase import FMT_AUTO
except ImportError:
    FMT_AUTO = None
__all__ = ('CouchbaseBackend',)

class CouchbaseBackend(KeyValueStoreBackend):
    """Couchbase backend.

    Raises:
        celery.exceptions.ImproperlyConfigured:
            if module :pypi:`couchbase` is not available.
    """
    bucket = 'default'
    host = 'localhost'
    port = 8091
    username = None
    password = None
    quiet = False
    supports_autoexpire = True
    timeout = 2.5
    key_t = str

    def __init__(self, url=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.setdefault('expires_type', int)
        super().__init__(*args, **kwargs)
        self.url = url
        if Cluster is None:
            raise ImproperlyConfigured('You need to install the couchbase library to use the Couchbase backend.')
        uhost = uport = uname = upass = ubucket = None
        if url:
            (_, uhost, uport, uname, upass, ubucket, _) = _parse_url(url)
            ubucket = ubucket.strip('/') if ubucket else None
        config = self.app.conf.get('couchbase_backend_settings', None)
        if config is not None:
            if not isinstance(config, dict):
                raise ImproperlyConfigured('Couchbase backend settings should be grouped in a dict')
        else:
            config = {}
        self.host = uhost or config.get('host', self.host)
        self.port = int(uport or config.get('port', self.port))
        self.bucket = ubucket or config.get('bucket', self.bucket)
        self.username = uname or config.get('username', self.username)
        self.password = upass or config.get('password', self.password)
        self._connection = None

    def _get_connection(self):
        if False:
            i = 10
            return i + 15
        'Connect to the Couchbase server.'
        if self._connection is None:
            if self.host and self.port:
                uri = f'couchbase://{self.host}:{self.port}'
            else:
                uri = f'couchbase://{self.host}'
            if self.username and self.password:
                opt = PasswordAuthenticator(self.username, self.password)
            else:
                opt = None
            cluster = Cluster(uri, opt)
            bucket = cluster.bucket(self.bucket)
            self._connection = bucket.default_collection()
        return self._connection

    @property
    def connection(self):
        if False:
            while True:
                i = 10
        return self._get_connection()

    def get(self, key):
        if False:
            print('Hello World!')
        return self.connection.get(key).content

    def set(self, key, value):
        if False:
            print('Hello World!')
        if FMT_AUTO is not None:
            self.connection.upsert(key, value, ttl=self.expires, format=FMT_AUTO)
        else:
            self.connection.upsert(key, value, ttl=self.expires)

    def mget(self, keys):
        if False:
            return 10
        return self.connection.get_multi(keys)

    def delete(self, key):
        if False:
            while True:
                i = 10
        self.connection.remove(key)