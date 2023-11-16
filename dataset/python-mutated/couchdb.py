"""CouchDB result store backend."""
from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import _parse_url
from celery.exceptions import ImproperlyConfigured
from .base import KeyValueStoreBackend
try:
    import pycouchdb
except ImportError:
    pycouchdb = None
__all__ = ('CouchBackend',)
ERR_LIB_MISSING = 'You need to install the pycouchdb library to use the CouchDB result backend'

class CouchBackend(KeyValueStoreBackend):
    """CouchDB backend.

    Raises:
        celery.exceptions.ImproperlyConfigured:
            if module :pypi:`pycouchdb` is not available.
    """
    container = 'default'
    scheme = 'http'
    host = 'localhost'
    port = 5984
    username = None
    password = None

    def __init__(self, url=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.url = url
        if pycouchdb is None:
            raise ImproperlyConfigured(ERR_LIB_MISSING)
        uscheme = uhost = uport = uname = upass = ucontainer = None
        if url:
            (_, uhost, uport, uname, upass, ucontainer, _) = _parse_url(url)
            ucontainer = ucontainer.strip('/') if ucontainer else None
        self.scheme = uscheme or self.scheme
        self.host = uhost or self.host
        self.port = int(uport or self.port)
        self.container = ucontainer or self.container
        self.username = uname or self.username
        self.password = upass or self.password
        self._connection = None

    def _get_connection(self):
        if False:
            while True:
                i = 10
        'Connect to the CouchDB server.'
        if self.username and self.password:
            conn_string = f'{self.scheme}://{self.username}:{self.password}@{self.host}:{self.port}'
            server = pycouchdb.Server(conn_string, authmethod='basic')
        else:
            conn_string = f'{self.scheme}://{self.host}:{self.port}'
            server = pycouchdb.Server(conn_string)
        try:
            return server.database(self.container)
        except pycouchdb.exceptions.NotFound:
            return server.create(self.container)

    @property
    def connection(self):
        if False:
            print('Hello World!')
        if self._connection is None:
            self._connection = self._get_connection()
        return self._connection

    def get(self, key):
        if False:
            print('Hello World!')
        key = bytes_to_str(key)
        try:
            return self.connection.get(key)['value']
        except pycouchdb.exceptions.NotFound:
            return None

    def set(self, key, value):
        if False:
            i = 10
            return i + 15
        key = bytes_to_str(key)
        data = {'_id': key, 'value': value}
        try:
            self.connection.save(data)
        except pycouchdb.exceptions.Conflict:
            data = self.connection.get(key)
            data['value'] = value
            self.connection.save(data)

    def mget(self, keys):
        if False:
            i = 10
            return i + 15
        return [self.get(key) for key in keys]

    def delete(self, key):
        if False:
            return 10
        self.connection.delete(key)