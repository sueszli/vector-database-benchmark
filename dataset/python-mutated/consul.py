"""Consul result store backend.

- :class:`ConsulBackend` implements KeyValueStoreBackend to store results
    in the key-value store of Consul.
"""
from kombu.utils.encoding import bytes_to_str
from kombu.utils.url import parse_url
from celery.backends.base import KeyValueStoreBackend
from celery.exceptions import ImproperlyConfigured
from celery.utils.log import get_logger
try:
    import consul
except ImportError:
    consul = None
logger = get_logger(__name__)
__all__ = ('ConsulBackend',)
CONSUL_MISSING = 'You need to install the python-consul library in order to use the Consul result store backend.'

class ConsulBackend(KeyValueStoreBackend):
    """Consul.io K/V store backend for Celery."""
    consul = consul
    supports_autoexpire = True
    consistency = 'consistent'
    path = None

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        if self.consul is None:
            raise ImproperlyConfigured(CONSUL_MISSING)
        self.one_client = None
        self._init_from_params(**parse_url(self.url))

    def _init_from_params(self, hostname, port, virtual_host, **params):
        if False:
            while True:
                i = 10
        logger.debug('Setting on Consul client to connect to %s:%d', hostname, port)
        self.path = virtual_host
        self.hostname = hostname
        self.port = port
        if params.get('one_client', None):
            self.one_client = self.client()

    def client(self):
        if False:
            print('Hello World!')
        return self.one_client or consul.Consul(host=self.hostname, port=self.port, consistency=self.consistency)

    def _key_to_consul_key(self, key):
        if False:
            i = 10
            return i + 15
        key = bytes_to_str(key)
        return key if self.path is None else f'{self.path}/{key}'

    def get(self, key):
        if False:
            return 10
        key = self._key_to_consul_key(key)
        logger.debug('Trying to fetch key %s from Consul', key)
        try:
            (_, data) = self.client().kv.get(key)
            return data['Value']
        except TypeError:
            pass

    def mget(self, keys):
        if False:
            for i in range(10):
                print('nop')
        for key in keys:
            yield self.get(key)

    def set(self, key, value):
        if False:
            i = 10
            return i + 15
        "Set a key in Consul.\n\n        Before creating the key it will create a session inside Consul\n        where it creates a session with a TTL\n\n        The key created afterwards will reference to the session's ID.\n\n        If the session expires it will remove the key so that results\n        can auto expire from the K/V store\n        "
        session_name = bytes_to_str(key)
        key = self._key_to_consul_key(key)
        logger.debug('Trying to create Consul session %s with TTL %d', session_name, self.expires)
        client = self.client()
        session_id = client.session.create(name=session_name, behavior='delete', ttl=self.expires)
        logger.debug('Created Consul session %s', session_id)
        logger.debug('Writing key %s to Consul', key)
        return client.kv.put(key=key, value=value, acquire=session_id)

    def delete(self, key):
        if False:
            for i in range(10):
                print('nop')
        key = self._key_to_consul_key(key)
        logger.debug('Removing key %s from Consul', key)
        return self.client().kv.delete(key)