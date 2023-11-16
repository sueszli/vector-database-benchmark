import copy
import xmlrpc.client as xmlrpc
from .errors import SAMPHubError
from .lockfile_helpers import get_main_running_hub
from .utils import ServerProxyPool
__all__ = ['SAMPHubProxy']

class SAMPHubProxy:
    """
    Proxy class to simplify the client interaction with a SAMP hub (via the
    standard profile).
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.proxy = None
        self._connected = False

    @property
    def is_connected(self):
        if False:
            print('Hello World!')
        '\n        Whether the hub proxy is currently connected to a hub.\n        '
        return self._connected

    def connect(self, hub=None, hub_params=None, pool_size=20):
        if False:
            print('Hello World!')
        '\n        Connect to the current SAMP Hub.\n\n        Parameters\n        ----------\n        hub : `~astropy.samp.SAMPHubServer`, optional\n            The hub to connect to.\n\n        hub_params : dict, optional\n            Optional dictionary containing the lock-file content of the Hub\n            with which to connect. This dictionary has the form\n            ``{<token-name>: <token-string>, ...}``.\n\n        pool_size : int, optional\n            The number of socket connections opened to communicate with the\n            Hub.\n        '
        self._connected = False
        self.lockfile = {}
        if hub is not None and hub_params is not None:
            raise ValueError('Cannot specify both hub and hub_params')
        if hub_params is None:
            if hub is not None:
                if not hub.is_running:
                    raise SAMPHubError('Hub is not running')
                else:
                    hub_params = hub.params
            else:
                hub_params = get_main_running_hub()
        try:
            url = hub_params['samp.hub.xmlrpc.url'].replace('\\', '')
            self.proxy = ServerProxyPool(pool_size, xmlrpc.ServerProxy, url, allow_none=1)
            self.ping()
            self.lockfile = copy.deepcopy(hub_params)
            self._connected = True
        except xmlrpc.ProtocolError as p:
            if p.errcode == 401:
                raise SAMPHubError('Unauthorized access. Basic Authentication required or failed.')
            else:
                raise SAMPHubError(f'Protocol Error {p.errcode}: {p.errmsg}')

    def disconnect(self):
        if False:
            while True:
                i = 10
        '\n        Disconnect from the current SAMP Hub.\n        '
        if self.proxy is not None:
            self.proxy.shutdown()
            self.proxy = None
        self._connected = False
        self.lockfile = {}

    @property
    def _samp_hub(self):
        if False:
            print('Hello World!')
        '\n        Property to abstract away the path to the hub, which allows this class\n        to be used for other profiles.\n        '
        return self.proxy.samp.hub

    def ping(self):
        if False:
            print('Hello World!')
        '\n        Proxy to ``ping`` SAMP Hub method (Standard Profile only).\n        '
        return self._samp_hub.ping()

    def set_xmlrpc_callback(self, private_key, xmlrpc_addr):
        if False:
            while True:
                i = 10
        '\n        Proxy to ``setXmlrpcCallback`` SAMP Hub method (Standard Profile only).\n        '
        return self._samp_hub.setXmlrpcCallback(private_key, xmlrpc_addr)

    def register(self, secret):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy to ``register`` SAMP Hub method.\n        '
        return self._samp_hub.register(secret)

    def unregister(self, private_key):
        if False:
            print('Hello World!')
        '\n        Proxy to ``unregister`` SAMP Hub method.\n        '
        return self._samp_hub.unregister(private_key)

    def declare_metadata(self, private_key, metadata):
        if False:
            print('Hello World!')
        '\n        Proxy to ``declareMetadata`` SAMP Hub method.\n        '
        return self._samp_hub.declareMetadata(private_key, metadata)

    def get_metadata(self, private_key, client_id):
        if False:
            return 10
        '\n        Proxy to ``getMetadata`` SAMP Hub method.\n        '
        return self._samp_hub.getMetadata(private_key, client_id)

    def declare_subscriptions(self, private_key, subscriptions):
        if False:
            print('Hello World!')
        '\n        Proxy to ``declareSubscriptions`` SAMP Hub method.\n        '
        return self._samp_hub.declareSubscriptions(private_key, subscriptions)

    def get_subscriptions(self, private_key, client_id):
        if False:
            print('Hello World!')
        '\n        Proxy to ``getSubscriptions`` SAMP Hub method.\n        '
        return self._samp_hub.getSubscriptions(private_key, client_id)

    def get_registered_clients(self, private_key):
        if False:
            while True:
                i = 10
        '\n        Proxy to ``getRegisteredClients`` SAMP Hub method.\n        '
        return self._samp_hub.getRegisteredClients(private_key)

    def get_subscribed_clients(self, private_key, mtype):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy to ``getSubscribedClients`` SAMP Hub method.\n        '
        return self._samp_hub.getSubscribedClients(private_key, mtype)

    def notify(self, private_key, recipient_id, message):
        if False:
            while True:
                i = 10
        '\n        Proxy to ``notify`` SAMP Hub method.\n        '
        return self._samp_hub.notify(private_key, recipient_id, message)

    def notify_all(self, private_key, message):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy to ``notifyAll`` SAMP Hub method.\n        '
        return self._samp_hub.notifyAll(private_key, message)

    def call(self, private_key, recipient_id, msg_tag, message):
        if False:
            i = 10
            return i + 15
        '\n        Proxy to ``call`` SAMP Hub method.\n        '
        return self._samp_hub.call(private_key, recipient_id, msg_tag, message)

    def call_all(self, private_key, msg_tag, message):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy to ``callAll`` SAMP Hub method.\n        '
        return self._samp_hub.callAll(private_key, msg_tag, message)

    def call_and_wait(self, private_key, recipient_id, message, timeout):
        if False:
            return 10
        '\n        Proxy to ``callAndWait`` SAMP Hub method.\n        '
        return self._samp_hub.callAndWait(private_key, recipient_id, message, timeout)

    def reply(self, private_key, msg_id, response):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy to ``reply`` SAMP Hub method.\n        '
        return self._samp_hub.reply(private_key, msg_id, response)