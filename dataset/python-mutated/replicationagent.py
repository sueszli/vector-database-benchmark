import logging
from typing import Dict, Optional
from zope.interface import implementer
from twisted.internet import defer
from twisted.internet.endpoints import HostnameEndpoint, UNIXClientEndpoint, wrapClientTLS
from twisted.internet.interfaces import IStreamClientEndpoint
from twisted.python.failure import Failure
from twisted.web.client import URI, HTTPConnectionPool, _AgentBase
from twisted.web.error import SchemeNotSupported
from twisted.web.http_headers import Headers
from twisted.web.iweb import IAgent, IAgentEndpointFactory, IBodyProducer, IPolicyForHTTPS, IResponse
from synapse.config.workers import InstanceLocationConfig, InstanceTcpLocationConfig, InstanceUnixLocationConfig
from synapse.types import ISynapseReactor
logger = logging.getLogger(__name__)

@implementer(IAgentEndpointFactory)
class ReplicationEndpointFactory:
    """Connect to a given TCP or UNIX socket"""

    def __init__(self, reactor: ISynapseReactor, instance_map: Dict[str, InstanceLocationConfig], context_factory: IPolicyForHTTPS) -> None:
        if False:
            i = 10
            return i + 15
        self.reactor = reactor
        self.instance_map = instance_map
        self.context_factory = context_factory

    def endpointForURI(self, uri: URI) -> IStreamClientEndpoint:
        if False:
            for i in range(10):
                print('nop')
        '\n        This part of the factory decides what kind of endpoint is being connected to.\n\n        Args:\n            uri: The pre-parsed URI object containing all the uri data\n\n        Returns: The correct client endpoint object\n        '
        worker_name = uri.netloc.decode('utf-8')
        location_config = self.instance_map[worker_name]
        scheme = location_config.scheme()
        if isinstance(location_config, InstanceTcpLocationConfig):
            endpoint = HostnameEndpoint(self.reactor, location_config.host, location_config.port)
            if scheme == 'https':
                endpoint = wrapClientTLS(self.context_factory.creatorForNetloc(location_config.host.encode('utf-8'), location_config.port), endpoint)
            return endpoint
        elif isinstance(location_config, InstanceUnixLocationConfig):
            return UNIXClientEndpoint(self.reactor, location_config.path)
        else:
            raise SchemeNotSupported(f'Unsupported scheme: {scheme}')

@implementer(IAgent)
class ReplicationAgent(_AgentBase):
    """
    Client for connecting to replication endpoints via HTTP and HTTPS.

    Much of this code is copied from Twisted's twisted.web.client.Agent.
    """

    def __init__(self, reactor: ISynapseReactor, instance_map: Dict[str, InstanceLocationConfig], contextFactory: IPolicyForHTTPS, connectTimeout: Optional[float]=None, bindAddress: Optional[bytes]=None, pool: Optional[HTTPConnectionPool]=None):
        if False:
            while True:
                i = 10
        '\n        Create a ReplicationAgent.\n\n        Args:\n            reactor: A reactor for this Agent to place outgoing connections.\n            contextFactory: A factory for TLS contexts, to control the\n                verification parameters of OpenSSL.  The default is to use a\n                BrowserLikePolicyForHTTPS, so unless you have special\n                requirements you can leave this as-is.\n            connectTimeout: The amount of time that this Agent will wait\n                for the peer to accept a connection.\n            bindAddress: The local address for client sockets to bind to.\n            pool: An HTTPConnectionPool instance, or None, in which\n                case a non-persistent HTTPConnectionPool instance will be\n                created.\n        '
        _AgentBase.__init__(self, reactor, pool)
        endpoint_factory = ReplicationEndpointFactory(reactor, instance_map, contextFactory)
        self._endpointFactory = endpoint_factory

    def request(self, method: bytes, uri: bytes, headers: Optional[Headers]=None, bodyProducer: Optional[IBodyProducer]=None) -> 'defer.Deferred[IResponse]':
        if False:
            while True:
                i = 10
        "\n        Issue a request to the server indicated by the given uri.\n\n        An existing connection from the connection pool may be used or a new\n        one may be created.\n\n        Currently, HTTP, HTTPS and UNIX schemes are supported in uri.\n\n        This is copied from twisted.web.client.Agent, except:\n\n        * It uses a different pool key (combining the scheme with either host & port or\n          socket path).\n        * It does not call _ensureValidURI(...) as the strictness of IDNA2008 is not\n          required when using a worker's name as a 'hostname' for Synapse HTTP\n          Replication machinery. Specifically, this allows a range of ascii characters\n          such as '+' and '_' in hostnames/worker's names.\n\n        See: twisted.web.iweb.IAgent.request\n        "
        parsedURI = URI.fromBytes(uri)
        try:
            endpoint = self._endpointFactory.endpointForURI(parsedURI)
        except SchemeNotSupported:
            return defer.fail(Failure())
        worker_name = parsedURI.netloc.decode('utf-8')
        key_scheme = self._endpointFactory.instance_map[worker_name].scheme()
        key_netloc = self._endpointFactory.instance_map[worker_name].netloc()
        key = (key_scheme, key_netloc)
        return self._requestWithEndpoint(key, endpoint, method, parsedURI, headers, bodyProducer, parsedURI.originForm)