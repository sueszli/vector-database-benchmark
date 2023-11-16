import random
from zope.interface import implementer
from twisted.internet import error, interfaces
from twisted.names import client, dns
from twisted.names.error import DNSNameError
from twisted.python.compat import nativeString

class _SRVConnector_ClientFactoryWrapper:

    def __init__(self, connector, wrappedFactory):
        if False:
            for i in range(10):
                print('nop')
        self.__connector = connector
        self.__wrappedFactory = wrappedFactory

    def startedConnecting(self, connector):
        if False:
            print('Hello World!')
        self.__wrappedFactory.startedConnecting(self.__connector)

    def clientConnectionFailed(self, connector, reason):
        if False:
            i = 10
            return i + 15
        self.__connector.connectionFailed(reason)

    def clientConnectionLost(self, connector, reason):
        if False:
            while True:
                i = 10
        self.__connector.connectionLost(reason)

    def __getattr__(self, key):
        if False:
            i = 10
            return i + 15
        return getattr(self.__wrappedFactory, key)

@implementer(interfaces.IConnector)
class SRVConnector:
    """
    A connector that looks up DNS SRV records.

    RFC 2782 details how SRV records should be interpreted and selected
    for subsequent connection attempts. The algorithm for using the records'
    priority and weight is implemented in L{pickServer}.

    @ivar servers: List of candidate server records for future connection
        attempts.
    @type servers: L{list} of L{dns.Record_SRV}

    @ivar orderedServers: List of server records that have already been tried
        in this round of connection attempts.
    @type orderedServers: L{list} of L{dns.Record_SRV}
    """
    stopAfterDNS = 0

    def __init__(self, reactor, service, domain, factory, protocol='tcp', connectFuncName='connectTCP', connectFuncArgs=(), connectFuncKwArgs={}, defaultPort=None):
        if False:
            return 10
        '\n        @param domain: The domain to connect to.  If passed as a text\n            string, it will be encoded using C{idna} encoding.\n        @type domain: L{bytes} or L{str}\n\n        @param defaultPort: Optional default port number to be used when SRV\n            lookup fails and the service name is unknown. This should be the\n            port number associated with the service name as defined by the IANA\n            registry.\n        @type defaultPort: L{int}\n        '
        self.reactor = reactor
        self.service = service
        self.domain = None if domain is None else dns.domainString(domain)
        self.factory = factory
        self.protocol = protocol
        self.connectFuncName = connectFuncName
        self.connectFuncArgs = connectFuncArgs
        self.connectFuncKwArgs = connectFuncKwArgs
        self._defaultPort = defaultPort
        self.connector = None
        self.servers = None
        self.orderedServers = None

    def connect(self):
        if False:
            for i in range(10):
                print('nop')
        'Start connection to remote server.'
        self.factory.doStart()
        self.factory.startedConnecting(self)
        if not self.servers:
            if self.domain is None:
                self.connectionFailed(error.DNSLookupError('Domain is not defined.'))
                return
            d = client.lookupService('_%s._%s.%s' % (nativeString(self.service), nativeString(self.protocol), nativeString(self.domain)))
            d.addCallbacks(self._cbGotServers, self._ebGotServers)
            d.addCallback(lambda x, self=self: self._reallyConnect())
            if self._defaultPort:
                d.addErrback(self._ebServiceUnknown)
            d.addErrback(self.connectionFailed)
        elif self.connector is None:
            self._reallyConnect()
        else:
            self.connector.connect()

    def _ebGotServers(self, failure):
        if False:
            while True:
                i = 10
        failure.trap(DNSNameError)
        self.servers = []
        self.orderedServers = []

    def _cbGotServers(self, result):
        if False:
            i = 10
            return i + 15
        (answers, auth, add) = result
        if len(answers) == 1 and answers[0].type == dns.SRV and answers[0].payload and (answers[0].payload.target == dns.Name(b'.')):
            raise error.DNSLookupError('Service %s not available for domain %s.' % (repr(self.service), repr(self.domain)))
        self.servers = []
        self.orderedServers = []
        for a in answers:
            if a.type != dns.SRV or not a.payload:
                continue
            self.orderedServers.append(a.payload)

    def _ebServiceUnknown(self, failure):
        if False:
            print('Hello World!')
        '\n        Connect to the default port when the service name is unknown.\n\n        If no SRV records were found, the service name will be passed as the\n        port. If resolving the name fails with\n        L{error.ServiceNameUnknownError}, a final attempt is done using the\n        default port.\n        '
        failure.trap(error.ServiceNameUnknownError)
        self.servers = [dns.Record_SRV(0, 0, self._defaultPort, self.domain)]
        self.orderedServers = []
        self.connect()

    def pickServer(self):
        if False:
            return 10
        '\n        Pick the next server.\n\n        This selects the next server from the list of SRV records according\n        to their priority and weight values, as set out by the default\n        algorithm specified in RFC 2782.\n\n        At the beginning of a round, L{servers} is populated with\n        L{orderedServers}, and the latter is made empty. L{servers}\n        is the list of candidates, and L{orderedServers} is the list of servers\n        that have already been tried.\n\n        First, all records are ordered by priority and weight in ascending\n        order. Then for each priority level, a running sum is calculated\n        over the sorted list of records for that priority. Then a random value\n        between 0 and the final sum is compared to each record in order. The\n        first record that is greater than or equal to that random value is\n        chosen and removed from the list of candidates for this round.\n\n        @return: A tuple of target hostname and port from the chosen DNS SRV\n            record.\n        @rtype: L{tuple} of native L{str} and L{int}\n        '
        assert self.servers is not None
        assert self.orderedServers is not None
        if not self.servers and (not self.orderedServers):
            return (nativeString(self.domain), self.service)
        if not self.servers and self.orderedServers:
            self.servers = self.orderedServers
            self.orderedServers = []
        assert self.servers
        self.servers.sort(key=lambda record: (record.priority, record.weight))
        minPriority = self.servers[0].priority
        index = 0
        weightSum = 0
        weightIndex = []
        for x in self.servers:
            if x.priority == minPriority:
                weightSum += x.weight
                weightIndex.append((index, weightSum))
                index += 1
        rand = random.randint(0, weightSum)
        for (index, weight) in weightIndex:
            if weight >= rand:
                chosen = self.servers[index]
                del self.servers[index]
                self.orderedServers.append(chosen)
                return (str(chosen.target), chosen.port)
        raise RuntimeError(f'Impossible {self.__class__.__name__} pickServer result.')

    def _reallyConnect(self):
        if False:
            while True:
                i = 10
        if self.stopAfterDNS:
            self.stopAfterDNS = 0
            return
        (self.host, self.port) = self.pickServer()
        assert self.host is not None, 'Must have a host to connect to.'
        assert self.port is not None, 'Must have a port to connect to.'
        connectFunc = getattr(self.reactor, self.connectFuncName)
        self.connector = connectFunc(self.host, self.port, _SRVConnector_ClientFactoryWrapper(self, self.factory), *self.connectFuncArgs, **self.connectFuncKwArgs)

    def stopConnecting(self):
        if False:
            while True:
                i = 10
        'Stop attempting to connect.'
        if self.connector:
            self.connector.stopConnecting()
        else:
            self.stopAfterDNS = 1

    def disconnect(self):
        if False:
            return 10
        'Disconnect whatever our are state is.'
        if self.connector is not None:
            self.connector.disconnect()
        else:
            self.stopConnecting()

    def getDestination(self):
        if False:
            i = 10
            return i + 15
        assert self.connector
        return self.connector.getDestination()

    def connectionFailed(self, reason):
        if False:
            print('Hello World!')
        self.factory.clientConnectionFailed(self, reason)
        self.factory.doStop()

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        self.factory.clientConnectionLost(self, reason)
        self.factory.doStop()