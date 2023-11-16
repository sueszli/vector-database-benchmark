"""
Utilities and helpers for simulating a network
"""
import itertools
try:
    from OpenSSL.SSL import Error as NativeOpenSSLError
except ImportError:
    pass
from zope.interface import directlyProvides, implementer
from twisted.internet import error, interfaces
from twisted.internet.endpoints import TCP4ClientEndpoint, TCP4ServerEndpoint
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure

class TLSNegotiation:

    def __init__(self, obj, connectState):
        if False:
            return 10
        self.obj = obj
        self.connectState = connectState
        self.sent = False
        self.readyToSend = connectState

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'TLSNegotiation({self.obj!r})'

    def pretendToVerify(self, other, tpt):
        if False:
            for i in range(10):
                print('nop')
        if not self.obj.iosimVerify(other.obj):
            tpt.disconnectReason = NativeOpenSSLError()
            tpt.loseConnection()

@implementer(interfaces.IAddress)
class FakeAddress:
    """
    The default address type for the host and peer of L{FakeTransport}
    connections.
    """

@implementer(interfaces.ITransport, interfaces.ITLSTransport)
class FakeTransport:
    """
    A wrapper around a file-like object to make it behave as a Transport.

    This doesn't actually stream the file to the attached protocol,
    and is thus useful mainly as a utility for debugging protocols.
    """
    _nextserial = staticmethod(lambda counter=itertools.count(): int(next(counter)))
    closed = 0
    disconnecting = 0
    disconnected = 0
    disconnectReason = error.ConnectionDone('Connection done')
    producer = None
    streamingProducer = 0
    tls = None

    def __init__(self, protocol, isServer, hostAddress=None, peerAddress=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param protocol: This transport will deliver bytes to this protocol.\n        @type protocol: L{IProtocol} provider\n\n        @param isServer: C{True} if this is the accepting side of the\n            connection, C{False} if it is the connecting side.\n        @type isServer: L{bool}\n\n        @param hostAddress: The value to return from C{getHost}.  L{None}\n            results in a new L{FakeAddress} being created to use as the value.\n        @type hostAddress: L{IAddress} provider or L{None}\n\n        @param peerAddress: The value to return from C{getPeer}.  L{None}\n            results in a new L{FakeAddress} being created to use as the value.\n        @type peerAddress: L{IAddress} provider or L{None}\n        '
        self.protocol = protocol
        self.isServer = isServer
        self.stream = []
        self.serial = self._nextserial()
        if hostAddress is None:
            hostAddress = FakeAddress()
        self.hostAddress = hostAddress
        if peerAddress is None:
            peerAddress = FakeAddress()
        self.peerAddress = peerAddress

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'FakeTransport<{},{},{}>'.format(self.isServer and 'S' or 'C', self.serial, self.protocol.__class__.__name__)

    def write(self, data):
        if False:
            print('Hello World!')
        if self.disconnecting:
            return
        if self.tls is not None:
            self.tlsbuf.append(data)
        else:
            self.stream.append(data)

    def _checkProducer(self):
        if False:
            print('Hello World!')
        if self.producer and (not self.streamingProducer):
            self.producer.resumeProducing()

    def registerProducer(self, producer, streaming):
        if False:
            print('Hello World!')
        '\n        From abstract.FileDescriptor\n        '
        self.producer = producer
        self.streamingProducer = streaming
        if not streaming:
            producer.resumeProducing()

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        self.producer = None

    def stopConsuming(self):
        if False:
            i = 10
            return i + 15
        self.unregisterProducer()
        self.loseConnection()

    def writeSequence(self, iovec):
        if False:
            for i in range(10):
                print('nop')
        self.write(b''.join(iovec))

    def loseConnection(self):
        if False:
            i = 10
            return i + 15
        self.disconnecting = True

    def abortConnection(self):
        if False:
            while True:
                i = 10
        '\n        For the time being, this is the same as loseConnection; no buffered\n        data will be lost.\n        '
        self.disconnecting = True

    def reportDisconnect(self):
        if False:
            print('Hello World!')
        if self.tls is not None:
            err = NativeOpenSSLError()
        else:
            err = self.disconnectReason
        self.protocol.connectionLost(Failure(err))

    def logPrefix(self):
        if False:
            return 10
        '\n        Identify this transport/event source to the logging system.\n        '
        return 'iosim'

    def getPeer(self):
        if False:
            while True:
                i = 10
        return self.peerAddress

    def getHost(self):
        if False:
            while True:
                i = 10
        return self.hostAddress

    def resumeProducing(self):
        if False:
            while True:
                i = 10
        pass

    def pauseProducing(self):
        if False:
            return 10
        pass

    def stopProducing(self):
        if False:
            return 10
        self.loseConnection()

    def startTLS(self, contextFactory, beNormal=True):
        if False:
            i = 10
            return i + 15
        connectState = self.isServer ^ beNormal
        self.tls = TLSNegotiation(contextFactory, connectState)
        self.tlsbuf = []

    def getOutBuffer(self):
        if False:
            print('Hello World!')
        '\n        Get the pending writes from this transport, clearing them from the\n        pending buffer.\n\n        @return: the bytes written with C{transport.write}\n        @rtype: L{bytes}\n        '
        S = self.stream
        if S:
            self.stream = []
            return b''.join(S)
        elif self.tls is not None:
            if self.tls.readyToSend:
                self.tls.sent = True
                return self.tls
            else:
                return None
        else:
            return None

    def bufferReceived(self, buf):
        if False:
            i = 10
            return i + 15
        if isinstance(buf, TLSNegotiation):
            assert self.tls is not None
            if self.tls.sent:
                self.tls.pretendToVerify(buf, self)
                self.tls = None
                (b, self.tlsbuf) = (self.tlsbuf, None)
                self.writeSequence(b)
                directlyProvides(self, interfaces.ISSLTransport)
            else:
                self.tls.readyToSend = True
        else:
            self.protocol.dataReceived(buf)

    def getTcpKeepAlive(self):
        if False:
            i = 10
            return i + 15
        pass

    def getTcpNoDelay(self):
        if False:
            while True:
                i = 10
        pass

    def loseWriteConnection(self):
        if False:
            print('Hello World!')
        pass

    def setTcpKeepAlive(self, enabled):
        if False:
            return 10
        pass

    def setTcpNoDelay(self, enabled):
        if False:
            while True:
                i = 10
        pass

def makeFakeClient(clientProtocol):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create and return a new in-memory transport hooked up to the given protocol.\n\n    @param clientProtocol: The client protocol to use.\n    @type clientProtocol: L{IProtocol} provider\n\n    @return: The transport.\n    @rtype: L{FakeTransport}\n    '
    return FakeTransport(clientProtocol, isServer=False)

def makeFakeServer(serverProtocol):
    if False:
        print('Hello World!')
    '\n    Create and return a new in-memory transport hooked up to the given protocol.\n\n    @param serverProtocol: The server protocol to use.\n    @type serverProtocol: L{IProtocol} provider\n\n    @return: The transport.\n    @rtype: L{FakeTransport}\n    '
    return FakeTransport(serverProtocol, isServer=True)

class IOPump:
    """
    Utility to pump data between clients and servers for protocol testing.

    Perhaps this is a utility worthy of being in protocol.py?
    """

    def __init__(self, client, server, clientIO, serverIO, debug, clock=None):
        if False:
            for i in range(10):
                print('nop')
        self.client = client
        self.server = server
        self.clientIO = clientIO
        self.serverIO = serverIO
        self.debug = debug
        if clock is None:
            clock = MemoryReactorClock()
        self.clock = clock

    def flush(self, debug=False, advanceClock=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Pump until there is no more input or output.\n\n        Returns whether any data was moved.\n        '
        result = False
        for _ in range(1000):
            if self.pump(debug, advanceClock):
                result = True
            else:
                break
        else:
            assert 0, 'Too long'
        return result

    def pump(self, debug=False, advanceClock=True):
        if False:
            return 10
        '\n        Move data back and forth, while also triggering any currently pending\n        scheduled calls (i.e. C{callLater(0, f)}).\n\n        Returns whether any data was moved.\n        '
        if advanceClock:
            self.clock.advance(0)
        if self.debug or debug:
            print('-- GLUG --')
        sData = self.serverIO.getOutBuffer()
        cData = self.clientIO.getOutBuffer()
        self.clientIO._checkProducer()
        self.serverIO._checkProducer()
        if self.debug or debug:
            print('.')
            if cData:
                print('C: ' + repr(cData))
            if sData:
                print('S: ' + repr(sData))
        if cData:
            self.serverIO.bufferReceived(cData)
        if sData:
            self.clientIO.bufferReceived(sData)
        if cData or sData:
            return True
        if self.serverIO.disconnecting and (not self.serverIO.disconnected):
            if self.debug or debug:
                print('* C')
            self.serverIO.disconnected = True
            self.clientIO.disconnecting = True
            self.clientIO.reportDisconnect()
            return True
        if self.clientIO.disconnecting and (not self.clientIO.disconnected):
            if self.debug or debug:
                print('* S')
            self.clientIO.disconnected = True
            self.serverIO.disconnecting = True
            self.serverIO.reportDisconnect()
            return True
        return False

def connect(serverProtocol, serverTransport, clientProtocol, clientTransport, debug=False, greet=True, clock=None):
    if False:
        i = 10
        return i + 15
    '\n    Create a new L{IOPump} connecting two protocols.\n\n    @param serverProtocol: The protocol to use on the accepting side of the\n        connection.\n    @type serverProtocol: L{IProtocol} provider\n\n    @param serverTransport: The transport to associate with C{serverProtocol}.\n    @type serverTransport: L{FakeTransport}\n\n    @param clientProtocol: The protocol to use on the initiating side of the\n        connection.\n    @type clientProtocol: L{IProtocol} provider\n\n    @param clientTransport: The transport to associate with C{clientProtocol}.\n    @type clientTransport: L{FakeTransport}\n\n    @param debug: A flag indicating whether to log information about what the\n        L{IOPump} is doing.\n    @type debug: L{bool}\n\n    @param greet: Should the L{IOPump} be L{flushed <IOPump.flush>} once before\n        returning to put the protocols into their post-handshake or\n        post-server-greeting state?\n    @type greet: L{bool}\n\n    @param clock: An optional L{Clock}. Pumping the resulting L{IOPump} will\n        also increase clock time by a small increment.\n\n    @return: An L{IOPump} which connects C{serverProtocol} and\n        C{clientProtocol} and delivers bytes between them when it is pumped.\n    @rtype: L{IOPump}\n    '
    serverProtocol.makeConnection(serverTransport)
    clientProtocol.makeConnection(clientTransport)
    pump = IOPump(clientProtocol, serverProtocol, clientTransport, serverTransport, debug, clock=clock)
    if greet:
        pump.flush()
    return pump

def connectedServerAndClient(ServerClass, ClientClass, clientTransportFactory=makeFakeClient, serverTransportFactory=makeFakeServer, debug=False, greet=True, clock=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Connect a given server and client class to each other.\n\n    @param ServerClass: a callable that produces the server-side protocol.\n    @type ServerClass: 0-argument callable returning L{IProtocol} provider.\n\n    @param ClientClass: like C{ServerClass} but for the other side of the\n        connection.\n    @type ClientClass: 0-argument callable returning L{IProtocol} provider.\n\n    @param clientTransportFactory: a callable that produces the transport which\n        will be attached to the protocol returned from C{ClientClass}.\n    @type clientTransportFactory: callable taking (L{IProtocol}) and returning\n        L{FakeTransport}\n\n    @param serverTransportFactory: a callable that produces the transport which\n        will be attached to the protocol returned from C{ServerClass}.\n    @type serverTransportFactory: callable taking (L{IProtocol}) and returning\n        L{FakeTransport}\n\n    @param debug: Should this dump an escaped version of all traffic on this\n        connection to stdout for inspection?\n    @type debug: L{bool}\n\n    @param greet: Should the L{IOPump} be L{flushed <IOPump.flush>} once before\n        returning to put the protocols into their post-handshake or\n        post-server-greeting state?\n    @type greet: L{bool}\n\n    @param clock: An optional L{Clock}. Pumping the resulting L{IOPump} will\n        also increase clock time by a small increment.\n\n    @return: the client protocol, the server protocol, and an L{IOPump} which,\n        when its C{pump} and C{flush} methods are called, will move data\n        between the created client and server protocol instances.\n    @rtype: 3-L{tuple} of L{IProtocol}, L{IProtocol}, L{IOPump}\n    '
    c = ClientClass()
    s = ServerClass()
    cio = clientTransportFactory(c)
    sio = serverTransportFactory(s)
    return (c, s, connect(s, sio, c, cio, debug, greet, clock=clock))

def _factoriesShouldConnect(clientInfo, serverInfo):
    if False:
        while True:
            i = 10
    "\n    Should the client and server described by the arguments be connected to\n    each other, i.e. do their port numbers match?\n\n    @param clientInfo: the args for connectTCP\n    @type clientInfo: L{tuple}\n\n    @param serverInfo: the args for listenTCP\n    @type serverInfo: L{tuple}\n\n    @return: If they do match, return factories for the client and server that\n        should connect; otherwise return L{None}, indicating they shouldn't be\n        connected.\n    @rtype: L{None} or 2-L{tuple} of (L{ClientFactory},\n        L{IProtocolFactory})\n    "
    (clientHost, clientPort, clientFactory, clientTimeout, clientBindAddress) = clientInfo
    (serverPort, serverFactory, serverBacklog, serverInterface) = serverInfo
    if serverPort == clientPort:
        return (clientFactory, serverFactory)
    else:
        return None

class ConnectionCompleter:
    """
    A L{ConnectionCompleter} can cause synthetic TCP connections established by
    L{MemoryReactor.connectTCP} and L{MemoryReactor.listenTCP} to succeed or
    fail.
    """

    def __init__(self, memoryReactor):
        if False:
            print('Hello World!')
        '\n        Create a L{ConnectionCompleter} from a L{MemoryReactor}.\n\n        @param memoryReactor: The reactor to attach to.\n        @type memoryReactor: L{MemoryReactor}\n        '
        self._reactor = memoryReactor

    def succeedOnce(self, debug=False):
        if False:
            i = 10
            return i + 15
        "\n        Complete a single TCP connection established on this\n        L{ConnectionCompleter}'s L{MemoryReactor}.\n\n        @param debug: A flag; whether to dump output from the established\n            connection to stdout.\n        @type debug: L{bool}\n\n        @return: a pump for the connection, or L{None} if no connection could\n            be established.\n        @rtype: L{IOPump} or L{None}\n        "
        memoryReactor = self._reactor
        for (clientIdx, clientInfo) in enumerate(memoryReactor.tcpClients):
            for serverInfo in memoryReactor.tcpServers:
                factories = _factoriesShouldConnect(clientInfo, serverInfo)
                if factories:
                    memoryReactor.tcpClients.remove(clientInfo)
                    memoryReactor.connectors.pop(clientIdx)
                    (clientFactory, serverFactory) = factories
                    clientProtocol = clientFactory.buildProtocol(None)
                    serverProtocol = serverFactory.buildProtocol(None)
                    serverTransport = makeFakeServer(serverProtocol)
                    clientTransport = makeFakeClient(clientProtocol)
                    return connect(serverProtocol, serverTransport, clientProtocol, clientTransport, debug)

    def failOnce(self, reason=Failure(ConnectionRefusedError())):
        if False:
            for i in range(10):
                print('nop')
        "\n        Fail a single TCP connection established on this\n        L{ConnectionCompleter}'s L{MemoryReactor}.\n\n        @param reason: the reason to provide that the connection failed.\n        @type reason: L{Failure}\n        "
        self._reactor.tcpClients.pop(0)[2].clientConnectionFailed(self._reactor.connectors.pop(0), reason)

def connectableEndpoint(debug=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create an endpoint that can be fired on demand.\n\n    @param debug: A flag; whether to dump output from the established\n        connection to stdout.\n    @type debug: L{bool}\n\n    @return: A client endpoint, and an object that will cause one of the\n        L{Deferred}s returned by that client endpoint.\n    @rtype: 2-L{tuple} of (L{IStreamClientEndpoint}, L{ConnectionCompleter})\n    '
    reactor = MemoryReactorClock()
    clientEndpoint = TCP4ClientEndpoint(reactor, '0.0.0.0', 4321)
    serverEndpoint = TCP4ServerEndpoint(reactor, 4321)
    serverEndpoint.listen(Factory.forProtocol(Protocol))
    return (clientEndpoint, ConnectionCompleter(reactor))