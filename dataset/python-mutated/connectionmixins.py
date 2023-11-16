"""
Various helpers for tests for connection-oriented transports.
"""
import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest

def findFreePort(interface='127.0.0.1', family=socket.AF_INET, type=socket.SOCK_STREAM):
    if False:
        while True:
            i = 10
    '\n    Ask the platform to allocate a free port on the specified interface, then\n    release the socket and return the address which was allocated.\n\n    @param interface: The local address to try to bind the port on.\n    @type interface: C{str}\n\n    @param type: The socket type which will use the resulting port.\n\n    @return: A two-tuple of address and port, like that returned by\n        L{socket.getsockname}.\n    '
    addr = socket.getaddrinfo(interface, 0)[0][4]
    probe = socket.socket(family, type)
    try:
        probe.bind(addr)
        if family == socket.AF_INET6:
            sockname = probe.getsockname()
            hostname = socket.getnameinfo(sockname, socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)[0]
            return (hostname, sockname[1])
        else:
            return probe.getsockname()
    finally:
        probe.close()

class ConnectableProtocol(Protocol):
    """
    A protocol to be used with L{runProtocolsWithReactor}.

    The protocol and its pair should eventually disconnect from each other.

    @ivar reactor: The reactor used in this test.

    @ivar disconnectReason: The L{Failure} passed to C{connectionLost}.

    @ivar _done: A L{Deferred} which will be fired when the connection is
        lost.
    """
    disconnectReason = None

    def _setAttributes(self, reactor, done):
        if False:
            print('Hello World!')
        '\n        Set attributes on the protocol that are known only externally; this\n        will be called by L{runProtocolsWithReactor} when this protocol is\n        instantiated.\n\n        @param reactor: The reactor used in this test.\n\n        @param done: A L{Deferred} which will be fired when the connection is\n           lost.\n        '
        self.reactor = reactor
        self._done = done

    def connectionLost(self, reason):
        if False:
            return 10
        self.disconnectReason = reason
        self._done.callback(None)
        del self._done

class EndpointCreator:
    """
    Create client and server endpoints that know how to connect to each other.
    """

    def server(self, reactor):
        if False:
            while True:
                i = 10
        '\n        Return an object providing C{IStreamServerEndpoint} for use in creating\n        a server to use to establish the connection type to be tested.\n        '
        raise NotImplementedError()

    def client(self, reactor, serverAddress):
        if False:
            while True:
                i = 10
        '\n        Return an object providing C{IStreamClientEndpoint} for use in creating\n        a client to use to establish the connection type to be tested.\n        '
        raise NotImplementedError()

class _SingleProtocolFactory(ClientFactory):
    """
    Factory to be used by L{runProtocolsWithReactor}.

    It always returns the same protocol (i.e. is intended for only a single
    connection).
    """

    def __init__(self, protocol):
        if False:
            while True:
                i = 10
        self._protocol = protocol

    def buildProtocol(self, addr):
        if False:
            i = 10
            return i + 15
        return self._protocol

def runProtocolsWithReactor(reactorBuilder, serverProtocol, clientProtocol, endpointCreator):
    if False:
        return 10
    '\n    Connect two protocols using endpoints and a new reactor instance.\n\n    A new reactor will be created and run, with the client and server protocol\n    instances connected to each other using the given endpoint creator. The\n    protocols should run through some set of tests, then disconnect; when both\n    have disconnected the reactor will be stopped and the function will\n    return.\n\n    @param reactorBuilder: A L{ReactorBuilder} instance.\n\n    @param serverProtocol: A L{ConnectableProtocol} that will be the server.\n\n    @param clientProtocol: A L{ConnectableProtocol} that will be the client.\n\n    @param endpointCreator: An instance of L{EndpointCreator}.\n\n    @return: The reactor run by this test.\n    '
    reactor = reactorBuilder.buildReactor()
    serverProtocol._setAttributes(reactor, Deferred())
    clientProtocol._setAttributes(reactor, Deferred())
    serverFactory = _SingleProtocolFactory(serverProtocol)
    clientFactory = _SingleProtocolFactory(clientProtocol)
    serverEndpoint = endpointCreator.server(reactor)
    d = serverEndpoint.listen(serverFactory)

    def gotPort(p):
        if False:
            return 10
        clientEndpoint = endpointCreator.client(reactor, p.getHost())
        return clientEndpoint.connect(clientFactory)
    d.addCallback(gotPort)

    def failed(result):
        if False:
            for i in range(10):
                print('nop')
        log.err(result, 'Connection setup failed.')
    disconnected = gatherResults([serverProtocol._done, clientProtocol._done])
    d.addCallback(lambda _: disconnected)
    d.addErrback(failed)
    d.addCallback(lambda _: needsRunningReactor(reactor, reactor.stop))
    reactorBuilder.runReactor(reactor)
    return reactor

def _getWriters(reactor):
    if False:
        for i in range(10):
            print('nop')
    '\n    Like L{IReactorFDSet.getWriters}, but with support for IOCP reactor as\n    well.\n    '
    if IReactorFDSet.providedBy(reactor):
        return reactor.getWriters()
    elif 'IOCP' in reactor.__class__.__name__:
        return reactor.handles
    else:
        raise Exception(f'Cannot find writers on {reactor!r}')

class _AcceptOneClient(ServerFactory):
    """
    This factory fires a L{Deferred} with a protocol instance shortly after it
    is constructed (hopefully long enough afterwards so that it has been
    connected to a transport).

    @ivar reactor: The reactor used to schedule the I{shortly}.

    @ivar result: A L{Deferred} which will be fired with the protocol instance.
    """

    def __init__(self, reactor, result):
        if False:
            i = 10
            return i + 15
        self.reactor = reactor
        self.result = result

    def buildProtocol(self, addr):
        if False:
            print('Hello World!')
        protocol = ServerFactory.buildProtocol(self, addr)
        self.reactor.callLater(0, self.result.callback, protocol)
        return protocol

class _SimplePullProducer:
    """
    A pull producer which writes one byte whenever it is resumed.  For use by
    C{test_unregisterProducerAfterDisconnect}.
    """

    def __init__(self, consumer):
        if False:
            i = 10
            return i + 15
        self.consumer = consumer

    def stopProducing(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def resumeProducing(self):
        if False:
            print('Hello World!')
        log.msg('Producer.resumeProducing')
        self.consumer.write(b'x')

class Stop(ClientFactory):
    """
    A client factory which stops a reactor when a connection attempt fails.
    """
    failReason = None

    def __init__(self, reactor):
        if False:
            for i in range(10):
                print('nop')
        self.reactor = reactor

    def clientConnectionFailed(self, connector, reason):
        if False:
            print('Hello World!')
        self.failReason = reason
        msg(f'Stop(CF) cCFailed: {reason.getErrorMessage()}')
        self.reactor.stop()

class ClosingLaterProtocol(ConnectableProtocol):
    """
    ClosingLaterProtocol exchanges one byte with its peer and then disconnects
    itself.  This is mostly a work-around for the fact that connectionMade is
    called before the SSL handshake has completed.
    """

    def __init__(self, onConnectionLost):
        if False:
            for i in range(10):
                print('nop')
        self.lostConnectionReason = None
        self.onConnectionLost = onConnectionLost

    def connectionMade(self):
        if False:
            print('Hello World!')
        msg('ClosingLaterProtocol.connectionMade')

    def dataReceived(self, bytes):
        if False:
            i = 10
            return i + 15
        msg(f'ClosingLaterProtocol.dataReceived {bytes!r}')
        self.transport.loseConnection()

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        msg('ClosingLaterProtocol.connectionLost')
        self.lostConnectionReason = reason
        self.onConnectionLost.callback(self)

class ConnectionTestsMixin:
    """
    This mixin defines test methods which should apply to most L{ITransport}
    implementations.
    """
    endpoints: Optional[EndpointCreator] = None

    def test_logPrefix(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Client and server transports implement L{ILoggingContext.logPrefix} to\n        return a message reflecting the protocol they are running.\n        '

        class CustomLogPrefixProtocol(ConnectableProtocol):

            def __init__(self, prefix):
                if False:
                    for i in range(10):
                        print('nop')
                self._prefix = prefix
                self.system = None

            def connectionMade(self):
                if False:
                    while True:
                        i = 10
                self.transport.write(b'a')

            def logPrefix(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self._prefix

            def dataReceived(self, bytes):
                if False:
                    print('Hello World!')
                self.system = context.get(ILogContext)['system']
                self.transport.write(b'b')
                if b'b' in bytes:
                    self.transport.loseConnection()
        client = CustomLogPrefixProtocol('Custom Client')
        server = CustomLogPrefixProtocol('Custom Server')
        runProtocolsWithReactor(self, server, client, self.endpoints)
        self.assertIn('Custom Client', client.system)
        self.assertIn('Custom Server', server.system)

    def test_writeAfterDisconnect(self):
        if False:
            i = 10
            return i + 15
        '\n        After a connection is disconnected, L{ITransport.write} and\n        L{ITransport.writeSequence} are no-ops.\n        '
        reactor = self.buildReactor()
        finished = []
        serverConnectionLostDeferred = Deferred()
        protocol = lambda : ClosingLaterProtocol(serverConnectionLostDeferred)
        portDeferred = self.endpoints.server(reactor).listen(ServerFactory.forProtocol(protocol))

        def listening(port):
            if False:
                for i in range(10):
                    print('nop')
            msg(f'Listening on {port.getHost()!r}')
            endpoint = self.endpoints.client(reactor, port.getHost())
            lostConnectionDeferred = Deferred()
            protocol = lambda : ClosingLaterProtocol(lostConnectionDeferred)
            client = endpoint.connect(ClientFactory.forProtocol(protocol))

            def write(proto):
                if False:
                    print('Hello World!')
                msg(f'About to write to {proto!r}')
                proto.transport.write(b'x')
            client.addCallbacks(write, lostConnectionDeferred.errback)

            def disconnected(proto):
                if False:
                    return 10
                msg(f'{proto!r} disconnected')
                proto.transport.write(b'some bytes to get lost')
                proto.transport.writeSequence([b'some', b'more'])
                finished.append(True)
            lostConnectionDeferred.addCallback(disconnected)
            serverConnectionLostDeferred.addCallback(disconnected)
            return gatherResults([lostConnectionDeferred, serverConnectionLostDeferred])

        def onListen():
            if False:
                print('Hello World!')
            portDeferred.addCallback(listening)
            portDeferred.addErrback(err)
            portDeferred.addCallback(lambda ignored: reactor.stop())
        needsRunningReactor(reactor, onListen)
        self.runReactor(reactor)
        self.assertEqual(finished, [True, True])

    def test_protocolGarbageAfterLostConnection(self):
        if False:
            i = 10
            return i + 15
        '\n        After the connection a protocol is being used for is closed, the\n        reactor discards all of its references to the protocol.\n        '
        lostConnectionDeferred = Deferred()
        clientProtocol = ClosingLaterProtocol(lostConnectionDeferred)
        clientRef = ref(clientProtocol)
        reactor = self.buildReactor()
        portDeferred = self.endpoints.server(reactor).listen(ServerFactory.forProtocol(Protocol))

        def listening(port):
            if False:
                return 10
            msg(f'Listening on {port.getHost()!r}')
            endpoint = self.endpoints.client(reactor, port.getHost())
            client = endpoint.connect(ClientFactory.forProtocol(lambda : clientProtocol))

            def disconnect(proto):
                if False:
                    while True:
                        i = 10
                msg(f'About to disconnect {proto!r}')
                proto.transport.loseConnection()
            client.addCallback(disconnect)
            client.addErrback(lostConnectionDeferred.errback)
            return lostConnectionDeferred

        def onListening():
            if False:
                print('Hello World!')
            portDeferred.addCallback(listening)
            portDeferred.addErrback(err)
            portDeferred.addBoth(lambda ignored: reactor.stop())
        needsRunningReactor(reactor, onListening)
        self.runReactor(reactor)
        clientProtocol = None
        collect()
        self.assertIsNone(clientRef())

class LogObserverMixin:
    """
    Mixin for L{TestCase} subclasses which want to observe log events.
    """

    def observe(self):
        if False:
            return 10
        loggedMessages = []
        log.addObserver(loggedMessages.append)
        self.addCleanup(log.removeObserver, loggedMessages.append)
        return loggedMessages

class BrokenContextFactory:
    """
    A context factory with a broken C{getContext} method, for exercising the
    error handling for such a case.
    """
    message = 'Some path was wrong maybe'

    def getContext(self):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError(self.message)

class StreamClientTestsMixin:
    """
    This mixin defines tests applicable to SOCK_STREAM client implementations.

    This must be mixed in to a L{ReactorBuilder
    <twisted.internet.test.reactormixins.ReactorBuilder>} subclass, as it
    depends on several of its methods.

    Then the methods C{connect} and C{listen} must defined, defining a client
    and a server communicating with each other.
    """

    def test_interface(self):
        if False:
            print('Hello World!')
        '\n        The C{connect} method returns an object providing L{IConnector}.\n        '
        reactor = self.buildReactor()
        connector = self.connect(reactor, ClientFactory())
        self.assertTrue(verifyObject(IConnector, connector))

    def test_clientConnectionFailedStopsReactor(self):
        if False:
            i = 10
            return i + 15
        "\n        The reactor can be stopped by a client factory's\n        C{clientConnectionFailed} method.\n        "
        reactor = self.buildReactor()
        needsRunningReactor(reactor, lambda : self.connect(reactor, Stop(reactor)))
        self.runReactor(reactor)

    def test_connectEvent(self):
        if False:
            print('Hello World!')
        '\n        This test checks that we correctly get notifications event for a\n        client.  This ought to prevent a regression under Windows using the\n        GTK2 reactor.  See #3925.\n        '
        reactor = self.buildReactor()
        self.listen(reactor, ServerFactory.forProtocol(Protocol))
        connected = []

        class CheckConnection(Protocol):

            def connectionMade(self):
                if False:
                    print('Hello World!')
                connected.append(self)
                reactor.stop()
        clientFactory = Stop(reactor)
        clientFactory.protocol = CheckConnection
        needsRunningReactor(reactor, lambda : self.connect(reactor, clientFactory))
        reactor.run()
        self.assertTrue(connected)

    def test_unregisterProducerAfterDisconnect(self):
        if False:
            i = 10
            return i + 15
        '\n        If a producer is unregistered from a transport after the transport has\n        been disconnected (by the peer) and after C{loseConnection} has been\n        called, the transport is not re-added to the reactor as a writer as\n        would be necessary if the transport were still connected.\n        '
        reactor = self.buildReactor()
        self.listen(reactor, ServerFactory.forProtocol(ClosingProtocol))
        finished = Deferred()
        finished.addErrback(log.err)
        finished.addCallback(lambda ign: reactor.stop())
        writing = []

        class ClientProtocol(Protocol):
            """
            Protocol to connect, register a producer, try to lose the
            connection, wait for the server to disconnect from us, and then
            unregister the producer.
            """

            def connectionMade(self):
                if False:
                    return 10
                log.msg('ClientProtocol.connectionMade')
                self.transport.registerProducer(_SimplePullProducer(self.transport), False)
                self.transport.loseConnection()

            def connectionLost(self, reason):
                if False:
                    print('Hello World!')
                log.msg('ClientProtocol.connectionLost')
                self.unregister()
                writing.append(self.transport in _getWriters(reactor))
                finished.callback(None)

            def unregister(self):
                if False:
                    i = 10
                    return i + 15
                log.msg('ClientProtocol unregister')
                self.transport.unregisterProducer()
        clientFactory = ClientFactory()
        clientFactory.protocol = ClientProtocol
        self.connect(reactor, clientFactory)
        self.runReactor(reactor)
        self.assertFalse(writing[0], 'Transport was writing after unregisterProducer.')

    def test_disconnectWhileProducing(self):
        if False:
            print('Hello World!')
        '\n        If C{loseConnection} is called while a producer is registered with the\n        transport, the connection is closed after the producer is unregistered.\n        '
        reactor = self.buildReactor()
        skippedReactors = ['Glib2Reactor', 'Gtk2Reactor']
        reactorClassName = reactor.__class__.__name__
        if reactorClassName in skippedReactors and platform.isWindows():
            raise SkipTest('A pygobject/pygtk bug disables this functionality on Windows.')

        class Producer:

            def resumeProducing(self):
                if False:
                    print('Hello World!')
                log.msg('Producer.resumeProducing')
        self.listen(reactor, ServerFactory.forProtocol(Protocol))
        finished = Deferred()
        finished.addErrback(log.err)
        finished.addCallback(lambda ign: reactor.stop())

        class ClientProtocol(Protocol):
            """
            Protocol to connect, register a producer, try to lose the
            connection, unregister the producer, and wait for the connection to
            actually be lost.
            """

            def connectionMade(self):
                if False:
                    print('Hello World!')
                log.msg('ClientProtocol.connectionMade')
                self.transport.registerProducer(Producer(), False)
                self.transport.loseConnection()
                reactor.callLater(0, reactor.callLater, 0, self.unregister)

            def unregister(self):
                if False:
                    return 10
                log.msg('ClientProtocol unregister')
                self.transport.unregisterProducer()
                reactor.callLater(1.0, finished.errback, Failure(Exception('Connection was not lost')))

            def connectionLost(self, reason):
                if False:
                    while True:
                        i = 10
                log.msg('ClientProtocol.connectionLost')
                finished.callback(None)
        clientFactory = ClientFactory()
        clientFactory.protocol = ClientProtocol
        self.connect(reactor, clientFactory)
        self.runReactor(reactor)