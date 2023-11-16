"""
Protocol wrapper that will detect hung connections.

In particular, since PB expects the server to talk first and HTTP
expects the client to talk first, when a PB client talks to an HTTP
server, neither side will talk, leading to a hung connection. This
wrapper will disconnect in that case, and inform the caller.
"""
from __future__ import absolute_import
from __future__ import print_function
from twisted.internet.interfaces import IProtocol
from twisted.internet.interfaces import IProtocolFactory
from twisted.python.components import proxyForInterface

def _noop():
    if False:
        while True:
            i = 10
    pass

class HangCheckProtocol(proxyForInterface(IProtocol, '_wrapped_protocol'), object):
    """
    Wrap a protocol, so the underlying connection will disconnect if
    the other end doesn't send data within a given timeout.
    """
    transport = None
    _hungConnectionTimer = None
    _HUNG_CONNECTION_TIMEOUT = 120

    def __init__(self, wrapped_protocol, hung_callback=_noop, reactor=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param IProtocol wrapped_protocol: The protocol to wrap.\n        :param hung_callback: Called when the connection has hung.\n        :type hung_callback: callable taking no arguments.\n        :param IReactorTime reactor: The reactor to use to schedule\n            the hang check.\n        '
        if reactor is None:
            from twisted.internet import reactor
        self._wrapped_protocol = wrapped_protocol
        self._reactor = reactor
        self._hung_callback = hung_callback

    def makeConnection(self, transport):
        if False:
            return 10
        self.transport = transport
        super(HangCheckProtocol, self).makeConnection(transport)
        self._startHungConnectionTimer()

    def dataReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        self._stopHungConnectionTimer()
        super(HangCheckProtocol, self).dataReceived(data)

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        self._stopHungConnectionTimer()
        super(HangCheckProtocol, self).connectionLost(reason)

    def _startHungConnectionTimer(self):
        if False:
            while True:
                i = 10
        '\n        Start a timer to detect if the connection is hung.\n        '

        def hungConnection():
            if False:
                i = 10
                return i + 15
            self._hung_callback()
            self._hungConnectionTimer = None
            self.transport.loseConnection()
        self._hungConnectionTimer = self._reactor.callLater(self._HUNG_CONNECTION_TIMEOUT, hungConnection)

    def _stopHungConnectionTimer(self):
        if False:
            return 10
        '\n        Cancel the hang check timer, since we have received data or\n        been closed.\n        '
        if self._hungConnectionTimer:
            self._hungConnectionTimer.cancel()
        self._hungConnectionTimer = None

class HangCheckFactory(proxyForInterface(IProtocolFactory, '_wrapped_factory'), object):
    """
    Wrap a protocol factory, so the underlying connection will
    disconnect if the other end doesn't send data within a given
    timeout.
    """

    def __init__(self, wrapped_factory, hung_callback):
        if False:
            i = 10
            return i + 15
        '\n        :param IProtocolFactory wrapped_factory: The factory to wrap.\n        :param hung_callback: Called when the connection has hung.\n        :type hung_callback: callable taking no arguments.\n        '
        self._wrapped_factory = wrapped_factory
        self._hung_callback = hung_callback

    def buildProtocol(self, addr):
        if False:
            while True:
                i = 10
        protocol = self._wrapped_factory.buildProtocol(addr)
        return HangCheckProtocol(protocol, hung_callback=self._hung_callback)

    def startedConnecting(self, connector):
        if False:
            while True:
                i = 10
        self._wrapped_factory.startedConnecting(connector)

    def clientConnectionFailed(self, connector, reason):
        if False:
            print('Hello World!')
        self._wrapped_factory.clientConnectionFailed(connector, reason)

    def clientConnectionLost(self, connector, reason):
        if False:
            for i in range(10):
                print('nop')
        self._wrapped_factory.clientConnectionLost(connector, reason)