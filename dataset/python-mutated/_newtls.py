"""
This module implements memory BIO based TLS support.  It is the preferred
implementation and will be used whenever pyOpenSSL 0.10 or newer is installed
(whenever L{twisted.protocols.tls} is importable).

@since: 11.1
"""
from zope.interface import directlyProvides
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import ISSLTransport
from twisted.protocols.tls import TLSMemoryBIOFactory

class _BypassTLS:
    """
    L{_BypassTLS} is used as the transport object for the TLS protocol object
    used to implement C{startTLS}.  Its methods skip any TLS logic which
    C{startTLS} enables.

    @ivar _base: A transport class L{_BypassTLS} has been mixed in with to which
        methods will be forwarded.  This class is only responsible for sending
        bytes over the connection, not doing TLS.

    @ivar _connection: A L{Connection} which TLS has been started on which will
        be proxied to by this object.  Any method which has its behavior
        altered after C{startTLS} will be skipped in favor of the base class's
        implementation.  This allows the TLS protocol object to have direct
        access to the transport, necessary to actually implement TLS.
    """

    def __init__(self, base, connection):
        if False:
            i = 10
            return i + 15
        self._base = base
        self._connection = connection

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        '\n        Forward any extra attribute access to the original transport object.\n        For example, this exposes C{getHost}, the behavior of which does not\n        change after TLS is enabled.\n        '
        return getattr(self._connection, name)

    def write(self, data):
        if False:
            while True:
                i = 10
        '\n        Write some bytes directly to the connection.\n        '
        return self._base.write(self._connection, data)

    def writeSequence(self, iovec):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write a some bytes directly to the connection.\n        '
        return self._base.writeSequence(self._connection, iovec)

    def loseConnection(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close the underlying connection.\n        '
        return self._base.loseConnection(self._connection, *args, **kwargs)

    def registerProducer(self, producer, streaming):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register a producer with the underlying connection.\n        '
        return self._base.registerProducer(self._connection, producer, streaming)

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        '\n        Unregister a producer with the underlying connection.\n        '
        return self._base.unregisterProducer(self._connection)

def startTLS(transport, contextFactory, normal, bypass):
    if False:
        i = 10
        return i + 15
    '\n    Add a layer of SSL to a transport.\n\n    @param transport: The transport which will be modified.  This can either by\n        a L{FileDescriptor<twisted.internet.abstract.FileDescriptor>} or a\n        L{FileHandle<twisted.internet.iocpreactor.abstract.FileHandle>}.  The\n        actual requirements of this instance are that it have:\n\n          - a C{_tlsClientDefault} attribute indicating whether the transport is\n            a client (C{True}) or a server (C{False})\n          - a settable C{TLS} attribute which can be used to mark the fact\n            that SSL has been started\n          - settable C{getHandle} and C{getPeerCertificate} attributes so\n            these L{ISSLTransport} methods can be added to it\n          - a C{protocol} attribute referring to the L{IProtocol} currently\n            connected to the transport, which can also be set to a new\n            L{IProtocol} for the transport to deliver data to\n\n    @param contextFactory: An SSL context factory defining SSL parameters for\n        the new SSL layer.\n    @type contextFactory: L{twisted.internet.interfaces.IOpenSSLContextFactory}\n\n    @param normal: A flag indicating whether SSL will go in the same direction\n        as the underlying transport goes.  That is, if the SSL client will be\n        the underlying client and the SSL server will be the underlying server.\n        C{True} means it is the same, C{False} means they are switched.\n    @type normal: L{bool}\n\n    @param bypass: A transport base class to call methods on to bypass the new\n        SSL layer (so that the SSL layer itself can send its bytes).\n    @type bypass: L{type}\n    '
    if normal:
        client = transport._tlsClientDefault
    else:
        client = not transport._tlsClientDefault
    (producer, streaming) = (None, None)
    if transport.producer is not None:
        (producer, streaming) = (transport.producer, transport.streamingProducer)
        transport.unregisterProducer()
    tlsFactory = TLSMemoryBIOFactory(contextFactory, client, None)
    tlsProtocol = tlsFactory.protocol(tlsFactory, transport.protocol, False)
    transport.protocol = tlsProtocol
    transport.getHandle = tlsProtocol.getHandle
    transport.getPeerCertificate = tlsProtocol.getPeerCertificate
    directlyProvides(transport, ISSLTransport)
    transport.TLS = True
    transport.protocol.makeConnection(_BypassTLS(bypass, transport))
    if producer:
        transport.registerProducer(producer, streaming)

class ConnectionMixin:
    """
    A mixin for L{twisted.internet.abstract.FileDescriptor} which adds an
    L{ITLSTransport} implementation.

    @ivar TLS: A flag indicating whether TLS is currently in use on this
        transport.  This is not a good way for applications to check for TLS,
        instead use L{twisted.internet.interfaces.ISSLTransport}.
    """
    TLS = False

    def startTLS(self, ctx, normal=True):
        if False:
            print('Hello World!')
        '\n        @see: L{ITLSTransport.startTLS}\n        '
        startTLS(self, ctx, normal, FileDescriptor)

    def write(self, bytes):
        if False:
            print('Hello World!')
        '\n        Write some bytes to this connection, passing them through a TLS layer if\n        necessary, or discarding them if the connection has already been lost.\n        '
        if self.TLS:
            if self.connected:
                self.protocol.write(bytes)
        else:
            FileDescriptor.write(self, bytes)

    def writeSequence(self, iovec):
        if False:
            while True:
                i = 10
        '\n        Write some bytes to this connection, scatter/gather-style, passing them\n        through a TLS layer if necessary, or discarding them if the connection\n        has already been lost.\n        '
        if self.TLS:
            if self.connected:
                self.protocol.writeSequence(iovec)
        else:
            FileDescriptor.writeSequence(self, iovec)

    def loseConnection(self):
        if False:
            return 10
        '\n        Close this connection after writing all pending data.\n\n        If TLS has been negotiated, perform a TLS shutdown.\n        '
        if self.TLS:
            if self.connected and (not self.disconnecting):
                self.protocol.loseConnection()
        else:
            FileDescriptor.loseConnection(self)

    def registerProducer(self, producer, streaming):
        if False:
            return 10
        '\n        Register a producer.\n\n        If TLS is enabled, the TLS connection handles this.\n        '
        if self.TLS:
            self.protocol.registerProducer(producer, streaming)
        else:
            FileDescriptor.registerProducer(self, producer, streaming)

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        '\n        Unregister a producer.\n\n        If TLS is enabled, the TLS connection handles this.\n        '
        if self.TLS:
            self.protocol.unregisterProducer()
        else:
            FileDescriptor.unregisterProducer(self)

class ClientMixin:
    """
    A mixin for L{twisted.internet.tcp.Client} which just marks it as a client
    for the purposes of the default TLS handshake.

    @ivar _tlsClientDefault: Always C{True}, indicating that this is a client
        connection, and by default when TLS is negotiated this class will act as
        a TLS client.
    """
    _tlsClientDefault = True

class ServerMixin:
    """
    A mixin for L{twisted.internet.tcp.Server} which just marks it as a server
    for the purposes of the default TLS handshake.

    @ivar _tlsClientDefault: Always C{False}, indicating that this is a server
        connection, and by default when TLS is negotiated this class will act as
        a TLS server.
    """
    _tlsClientDefault = False