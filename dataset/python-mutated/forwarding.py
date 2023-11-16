"""
This module contains the implementation of the TCP forwarding, which allows
clients and servers to forward arbitrary TCP data across the connection.

Maintainer: Paul Swartz
"""
import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol

class SSHListenForwardingFactory(protocol.Factory):

    def __init__(self, connection, hostport, klass):
        if False:
            while True:
                i = 10
        self.conn = connection
        self.hostport = hostport
        self.klass = klass

    def buildProtocol(self, addr):
        if False:
            i = 10
            return i + 15
        channel = self.klass(conn=self.conn)
        client = SSHForwardingClient(channel)
        channel.client = client
        addrTuple = (addr.host, addr.port)
        channelOpenData = packOpen_direct_tcpip(self.hostport, addrTuple)
        self.conn.openChannel(channel, channelOpenData)
        return client

class SSHListenForwardingChannel(channel.SSHChannel):

    def channelOpen(self, specificData):
        if False:
            while True:
                i = 10
        self._log.info('opened forwarding channel {id}', id=self.id)
        if len(self.client.buf) > 1:
            b = self.client.buf[1:]
            self.write(b)
        self.client.buf = b''

    def openFailed(self, reason):
        if False:
            return 10
        self.closed()

    def dataReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.client.transport.write(data)

    def eofReceived(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.transport.loseConnection()

    def closed(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'client'):
            self._log.info('closing local forwarding channel {id}', id=self.id)
            self.client.transport.loseConnection()
            del self.client

class SSHListenClientForwardingChannel(SSHListenForwardingChannel):
    name = b'direct-tcpip'

class SSHListenServerForwardingChannel(SSHListenForwardingChannel):
    name = b'forwarded-tcpip'

class SSHConnectForwardingChannel(channel.SSHChannel):
    """
    Channel used for handling server side forwarding request.
    It acts as a client for the remote forwarding destination.

    @ivar hostport: C{(host, port)} requested by client as forwarding
        destination.
    @type hostport: L{tuple} or a C{sequence}

    @ivar client: Protocol connected to the forwarding destination.
    @type client: L{protocol.Protocol}

    @ivar clientBuf: Data received while forwarding channel is not yet
        connected.
    @type clientBuf: L{bytes}

    @var  _reactor: Reactor used for TCP connections.
    @type _reactor: A reactor.

    @ivar _channelOpenDeferred: Deferred used in testing to check the
        result of C{channelOpen}.
    @type _channelOpenDeferred: L{twisted.internet.defer.Deferred}
    """
    _reactor = reactor

    def __init__(self, hostport, *args, **kw):
        if False:
            print('Hello World!')
        channel.SSHChannel.__init__(self, *args, **kw)
        self.hostport = hostport
        self.client = None
        self.clientBuf = b''

    def channelOpen(self, specificData):
        if False:
            for i in range(10):
                print('nop')
        '\n        See: L{channel.SSHChannel}\n        '
        self._log.info('connecting to {host}:{port}', host=self.hostport[0], port=self.hostport[1])
        ep = HostnameEndpoint(self._reactor, self.hostport[0], self.hostport[1])
        d = connectProtocol(ep, SSHForwardingClient(self))
        d.addCallbacks(self._setClient, self._close)
        self._channelOpenDeferred = d

    def _setClient(self, client):
        if False:
            return 10
        '\n        Called when the connection was established to the forwarding\n        destination.\n\n        @param client: Client protocol connected to the forwarding destination.\n        @type  client: L{protocol.Protocol}\n        '
        self.client = client
        self._log.info('connected to {host}:{port}', host=self.hostport[0], port=self.hostport[1])
        if self.clientBuf:
            self.client.transport.write(self.clientBuf)
            self.clientBuf = None
        if self.client.buf[1:]:
            self.write(self.client.buf[1:])
        self.client.buf = b''

    def _close(self, reason):
        if False:
            return 10
        '\n        Called when failed to connect to the forwarding destination.\n\n        @param reason: Reason why connection failed.\n        @type  reason: L{twisted.python.failure.Failure}\n        '
        self._log.error('failed to connect to {host}:{port}: {reason}', host=self.hostport[0], port=self.hostport[1], reason=reason)
        self.loseConnection()

    def dataReceived(self, data):
        if False:
            i = 10
            return i + 15
        '\n        See: L{channel.SSHChannel}\n        '
        if self.client:
            self.client.transport.write(data)
        else:
            self.clientBuf += data

    def closed(self):
        if False:
            while True:
                i = 10
        '\n        See: L{channel.SSHChannel}\n        '
        if self.client:
            self._log.info('closed remote forwarding channel {id}', id=self.id)
            if self.client.channel:
                self.loseConnection()
            self.client.transport.loseConnection()
            del self.client

def openConnectForwardingClient(remoteWindow, remoteMaxPacket, data, avatar):
    if False:
        return 10
    (remoteHP, origHP) = unpackOpen_direct_tcpip(data)
    return SSHConnectForwardingChannel(remoteHP, remoteWindow=remoteWindow, remoteMaxPacket=remoteMaxPacket, avatar=avatar)

class SSHForwardingClient(protocol.Protocol):

    def __init__(self, channel):
        if False:
            print('Hello World!')
        self.channel = channel
        self.buf = b'\x00'

    def dataReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.buf:
            self.buf += data
        else:
            self.channel.write(data)

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        if self.channel:
            self.channel.loseConnection()
            self.channel = None

def packOpen_direct_tcpip(destination, source):
    if False:
        print('Hello World!')
    '\n    Pack the data suitable for sending in a CHANNEL_OPEN packet.\n\n    @type destination: L{tuple}\n    @param destination: A tuple of the (host, port) of the destination host.\n\n    @type source: L{tuple}\n    @param source: A tuple of the (host, port) of the source host.\n    '
    (connHost, connPort) = destination
    (origHost, origPort) = source
    if isinstance(connHost, str):
        connHost = connHost.encode('utf-8')
    if isinstance(origHost, str):
        origHost = origHost.encode('utf-8')
    conn = common.NS(connHost) + struct.pack('>L', connPort)
    orig = common.NS(origHost) + struct.pack('>L', origPort)
    return conn + orig
packOpen_forwarded_tcpip = packOpen_direct_tcpip

def unpackOpen_direct_tcpip(data):
    if False:
        for i in range(10):
            print('nop')
    'Unpack the data to a usable format.'
    (connHost, rest) = common.getNS(data)
    if isinstance(connHost, bytes):
        connHost = connHost.decode('utf-8')
    connPort = int(struct.unpack('>L', rest[:4])[0])
    (origHost, rest) = common.getNS(rest[4:])
    if isinstance(origHost, bytes):
        origHost = origHost.decode('utf-8')
    origPort = int(struct.unpack('>L', rest[:4])[0])
    return ((connHost, connPort), (origHost, origPort))
unpackOpen_forwarded_tcpip = unpackOpen_direct_tcpip

def packGlobal_tcpip_forward(peer):
    if False:
        while True:
            i = 10
    '\n    Pack the data for tcpip forwarding.\n\n    @param peer: A tuple of the (host, port) .\n    @type peer: L{tuple}\n    '
    (host, port) = peer
    return common.NS(host) + struct.pack('>L', port)

def unpackGlobal_tcpip_forward(data):
    if False:
        i = 10
        return i + 15
    (host, rest) = common.getNS(data)
    if isinstance(host, bytes):
        host = host.decode('utf-8')
    port = int(struct.unpack('>L', rest[:4])[0])
    return (host, port)
'This is how the data -> eof -> close stuff /should/ work.\n\ndebug3: channel 1: waiting for connection\ndebug1: channel 1: connected\ndebug1: channel 1: read<=0 rfd 7 len 0\ndebug1: channel 1: read failed\ndebug1: channel 1: close_read\ndebug1: channel 1: input open -> drain\ndebug1: channel 1: ibuf empty\ndebug1: channel 1: send eof\ndebug1: channel 1: input drain -> closed\ndebug1: channel 1: rcvd eof\ndebug1: channel 1: output open -> drain\ndebug1: channel 1: obuf empty\ndebug1: channel 1: close_write\ndebug1: channel 1: output drain -> closed\ndebug1: channel 1: rcvd close\ndebug3: channel 1: will not send data after close\ndebug1: channel 1: send close\ndebug1: channel 1: is dead\n'