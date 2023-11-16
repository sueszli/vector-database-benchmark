"""
Helpers related to HTTP requests, used by tests.
"""
from __future__ import annotations
__all__ = ['DummyChannel', 'DummyRequest']
from io import BytesIO
from typing import Dict, List, Optional
from zope.interface import implementer, verify
from incremental import Version
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, ISSLTransport
from twisted.internet.task import Clock
from twisted.python.deprecate import deprecated
from twisted.trial import unittest
from twisted.web._responses import FOUND
from twisted.web.http_headers import Headers
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET, Session, Site
textLinearWhitespaceComponents = [f'Foo{lw}bar' for lw in ['\r', '\n', '\r\n']]
sanitizedText = 'Foo bar'
bytesLinearWhitespaceComponents = [component.encode('ascii') for component in textLinearWhitespaceComponents]
sanitizedBytes = sanitizedText.encode('ascii')

@implementer(IAddress)
class NullAddress:
    """
    A null implementation of L{IAddress}.
    """

class DummyChannel:

    class TCP:
        port = 80
        disconnected = False

        def __init__(self, peer=None):
            if False:
                while True:
                    i = 10
            if peer is None:
                peer = IPv4Address('TCP', '192.168.1.1', 12344)
            self._peer = peer
            self.written = BytesIO()
            self.producers = []

        def getPeer(self):
            if False:
                return 10
            return self._peer

        def write(self, data):
            if False:
                return 10
            if not isinstance(data, bytes):
                raise TypeError(f'Can only write bytes to a transport, not {data!r}')
            self.written.write(data)

        def writeSequence(self, iovec):
            if False:
                i = 10
                return i + 15
            for data in iovec:
                self.write(data)

        def getHost(self):
            if False:
                return 10
            return IPv4Address('TCP', '10.0.0.1', self.port)

        def registerProducer(self, producer, streaming):
            if False:
                return 10
            self.producers.append((producer, streaming))

        def unregisterProducer(self):
            if False:
                print('Hello World!')
            pass

        def loseConnection(self):
            if False:
                i = 10
                return i + 15
            self.disconnected = True

    @implementer(ISSLTransport)
    class SSL(TCP):

        def abortConnection(self):
            if False:
                i = 10
                return i + 15
            pass

        def getTcpKeepAlive(self):
            if False:
                return 10
            pass

        def getTcpNoDelay(self):
            if False:
                while True:
                    i = 10
            pass

        def loseWriteConnection(self):
            if False:
                i = 10
                return i + 15
            pass

        def setTcpKeepAlive(self, enabled):
            if False:
                while True:
                    i = 10
            pass

        def setTcpNoDelay(self, enabled):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def getPeerCertificate(self):
            if False:
                return 10
            pass
    site = Site(Resource())

    def __init__(self, peer=None):
        if False:
            return 10
        self.transport = self.TCP(peer)

    def requestDone(self, request):
        if False:
            while True:
                i = 10
        pass

    def writeHeaders(self, version, code, reason, headers):
        if False:
            i = 10
            return i + 15
        response_line = version + b' ' + code + b' ' + reason + b'\r\n'
        headerSequence = [response_line]
        headerSequence.extend((name + b': ' + value + b'\r\n' for (name, value) in headers))
        headerSequence.append(b'\r\n')
        self.transport.writeSequence(headerSequence)

    def getPeer(self):
        if False:
            i = 10
            return i + 15
        return self.transport.getPeer()

    def getHost(self):
        if False:
            print('Hello World!')
        return self.transport.getHost()

    def registerProducer(self, producer, streaming):
        if False:
            return 10
        self.transport.registerProducer(producer, streaming)

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        self.transport.unregisterProducer()

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.transport.write(data)

    def writeSequence(self, iovec):
        if False:
            for i in range(10):
                print('nop')
        self.transport.writeSequence(iovec)

    def loseConnection(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport.loseConnection()

    def endRequest(self):
        if False:
            while True:
                i = 10
        pass

    def isSecure(self):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(self.transport, self.SSL)

    def abortConnection(self):
        if False:
            i = 10
            return i + 15
        pass

    def getTcpKeepAlive(self):
        if False:
            print('Hello World!')
        pass

    def getTcpNoDelay(self):
        if False:
            while True:
                i = 10
        pass

    def loseWriteConnection(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def setTcpKeepAlive(self):
        if False:
            while True:
                i = 10
        pass

    def setTcpNoDelay(self):
        if False:
            print('Hello World!')
        pass

    def getPeerCertificate(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class DummyRequest:
    """
    Represents a dummy or fake request. See L{twisted.web.server.Request}.

    @ivar _finishedDeferreds: L{None} or a C{list} of L{Deferreds} which will
        be called back with L{None} when C{finish} is called or which will be
        errbacked if C{processingFailed} is called.

    @type requestheaders: C{Headers}
    @ivar requestheaders: A Headers instance that stores values for all request
        headers.

    @type responseHeaders: C{Headers}
    @ivar responseHeaders: A Headers instance that stores values for all
        response headers.

    @type responseCode: C{int}
    @ivar responseCode: The response code which was passed to
        C{setResponseCode}.

    @type written: C{list} of C{bytes}
    @ivar written: The bytes which have been written to the request.
    """
    uri = b'http://dummy/'
    method = b'GET'
    client: Optional[IAddress] = None
    sitepath: List[bytes]
    written: List[bytes]
    prepath: List[bytes]
    args: Dict[bytes, List[bytes]]
    _finishedDeferreds: List[Deferred[None]]

    def registerProducer(self, prod, s):
        if False:
            return 10
        "\n        Call an L{IPullProducer}'s C{resumeProducing} method in a\n        loop until it unregisters itself.\n\n        @param prod: The producer.\n        @type prod: L{IPullProducer}\n\n        @param s: Whether or not the producer is streaming.\n        "
        self.go = 1
        while self.go:
            prod.resumeProducing()

    def unregisterProducer(self):
        if False:
            return 10
        self.go = 0

    def __init__(self, postpath: list[bytes], session: Optional[Session]=None, client: Optional[IAddress]=None) -> None:
        if False:
            i = 10
            return i + 15
        self.sitepath = []
        self.written = []
        self.finished = 0
        self.postpath = postpath
        self.prepath = []
        self.session = None
        self.protoSession = session or Session(site=None, uid=b'0', reactor=Clock())
        self.args = {}
        self.requestHeaders = Headers()
        self.responseHeaders = Headers()
        self.responseCode = None
        self._finishedDeferreds = []
        self._serverName = b'dummy'
        self.clientproto = b'HTTP/1.0'

    def getAllHeaders(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return dictionary mapping the names of all received headers to the last\n        value received for each.\n\n        Since this method does not return all header information,\n        C{self.requestHeaders.getAllRawHeaders()} may be preferred.\n\n        NOTE: This function is a direct copy of\n        C{twisted.web.http.Request.getAllRawHeaders}.\n        '
        headers = {}
        for (k, v) in self.requestHeaders.getAllRawHeaders():
            headers[k.lower()] = v[-1]
        return headers

    def getHeader(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve the value of a request header.\n\n        @type name: C{bytes}\n        @param name: The name of the request header for which to retrieve the\n            value.  Header names are compared case-insensitively.\n\n        @rtype: C{bytes} or L{None}\n        @return: The value of the specified request header.\n        '
        return self.requestHeaders.getRawHeaders(name.lower(), [None])[0]

    def setHeader(self, name, value):
        if False:
            print('Hello World!')
        'TODO: make this assert on write() if the header is content-length'
        self.responseHeaders.addRawHeader(name, value)

    def getSession(self, sessionInterface=None):
        if False:
            print('Hello World!')
        if self.session:
            return self.session
        assert not self.written, 'Session cannot be requested after data has been written.'
        self.session = self.protoSession
        return self.session

    def render(self, resource):
        if False:
            return 10
        "\n        Render the given resource as a response to this request.\n\n        This implementation only handles a few of the most common behaviors of\n        resources.  It can handle a render method that returns a string or\n        C{NOT_DONE_YET}.  It doesn't know anything about the semantics of\n        request methods (eg HEAD) nor how to set any particular headers.\n        Basically, it's largely broken, but sufficient for some tests at least.\n        It should B{not} be expanded to do all the same stuff L{Request} does.\n        Instead, L{DummyRequest} should be phased out and L{Request} (or some\n        other real code factored in a different way) used.\n        "
        result = resource.render(self)
        if result is NOT_DONE_YET:
            return
        self.write(result)
        self.finish()

    def write(self, data):
        if False:
            print('Hello World!')
        if not isinstance(data, bytes):
            raise TypeError('write() only accepts bytes')
        self.written.append(data)

    def notifyFinish(self) -> Deferred[None]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a L{Deferred} which is called back with L{None} when the request\n        is finished.  This will probably only work if you haven't called\n        C{finish} yet.\n        "
        finished: Deferred[None] = Deferred()
        self._finishedDeferreds.append(finished)
        return finished

    def finish(self):
        if False:
            print('Hello World!')
        '\n        Record that the request is finished and callback and L{Deferred}s\n        waiting for notification of this.\n        '
        self.finished = self.finished + 1
        if self._finishedDeferreds is not None:
            observers = self._finishedDeferreds
            self._finishedDeferreds = None
            for obs in observers:
                obs.callback(None)

    def processingFailed(self, reason):
        if False:
            return 10
        '\n        Errback and L{Deferreds} waiting for finish notification.\n        '
        if self._finishedDeferreds is not None:
            observers = self._finishedDeferreds
            self._finishedDeferreds = None
            for obs in observers:
                obs.errback(reason)

    def addArg(self, name, value):
        if False:
            i = 10
            return i + 15
        self.args[name] = [value]

    def setResponseCode(self, code, message=None):
        if False:
            return 10
        '\n        Set the HTTP status response code, but takes care that this is called\n        before any data is written.\n        '
        assert not self.written, 'Response code cannot be set after data hasbeen written: {}.'.format('@@@@'.join(self.written))
        self.responseCode = code
        self.responseMessage = message

    def setLastModified(self, when):
        if False:
            print('Hello World!')
        assert not self.written, 'Last-Modified cannot be set after data has been written: {}.'.format('@@@@'.join(self.written))

    def setETag(self, tag):
        if False:
            while True:
                i = 10
        assert not self.written, 'ETag cannot be set after data has been written: {}.'.format('@@@@'.join(self.written))

    @deprecated(Version('Twisted', 18, 4, 0), replacement='getClientAddress')
    def getClientIP(self):
        if False:
            print('Hello World!')
        '\n        Return the IPv4 address of the client which made this request, if there\n        is one, otherwise L{None}.\n        '
        if isinstance(self.client, (IPv4Address, IPv6Address)):
            return self.client.host
        return None

    def getClientAddress(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the L{IAddress} of the client that made this request.\n\n        @return: an address.\n        @rtype: an L{IAddress} provider.\n        '
        if self.client is None:
            return NullAddress()
        return self.client

    def getRequestHostname(self):
        if False:
            print('Hello World!')
        '\n        Get a dummy hostname associated to the HTTP request.\n\n        @rtype: C{bytes}\n        @returns: a dummy hostname\n        '
        return self._serverName

    def getHost(self):
        if False:
            return 10
        "\n        Get a dummy transport's host.\n\n        @rtype: C{IPv4Address}\n        @returns: a dummy transport's host\n        "
        return IPv4Address('TCP', '127.0.0.1', 80)

    def setHost(self, host, port, ssl=0):
        if False:
            while True:
                i = 10
        "\n        Change the host and port the request thinks it's using.\n\n        @type host: C{bytes}\n        @param host: The value to which to change the host header.\n\n        @type ssl: C{bool}\n        @param ssl: A flag which, if C{True}, indicates that the request is\n            considered secure (if C{True}, L{isSecure} will return C{True}).\n        "
        self._forceSSL = ssl
        if self.isSecure():
            default = 443
        else:
            default = 80
        if port == default:
            hostHeader = host
        else:
            hostHeader = b'%b:%d' % (host, port)
        self.requestHeaders.addRawHeader(b'host', hostHeader)

    def redirect(self, url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Utility function that does a redirect.\n\n        The request should have finish() called after this.\n        '
        self.setResponseCode(FOUND)
        self.setHeader(b'location', url)

class DummyRequestTests(unittest.SynchronousTestCase):
    """
    Tests for L{DummyRequest}.
    """

    def test_getClientIPDeprecated(self):
        if False:
            while True:
                i = 10
        '\n        L{DummyRequest.getClientIP} is deprecated in favor of\n        L{DummyRequest.getClientAddress}\n        '
        request = DummyRequest([])
        request.getClientIP()
        warnings = self.flushWarnings(offendingFunctions=[self.test_getClientIPDeprecated])
        self.assertEqual(1, len(warnings))
        [warning] = warnings
        self.assertEqual(warning.get('category'), DeprecationWarning)
        self.assertEqual(warning.get('message'), 'twisted.web.test.requesthelper.DummyRequest.getClientIP was deprecated in Twisted 18.4.0; please use getClientAddress instead')

    def test_getClientIPSupportsIPv6(self):
        if False:
            return 10
        '\n        L{DummyRequest.getClientIP} supports IPv6 addresses, just like\n        L{twisted.web.http.Request.getClientIP}.\n        '
        request = DummyRequest([])
        client = IPv6Address('TCP', '::1', 12345)
        request.client = client
        self.assertEqual('::1', request.getClientIP())

    def test_getClientAddressWithoutClient(self):
        if False:
            print('Hello World!')
        '\n        L{DummyRequest.getClientAddress} returns an L{IAddress}\n        provider no C{client} has been set.\n        '
        request = DummyRequest([])
        null = request.getClientAddress()
        verify.verifyObject(IAddress, null)

    def test_getClientAddress(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{DummyRequest.getClientAddress} returns the C{client}.\n        '
        request = DummyRequest([])
        client = IPv4Address('TCP', '127.0.0.1', 12345)
        request.client = client
        address = request.getClientAddress()
        self.assertIs(address, client)