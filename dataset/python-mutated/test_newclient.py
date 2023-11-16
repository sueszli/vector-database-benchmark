"""
Tests for L{twisted.web._newclient}.
"""
from typing import Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError, Deferred, fail, succeed
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol, EventLoggingObserver, StringTransport, StringTransportWithDisconnection
from twisted.logger import globalLogPublisher
from twisted.protocols.basic import LineReceiver
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from twisted.web._newclient import BODY, DONE, HEADER, STATUS, UNKNOWN_LENGTH, BadHeaders, BadResponseVersion, ChunkedEncoder, ConnectionAborted, ExcessWrite, HTTPClientParser, HTTPParser, LengthEnforcingConsumer, ParseError, RequestNotSent, TransportProxyProducer, WrongBodyLength, makeStatefulDispatcher
from twisted.web.client import HTTP11ClientProtocol, PotentialDataLoss, Request, RequestGenerationFailed, RequestTransmissionFailed, Response, ResponseDone, ResponseFailed, ResponseNeverReceived
from twisted.web.http import _DataLoss
from twisted.web.http_headers import Headers
from twisted.web.iweb import IBodyProducer, IResponse
from twisted.web.test.requesthelper import bytesLinearWhitespaceComponents, sanitizedBytes

class ArbitraryException(Exception):
    """
    A unique, arbitrary exception type which L{twisted.web._newclient} knows
    nothing about.
    """

class AnotherArbitraryException(Exception):
    """
    Similar to L{ArbitraryException} but with a different identity.
    """
_boringHeaders = Headers({b'host': [b'example.com']})

def assertWrapperExceptionTypes(self, deferred, mainType, reasonTypes):
    if False:
        i = 10
        return i + 15
    "\n    Assert that the given L{Deferred} fails with the exception given by\n    C{mainType} and that the exceptions wrapped by the instance of C{mainType}\n    it fails with match the list of exception types given by C{reasonTypes}.\n\n    This is a helper for testing failures of exceptions which subclass\n    L{_newclient._WrapperException}.\n\n    @param self: A L{TestCase} instance which will be used to make the\n        assertions.\n\n    @param deferred: The L{Deferred} which is expected to fail with\n        C{mainType}.\n\n    @param mainType: A L{_newclient._WrapperException} subclass which will be\n        trapped on C{deferred}.\n\n    @param reasonTypes: A sequence of exception types which will be trapped on\n        the resulting C{mainType} exception instance's C{reasons} sequence.\n\n    @return: A L{Deferred} which fires with the C{mainType} instance\n        C{deferred} fails with, or which fails somehow.\n    "

    def cbFailed(err):
        if False:
            while True:
                i = 10
        for (reason, type) in zip(err.reasons, reasonTypes):
            reason.trap(type)
        self.assertEqual(len(err.reasons), len(reasonTypes), f'len({err.reasons}) != len({reasonTypes})')
        return err
    d = self.assertFailure(deferred, mainType)
    d.addCallback(cbFailed)
    return d

def assertResponseFailed(self, deferred, reasonTypes):
    if False:
        for i in range(10):
            print('nop')
    '\n    A simple helper to invoke L{assertWrapperExceptionTypes} with a C{mainType}\n    of L{ResponseFailed}.\n    '
    return assertWrapperExceptionTypes(self, deferred, ResponseFailed, reasonTypes)

def assertRequestGenerationFailed(self, deferred, reasonTypes):
    if False:
        while True:
            i = 10
    '\n    A simple helper to invoke L{assertWrapperExceptionTypes} with a C{mainType}\n    of L{RequestGenerationFailed}.\n    '
    return assertWrapperExceptionTypes(self, deferred, RequestGenerationFailed, reasonTypes)

def assertRequestTransmissionFailed(self, deferred, reasonTypes):
    if False:
        for i in range(10):
            print('nop')
    '\n    A simple helper to invoke L{assertWrapperExceptionTypes} with a C{mainType}\n    of L{RequestTransmissionFailed}.\n    '
    return assertWrapperExceptionTypes(self, deferred, RequestTransmissionFailed, reasonTypes)

def justTransportResponse(transport):
    if False:
        print('Hello World!')
    "\n    Helper function for creating a Response which uses the given transport.\n    All of the other parameters to L{Response.__init__} are filled with\n    arbitrary values.  Only use this method if you don't care about any of\n    them.\n    "
    return Response((b'HTTP', 1, 1), 200, b'OK', _boringHeaders, transport)

class MakeStatefulDispatcherTests(TestCase):
    """
    Tests for L{makeStatefulDispatcher}.
    """

    def test_functionCalledByState(self):
        if False:
            return 10
        '\n        A method defined with L{makeStatefulDispatcher} invokes a second\n        method based on the current state of the object.\n        '

        class Foo:
            _state = 'A'

            def bar(self):
                if False:
                    i = 10
                    return i + 15
                pass
            bar = makeStatefulDispatcher('quux', bar)

            def _quux_A(self):
                if False:
                    i = 10
                    return i + 15
                return 'a'

            def _quux_B(self):
                if False:
                    i = 10
                    return i + 15
                return 'b'
        stateful = Foo()
        self.assertEqual(stateful.bar(), 'a')
        stateful._state = 'B'
        self.assertEqual(stateful.bar(), 'b')
        stateful._state = 'C'
        self.assertRaises(RuntimeError, stateful.bar)

class _HTTPParserTests:
    """
    Base test class for L{HTTPParser} which is responsible for the bulk of
    the task of parsing HTTP bytes.
    """
    sep: Optional[bytes] = None

    def test_statusCallback(self):
        if False:
            i = 10
            return i + 15
        '\n        L{HTTPParser} calls its C{statusReceived} method when it receives a\n        status line.\n        '
        status = []
        protocol = HTTPParser()
        protocol.statusReceived = status.append
        protocol.makeConnection(StringTransport())
        self.assertEqual(protocol.state, STATUS)
        protocol.dataReceived(b'HTTP/1.1 200 OK' + self.sep)
        self.assertEqual(status, [b'HTTP/1.1 200 OK'])
        self.assertEqual(protocol.state, HEADER)

    def _headerTestSetup(self):
        if False:
            for i in range(10):
                print('nop')
        header = {}
        protocol = HTTPParser()
        protocol.headerReceived = header.__setitem__
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK' + self.sep)
        return (header, protocol)

    def test_headerCallback(self):
        if False:
            return 10
        '\n        L{HTTPParser} calls its C{headerReceived} method when it receives a\n        header.\n        '
        (header, protocol) = self._headerTestSetup()
        protocol.dataReceived(b'X-Foo:bar' + self.sep)
        protocol.dataReceived(self.sep)
        self.assertEqual(header, {b'X-Foo': b'bar'})
        self.assertEqual(protocol.state, BODY)

    def test_continuedHeaderCallback(self):
        if False:
            print('Hello World!')
        '\n        If a header is split over multiple lines, L{HTTPParser} calls\n        C{headerReceived} with the entire value once it is received.\n        '
        (header, protocol) = self._headerTestSetup()
        protocol.dataReceived(b'X-Foo: bar' + self.sep)
        protocol.dataReceived(b' baz' + self.sep)
        protocol.dataReceived(b'\tquux' + self.sep)
        protocol.dataReceived(self.sep)
        self.assertEqual(header, {b'X-Foo': b'bar baz\tquux'})
        self.assertEqual(protocol.state, BODY)

    def test_fieldContentWhitespace(self):
        if False:
            i = 10
            return i + 15
        '\n        Leading and trailing linear whitespace is stripped from the header\n        value passed to the C{headerReceived} callback.\n        '
        (header, protocol) = self._headerTestSetup()
        value = self.sep.join([b' \t ', b' bar \t', b' \t', b''])
        protocol.dataReceived(b'X-Bar:' + value)
        protocol.dataReceived(b'X-Foo:' + value)
        protocol.dataReceived(self.sep)
        self.assertEqual(header, {b'X-Foo': b'bar', b'X-Bar': b'bar'})

    def test_allHeadersCallback(self):
        if False:
            while True:
                i = 10
        '\n        After the last header is received, L{HTTPParser} calls\n        C{allHeadersReceived}.\n        '
        called = []
        (header, protocol) = self._headerTestSetup()

        def allHeadersReceived():
            if False:
                for i in range(10):
                    print('nop')
            called.append(protocol.state)
            protocol.state = STATUS
        protocol.allHeadersReceived = allHeadersReceived
        protocol.dataReceived(self.sep)
        self.assertEqual(called, [HEADER])
        self.assertEqual(protocol.state, STATUS)

    def test_noHeaderCallback(self):
        if False:
            return 10
        '\n        If there are no headers in the message, L{HTTPParser} does not call\n        C{headerReceived}.\n        '
        (header, protocol) = self._headerTestSetup()
        protocol.dataReceived(self.sep)
        self.assertEqual(header, {})
        self.assertEqual(protocol.state, BODY)

    def test_headersSavedOnResponse(self):
        if False:
            return 10
        '\n        All headers received by L{HTTPParser} are added to\n        L{HTTPParser.headers}.\n        '
        protocol = HTTPParser()
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK' + self.sep)
        protocol.dataReceived(b'X-Foo: bar' + self.sep)
        protocol.dataReceived(b'X-Foo: baz' + self.sep)
        protocol.dataReceived(self.sep)
        expected = [(b'X-Foo', [b'bar', b'baz'])]
        self.assertEqual(expected, list(protocol.headers.getAllRawHeaders()))

    def test_connectionControlHeaders(self):
        if False:
            i = 10
            return i + 15
        '\n        L{HTTPParser.isConnectionControlHeader} returns C{True} for headers\n        which are always connection control headers (similar to "hop-by-hop"\n        headers from RFC 2616 section 13.5.1) and C{False} for other headers.\n        '
        protocol = HTTPParser()
        connHeaderNames = [b'content-length', b'connection', b'keep-alive', b'te', b'trailers', b'transfer-encoding', b'upgrade', b'proxy-connection']
        for header in connHeaderNames:
            self.assertTrue(protocol.isConnectionControlHeader(header), "Expecting %r to be a connection control header, but wasn't" % (header,))
        self.assertFalse(protocol.isConnectionControlHeader(b'date'), "Expecting the arbitrarily selected 'date' header to not be a connection control header, but was.")

    def test_switchToBodyMode(self):
        if False:
            while True:
                i = 10
        '\n        L{HTTPParser.switchToBodyMode} raises L{RuntimeError} if called more\n        than once.\n        '
        protocol = HTTPParser()
        protocol.makeConnection(StringTransport())
        protocol.switchToBodyMode(object())
        self.assertRaises(RuntimeError, protocol.switchToBodyMode, object())

class HTTPParserRFCComplaintDelimeterTests(_HTTPParserTests, TestCase):
    """
    L{_HTTPParserTests} using standard CR LF newlines.
    """
    sep = b'\r\n'

class HTTPParserNonRFCComplaintDelimeterTests(_HTTPParserTests, TestCase):
    """
    L{_HTTPParserTests} using bare LF newlines.
    """
    sep = b'\n'

class HTTPClientParserTests(TestCase):
    """
    Tests for L{HTTPClientParser} which is responsible for parsing HTTP
    response messages.
    """

    def test_parseVersion(self):
        if False:
            while True:
                i = 10
        '\n        L{HTTPClientParser.parseVersion} parses a status line into its three\n        components.\n        '
        protocol = HTTPClientParser(None, None)
        self.assertEqual(protocol.parseVersion(b'CANDY/7.2'), (b'CANDY', 7, 2))

    def test_parseBadVersion(self):
        if False:
            i = 10
            return i + 15
        '\n        L{HTTPClientParser.parseVersion} raises L{ValueError} when passed an\n        unparsable version.\n        '
        protocol = HTTPClientParser(None, None)
        e = BadResponseVersion
        f = protocol.parseVersion

        def checkParsing(s):
            if False:
                for i in range(10):
                    print('nop')
            exc = self.assertRaises(e, f, s)
            self.assertEqual(exc.data, s)
        checkParsing(b'foo')
        checkParsing(b'foo/bar/baz')
        checkParsing(b'foo/')
        checkParsing(b'foo/..')
        checkParsing(b'foo/a.b')
        checkParsing(b'foo/-1.-1')

    def test_responseStatusParsing(self):
        if False:
            while True:
                i = 10
        '\n        L{HTTPClientParser.statusReceived} parses the version, code, and phrase\n        from the status line and stores them on the response object.\n        '
        request = Request(b'GET', b'/', _boringHeaders, None)
        protocol = HTTPClientParser(request, None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        self.assertEqual(protocol.response.version, (b'HTTP', 1, 1))
        self.assertEqual(protocol.response.code, 200)
        self.assertEqual(protocol.response.phrase, b'OK')

    def test_responseStatusWithoutPhrase(self):
        if False:
            return 10
        '\n        L{HTTPClientParser.statusReceived} can parse a status line without a\n        phrase (though such lines are a violation of RFC 7230, section 3.1.2;\n        nevertheless some broken servers omit the phrase).\n        '
        request = Request(b'GET', b'/', _boringHeaders, None)
        protocol = HTTPClientParser(request, None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200\r\n')
        self.assertEqual(protocol.response.version, (b'HTTP', 1, 1))
        self.assertEqual(protocol.response.code, 200)
        self.assertEqual(protocol.response.phrase, b'')

    def test_badResponseStatus(self):
        if False:
            i = 10
            return i + 15
        '\n        L{HTTPClientParser.statusReceived} raises L{ParseError} if it is called\n        with a status line which cannot be parsed.\n        '
        protocol = HTTPClientParser(None, None)

        def checkParsing(s):
            if False:
                print('Hello World!')
            exc = self.assertRaises(ParseError, protocol.statusReceived, s)
            self.assertEqual(exc.data, s)
        checkParsing(b'foo')
        checkParsing(b'HTTP/1.1 bar OK')

    def _noBodyTest(self, request, status, response):
        if False:
            i = 10
            return i + 15
        '\n        Assert that L{HTTPClientParser} parses the given C{response} to\n        C{request}, resulting in a response with no body and no extra bytes and\n        leaving the transport in the producing state.\n\n        @param request: A L{Request} instance which might have caused a server\n            to return the given response.\n        @param status: A string giving the status line of the response to be\n            parsed.\n        @param response: A string giving the response to be parsed.\n\n        @return: A C{dict} of headers from the response.\n        '
        header = {}
        finished = []
        body = []
        bodyDataFinished = []
        protocol = HTTPClientParser(request, finished.append)
        protocol.headerReceived = header.__setitem__
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(status)
        protocol.response._bodyDataReceived = body.append
        protocol.response._bodyDataFinished = lambda : bodyDataFinished.append(True)
        protocol.dataReceived(response)
        self.assertEqual(transport.producerState, 'producing')
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(body, [])
        self.assertEqual(finished, [b''])
        self.assertEqual(bodyDataFinished, [True])
        self.assertEqual(protocol.response.length, 0)
        return header

    def test_headResponse(self):
        if False:
            print('Hello World!')
        '\n        If the response is to a HEAD request, no body is expected, the body\n        callback is not invoked, and the I{Content-Length} header is passed to\n        the header callback.\n        '
        request = Request(b'HEAD', b'/', _boringHeaders, None)
        status = b'HTTP/1.1 200 OK\r\n'
        response = b'Content-Length: 10\r\n\r\n'
        header = self._noBodyTest(request, status, response)
        self.assertEqual(header, {b'Content-Length': b'10'})

    def test_noContentResponse(self):
        if False:
            i = 10
            return i + 15
        '\n        If the response code is I{NO CONTENT} (204), no body is expected and\n        the body callback is not invoked.\n        '
        request = Request(b'GET', b'/', _boringHeaders, None)
        status = b'HTTP/1.1 204 NO CONTENT\r\n'
        response = b'\r\n'
        self._noBodyTest(request, status, response)

    def test_notModifiedResponse(self):
        if False:
            print('Hello World!')
        '\n        If the response code is I{NOT MODIFIED} (304), no body is expected and\n        the body callback is not invoked.\n        '
        request = Request(b'GET', b'/', _boringHeaders, None)
        status = b'HTTP/1.1 304 NOT MODIFIED\r\n'
        response = b'\r\n'
        self._noBodyTest(request, status, response)

    def test_responseHeaders(self):
        if False:
            while True:
                i = 10
        "\n        The response headers are added to the response object's C{headers}\n        L{Headers} instance.\n        "
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda rest: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'X-Foo: bar\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.connHeaders, Headers({}))
        self.assertEqual(protocol.response.headers, Headers({b'x-foo': [b'bar']}))
        self.assertIdentical(protocol.response.length, UNKNOWN_LENGTH)

    def test_responseHeadersMultiline(self):
        if False:
            return 10
        "\n        The multi-line response headers are folded and added to the response\n        object's C{headers} L{Headers} instance.\n        "
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda rest: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'X-Multiline: a\r\n')
        protocol.dataReceived(b'    b\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.connHeaders, Headers({}))
        self.assertEqual(protocol.response.headers, Headers({b'x-multiline': [b'a    b']}))
        self.assertIdentical(protocol.response.length, UNKNOWN_LENGTH)

    def test_connectionHeaders(self):
        if False:
            return 10
        "\n        The connection control headers are added to the parser's C{connHeaders}\n        L{Headers} instance.\n        "
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda rest: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'Content-Length: 123\r\n')
        protocol.dataReceived(b'Connection: close\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.response.headers, Headers({}))
        self.assertEqual(protocol.connHeaders, Headers({b'content-length': [b'123'], b'connection': [b'close']}))
        self.assertEqual(protocol.response.length, 123)

    def test_headResponseContentLengthEntityHeader(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If a HEAD request is made, the I{Content-Length} header in the response\n        is added to the response headers, not the connection control headers.\n        '
        protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), lambda rest: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'Content-Length: 123\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.response.headers, Headers({b'content-length': [b'123']}))
        self.assertEqual(protocol.connHeaders, Headers({}))
        self.assertEqual(protocol.response.length, 0)

    def test_contentLength(self):
        if False:
            while True:
                i = 10
        '\n        If a response includes a body with a length given by the\n        I{Content-Length} header, the bytes which make up the body are passed\n        to the C{_bodyDataReceived} callback on the L{HTTPParser}.\n        '
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'Content-Length: 10\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(transport.producerState, 'paused')
        self.assertEqual(protocol.state, BODY)
        protocol.dataReceived(b'x' * 6)
        self.assertEqual(body, [b'x' * 6])
        self.assertEqual(protocol.state, BODY)
        protocol.dataReceived(b'y' * 4)
        self.assertEqual(body, [b'x' * 6, b'y' * 4])
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(finished, [b''])

    def test_zeroContentLength(self):
        if False:
            while True:
                i = 10
        '\n        If a response includes a I{Content-Length} header indicating zero bytes\n        in the response, L{Response.length} is set accordingly and no data is\n        delivered to L{Response._bodyDataReceived}.\n        '
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'Content-Length: 0\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(body, [])
        self.assertEqual(finished, [b''])
        self.assertEqual(protocol.response.length, 0)

    def test_multipleContentLengthHeaders(self):
        if False:
            print('Hello World!')
        '\n        If a response includes multiple I{Content-Length} headers,\n        L{HTTPClientParser.dataReceived} raises L{ValueError} to indicate that\n        the response is invalid and the transport is now unusable.\n        '
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), None)
        protocol.makeConnection(StringTransport())
        self.assertRaises(ValueError, protocol.dataReceived, b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\n')

    def test_extraBytesPassedBack(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If extra bytes are received past the end of a response, they are passed\n        to the finish callback.\n        '
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'Content-Length: 0\r\n')
        protocol.dataReceived(b'\r\nHere is another thing!')
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(finished, [b'Here is another thing!'])

    def test_extraBytesPassedBackHEAD(self):
        if False:
            return 10
        '\n        If extra bytes are received past the end of the headers of a response\n        to a HEAD request, they are passed to the finish callback.\n        '
        finished = []
        protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), finished.append)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        protocol.dataReceived(b'Content-Length: 12\r\n')
        protocol.dataReceived(b'\r\nHere is another thing!')
        self.assertEqual(protocol.state, DONE)
        self.assertEqual(finished, [b'Here is another thing!'])

    def test_chunkedResponseBody(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the response headers indicate the response body is encoded with the\n        I{chunked} transfer encoding, the body is decoded according to that\n        transfer encoding before being passed to L{Response._bodyDataReceived}.\n        '
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'Transfer-Encoding: chunked\r\n')
        protocol.dataReceived(b'\r\n')
        self.assertEqual(body, [])
        self.assertIdentical(protocol.response.length, UNKNOWN_LENGTH)
        protocol.dataReceived(b'3\r\na')
        self.assertEqual(body, [b'a'])
        protocol.dataReceived(b'bc\r\n')
        self.assertEqual(body, [b'a', b'bc'])
        protocol.dataReceived(b'0\r\n\r\nextra')
        self.assertEqual(finished, [b'extra'])

    def test_unknownContentLength(self):
        if False:
            while True:
                i = 10
        '\n        If a response does not include a I{Transfer-Encoding} or a\n        I{Content-Length}, the end of response body is indicated by the\n        connection being closed.\n        '
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'\r\n')
        protocol.dataReceived(b'foo')
        protocol.dataReceived(b'bar')
        self.assertEqual(body, [b'foo', b'bar'])
        protocol.connectionLost(ConnectionDone('simulated end of connection'))
        self.assertEqual(finished, [b''])

    def test_contentLengthAndTransferEncoding(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        According to RFC 2616, section 4.4, point 3, if I{Content-Length} and\n        I{Transfer-Encoding: chunked} are present, I{Content-Length} MUST be\n        ignored\n        '
        finished = []
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), finished.append)
        transport = StringTransport()
        protocol.makeConnection(transport)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        body = []
        protocol.response._bodyDataReceived = body.append
        protocol.dataReceived(b'Content-Length: 102\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n')
        self.assertEqual(body, [b'abc'])
        self.assertEqual(finished, [b''])

    def test_connectionLostBeforeBody(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If L{HTTPClientParser.connectionLost} is called before the headers are\n        finished, the C{_responseDeferred} is fired with the L{Failure} passed\n        to C{connectionLost}.\n        '
        transport = StringTransport()
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), None)
        protocol.makeConnection(transport)
        responseDeferred = protocol._responseDeferred
        protocol.connectionLost(Failure(ArbitraryException()))
        return assertResponseFailed(self, responseDeferred, [ArbitraryException])

    def test_connectionLostWithError(self):
        if False:
            while True:
                i = 10
        '\n        If one of the L{Response} methods called by\n        L{HTTPClientParser.connectionLost} raises an exception, the exception\n        is logged and not re-raised.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        transport = StringTransport()
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), None)
        protocol.makeConnection(transport)
        response = []
        protocol._responseDeferred.addCallback(response.append)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n')
        response = response[0]

        def fakeBodyDataFinished(err=None):
            if False:
                i = 10
                return i + 15
            raise ArbitraryException()
        response._bodyDataFinished = fakeBodyDataFinished
        protocol.connectionLost(None)
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, ArbitraryException)
        self.flushLoggedErrors(ArbitraryException)

    def test_noResponseAtAll(self):
        if False:
            i = 10
            return i + 15
        '\n        If no response at all was received and the connection is lost, the\n        resulting error is L{ResponseNeverReceived}.\n        '
        protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), lambda ign: None)
        d = protocol._responseDeferred
        protocol.makeConnection(StringTransport())
        protocol.connectionLost(ConnectionLost())
        return self.assertFailure(d, ResponseNeverReceived)

    def test_someResponseButNotAll(self):
        if False:
            while True:
                i = 10
        '\n        If a partial response was received and the connection is lost, the\n        resulting error is L{ResponseFailed}, but not\n        L{ResponseNeverReceived}.\n        '
        protocol = HTTPClientParser(Request(b'HEAD', b'/', _boringHeaders, None), lambda ign: None)
        d = protocol._responseDeferred
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(b'2')
        protocol.connectionLost(ConnectionLost())
        return self.assertFailure(d, ResponseFailed).addCallback(self.assertIsInstance, ResponseFailed)

    def test_1XXResponseIsSwallowed(self):
        if False:
            return 10
        '\n        If a response in the 1XX range is received it just gets swallowed and\n        the parser resets itself.\n        '
        sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(sample103Response)
        self.assertTrue(getattr(protocol, 'response', None) is None)
        self.assertEqual(protocol.state, STATUS)
        self.assertEqual(len(list(protocol.headers.getAllRawHeaders())), 0)
        self.assertEqual(len(list(protocol.connHeaders.getAllRawHeaders())), 0)
        self.assertTrue(protocol._everReceivedData)

    def test_1XXFollowedByFinalResponseOnlyEmitsFinal(self):
        if False:
            i = 10
            return i + 15
        '\n        When a 1XX response is swallowed, the final response that follows it is\n        the only one that gets sent to the application.\n        '
        sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
        following200Response = b'HTTP/1.1 200 OK\r\nContent-Length: 123\r\n\r\n'
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(sample103Response + following200Response)
        self.assertEqual(protocol.response.code, 200)
        self.assertEqual(protocol.response.headers, Headers({}))
        self.assertEqual(protocol.connHeaders, Headers({b'content-length': [b'123']}))
        self.assertEqual(protocol.response.length, 123)

    def test_multiple1XXResponsesAreIgnored(self):
        if False:
            print('Hello World!')
        '\n        It is acceptable for multiple 1XX responses to come through, all of\n        which get ignored.\n        '
        sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
        following200Response = b'HTTP/1.1 200 OK\r\nContent-Length: 123\r\n\r\n'
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(sample103Response + sample103Response + sample103Response + following200Response)
        self.assertEqual(protocol.response.code, 200)
        self.assertEqual(protocol.response.headers, Headers({}))
        self.assertEqual(protocol.connHeaders, Headers({b'content-length': [b'123']}))
        self.assertEqual(protocol.response.length, 123)

    def test_ignored1XXResponseCausesLog(self):
        if False:
            i = 10
            return i + 15
        '\n        When a 1XX response is ignored, Twisted emits a log.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        sample103Response = b'HTTP/1.1 103 Early Hints\r\nServer: socketserver/1.0.0\r\nLink: </other/styles.css>; rel=preload; as=style\r\nLink: </other/action.js>; rel=preload; as=script\r\n\r\n'
        protocol = HTTPClientParser(Request(b'GET', b'/', _boringHeaders, None), lambda ign: None)
        protocol.makeConnection(StringTransport())
        protocol.dataReceived(sample103Response)
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        self.assertEquals(event['log_format'], 'Ignoring unexpected {code} response')
        self.assertEquals(event['code'], 103)

class SlowRequest:
    """
    L{SlowRequest} is a fake implementation of L{Request} which is easily
    controlled externally (for example, by code in a test method).

    @ivar stopped: A flag indicating whether C{stopWriting} has been called.

    @ivar finished: After C{writeTo} is called, a L{Deferred} which was
        returned by that method.  L{SlowRequest} will never fire this
        L{Deferred}.
    """
    method = b'GET'
    stopped = False
    persistent = False

    def writeTo(self, transport):
        if False:
            while True:
                i = 10
        self.finished = Deferred()
        return self.finished

    def stopWriting(self):
        if False:
            i = 10
            return i + 15
        self.stopped = True

class SimpleRequest:
    """
    L{SimpleRequest} is a fake implementation of L{Request} which writes a
    short, fixed string to the transport passed to its C{writeTo} method and
    returns a succeeded L{Deferred}.  This vaguely emulates the behavior of a
    L{Request} with no body producer.
    """
    persistent = False

    def writeTo(self, transport):
        if False:
            print('Hello World!')
        transport.write(b'SOME BYTES')
        return succeed(None)

class HTTP11ClientProtocolTests(TestCase):
    """
    Tests for the HTTP 1.1 client protocol implementation,
    L{HTTP11ClientProtocol}.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an L{HTTP11ClientProtocol} connected to a fake transport.\n        '
        self.transport = StringTransport()
        self.protocol = HTTP11ClientProtocol()
        self.protocol.makeConnection(self.transport)

    def test_request(self):
        if False:
            print('Hello World!')
        '\n        L{HTTP11ClientProtocol.request} accepts a L{Request} and calls its\n        C{writeTo} method with its own transport.\n        '
        self.protocol.request(SimpleRequest())
        self.assertEqual(self.transport.value(), b'SOME BYTES')

    def test_secondRequest(self):
        if False:
            print('Hello World!')
        '\n        The second time L{HTTP11ClientProtocol.request} is called, it returns a\n        L{Deferred} which immediately fires with a L{Failure} wrapping a\n        L{RequestNotSent} exception.\n        '
        self.protocol.request(SlowRequest())

        def cbNotSent(ignored):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(self.transport.value(), b'')
        d = self.assertFailure(self.protocol.request(SimpleRequest()), RequestNotSent)
        d.addCallback(cbNotSent)
        return d

    def test_requestAfterConnectionLost(self):
        if False:
            print('Hello World!')
        '\n        L{HTTP11ClientProtocol.request} returns a L{Deferred} which immediately\n        fires with a L{Failure} wrapping a L{RequestNotSent} if called after\n        the protocol has been disconnected.\n        '
        self.protocol.connectionLost(Failure(ConnectionDone('sad transport')))

        def cbNotSent(ignored):
            if False:
                while True:
                    i = 10
            self.assertEqual(self.transport.value(), b'')
        d = self.assertFailure(self.protocol.request(SimpleRequest()), RequestNotSent)
        d.addCallback(cbNotSent)
        return d

    def test_failedWriteTo(self):
        if False:
            i = 10
            return i + 15
        '\n        If the L{Deferred} returned by L{Request.writeTo} fires with a\n        L{Failure}, L{HTTP11ClientProtocol.request} disconnects its transport\n        and returns a L{Deferred} which fires with a L{Failure} of\n        L{RequestGenerationFailed} wrapping the underlying failure.\n        '

        class BrokenRequest:
            persistent = False

            def writeTo(self, transport):
                if False:
                    return 10
                return fail(ArbitraryException())
        d = self.protocol.request(BrokenRequest())

        def cbFailed(ignored):
            if False:
                return 10
            self.assertTrue(self.transport.disconnecting)
            self.protocol.connectionLost(Failure(ConnectionDone('you asked for it')))
        d = assertRequestGenerationFailed(self, d, [ArbitraryException])
        d.addCallback(cbFailed)
        return d

    def test_synchronousWriteToError(self):
        if False:
            print('Hello World!')
        '\n        If L{Request.writeTo} raises an exception,\n        L{HTTP11ClientProtocol.request} returns a L{Deferred} which fires with\n        a L{Failure} of L{RequestGenerationFailed} wrapping that exception.\n        '

        class BrokenRequest:
            persistent = False

            def writeTo(self, transport):
                if False:
                    for i in range(10):
                        print('nop')
                raise ArbitraryException()
        d = self.protocol.request(BrokenRequest())
        return assertRequestGenerationFailed(self, d, [ArbitraryException])

    def test_connectionLostDuringRequestGeneration(self, mode=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        If L{HTTP11ClientProtocol}'s transport is disconnected before the\n        L{Deferred} returned by L{Request.writeTo} fires, the L{Deferred}\n        returned by L{HTTP11ClientProtocol.request} fires with a L{Failure} of\n        L{RequestTransmissionFailed} wrapping the underlying failure.\n        "
        request = SlowRequest()
        d = self.protocol.request(request)
        d = assertRequestTransmissionFailed(self, d, [ArbitraryException])
        self.assertFalse(request.stopped)
        self.protocol.connectionLost(Failure(ArbitraryException()))
        self.assertTrue(request.stopped)
        if mode == 'callback':
            request.finished.callback(None)
        elif mode == 'errback':
            request.finished.errback(Failure(AnotherArbitraryException()))
            errors = self.flushLoggedErrors(AnotherArbitraryException)
            self.assertEqual(len(errors), 1)
        else:
            pass
        return d

    def test_connectionLostBeforeGenerationFinished(self):
        if False:
            while True:
                i = 10
        "\n        If the request passed to L{HTTP11ClientProtocol} finishes generation\n        successfully after the L{HTTP11ClientProtocol}'s connection has been\n        lost, nothing happens.\n        "
        return self.test_connectionLostDuringRequestGeneration('callback')

    def test_connectionLostBeforeGenerationFailed(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If the request passed to L{HTTP11ClientProtocol} finished generation\n        with an error after the L{HTTP11ClientProtocol}'s connection has been\n        lost, nothing happens.\n        "
        return self.test_connectionLostDuringRequestGeneration('errback')

    def test_errorMessageOnConnectionLostBeforeGenerationFailedDoesNotConfuse(self):
        if False:
            while True:
                i = 10
        "\n        If the request passed to L{HTTP11ClientProtocol} finished generation\n        with an error after the L{HTTP11ClientProtocol}'s connection has been\n        lost, an error is logged that gives a non-confusing hint to user on what\n        went wrong.\n        "
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

        def check(ignore):
            if False:
                print('Hello World!')
            self.assertEquals(1, len(logObserver))
            event = logObserver[0]
            self.assertIn('log_failure', event)
            self.assertEqual(event['log_format'], 'Error writing request, but not in valid state to finalize request: {state}')
            self.assertEqual(event['state'], 'CONNECTION_LOST')
        return self.test_connectionLostDuringRequestGeneration('errback').addCallback(check)

    def test_receiveSimplestResponse(self):
        if False:
            return 10
        '\n        When a response is delivered to L{HTTP11ClientProtocol}, the\n        L{Deferred} previously returned by the C{request} method is called back\n        with a L{Response} instance and the connection is closed.\n        '
        d = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))

        def cbRequest(response):
            if False:
                print('Hello World!')
            self.assertEqual(response.code, 200)
            self.assertEqual(response.headers, Headers())
            self.assertTrue(self.transport.disconnecting)
            self.assertEqual(self.protocol.state, 'QUIESCENT')
        d.addCallback(cbRequest)
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
        return d

    def test_receiveResponseHeaders(self):
        if False:
            return 10
        '\n        The headers included in a response delivered to L{HTTP11ClientProtocol}\n        are included on the L{Response} instance passed to the callback\n        returned by the C{request} method.\n        '
        d = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))

        def cbRequest(response):
            if False:
                i = 10
                return i + 15
            expected = Headers({b'x-foo': [b'bar', b'baz']})
            self.assertEqual(response.headers, expected)
        d.addCallback(cbRequest)
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nX-Foo: bar\r\nX-Foo: baz\r\n\r\n')
        return d

    def test_receiveResponseBeforeRequestGenerationDone(self):
        if False:
            while True:
                i = 10
        "\n        If response bytes are delivered to L{HTTP11ClientProtocol} before the\n        L{Deferred} returned by L{Request.writeTo} fires, those response bytes\n        are parsed as part of the response.\n\n        The connection is also closed, because we're in a confusing state, and\n        therefore the C{quiescentCallback} isn't called.\n        "
        quiescentResult = []
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(quiescentResult.append)
        protocol.makeConnection(transport)
        request = SlowRequest()
        d = protocol.request(request)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nX-Foo: bar\r\nContent-Length: 6\r\n\r\nfoobar')

        def cbResponse(response):
            if False:
                while True:
                    i = 10
            p = AccumulatingProtocol()
            whenFinished = p.closedDeferred = Deferred()
            response.deliverBody(p)
            self.assertEqual(protocol.state, 'TRANSMITTING_AFTER_RECEIVING_RESPONSE')
            self.assertTrue(transport.disconnecting)
            self.assertEqual(quiescentResult, [])
            return whenFinished.addCallback(lambda ign: (response, p.data))
        d.addCallback(cbResponse)

        def cbAllResponse(result):
            if False:
                for i in range(10):
                    print('nop')
            (response, body) = result
            self.assertEqual(response.version, (b'HTTP', 1, 1))
            self.assertEqual(response.code, 200)
            self.assertEqual(response.phrase, b'OK')
            self.assertEqual(response.headers, Headers({b'x-foo': [b'bar']}))
            self.assertEqual(body, b'foobar')
            request.finished.callback(None)
        d.addCallback(cbAllResponse)
        return d

    def test_receiveResponseHeadersTooLong(self):
        if False:
            return 10
        '\n        The connection is closed when the server respond with a header which\n        is above the maximum line.\n        '
        transport = StringTransportWithDisconnection()
        protocol = HTTP11ClientProtocol()
        transport.protocol = protocol
        protocol.makeConnection(transport)
        longLine = b'a' * LineReceiver.MAX_LENGTH
        d = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nX-Foo: ' + longLine + b'\r\nX-Ignored: ignored\r\n\r\n')
        return assertResponseFailed(self, d, [ConnectionDone])

    def test_connectionLostAfterReceivingResponseBeforeRequestGenerationDone(self):
        if False:
            print('Hello World!')
        "\n        If response bytes are delivered to L{HTTP11ClientProtocol} before the\n        request completes, calling C{connectionLost} on the protocol will\n        result in protocol being moved to C{'CONNECTION_LOST'} state.\n        "
        request = SlowRequest()
        d = self.protocol.request(request)
        self.protocol.dataReceived(b'HTTP/1.1 400 BAD REQUEST\r\nContent-Length: 9\r\n\r\ntisk tisk')

        def cbResponse(response):
            if False:
                i = 10
                return i + 15
            p = AccumulatingProtocol()
            whenFinished = p.closedDeferred = Deferred()
            response.deliverBody(p)
            return whenFinished.addCallback(lambda ign: (response, p.data))
        d.addCallback(cbResponse)

        def cbAllResponse(ignore):
            if False:
                print('Hello World!')
            request.finished.callback(None)
            self.protocol.connectionLost(Failure(ArbitraryException()))
            self.assertEqual(self.protocol._state, 'CONNECTION_LOST')
        d.addCallback(cbAllResponse)
        return d

    def test_receiveResponseBody(self):
        if False:
            i = 10
            return i + 15
        '\n        The C{deliverBody} method of the response object with which the\n        L{Deferred} returned by L{HTTP11ClientProtocol.request} fires can be\n        used to get the body of the response.\n        '
        protocol = AccumulatingProtocol()
        whenFinished = protocol.closedDeferred = Deferred()
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r')
        result = []
        requestDeferred.addCallback(result.append)
        self.assertEqual(result, [])
        self.protocol.dataReceived(b'\n')
        response = result[0]
        response.deliverBody(protocol)
        self.protocol.dataReceived(b'foo')
        self.protocol.dataReceived(b'bar')

        def cbAllResponse(ignored):
            if False:
                return 10
            self.assertEqual(protocol.data, b'foobar')
            protocol.closedReason.trap(ResponseDone)
        whenFinished.addCallback(cbAllResponse)
        return whenFinished

    def test_responseBodyFinishedWhenConnectionLostWhenContentLengthIsUnknown(self):
        if False:
            return 10
        "\n        If the length of the response body is unknown, the protocol passed to\n        the response's C{deliverBody} method has its C{connectionLost}\n        method called with a L{Failure} wrapping a L{PotentialDataLoss}\n        exception.\n        "
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        protocol = AccumulatingProtocol()
        response.deliverBody(protocol)
        self.protocol.dataReceived(b'foo')
        self.protocol.dataReceived(b'bar')
        self.assertEqual(protocol.data, b'foobar')
        self.protocol.connectionLost(Failure(ConnectionDone('low-level transport disconnected')))
        protocol.closedReason.trap(PotentialDataLoss)

    def test_chunkedResponseBodyUnfinishedWhenConnectionLost(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the final chunk has not been received when the connection is lost\n        (for any reason), the protocol passed to C{deliverBody} has its\n        C{connectionLost} method called with a L{Failure} wrapping the\n        exception for that reason.\n        '
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        self.protocol.dataReceived(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        protocol = AccumulatingProtocol()
        response.deliverBody(protocol)
        self.protocol.dataReceived(b'3\r\nfoo\r\n')
        self.protocol.dataReceived(b'3\r\nbar\r\n')
        self.assertEqual(protocol.data, b'foobar')
        self.protocol.connectionLost(Failure(ArbitraryException()))
        return assertResponseFailed(self, fail(protocol.closedReason), [ArbitraryException, _DataLoss])

    def test_parserDataReceivedException(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If the parser L{HTTP11ClientProtocol} delivers bytes to in\n        C{dataReceived} raises an exception, the exception is wrapped in a\n        L{Failure} and passed to the parser's C{connectionLost} and then the\n        L{HTTP11ClientProtocol}'s transport is disconnected.\n        "
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        self.protocol.dataReceived(b'unparseable garbage goes here\r\n')
        d = assertResponseFailed(self, requestDeferred, [ParseError])

        def cbFailed(exc):
            if False:
                i = 10
                return i + 15
            self.assertTrue(self.transport.disconnecting)
            self.assertEqual(exc.reasons[0].value.data, b'unparseable garbage goes here')
            self.protocol.connectionLost(Failure(ConnectionDone('it is done')))
        d.addCallback(cbFailed)
        return d

    def test_proxyStopped(self):
        if False:
            print('Hello World!')
        '\n        When the HTTP response parser is disconnected, the\n        L{TransportProxyProducer} which was connected to it as a transport is\n        stopped.\n        '
        requestDeferred = self.protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        transport = self.protocol._parser.transport
        self.assertIdentical(transport._producer, self.transport)
        self.protocol._disconnectParser(Failure(ConnectionDone('connection done')))
        self.assertIdentical(transport._producer, None)
        return assertResponseFailed(self, requestDeferred, [ConnectionDone])

    def test_abortClosesConnection(self):
        if False:
            i = 10
            return i + 15
        '\n        L{HTTP11ClientProtocol.abort} will tell the transport to close its\n        connection when it is invoked, and returns a C{Deferred} that fires\n        when the connection is lost.\n        '
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        r1 = []
        r2 = []
        protocol.abort().addCallback(r1.append)
        protocol.abort().addCallback(r2.append)
        self.assertEqual((r1, r2), ([], []))
        self.assertTrue(transport.disconnecting)
        protocol.connectionLost(Failure(ConnectionDone()))
        self.assertEqual(r1, [None])
        self.assertEqual(r2, [None])

    def test_abortAfterConnectionLost(self):
        if False:
            while True:
                i = 10
        '\n        L{HTTP11ClientProtocol.abort} called after the connection is lost\n        returns a C{Deferred} that fires immediately.\n        '
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        protocol.connectionLost(Failure(ConnectionDone()))
        result = []
        protocol.abort().addCallback(result.append)
        self.assertEqual(result, [None])
        self.assertEqual(protocol._state, 'CONNECTION_LOST')

    def test_abortBeforeResponseBody(self):
        if False:
            i = 10
            return i + 15
        '\n        The Deferred returned by L{HTTP11ClientProtocol.request} will fire\n        with a L{ResponseFailed} failure containing a L{ConnectionAborted}\n        exception, if the connection was aborted before all response headers\n        have been received.\n        '
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        result = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        protocol.abort()
        self.assertTrue(transport.disconnecting)
        protocol.connectionLost(Failure(ConnectionDone()))
        return assertResponseFailed(self, result, [ConnectionAborted])

    def test_abortAfterResponseHeaders(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        When the connection is aborted after the response headers have\n        been received and the L{Response} has been made available to\n        application code, the response body protocol's C{connectionLost}\n        method will be invoked with a L{ResponseFailed} failure containing a\n        L{ConnectionAborted} exception.\n        "
        transport = StringTransport(lenient=True)
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        result = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-Length: 1\r\n\r\n')
        testResult = Deferred()

        class BodyDestination(Protocol):
            """
            A body response protocol which immediately aborts the HTTP
            connection.
            """

            def connectionMade(self):
                if False:
                    return 10
                '\n                Abort the HTTP connection.\n                '
                protocol.abort()

            def connectionLost(self, reason):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Make the reason for the losing of the connection available to\n                the unit test via C{testResult}.\n                '
                testResult.errback(reason)

        def deliverBody(response):
            if False:
                while True:
                    i = 10
            '\n            Connect the L{BodyDestination} response body protocol to the\n            response, and then simulate connection loss after ensuring that\n            the HTTP connection has been aborted.\n            '
            response.deliverBody(BodyDestination())
            self.assertTrue(transport.disconnecting)
            protocol.connectionLost(Failure(ConnectionDone()))

        def checkError(error):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(error.response, Response)
        result.addCallback(deliverBody)
        deferred = assertResponseFailed(self, testResult, [ConnectionAborted, _DataLoss])
        return deferred.addCallback(checkError)

    def test_quiescentCallbackCalled(self):
        if False:
            print('Hello World!')
        "\n        If after a response is done the {HTTP11ClientProtocol} stays open and\n        returns to QUIESCENT state, all per-request state is reset and the\n        C{quiescentCallback} is called with the protocol instance.\n\n        This is useful for implementing a persistent connection pool.\n\n        The C{quiescentCallback} is called *before* the response-receiving\n        protocol's C{connectionLost}, so that new requests triggered by end of\n        first request can re-use a persistent connection.\n        "
        quiescentResult = []

        def callback(p):
            if False:
                i = 10
                return i + 15
            self.assertEqual(p, protocol)
            self.assertEqual(p.state, 'QUIESCENT')
            quiescentResult.append(p)
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(callback)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 3\r\n\r\n')
        self.assertEqual(quiescentResult, [])
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        bodyProtocol = AccumulatingProtocol()
        bodyProtocol.closedDeferred = Deferred()
        bodyProtocol.closedDeferred.addCallback(lambda ign: quiescentResult.append('response done'))
        response.deliverBody(bodyProtocol)
        protocol.dataReceived(b'abc')
        bodyProtocol.closedReason.trap(ResponseDone)
        self.assertEqual(quiescentResult, [protocol, 'response done'])
        self.assertEqual(protocol._parser, None)
        self.assertEqual(protocol._finishedRequest, None)
        self.assertEqual(protocol._currentRequest, None)
        self.assertEqual(protocol._transportProxy, None)
        self.assertEqual(protocol._responseDeferred, None)

    def test_transportProducingWhenQuiescentAfterFullBody(self):
        if False:
            print('Hello World!')
        "\n        The C{quiescentCallback} passed to L{HTTP11ClientProtocol} will only be\n        invoked once that protocol is in a state similar to its initial state.\n        One of the aspects of this initial state is the producer-state of its\n        transport; an L{HTTP11ClientProtocol} begins with a transport that is\n        producing, i.e. not C{pauseProducing}'d.\n\n        Therefore, when C{quiescentCallback} is invoked the protocol will still\n        be producing.\n        "
        quiescentResult = []

        def callback(p):
            if False:
                while True:
                    i = 10
            self.assertEqual(p, protocol)
            self.assertEqual(p.state, 'QUIESCENT')
            quiescentResult.append(p)
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(callback)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 3\r\n\r\nBBB')
        response = self.successResultOf(requestDeferred)
        self.assertEqual(response._state, 'DEFERRED_CLOSE')
        self.assertEqual(len(quiescentResult), 1)
        self.assertEqual(transport.producerState, 'producing')

    def test_quiescentCallbackCalledEmptyResponse(self):
        if False:
            while True:
                i = 10
        '\n        The quiescentCallback is called before the request C{Deferred} fires,\n        in cases where the response has no body.\n        '
        quiescentResult = []

        def callback(p):
            if False:
                return 10
            self.assertEqual(p, protocol)
            self.assertEqual(p.state, 'QUIESCENT')
            quiescentResult.append(p)
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(callback)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        requestDeferred.addCallback(quiescentResult.append)
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\n\r\n')
        self.assertEqual(len(quiescentResult), 2)
        self.assertIdentical(quiescentResult[0], protocol)
        self.assertIsInstance(quiescentResult[1], Response)

    def test_quiescentCallbackNotCalled(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If after a response is done the {HTTP11ClientProtocol} returns a\n        C{Connection: close} header in the response, the C{quiescentCallback}\n        is not called and the connection is lost.\n        '
        quiescentResult = []
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(quiescentResult.append)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\nConnection: close\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        bodyProtocol = AccumulatingProtocol()
        response.deliverBody(bodyProtocol)
        bodyProtocol.closedReason.trap(ResponseDone)
        self.assertEqual(quiescentResult, [])
        self.assertTrue(transport.disconnecting)

    def test_quiescentCallbackNotCalledNonPersistentQuery(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the request was non-persistent (i.e. sent C{Connection: close}),\n        the C{quiescentCallback} is not called and the connection is lost.\n        '
        quiescentResult = []
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(quiescentResult.append)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=False))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        bodyProtocol = AccumulatingProtocol()
        response.deliverBody(bodyProtocol)
        bodyProtocol.closedReason.trap(ResponseDone)
        self.assertEqual(quiescentResult, [])
        self.assertTrue(transport.disconnecting)

    def test_quiescentCallbackThrows(self):
        if False:
            print('Hello World!')
        '\n        If C{quiescentCallback} throws an exception, the error is logged and\n        protocol is disconnected.\n        '

        def callback(p):
            if False:
                print('Hello World!')
            raise ZeroDivisionError()
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        transport = StringTransport()
        protocol = HTTP11ClientProtocol(callback)
        protocol.makeConnection(transport)
        requestDeferred = protocol.request(Request(b'GET', b'/', _boringHeaders, None, persistent=True))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\nContent-length: 0\r\n\r\n')
        result = []
        requestDeferred.addCallback(result.append)
        response = result[0]
        bodyProtocol = AccumulatingProtocol()
        response.deliverBody(bodyProtocol)
        bodyProtocol.closedReason.trap(ResponseDone)
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        f = event['log_failure']
        self.assertIsInstance(f.value, ZeroDivisionError)
        self.flushLoggedErrors(ZeroDivisionError)
        self.assertTrue(transport.disconnecting)

    def test_cancelBeforeResponse(self):
        if False:
            print('Hello World!')
        '\n        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire\n        with a L{ResponseNeverReceived} failure containing a L{CancelledError}\n        exception if the request was cancelled before any response headers were\n        received.\n        '
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        result = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        result.cancel()
        self.assertTrue(transport.disconnected)
        return assertWrapperExceptionTypes(self, result, ResponseNeverReceived, [CancelledError])

    def test_cancelDuringResponse(self):
        if False:
            while True:
                i = 10
        '\n        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire\n        with a L{ResponseFailed} failure containing a L{CancelledError}\n        exception if the request was cancelled before all response headers were\n        received.\n        '
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        result = protocol.request(Request(b'GET', b'/', _boringHeaders, None))
        protocol.dataReceived(b'HTTP/1.1 200 OK\r\n')
        result.cancel()
        self.assertTrue(transport.disconnected)
        return assertResponseFailed(self, result, [CancelledError])

    def assertCancelDuringBodyProduction(self, producerLength):
        if False:
            print('Hello World!')
        '\n        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire\n        with a L{RequestGenerationFailed} failure containing a\n        L{CancelledError} exception if the request was cancelled before a\n        C{bodyProducer} has finished producing.\n        '
        transport = StringTransport()
        protocol = HTTP11ClientProtocol()
        protocol.makeConnection(transport)
        producer = StringProducer(producerLength)
        nonLocal = {'cancelled': False}

        def cancel(ign):
            if False:
                for i in range(10):
                    print('nop')
            nonLocal['cancelled'] = True

        def startProducing(consumer):
            if False:
                i = 10
                return i + 15
            producer.consumer = consumer
            producer.finished = Deferred(cancel)
            return producer.finished
        producer.startProducing = startProducing
        result = protocol.request(Request(b'POST', b'/bar', _boringHeaders, producer))
        producer.consumer.write(b'x' * 5)
        result.cancel()
        self.assertTrue(transport.disconnected)
        self.assertTrue(nonLocal['cancelled'])
        return assertRequestGenerationFailed(self, result, [CancelledError])

    def test_cancelDuringBodyProduction(self):
        if False:
            i = 10
            return i + 15
        '\n        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire\n        with a L{RequestGenerationFailed} failure containing a\n        L{CancelledError} exception if the request was cancelled before a\n        C{bodyProducer} with an explicit length has finished producing.\n        '
        return self.assertCancelDuringBodyProduction(10)

    def test_cancelDuringChunkedBodyProduction(self):
        if False:
            while True:
                i = 10
        '\n        The L{Deferred} returned by L{HTTP11ClientProtocol.request} will fire\n        with a L{RequestGenerationFailed} failure containing a\n        L{CancelledError} exception if the request was cancelled before a\n        C{bodyProducer} with C{UNKNOWN_LENGTH} has finished producing.\n        '
        return self.assertCancelDuringBodyProduction(UNKNOWN_LENGTH)

@implementer(IBodyProducer)
class StringProducer:
    """
    L{StringProducer} is a dummy body producer.

    @ivar stopped: A flag which indicates whether or not C{stopProducing} has
        been called.
    @ivar consumer: After C{startProducing} is called, the value of the
        C{consumer} argument to that method.
    @ivar finished: After C{startProducing} is called, a L{Deferred} which was
        returned by that method.  L{StringProducer} will never fire this
        L{Deferred}.
    """
    stopped = False

    def __init__(self, length):
        if False:
            i = 10
            return i + 15
        self.length = length

    def startProducing(self, consumer):
        if False:
            while True:
                i = 10
        self.consumer = consumer
        self.finished = Deferred()
        return self.finished

    def stopProducing(self):
        if False:
            while True:
                i = 10
        self.stopped = True

    def pauseProducing(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def resumeProducing(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class RequestTests(TestCase):
    """
    Tests for L{Request}.
    """

    def setUp(self):
        if False:
            return 10
        self.transport = StringTransport()

    def test_sendSimplestRequest(self):
        if False:
            print('Hello World!')
        '\n        L{Request.writeTo} formats the request data and writes it to the given\n        transport.\n        '
        Request(b'GET', b'/', _boringHeaders, None).writeTo(self.transport)
        self.assertEqual(self.transport.value(), b'GET / HTTP/1.1\r\nConnection: close\r\nHost: example.com\r\n\r\n')

    def test_sendSimplestPersistentRequest(self):
        if False:
            i = 10
            return i + 15
        "\n        A pesistent request does not send 'Connection: close' header.\n        "
        req = Request(b'GET', b'/', _boringHeaders, None, persistent=True)
        req.writeTo(self.transport)
        self.assertEqual(self.transport.value(), b'GET / HTTP/1.1\r\nHost: example.com\r\n\r\n')

    def test_sendRequestHeaders(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Request.writeTo} formats header data and writes it to the given\n        transport.\n        '
        headers = Headers({b'x-foo': [b'bar', b'baz'], b'host': [b'example.com']})
        Request(b'GET', b'/foo', headers, None).writeTo(self.transport)
        lines = self.transport.value().split(b'\r\n')
        self.assertEqual(lines[0], b'GET /foo HTTP/1.1')
        self.assertEqual(lines[-2:], [b'', b''])
        del lines[0], lines[-2:]
        lines.sort()
        self.assertEqual(lines, [b'Connection: close', b'Host: example.com', b'X-Foo: bar', b'X-Foo: baz'])

    def test_sanitizeLinearWhitespaceInRequestHeaders(self):
        if False:
            print('Hello World!')
        '\n        Linear whitespace in request headers is replaced with a single\n        space.\n        '
        for component in bytesLinearWhitespaceComponents:
            headers = Headers({component: [component], b'host': [b'example.invalid']})
            transport = StringTransport()
            Request(b'GET', b'/foo', headers, None).writeTo(transport)
            lines = transport.value().split(b'\r\n')
            self.assertEqual(lines[0], b'GET /foo HTTP/1.1')
            self.assertEqual(lines[-2:], [b'', b''])
            del lines[0], lines[-2:]
            lines.remove(b'Connection: close')
            lines.remove(b'Host: example.invalid')
            sanitizedHeaderLine = b': '.join([sanitizedBytes, sanitizedBytes])
            self.assertEqual(lines, [sanitizedHeaderLine])

    def test_sendChunkedRequestBody(self):
        if False:
            while True:
                i = 10
        '\n        L{Request.writeTo} uses chunked encoding to write data from the request\n        body producer to the given transport.  It registers the request body\n        producer with the transport.\n        '
        producer = StringProducer(UNKNOWN_LENGTH)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        request.writeTo(self.transport)
        self.assertNotIdentical(producer.consumer, None)
        self.assertIdentical(self.transport.producer, producer)
        self.assertTrue(self.transport.streaming)
        self.assertEqual(self.transport.value(), b'POST /bar HTTP/1.1\r\nConnection: close\r\nTransfer-Encoding: chunked\r\nHost: example.com\r\n\r\n')
        self.transport.clear()
        producer.consumer.write(b'x' * 3)
        producer.consumer.write(b'y' * 15)
        producer.finished.callback(None)
        self.assertIdentical(self.transport.producer, None)
        self.assertEqual(self.transport.value(), b'3\r\nxxx\r\nf\r\nyyyyyyyyyyyyyyy\r\n0\r\n\r\n')

    def test_sendChunkedRequestBodyWithError(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If L{Request} is created with a C{bodyProducer} without a known length\n        and the L{Deferred} returned from its C{startProducing} method fires\n        with a L{Failure}, the L{Deferred} returned by L{Request.writeTo} fires\n        with that L{Failure} and the body producer is unregistered from the\n        transport.  The final zero-length chunk is not written to the\n        transport.\n        '
        producer = StringProducer(UNKNOWN_LENGTH)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        writeDeferred = request.writeTo(self.transport)
        self.transport.clear()
        producer.finished.errback(ArbitraryException())

        def cbFailed(ignored):
            if False:
                print('Hello World!')
            self.assertEqual(self.transport.value(), b'')
            self.assertIdentical(self.transport.producer, None)
        d = self.assertFailure(writeDeferred, ArbitraryException)
        d.addCallback(cbFailed)
        return d

    def test_sendRequestBodyWithLength(self):
        if False:
            while True:
                i = 10
        '\n        If L{Request} is created with a C{bodyProducer} with a known length,\n        that length is sent as the value for the I{Content-Length} header and\n        chunked encoding is not used.\n        '
        producer = StringProducer(3)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        request.writeTo(self.transport)
        self.assertNotIdentical(producer.consumer, None)
        self.assertIdentical(self.transport.producer, producer)
        self.assertTrue(self.transport.streaming)
        self.assertEqual(self.transport.value(), b'POST /bar HTTP/1.1\r\nConnection: close\r\nContent-Length: 3\r\nHost: example.com\r\n\r\n')
        self.transport.clear()
        producer.consumer.write(b'abc')
        producer.finished.callback(None)
        self.assertIdentical(self.transport.producer, None)
        self.assertEqual(self.transport.value(), b'abc')

    def _sendRequestEmptyBodyWithLength(self, method):
        if False:
            return 10
        '\n        Verify that the message generated by a L{Request} initialized with\n        the given method and C{None} as the C{bodyProducer} includes\n        I{Content-Length: 0} in the header.\n\n        @param method: The HTTP method issue in the request.\n        @type method: L{bytes}\n        '
        request = Request(method, b'/foo', _boringHeaders, None)
        request.writeTo(self.transport)
        self.assertEqual(self.transport.value(), method + b' /foo HTTP/1.1\r\nConnection: close\r\nContent-Length: 0\r\nHost: example.com\r\n\r\n')

    def test_sendPUTRequestEmptyBody(self):
        if False:
            i = 10
            return i + 15
        '\n        If I{PUT} L{Request} is created without a C{bodyProducer},\n        I{Content-Length: 0} is included in the header and chunked\n        encoding is not used.\n        '
        self._sendRequestEmptyBodyWithLength(b'PUT')

    def test_sendPOSTRequestEmptyBody(self):
        if False:
            i = 10
            return i + 15
        '\n        If I{POST} L{Request} is created without a C{bodyProducer},\n        I{Content-Length: 0} is included in the header and chunked\n        encoding is not used.\n        '
        self._sendRequestEmptyBodyWithLength(b'POST')

    def test_sendRequestBodyWithTooFewBytes(self):
        if False:
            print('Hello World!')
        '\n        If L{Request} is created with a C{bodyProducer} with a known length and\n        the producer does not produce that many bytes, the L{Deferred} returned\n        by L{Request.writeTo} fires with a L{Failure} wrapping a\n        L{WrongBodyLength} exception.\n        '
        producer = StringProducer(3)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        writeDeferred = request.writeTo(self.transport)
        producer.consumer.write(b'ab')
        producer.finished.callback(None)
        self.assertIdentical(self.transport.producer, None)
        return self.assertFailure(writeDeferred, WrongBodyLength)

    def _sendRequestBodyWithTooManyBytesTest(self, finisher):
        if False:
            print('Hello World!')
        "\n        Verify that when too many bytes have been written by a body producer\n        and then the body producer's C{startProducing} L{Deferred} fires that\n        the producer is unregistered from the transport and that the\n        L{Deferred} returned from L{Request.writeTo} is fired with a L{Failure}\n        wrapping a L{WrongBodyLength}.\n\n        @param finisher: A callable which will be invoked with the body\n            producer after too many bytes have been written to the transport.\n            It should fire the startProducing Deferred somehow.\n        "
        producer = StringProducer(3)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        writeDeferred = request.writeTo(self.transport)
        producer.consumer.write(b'ab')
        self.assertFalse(producer.stopped)
        producer.consumer.write(b'cd')
        self.assertTrue(producer.stopped)
        self.assertIdentical(self.transport.producer, None)

        def cbFailed(exc):
            if False:
                print('Hello World!')
            self.assertEqual(self.transport.value(), b'POST /bar HTTP/1.1\r\nConnection: close\r\nContent-Length: 3\r\nHost: example.com\r\n\r\nab')
            self.transport.clear()
            self.assertRaises(ExcessWrite, producer.consumer.write, b'ef')
            finisher(producer)
            self.assertEqual(self.transport.value(), b'')
        d = self.assertFailure(writeDeferred, WrongBodyLength)
        d.addCallback(cbFailed)
        return d

    def test_sendRequestBodyWithTooManyBytes(self):
        if False:
            print('Hello World!')
        '\n        If L{Request} is created with a C{bodyProducer} with a known length and\n        the producer tries to produce more than than many bytes, the\n        L{Deferred} returned by L{Request.writeTo} fires with a L{Failure}\n        wrapping a L{WrongBodyLength} exception.\n        '

        def finisher(producer):
            if False:
                print('Hello World!')
            producer.finished.callback(None)
        return self._sendRequestBodyWithTooManyBytesTest(finisher)

    def test_sendRequestBodyErrorWithTooManyBytes(self):
        if False:
            print('Hello World!')
        '\n        If L{Request} is created with a C{bodyProducer} with a known length and\n        the producer tries to produce more than than many bytes, the\n        L{Deferred} returned by L{Request.writeTo} fires with a L{Failure}\n        wrapping a L{WrongBodyLength} exception.\n        '
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

        def finisher(producer):
            if False:
                for i in range(10):
                    print('nop')
            producer.finished.errback(ArbitraryException())
            event = logObserver[0]
            self.assertIn('log_failure', event)
            f = event['log_failure']
            self.assertIsInstance(f.value, ArbitraryException)
            errors = self.flushLoggedErrors(ArbitraryException)
            self.assertEqual(len(errors), 1)
        return self._sendRequestBodyWithTooManyBytesTest(finisher)

    def test_sendRequestBodyErrorWithConsumerError(self):
        if False:
            print('Hello World!')
        "\n        Though there should be no way for the internal C{finishedConsuming}\n        L{Deferred} in L{Request._writeToBodyProducerContentLength} to fire a\n        L{Failure} after the C{finishedProducing} L{Deferred} has fired, in\n        case this does happen, the error should be logged with a message about\n        how there's probably a bug in L{Request}.\n\n        This is a whitebox test.\n        "
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        producer = StringProducer(3)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        request.writeTo(self.transport)
        finishedConsuming = producer.consumer._finished
        producer.consumer.write(b'abc')
        producer.finished.callback(None)
        finishedConsuming.errback(ArbitraryException())
        event = logObserver[0]
        self.assertIn('log_failure', event)
        f = event['log_failure']
        self.assertIsInstance(f.value, ArbitraryException)
        self.assertEqual(len(self.flushLoggedErrors(ArbitraryException)), 1)

    def _sendRequestBodyFinishedEarlyThenTooManyBytes(self, finisher):
        if False:
            print('Hello World!')
        '\n        Verify that if the body producer fires its Deferred and then keeps\n        writing to the consumer that the extra writes are ignored and the\n        L{Deferred} returned by L{Request.writeTo} fires with a L{Failure}\n        wrapping the most appropriate exception type.\n        '
        producer = StringProducer(3)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        writeDeferred = request.writeTo(self.transport)
        producer.consumer.write(b'ab')
        finisher(producer)
        self.assertIdentical(self.transport.producer, None)
        self.transport.clear()
        self.assertRaises(ExcessWrite, producer.consumer.write, b'cd')
        self.assertEqual(self.transport.value(), b'')
        return writeDeferred

    def test_sendRequestBodyFinishedEarlyThenTooManyBytes(self):
        if False:
            i = 10
            return i + 15
        '\n        If the request body producer indicates it is done by firing the\n        L{Deferred} returned from its C{startProducing} method but then goes on\n        to write too many bytes, the L{Deferred} returned by {Request.writeTo}\n        fires with a L{Failure} wrapping L{WrongBodyLength}.\n        '

        def finisher(producer):
            if False:
                print('Hello World!')
            producer.finished.callback(None)
        return self.assertFailure(self._sendRequestBodyFinishedEarlyThenTooManyBytes(finisher), WrongBodyLength)

    def test_sendRequestBodyErroredEarlyThenTooManyBytes(self):
        if False:
            i = 10
            return i + 15
        '\n        If the request body producer indicates an error by firing the\n        L{Deferred} returned from its C{startProducing} method but then goes on\n        to write too many bytes, the L{Deferred} returned by {Request.writeTo}\n        fires with that L{Failure} and L{WrongBodyLength} is logged.\n        '

        def finisher(producer):
            if False:
                i = 10
                return i + 15
            producer.finished.errback(ArbitraryException())
        return self.assertFailure(self._sendRequestBodyFinishedEarlyThenTooManyBytes(finisher), ArbitraryException)

    def test_sendChunkedRequestBodyFinishedThenWriteMore(self, _with=None):
        if False:
            i = 10
            return i + 15
        '\n        If the request body producer with an unknown length tries to write\n        after firing the L{Deferred} returned by its C{startProducing} method,\n        the C{write} call raises an exception and does not write anything to\n        the underlying transport.\n        '
        producer = StringProducer(UNKNOWN_LENGTH)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        writeDeferred = request.writeTo(self.transport)
        producer.finished.callback(_with)
        self.transport.clear()
        self.assertRaises(ExcessWrite, producer.consumer.write, b'foo')
        self.assertEqual(self.transport.value(), b'')
        return writeDeferred

    def test_sendChunkedRequestBodyFinishedWithErrorThenWriteMore(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the request body producer with an unknown length tries to write\n        after firing the L{Deferred} returned by its C{startProducing} method\n        with a L{Failure}, the C{write} call raises an exception and does not\n        write anything to the underlying transport.\n        '
        d = self.test_sendChunkedRequestBodyFinishedThenWriteMore(Failure(ArbitraryException()))
        return self.assertFailure(d, ArbitraryException)

    def test_sendRequestBodyWithError(self):
        if False:
            while True:
                i = 10
        '\n        If the L{Deferred} returned from the C{startProducing} method of the\n        L{IBodyProducer} passed to L{Request} fires with a L{Failure}, the\n        L{Deferred} returned from L{Request.writeTo} fails with that\n        L{Failure}.\n        '
        producer = StringProducer(5)
        request = Request(b'POST', b'/bar', _boringHeaders, producer)
        writeDeferred = request.writeTo(self.transport)
        self.assertIdentical(self.transport.producer, producer)
        self.assertTrue(self.transport.streaming)
        producer.consumer.write(b'ab')
        self.assertEqual(self.transport.value(), b'POST /bar HTTP/1.1\r\nConnection: close\r\nContent-Length: 5\r\nHost: example.com\r\n\r\nab')
        self.assertFalse(self.transport.disconnecting)
        producer.finished.errback(Failure(ArbitraryException()))
        self.assertFalse(self.transport.disconnecting)
        self.assertIdentical(self.transport.producer, None)
        return self.assertFailure(writeDeferred, ArbitraryException)

    def test_hostHeaderRequired(self):
        if False:
            return 10
        '\n        L{Request.writeTo} raises L{BadHeaders} if there is not exactly one\n        I{Host} header and writes nothing to the given transport.\n        '
        request = Request(b'GET', b'/', Headers({}), None)
        self.assertRaises(BadHeaders, request.writeTo, self.transport)
        self.assertEqual(self.transport.value(), b'')
        request = Request(b'GET', b'/', Headers({b'Host': [b'example.com', b'example.org']}), None)
        self.assertRaises(BadHeaders, request.writeTo, self.transport)
        self.assertEqual(self.transport.value(), b'')

    def test_stopWriting(self):
        if False:
            while True:
                i = 10
        "\n        L{Request.stopWriting} calls its body producer's C{stopProducing}\n        method.\n        "
        producer = StringProducer(3)
        request = Request(b'GET', b'/', _boringHeaders, producer)
        request.writeTo(self.transport)
        self.assertFalse(producer.stopped)
        request.stopWriting()
        self.assertTrue(producer.stopped)

    def test_brokenStopProducing(self):
        if False:
            return 10
        "\n        If the body producer's C{stopProducing} method raises an exception,\n        L{Request.stopWriting} logs it and does not re-raise it.\n        "
        logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)
        producer = StringProducer(3)

        def brokenStopProducing():
            if False:
                while True:
                    i = 10
            raise ArbitraryException('stopProducing is busted')
        producer.stopProducing = brokenStopProducing
        request = Request(b'GET', b'/', _boringHeaders, producer)
        request.writeTo(self.transport)
        request.stopWriting()
        self.assertEqual(len(self.flushLoggedErrors(ArbitraryException)), 1)
        self.assertEquals(1, len(logObserver))
        event = logObserver[0]
        self.assertIn('log_failure', event)
        f = event['log_failure']
        self.assertIsInstance(f.value, ArbitraryException)

class LengthEnforcingConsumerTests(TestCase):
    """
    Tests for L{LengthEnforcingConsumer}.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.result = Deferred()
        self.producer = StringProducer(10)
        self.transport = StringTransport()
        self.enforcer = LengthEnforcingConsumer(self.producer, self.transport, self.result)

    def test_write(self):
        if False:
            print('Hello World!')
        "\n        L{LengthEnforcingConsumer.write} calls the wrapped consumer's C{write}\n        method with the bytes it is passed as long as there are fewer of them\n        than the C{length} attribute indicates remain to be received.\n        "
        self.enforcer.write(b'abc')
        self.assertEqual(self.transport.value(), b'abc')
        self.transport.clear()
        self.enforcer.write(b'def')
        self.assertEqual(self.transport.value(), b'def')

    def test_finishedEarly(self):
        if False:
            return 10
        '\n        L{LengthEnforcingConsumer._noMoreWritesExpected} raises\n        L{WrongBodyLength} if it is called before the indicated number of bytes\n        have been written.\n        '
        self.enforcer.write(b'x' * 9)
        self.assertRaises(WrongBodyLength, self.enforcer._noMoreWritesExpected)

    def test_writeTooMany(self, _unregisterAfter=False):
        if False:
            return 10
        '\n        If it is called with a total number of bytes exceeding the indicated\n        limit passed to L{LengthEnforcingConsumer.__init__},\n        L{LengthEnforcingConsumer.write} fires the L{Deferred} with a\n        L{Failure} wrapping a L{WrongBodyLength} and also calls the\n        C{stopProducing} method of the producer.\n        '
        self.enforcer.write(b'x' * 10)
        self.assertFalse(self.producer.stopped)
        self.enforcer.write(b'x')
        self.assertTrue(self.producer.stopped)
        if _unregisterAfter:
            self.enforcer._noMoreWritesExpected()
        return self.assertFailure(self.result, WrongBodyLength)

    def test_writeAfterNoMoreExpected(self):
        if False:
            print('Hello World!')
        "\n        If L{LengthEnforcingConsumer.write} is called after\n        L{LengthEnforcingConsumer._noMoreWritesExpected}, it calls the\n        producer's C{stopProducing} method and raises L{ExcessWrite}.\n        "
        self.enforcer.write(b'x' * 10)
        self.enforcer._noMoreWritesExpected()
        self.assertFalse(self.producer.stopped)
        self.assertRaises(ExcessWrite, self.enforcer.write, b'x')
        self.assertTrue(self.producer.stopped)

    def test_finishedLate(self):
        if False:
            print('Hello World!')
        '\n        L{LengthEnforcingConsumer._noMoreWritesExpected} does nothing (in\n        particular, it does not raise any exception) if called after too many\n        bytes have been passed to C{write}.\n        '
        return self.test_writeTooMany(True)

    def test_finished(self):
        if False:
            while True:
                i = 10
        '\n        If L{LengthEnforcingConsumer._noMoreWritesExpected} is called after\n        the correct number of bytes have been written it returns L{None}.\n        '
        self.enforcer.write(b'x' * 10)
        self.assertIdentical(self.enforcer._noMoreWritesExpected(), None)

    def test_stopProducingRaises(self):
        if False:
            return 10
        "\n        If L{LengthEnforcingConsumer.write} calls the producer's\n        C{stopProducing} because too many bytes were written and the\n        C{stopProducing} method raises an exception, the exception is logged\n        and the L{LengthEnforcingConsumer} still errbacks the finished\n        L{Deferred}.\n        "

        def brokenStopProducing():
            if False:
                for i in range(10):
                    print('nop')
            StringProducer.stopProducing(self.producer)
            raise ArbitraryException('stopProducing is busted')
        self.producer.stopProducing = brokenStopProducing

        def cbFinished(ignored):
            if False:
                i = 10
                return i + 15
            self.assertEqual(len(self.flushLoggedErrors(ArbitraryException)), 1)
        d = self.test_writeTooMany()
        d.addCallback(cbFinished)
        return d

class RequestBodyConsumerTests(TestCase):
    """
    Tests for L{ChunkedEncoder} which sits between an L{ITransport} and a
    request/response body producer and chunked encodes everything written to
    it.
    """

    def test_interface(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ChunkedEncoder} instances provide L{IConsumer}.\n        '
        self.assertTrue(verifyObject(IConsumer, ChunkedEncoder(StringTransport())))

    def test_write(self):
        if False:
            while True:
                i = 10
        '\n        L{ChunkedEncoder.write} writes to the transport the chunked encoded\n        form of the bytes passed to it.\n        '
        transport = StringTransport()
        encoder = ChunkedEncoder(transport)
        encoder.write(b'foo')
        self.assertEqual(transport.value(), b'3\r\nfoo\r\n')
        transport.clear()
        encoder.write(b'x' * 16)
        self.assertEqual(transport.value(), b'10\r\n' + b'x' * 16 + b'\r\n')

    def test_producerRegistration(self):
        if False:
            print('Hello World!')
        "\n        L{ChunkedEncoder.registerProducer} registers the given streaming\n        producer with its transport and L{ChunkedEncoder.unregisterProducer}\n        writes a zero-length chunk to its transport and unregisters the\n        transport's producer.\n        "
        transport = StringTransport()
        producer = object()
        encoder = ChunkedEncoder(transport)
        encoder.registerProducer(producer, True)
        self.assertIdentical(transport.producer, producer)
        self.assertTrue(transport.streaming)
        encoder.unregisterProducer()
        self.assertIdentical(transport.producer, None)
        self.assertEqual(transport.value(), b'0\r\n\r\n')

class TransportProxyProducerTests(TestCase):
    """
    Tests for L{TransportProxyProducer} which proxies the L{IPushProducer}
    interface of a transport.
    """

    def test_interface(self):
        if False:
            return 10
        '\n        L{TransportProxyProducer} instances provide L{IPushProducer}.\n        '
        self.assertTrue(verifyObject(IPushProducer, TransportProxyProducer(None)))

    def test_stopProxyingUnreferencesProducer(self):
        if False:
            i = 10
            return i + 15
        '\n        L{TransportProxyProducer.stopProxying} drops the reference to the\n        wrapped L{IPushProducer} provider.\n        '
        transport = StringTransport()
        proxy = TransportProxyProducer(transport)
        self.assertIdentical(proxy._producer, transport)
        proxy.stopProxying()
        self.assertIdentical(proxy._producer, None)

    def test_resumeProducing(self):
        if False:
            print('Hello World!')
        "\n        L{TransportProxyProducer.resumeProducing} calls the wrapped\n        transport's C{resumeProducing} method unless told to stop proxying.\n        "
        transport = StringTransport()
        transport.pauseProducing()
        proxy = TransportProxyProducer(transport)
        self.assertEqual(transport.producerState, 'paused')
        proxy.resumeProducing()
        self.assertEqual(transport.producerState, 'producing')
        transport.pauseProducing()
        proxy.stopProxying()
        proxy.resumeProducing()
        self.assertEqual(transport.producerState, 'paused')

    def test_pauseProducing(self):
        if False:
            while True:
                i = 10
        "\n        L{TransportProxyProducer.pauseProducing} calls the wrapped transport's\n        C{pauseProducing} method unless told to stop proxying.\n        "
        transport = StringTransport()
        proxy = TransportProxyProducer(transport)
        self.assertEqual(transport.producerState, 'producing')
        proxy.pauseProducing()
        self.assertEqual(transport.producerState, 'paused')
        transport.resumeProducing()
        proxy.stopProxying()
        proxy.pauseProducing()
        self.assertEqual(transport.producerState, 'producing')

    def test_stopProducing(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{TransportProxyProducer.stopProducing} calls the wrapped transport's\n        C{stopProducing} method unless told to stop proxying.\n        "
        transport = StringTransport()
        proxy = TransportProxyProducer(transport)
        self.assertEqual(transport.producerState, 'producing')
        proxy.stopProducing()
        self.assertEqual(transport.producerState, 'stopped')
        transport = StringTransport()
        proxy = TransportProxyProducer(transport)
        proxy.stopProxying()
        proxy.stopProducing()
        self.assertEqual(transport.producerState, 'producing')

    def test_loseConnectionWhileProxying(self):
        if False:
            return 10
        "\n        L{TransportProxyProducer.loseConnection} calls the wrapped transport's\n        C{loseConnection}.\n        "
        transport = StringTransportWithDisconnection()
        protocol = AccumulatingProtocol()
        protocol.makeConnection(transport)
        transport.protocol = protocol
        proxy = TransportProxyProducer(transport)
        self.assertTrue(transport.connected)
        self.assertEqual(transport.producerState, 'producing')
        proxy.loseConnection()
        self.assertEqual(transport.producerState, 'producing')
        self.assertFalse(transport.connected)

    def test_loseConnectionNotProxying(self):
        if False:
            while True:
                i = 10
        '\n        L{TransportProxyProducer.loseConnection} does nothing when the\n        proxy is not active.\n        '
        transport = StringTransportWithDisconnection()
        protocol = AccumulatingProtocol()
        protocol.makeConnection(transport)
        transport.protocol = protocol
        proxy = TransportProxyProducer(transport)
        proxy.stopProxying()
        self.assertTrue(transport.connected)
        proxy.loseConnection()
        self.assertTrue(transport.connected)

class ResponseTests(TestCase):
    """
    Tests for L{Response}.
    """

    def test_verifyInterface(self):
        if False:
            while True:
                i = 10
        '\n        L{Response} instances provide L{IResponse}.\n        '
        response = justTransportResponse(StringTransport())
        self.assertTrue(verifyObject(IResponse, response))

    def test_makeConnection(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The L{IProtocol} provider passed to L{Response.deliverBody} has its\n        C{makeConnection} method called with an L{IPushProducer} provider\n        hooked up to the response as an argument.\n        '
        producers = []
        transport = StringTransport()

        class SomeProtocol(Protocol):

            def makeConnection(self, producer):
                if False:
                    i = 10
                    return i + 15
                producers.append(producer)
        consumer = SomeProtocol()
        response = justTransportResponse(transport)
        response.deliverBody(consumer)
        [theProducer] = producers
        theProducer.pauseProducing()
        self.assertEqual(transport.producerState, 'paused')
        theProducer.resumeProducing()
        self.assertEqual(transport.producerState, 'producing')

    def test_dataReceived(self):
        if False:
            return 10
        '\n        The L{IProtocol} provider passed to L{Response.deliverBody} has its\n        C{dataReceived} method called with bytes received as part of the\n        response body.\n        '
        bytes = []

        class ListConsumer(Protocol):

            def dataReceived(self, data):
                if False:
                    print('Hello World!')
                bytes.append(data)
        consumer = ListConsumer()
        response = justTransportResponse(StringTransport())
        response.deliverBody(consumer)
        response._bodyDataReceived(b'foo')
        self.assertEqual(bytes, [b'foo'])

    def test_connectionLost(self):
        if False:
            return 10
        "\n        The L{IProtocol} provider passed to L{Response.deliverBody} has its\n        C{connectionLost} method called with a L{Failure} wrapping\n        L{ResponseDone} when the response's C{_bodyDataFinished} method is\n        called.\n        "
        lost = []

        class ListConsumer(Protocol):

            def connectionLost(self, reason):
                if False:
                    return 10
                lost.append(reason)
        consumer = ListConsumer()
        response = justTransportResponse(StringTransport())
        response.deliverBody(consumer)
        response._bodyDataFinished()
        lost[0].trap(ResponseDone)
        self.assertEqual(len(lost), 1)
        self.assertIdentical(response._bodyProtocol, None)

    def test_bufferEarlyData(self):
        if False:
            return 10
        '\n        If data is delivered to the L{Response} before a protocol is registered\n        with C{deliverBody}, that data is buffered until the protocol is\n        registered and then is delivered.\n        '
        bytes = []

        class ListConsumer(Protocol):

            def dataReceived(self, data):
                if False:
                    return 10
                bytes.append(data)
        protocol = ListConsumer()
        response = justTransportResponse(StringTransport())
        response._bodyDataReceived(b'foo')
        response._bodyDataReceived(b'bar')
        response.deliverBody(protocol)
        response._bodyDataReceived(b'baz')
        self.assertEqual(bytes, [b'foo', b'bar', b'baz'])
        self.assertIdentical(response._bodyBuffer, None)

    def test_multipleStartProducingFails(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Response.deliverBody} raises L{RuntimeError} if called more than\n        once.\n        '
        response = justTransportResponse(StringTransport())
        response.deliverBody(Protocol())
        self.assertRaises(RuntimeError, response.deliverBody, Protocol())

    def test_startProducingAfterFinishedFails(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Response.deliverBody} raises L{RuntimeError} if called after\n        L{Response._bodyDataFinished}.\n        '
        response = justTransportResponse(StringTransport())
        response.deliverBody(Protocol())
        response._bodyDataFinished()
        self.assertRaises(RuntimeError, response.deliverBody, Protocol())

    def test_bodyDataReceivedAfterFinishedFails(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Response._bodyDataReceived} raises L{RuntimeError} if called after\n        L{Response._bodyDataFinished} but before L{Response.deliverBody}.\n        '
        response = justTransportResponse(StringTransport())
        response._bodyDataFinished()
        self.assertRaises(RuntimeError, response._bodyDataReceived, b'foo')

    def test_bodyDataReceivedAfterDeliveryFails(self):
        if False:
            while True:
                i = 10
        '\n        L{Response._bodyDataReceived} raises L{RuntimeError} if called after\n        L{Response._bodyDataFinished} and after L{Response.deliverBody}.\n        '
        response = justTransportResponse(StringTransport())
        response._bodyDataFinished()
        response.deliverBody(Protocol())
        self.assertRaises(RuntimeError, response._bodyDataReceived, b'foo')

    def test_bodyDataFinishedAfterFinishedFails(self):
        if False:
            return 10
        '\n        L{Response._bodyDataFinished} raises L{RuntimeError} if called more\n        than once.\n        '
        response = justTransportResponse(StringTransport())
        response._bodyDataFinished()
        self.assertRaises(RuntimeError, response._bodyDataFinished)

    def test_bodyDataFinishedAfterDeliveryFails(self):
        if False:
            while True:
                i = 10
        '\n        L{Response._bodyDataFinished} raises L{RuntimeError} if called after\n        the body has been delivered.\n        '
        response = justTransportResponse(StringTransport())
        response._bodyDataFinished()
        response.deliverBody(Protocol())
        self.assertRaises(RuntimeError, response._bodyDataFinished)

    def test_transportResumed(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        L{Response.deliverBody} resumes the HTTP connection's transport\n        after passing it to the consumer's C{makeConnection} method.\n        "
        transportState = []

        class ListConsumer(Protocol):

            def makeConnection(self, transport):
                if False:
                    for i in range(10):
                        print('nop')
                transportState.append(transport.producerState)
        transport = StringTransport()
        transport.pauseProducing()
        protocol = ListConsumer()
        response = justTransportResponse(transport)
        self.assertEqual(transport.producerState, 'paused')
        response.deliverBody(protocol)
        self.assertEqual(transportState, ['paused'])
        self.assertEqual(transport.producerState, 'producing')

    def test_bodyDataFinishedBeforeStartProducing(self):
        if False:
            return 10
        "\n        If the entire body is delivered to the L{Response} before the\n        response's C{deliverBody} method is called, the protocol passed to\n        C{deliverBody} is immediately given the body data and then\n        disconnected.\n        "
        transport = StringTransport()
        response = justTransportResponse(transport)
        response._bodyDataReceived(b'foo')
        response._bodyDataReceived(b'bar')
        response._bodyDataFinished()
        protocol = AccumulatingProtocol()
        response.deliverBody(protocol)
        self.assertEqual(protocol.data, b'foobar')
        protocol.closedReason.trap(ResponseDone)

    def test_finishedWithErrorWhenConnected(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The L{Failure} passed to L{Response._bodyDataFinished} when the response\n        is in the I{connected} state is passed to the C{connectionLost} method\n        of the L{IProtocol} provider passed to the L{Response}'s\n        C{deliverBody} method.\n        "
        transport = StringTransport()
        response = justTransportResponse(transport)
        protocol = AccumulatingProtocol()
        response.deliverBody(protocol)
        self.assertEqual(response._state, 'CONNECTED')
        response._bodyDataFinished(Failure(ArbitraryException()))
        protocol.closedReason.trap(ArbitraryException)

    def test_finishedWithErrorWhenInitial(self):
        if False:
            print('Hello World!')
        "\n        The L{Failure} passed to L{Response._bodyDataFinished} when the response\n        is in the I{initial} state is passed to the C{connectionLost} method of\n        the L{IProtocol} provider passed to the L{Response}'s C{deliverBody}\n        method.\n        "
        transport = StringTransport()
        response = justTransportResponse(transport)
        self.assertEqual(response._state, 'INITIAL')
        response._bodyDataFinished(Failure(ArbitraryException()))
        protocol = AccumulatingProtocol()
        response.deliverBody(protocol)
        protocol.closedReason.trap(ArbitraryException)