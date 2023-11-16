"""
An U{HTTP 1.1<http://www.w3.org/Protocols/rfc2616/rfc2616.html>} client.

The way to use the functionality provided by this module is to:

  - Connect a L{HTTP11ClientProtocol} to an HTTP server
  - Create a L{Request} with the appropriate data
  - Pass the request to L{HTTP11ClientProtocol.request}
  - The returned Deferred will fire with a L{Response} object
  - Create a L{IProtocol} provider which can handle the response body
  - Connect it to the response with L{Response.deliverBody}
  - When the protocol's C{connectionLost} method is called, the response is
    complete.  See L{Response.deliverBody} for details.

Various other classes in this module support this usage:

  - HTTPParser is the basic HTTP parser.  It can handle the parts of HTTP which
    are symmetric between requests and responses.

  - HTTPClientParser extends HTTPParser to handle response-specific parts of
    HTTP.  One instance is created for each request to parse the corresponding
    response.
"""
import re
from zope.interface import implementer
from twisted.internet.defer import CancelledError, Deferred, fail, maybeDeferred, succeed
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import NO_CONTENT, NOT_MODIFIED, PotentialDataLoss, _ChunkedTransferDecoder, _DataLoss, _IdentityTransferDecoder
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
STATUS = 'STATUS'
HEADER = 'HEADER'
BODY = 'BODY'
DONE = 'DONE'
_moduleLog = Logger()

class BadHeaders(Exception):
    """
    Headers passed to L{Request} were in some way invalid.
    """

class ExcessWrite(Exception):
    """
    The body L{IBodyProducer} for a request tried to write data after
    indicating it had finished writing data.
    """

class ParseError(Exception):
    """
    Some received data could not be parsed.

    @ivar data: The string which could not be parsed.
    """

    def __init__(self, reason, data):
        if False:
            while True:
                i = 10
        Exception.__init__(self, reason, data)
        self.data = data

class BadResponseVersion(ParseError):
    """
    The version string in a status line was unparsable.
    """

class _WrapperException(Exception):
    """
    L{_WrapperException} is the base exception type for exceptions which
    include one or more other exceptions as the low-level causes.

    @ivar reasons: A L{list} of one or more L{Failure} instances encountered
        during an HTTP request.  See subclass documentation for more details.
    """

    def __init__(self, reasons):
        if False:
            return 10
        Exception.__init__(self, reasons)
        self.reasons = reasons

class RequestGenerationFailed(_WrapperException):
    """
    There was an error while creating the bytes which make up a request.

    @ivar reasons: A C{list} of one or more L{Failure} instances giving the
        reasons the request generation was considered to have failed.
    """

class RequestTransmissionFailed(_WrapperException):
    """
    There was an error while sending the bytes which make up a request.

    @ivar reasons: A C{list} of one or more L{Failure} instances giving the
        reasons the request transmission was considered to have failed.
    """

class ConnectionAborted(Exception):
    """
    The connection was explicitly aborted by application code.
    """

class WrongBodyLength(Exception):
    """
    An L{IBodyProducer} declared the number of bytes it was going to
    produce (via its C{length} attribute) and then produced a different number
    of bytes.
    """

class ResponseDone(Exception):
    """
    L{ResponseDone} may be passed to L{IProtocol.connectionLost} on the
    protocol passed to L{Response.deliverBody} and indicates that the entire
    response has been delivered.
    """

class ResponseFailed(_WrapperException):
    """
    L{ResponseFailed} indicates that all of the response to a request was not
    received for some reason.

    @ivar reasons: A C{list} of one or more L{Failure} instances giving the
        reasons the response was considered to have failed.

    @ivar response: If specified, the L{Response} received from the server (and
        in particular the status code and the headers).
    """

    def __init__(self, reasons, response=None):
        if False:
            i = 10
            return i + 15
        _WrapperException.__init__(self, reasons)
        self.response = response

class ResponseNeverReceived(ResponseFailed):
    """
    A L{ResponseFailed} that knows no response bytes at all have been received.
    """

class RequestNotSent(Exception):
    """
    L{RequestNotSent} indicates that an attempt was made to issue a request but
    for reasons unrelated to the details of the request itself, the request
    could not be sent.  For example, this may indicate that an attempt was made
    to send a request using a protocol which is no longer connected to a
    server.
    """

def _callAppFunction(function):
    if False:
        for i in range(10):
            print('nop')
    '\n    Call C{function}.  If it raises an exception, log it with a minimal\n    description of the source.\n\n    @return: L{None}\n    '
    try:
        function()
    except BaseException:
        _moduleLog.failure('Unexpected exception from {name}', name=fullyQualifiedName(function))

class HTTPParser(LineReceiver):
    """
    L{HTTPParser} handles the parsing side of HTTP processing. With a suitable
    subclass, it can parse either the client side or the server side of the
    connection.

    @ivar headers: All of the non-connection control message headers yet
        received.

    @ivar state: State indicator for the response parsing state machine.  One
        of C{STATUS}, C{HEADER}, C{BODY}, C{DONE}.

    @ivar _partialHeader: L{None} or a C{list} of the lines of a multiline
        header while that header is being received.
    """
    delimiter = b'\n'
    CONNECTION_CONTROL_HEADERS = {b'content-length', b'connection', b'keep-alive', b'te', b'trailers', b'transfer-encoding', b'upgrade', b'proxy-connection'}

    def connectionMade(self):
        if False:
            while True:
                i = 10
        self.headers = Headers()
        self.connHeaders = Headers()
        self.state = STATUS
        self._partialHeader = None

    def switchToBodyMode(self, decoder):
        if False:
            for i in range(10):
                print('nop')
        '\n        Switch to body parsing mode - interpret any more bytes delivered as\n        part of the message body and deliver them to the given decoder.\n        '
        if self.state == BODY:
            raise RuntimeError('already in body mode')
        self.bodyDecoder = decoder
        self.state = BODY
        self.setRawMode()

    def lineReceived(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Handle one line from a response.\n        '
        if line[-1:] == b'\r':
            line = line[:-1]
        if self.state == STATUS:
            self.statusReceived(line)
            self.state = HEADER
        elif self.state == HEADER:
            if not line or line[0] not in b' \t':
                if self._partialHeader is not None:
                    header = b''.join(self._partialHeader)
                    (name, value) = header.split(b':', 1)
                    value = value.strip()
                    self.headerReceived(name, value)
                if not line:
                    self.allHeadersReceived()
                else:
                    self._partialHeader = [line]
            else:
                self._partialHeader.append(line)

    def rawDataReceived(self, data):
        if False:
            return 10
        '\n        Pass data from the message body to the body decoder object.\n        '
        self.bodyDecoder.dataReceived(data)

    def isConnectionControlHeader(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Return C{True} if the given lower-cased name is the name of a\n        connection control header (rather than an entity header).\n\n        According to RFC 2616, section 14.10, the tokens in the Connection\n        header are probably relevant here.  However, I am not sure what the\n        practical consequences of either implementing or ignoring that are.\n        So I leave it unimplemented for the time being.\n        '
        return name in self.CONNECTION_CONTROL_HEADERS

    def statusReceived(self, status):
        if False:
            i = 10
            return i + 15
        '\n        Callback invoked whenever the first line of a new message is received.\n        Override this.\n\n        @param status: The first line of an HTTP request or response message\n            without trailing I{CR LF}.\n        @type status: C{bytes}\n        '

    def headerReceived(self, name, value):
        if False:
            return 10
        '\n        Store the given header in C{self.headers}.\n        '
        name = name.lower()
        if self.isConnectionControlHeader(name):
            headers = self.connHeaders
        else:
            headers = self.headers
        headers.addRawHeader(name, value)

    def allHeadersReceived(self):
        if False:
            return 10
        '\n        Callback invoked after the last header is passed to C{headerReceived}.\n        Override this to change to the C{BODY} or C{DONE} state.\n        '
        self.switchToBodyMode(None)

class HTTPClientParser(HTTPParser):
    """
    An HTTP parser which only handles HTTP responses.

    @ivar request: The request with which the expected response is associated.
    @type request: L{Request}

    @ivar NO_BODY_CODES: A C{set} of response codes which B{MUST NOT} have a
        body.

    @ivar finisher: A callable to invoke when this response is fully parsed.

    @ivar _responseDeferred: A L{Deferred} which will be called back with the
        response when all headers in the response have been received.
        Thereafter, L{None}.

    @ivar _everReceivedData: C{True} if any bytes have been received.
    """
    NO_BODY_CODES = {NO_CONTENT, NOT_MODIFIED}
    _transferDecoders = {b'chunked': _ChunkedTransferDecoder}
    bodyDecoder = None
    _log = Logger()

    def __init__(self, request, finisher):
        if False:
            print('Hello World!')
        self.request = request
        self.finisher = finisher
        self._responseDeferred = Deferred()
        self._everReceivedData = False

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        '\n        Override so that we know if any response has been received.\n        '
        self._everReceivedData = True
        HTTPParser.dataReceived(self, data)

    def parseVersion(self, strversion):
        if False:
            return 10
        "\n        Parse version strings of the form Protocol '/' Major '.' Minor. E.g.\n        b'HTTP/1.1'.  Returns (protocol, major, minor).  Will raise ValueError\n        on bad syntax.\n        "
        try:
            (proto, strnumber) = strversion.split(b'/')
            (major, minor) = strnumber.split(b'.')
            (major, minor) = (int(major), int(minor))
        except ValueError as e:
            raise BadResponseVersion(str(e), strversion)
        if major < 0 or minor < 0:
            raise BadResponseVersion('version may not be negative', strversion)
        return (proto, major, minor)

    def statusReceived(self, status):
        if False:
            while True:
                i = 10
        "\n        Parse the status line into its components and create a response object\n        to keep track of this response's state.\n        "
        parts = status.split(b' ', 2)
        if len(parts) == 2:
            (version, codeBytes) = parts
            phrase = b''
        elif len(parts) == 3:
            (version, codeBytes, phrase) = parts
        else:
            raise ParseError('wrong number of parts', status)
        try:
            statusCode = int(codeBytes)
        except ValueError:
            raise ParseError('non-integer status code', status)
        self.response = Response._construct(self.parseVersion(version), statusCode, phrase, self.headers, self.transport, self.request)

    def _finished(self, rest):
        if False:
            return 10
        '\n        Called to indicate that an entire response has been received.  No more\n        bytes will be interpreted by this L{HTTPClientParser}.  Extra bytes are\n        passed up and the state of this L{HTTPClientParser} is set to I{DONE}.\n\n        @param rest: A C{bytes} giving any extra bytes delivered to this\n            L{HTTPClientParser} which are not part of the response being\n            parsed.\n        '
        self.state = DONE
        self.finisher(rest)

    def isConnectionControlHeader(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Content-Length in the response to a HEAD request is an entity header,\n        not a connection control header.\n        '
        if self.request.method == b'HEAD' and name == b'content-length':
            return False
        return HTTPParser.isConnectionControlHeader(self, name)

    def allHeadersReceived(self):
        if False:
            print('Hello World!')
        '\n        Figure out how long the response body is going to be by examining\n        headers and stuff.\n        '
        if 100 <= self.response.code < 200:
            self._log.info('Ignoring unexpected {code} response', code=self.response.code)
            self.connectionMade()
            del self.response
            return
        if self.response.code in self.NO_BODY_CODES or self.request.method == b'HEAD':
            self.response.length = 0
            self._finished(self.clearLineBuffer())
            self.response._bodyDataFinished()
        else:
            transferEncodingHeaders = self.connHeaders.getRawHeaders(b'transfer-encoding')
            if transferEncodingHeaders:
                transferDecoder = self._transferDecoders[transferEncodingHeaders[0].lower()]
            else:
                contentLengthHeaders = self.connHeaders.getRawHeaders(b'content-length')
                if contentLengthHeaders is None:
                    contentLength = None
                elif len(contentLengthHeaders) == 1:
                    contentLength = int(contentLengthHeaders[0])
                    self.response.length = contentLength
                else:
                    raise ValueError('Too many Content-Length headers; response is invalid')
                if contentLength == 0:
                    self._finished(self.clearLineBuffer())
                    transferDecoder = None
                else:
                    transferDecoder = lambda x, y: _IdentityTransferDecoder(contentLength, x, y)
            if transferDecoder is None:
                self.response._bodyDataFinished()
            else:
                self.transport.pauseProducing()
                self.switchToBodyMode(transferDecoder(self.response._bodyDataReceived, self._finished))
        self._responseDeferred.callback(self.response)
        del self._responseDeferred

    def connectionLost(self, reason):
        if False:
            return 10
        if self.bodyDecoder is not None:
            try:
                try:
                    self.bodyDecoder.noMoreData()
                except PotentialDataLoss:
                    self.response._bodyDataFinished(Failure())
                except _DataLoss:
                    self.response._bodyDataFinished(Failure(ResponseFailed([reason, Failure()], self.response)))
                else:
                    self.response._bodyDataFinished()
            except BaseException:
                self._log.failure('')
        elif self.state != DONE:
            if self._everReceivedData:
                exceptionClass = ResponseFailed
            else:
                exceptionClass = ResponseNeverReceived
            self._responseDeferred.errback(Failure(exceptionClass([reason])))
            del self._responseDeferred
_VALID_METHOD = re.compile(b'\\A[%s]+\\Z' % (bytes().join((b'!', b'#', b'$', b'%', b'&', b"'", b'*', b'+', b'-', b'.', b'^', b'_', b'`', b'|', b'~', b'0-9', b'A-Z', b'a-z')),))

def _ensureValidMethod(method):
    if False:
        i = 10
        return i + 15
    '\n    An HTTP method is an HTTP token, which consists of any visible\n    ASCII character that is not a delimiter (i.e. one of\n    C{"(),/:;<=>?@[\\]{}}.)\n\n    @param method: the method to check\n    @type method: L{bytes}\n\n    @return: the method if it is valid\n    @rtype: L{bytes}\n\n    @raise ValueError: if the method is not valid\n\n    @see: U{https://tools.ietf.org/html/rfc7230#section-3.1.1},\n        U{https://tools.ietf.org/html/rfc7230#section-3.2.6},\n        U{https://tools.ietf.org/html/rfc5234#appendix-B.1}\n    '
    if _VALID_METHOD.match(method):
        return method
    raise ValueError(f'Invalid method {method!r}')
_VALID_URI = re.compile(b'\\A[\\x21-\\x7e]+\\Z')

def _ensureValidURI(uri):
    if False:
        return 10
    '\n    A valid URI cannot contain control characters (i.e., characters\n    between 0-32, inclusive and 127) or non-ASCII characters (i.e.,\n    characters with values between 128-255, inclusive).\n\n    @param uri: the URI to check\n    @type uri: L{bytes}\n\n    @return: the URI if it is valid\n    @rtype: L{bytes}\n\n    @raise ValueError: if the URI is not valid\n\n    @see: U{https://tools.ietf.org/html/rfc3986#section-3.3},\n        U{https://tools.ietf.org/html/rfc3986#appendix-A},\n        U{https://tools.ietf.org/html/rfc5234#appendix-B.1}\n    '
    if _VALID_URI.match(uri):
        return uri
    raise ValueError(f'Invalid URI {uri!r}')

@implementer(IClientRequest)
class Request:
    """
    A L{Request} instance describes an HTTP request to be sent to an HTTP
    server.

    @ivar method: See L{__init__}.
    @ivar uri: See L{__init__}.
    @ivar headers: See L{__init__}.
    @ivar bodyProducer: See L{__init__}.
    @ivar persistent: See L{__init__}.

    @ivar _parsedURI: Parsed I{URI} for the request, or L{None}.
    @type _parsedURI: L{twisted.web.client.URI} or L{None}
    """
    _log = Logger()

    def __init__(self, method, uri, headers, bodyProducer, persistent=False):
        if False:
            print('Hello World!')
        "\n        @param method: The HTTP method for this request, ex: b'GET', b'HEAD',\n            b'POST', etc.\n        @type method: L{bytes}\n\n        @param uri: The relative URI of the resource to request.  For example,\n            C{b'/foo/bar?baz=quux'}.\n        @type uri: L{bytes}\n\n        @param headers: Headers to be sent to the server.  It is important to\n            note that this object does not create any implicit headers.  So it\n            is up to the HTTP Client to add required headers such as 'Host'.\n        @type headers: L{twisted.web.http_headers.Headers}\n\n        @param bodyProducer: L{None} or an L{IBodyProducer} provider which\n            produces the content body to send to the remote HTTP server.\n\n        @param persistent: Set to C{True} when you use HTTP persistent\n            connection, defaults to C{False}.\n        @type persistent: L{bool}\n        "
        self.method = _ensureValidMethod(method)
        self.uri = _ensureValidURI(uri)
        self.headers = headers
        self.bodyProducer = bodyProducer
        self.persistent = persistent
        self._parsedURI = None

    @classmethod
    def _construct(cls, method, uri, headers, bodyProducer, persistent=False, parsedURI=None):
        if False:
            print('Hello World!')
        '\n        Private constructor.\n\n        @param method: See L{__init__}.\n        @param uri: See L{__init__}.\n        @param headers: See L{__init__}.\n        @param bodyProducer: See L{__init__}.\n        @param persistent: See L{__init__}.\n        @param parsedURI: See L{Request._parsedURI}.\n\n        @return: L{Request} instance.\n        '
        request = cls(method, uri, headers, bodyProducer, persistent)
        request._parsedURI = parsedURI
        return request

    @property
    def absoluteURI(self):
        if False:
            while True:
                i = 10
        '\n        The absolute URI of the request as C{bytes}, or L{None} if the\n        absolute URI cannot be determined.\n        '
        return getattr(self._parsedURI, 'toBytes', lambda : None)()

    def _writeHeaders(self, transport, TEorCL):
        if False:
            print('Hello World!')
        hosts = self.headers.getRawHeaders(b'host', ())
        if len(hosts) != 1:
            raise BadHeaders('Exactly one Host header required')
        requestLines = []
        requestLines.append(b' '.join([_ensureValidMethod(self.method), _ensureValidURI(self.uri), b'HTTP/1.1\r\n']))
        if not self.persistent:
            requestLines.append(b'Connection: close\r\n')
        if TEorCL is not None:
            requestLines.append(TEorCL)
        for (name, values) in self.headers.getAllRawHeaders():
            requestLines.extend([name + b': ' + v + b'\r\n' for v in values])
        requestLines.append(b'\r\n')
        transport.writeSequence(requestLines)

    def _writeToBodyProducerChunked(self, transport):
        if False:
            return 10
        '\n        Write this request to the given transport using chunked\n        transfer-encoding to frame the body.\n\n        @param transport: See L{writeTo}.\n        @return: See L{writeTo}.\n        '
        self._writeHeaders(transport, b'Transfer-Encoding: chunked\r\n')
        encoder = ChunkedEncoder(transport)
        encoder.registerProducer(self.bodyProducer, True)
        d = self.bodyProducer.startProducing(encoder)

        def cbProduced(ignored):
            if False:
                for i in range(10):
                    print('nop')
            encoder.unregisterProducer()

        def ebProduced(err):
            if False:
                while True:
                    i = 10
            encoder._allowNoMoreWrites()
            transport.unregisterProducer()
            return err
        d.addCallbacks(cbProduced, ebProduced)
        return d

    def _writeToBodyProducerContentLength(self, transport):
        if False:
            while True:
                i = 10
        '\n        Write this request to the given transport using content-length to frame\n        the body.\n\n        @param transport: See L{writeTo}.\n        @return: See L{writeTo}.\n        '
        self._writeHeaders(transport, networkString('Content-Length: %d\r\n' % (self.bodyProducer.length,)))
        finishedConsuming = Deferred()
        encoder = LengthEnforcingConsumer(self.bodyProducer, transport, finishedConsuming)
        transport.registerProducer(self.bodyProducer, True)
        finishedProducing = self.bodyProducer.startProducing(encoder)

        def combine(consuming, producing):
            if False:
                return 10

            def cancelConsuming(ign):
                if False:
                    while True:
                        i = 10
                finishedProducing.cancel()
            ultimate = Deferred(cancelConsuming)
            state = [None]

            def ebConsuming(err):
                if False:
                    return 10
                if state == [None]:
                    state[0] = 1
                    ultimate.errback(err)
                else:
                    self._log.failure('Buggy state machine in {request}/[{state}]: ebConsuming called', failure=err, request=repr(self), state=state[0])

            def cbProducing(result):
                if False:
                    print('Hello World!')
                if state == [None]:
                    state[0] = 2
                    try:
                        encoder._noMoreWritesExpected()
                    except BaseException:
                        ultimate.errback()
                    else:
                        ultimate.callback(None)

            def ebProducing(err):
                if False:
                    while True:
                        i = 10
                if state == [None]:
                    state[0] = 3
                    encoder._allowNoMoreWrites()
                    ultimate.errback(err)
                else:
                    self._log.failure('Producer is buggy', failure=err)
            consuming.addErrback(ebConsuming)
            producing.addCallbacks(cbProducing, ebProducing)
            return ultimate
        d = combine(finishedConsuming, finishedProducing)

        def f(passthrough):
            if False:
                for i in range(10):
                    print('nop')
            transport.unregisterProducer()
            return passthrough
        d.addBoth(f)
        return d

    def _writeToEmptyBodyContentLength(self, transport):
        if False:
            i = 10
            return i + 15
        '\n        Write this request to the given transport using content-length to frame\n        the (empty) body.\n\n        @param transport: See L{writeTo}.\n        @return: See L{writeTo}.\n        '
        self._writeHeaders(transport, b'Content-Length: 0\r\n')
        return succeed(None)

    def writeTo(self, transport):
        if False:
            return 10
        '\n        Format this L{Request} as an HTTP/1.1 request and write it to the given\n        transport.  If bodyProducer is not None, it will be associated with an\n        L{IConsumer}.\n\n        @param transport: The transport to which to write.\n        @type transport: L{twisted.internet.interfaces.ITransport} provider\n\n        @return: A L{Deferred} which fires with L{None} when the request has\n            been completely written to the transport or with a L{Failure} if\n            there is any problem generating the request bytes.\n        '
        if self.bodyProducer is None:
            if self.method in (b'PUT', b'POST'):
                self._writeToEmptyBodyContentLength(transport)
            else:
                self._writeHeaders(transport, None)
        elif self.bodyProducer.length is UNKNOWN_LENGTH:
            return self._writeToBodyProducerChunked(transport)
        else:
            return self._writeToBodyProducerContentLength(transport)

    def stopWriting(self):
        if False:
            print('Hello World!')
        '\n        Stop writing this request to the transport.  This can only be called\n        after C{writeTo} and before the L{Deferred} returned by C{writeTo}\n        fires.  It should cancel any asynchronous task started by C{writeTo}.\n        The L{Deferred} returned by C{writeTo} need not be fired if this method\n        is called.\n        '
        _callAppFunction(self.bodyProducer.stopProducing)

class LengthEnforcingConsumer:
    """
    An L{IConsumer} proxy which enforces an exact length requirement on the
    total data written to it.

    @ivar _length: The number of bytes remaining to be written.

    @ivar _producer: The L{IBodyProducer} which is writing to this
        consumer.

    @ivar _consumer: The consumer to which at most C{_length} bytes will be
        forwarded.

    @ivar _finished: A L{Deferred} which will be fired with a L{Failure} if too
        many bytes are written to this consumer.
    """

    def __init__(self, producer, consumer, finished):
        if False:
            while True:
                i = 10
        self._length = producer.length
        self._producer = producer
        self._consumer = consumer
        self._finished = finished

    def _allowNoMoreWrites(self):
        if False:
            while True:
                i = 10
        '\n        Indicate that no additional writes are allowed.  Attempts to write\n        after calling this method will be met with an exception.\n        '
        self._finished = None

    def write(self, bytes):
        if False:
            while True:
                i = 10
        '\n        Write C{bytes} to the underlying consumer unless\n        C{_noMoreWritesExpected} has been called or there are/have been too\n        many bytes.\n        '
        if self._finished is None:
            self._producer.stopProducing()
            raise ExcessWrite()
        if len(bytes) <= self._length:
            self._length -= len(bytes)
            self._consumer.write(bytes)
        else:
            _callAppFunction(self._producer.stopProducing)
            self._finished.errback(WrongBodyLength('too many bytes written'))
            self._allowNoMoreWrites()

    def _noMoreWritesExpected(self):
        if False:
            i = 10
            return i + 15
        '\n        Called to indicate no more bytes will be written to this consumer.\n        Check to see that the correct number have been written.\n\n        @raise WrongBodyLength: If not enough bytes have been written.\n        '
        if self._finished is not None:
            self._allowNoMoreWrites()
            if self._length:
                raise WrongBodyLength('too few bytes written')

def makeStatefulDispatcher(name, template):
    if False:
        print('Hello World!')
    "\n    Given a I{dispatch} name and a function, return a function which can be\n    used as a method and which, when called, will call another method defined\n    on the instance and return the result.  The other method which is called is\n    determined by the value of the C{_state} attribute of the instance.\n\n    @param name: A string which is used to construct the name of the subsidiary\n        method to invoke.  The subsidiary method is named like C{'_%s_%s' %\n        (name, _state)}.\n\n    @param template: A function object which is used to give the returned\n        function a docstring.\n\n    @return: The dispatcher function.\n    "

    def dispatcher(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        func = getattr(self, '_' + name + '_' + self._state, None)
        if func is None:
            raise RuntimeError(f'{self!r} has no {name} method in state {self._state}')
        return func(*args, **kwargs)
    dispatcher.__doc__ = template.__doc__
    return dispatcher
_ClientRequestProxy = proxyForInterface(IClientRequest)

@implementer(IResponse)
class Response:
    """
    A L{Response} instance describes an HTTP response received from an HTTP
    server.

    L{Response} should not be subclassed or instantiated.

    @ivar _transport: See L{__init__}.

    @ivar _bodyProtocol: The L{IProtocol} provider to which the body is
        delivered.  L{None} before one has been registered with
        C{deliverBody}.

    @ivar _bodyBuffer: A C{list} of the strings passed to C{bodyDataReceived}
        before C{deliverBody} is called.  L{None} afterwards.

    @ivar _state: Indicates what state this L{Response} instance is in,
        particularly with respect to delivering bytes from the response body
        to an application-supplied protocol object.  This may be one of
        C{'INITIAL'}, C{'CONNECTED'}, C{'DEFERRED_CLOSE'}, or C{'FINISHED'},
        with the following meanings:

          - INITIAL: This is the state L{Response} objects start in.  No
            protocol has yet been provided and the underlying transport may
            still have bytes to deliver to it.

          - DEFERRED_CLOSE: If the underlying transport indicates all bytes
            have been delivered but no application-provided protocol is yet
            available, the L{Response} moves to this state.  Data is
            buffered and waiting for a protocol to be delivered to.

          - CONNECTED: If a protocol is provided when the state is INITIAL,
            the L{Response} moves to this state.  Any buffered data is
            delivered and any data which arrives from the transport
            subsequently is given directly to the protocol.

          - FINISHED: If a protocol is provided in the DEFERRED_CLOSE state,
            the L{Response} moves to this state after delivering all
            buffered data to the protocol.  Otherwise, if the L{Response} is
            in the CONNECTED state, if the transport indicates there is no
            more data, the L{Response} moves to this state.  Nothing else
            can happen once the L{Response} is in this state.
    @type _state: C{str}
    """
    length = UNKNOWN_LENGTH
    _bodyProtocol = None
    _bodyFinished = False

    def __init__(self, version, code, phrase, headers, _transport):
        if False:
            i = 10
            return i + 15
        "\n        @param version: HTTP version components protocol, major, minor. E.g.\n            C{(b'HTTP', 1, 1)} to mean C{b'HTTP/1.1'}.\n\n        @param code: HTTP status code.\n        @type code: L{int}\n\n        @param phrase: HTTP reason phrase, intended to give a short description\n            of the HTTP status code.\n\n        @param headers: HTTP response headers.\n        @type headers: L{twisted.web.http_headers.Headers}\n\n        @param _transport: The transport which is delivering this response.\n        "
        self.version = version
        self.code = code
        self.phrase = phrase
        self.headers = headers
        self._transport = _transport
        self._bodyBuffer = []
        self._state = 'INITIAL'
        self.request = None
        self.previousResponse = None

    @classmethod
    def _construct(cls, version, code, phrase, headers, _transport, request):
        if False:
            print('Hello World!')
        '\n        Private constructor.\n\n        @param version: See L{__init__}.\n        @param code: See L{__init__}.\n        @param phrase: See L{__init__}.\n        @param headers: See L{__init__}.\n        @param _transport: See L{__init__}.\n        @param request: See L{IResponse.request}.\n\n        @return: L{Response} instance.\n        '
        response = Response(version, code, phrase, headers, _transport)
        response.request = _ClientRequestProxy(request)
        return response

    def setPreviousResponse(self, previousResponse):
        if False:
            i = 10
            return i + 15
        self.previousResponse = previousResponse

    def deliverBody(self, protocol):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dispatch the given L{IProtocol} depending of the current state of the\n        response.\n        '
    deliverBody = makeStatefulDispatcher('deliverBody', deliverBody)

    def _deliverBody_INITIAL(self, protocol):
        if False:
            while True:
                i = 10
        "\n        Deliver any buffered data to C{protocol} and prepare to deliver any\n        future data to it.  Move to the C{'CONNECTED'} state.\n        "
        protocol.makeConnection(self._transport)
        self._bodyProtocol = protocol
        for data in self._bodyBuffer:
            self._bodyProtocol.dataReceived(data)
        self._bodyBuffer = None
        self._state = 'CONNECTED'
        self._transport.resumeProducing()

    def _deliverBody_CONNECTED(self, protocol):
        if False:
            i = 10
            return i + 15
        '\n        It is invalid to attempt to deliver data to a protocol when it is\n        already being delivered to another protocol.\n        '
        raise RuntimeError('Response already has protocol %r, cannot deliverBody again' % (self._bodyProtocol,))

    def _deliverBody_DEFERRED_CLOSE(self, protocol):
        if False:
            print('Hello World!')
        "\n        Deliver any buffered data to C{protocol} and then disconnect the\n        protocol.  Move to the C{'FINISHED'} state.\n        "
        protocol.makeConnection(self._transport)
        for data in self._bodyBuffer:
            protocol.dataReceived(data)
        self._bodyBuffer = None
        protocol.connectionLost(self._reason)
        self._state = 'FINISHED'

    def _deliverBody_FINISHED(self, protocol):
        if False:
            for i in range(10):
                print('nop')
        '\n        It is invalid to attempt to deliver data to a protocol after the\n        response body has been delivered to another protocol.\n        '
        raise RuntimeError('Response already finished, cannot deliverBody now.')

    def _bodyDataReceived(self, data):
        if False:
            print('Hello World!')
        '\n        Called by HTTPClientParser with chunks of data from the response body.\n        They will be buffered or delivered to the protocol passed to\n        deliverBody.\n        '
    _bodyDataReceived = makeStatefulDispatcher('bodyDataReceived', _bodyDataReceived)

    def _bodyDataReceived_INITIAL(self, data):
        if False:
            print('Hello World!')
        '\n        Buffer any data received for later delivery to a protocol passed to\n        C{deliverBody}.\n\n        Little or no data should be buffered by this method, since the\n        transport has been paused and will not be resumed until a protocol\n        is supplied.\n        '
        self._bodyBuffer.append(data)

    def _bodyDataReceived_CONNECTED(self, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deliver any data received to the protocol to which this L{Response}\n        is connected.\n        '
        self._bodyProtocol.dataReceived(data)

    def _bodyDataReceived_DEFERRED_CLOSE(self, data):
        if False:
            i = 10
            return i + 15
        '\n        It is invalid for data to be delivered after it has been indicated\n        that the response body has been completely delivered.\n        '
        raise RuntimeError('Cannot receive body data after _bodyDataFinished')

    def _bodyDataReceived_FINISHED(self, data):
        if False:
            i = 10
            return i + 15
        '\n        It is invalid for data to be delivered after the response body has\n        been delivered to a protocol.\n        '
        raise RuntimeError('Cannot receive body data after protocol disconnected')

    def _bodyDataFinished(self, reason=None):
        if False:
            return 10
        '\n        Called by HTTPClientParser when no more body data is available.  If the\n        optional reason is supplied, this indicates a problem or potential\n        problem receiving all of the response body.\n        '
    _bodyDataFinished = makeStatefulDispatcher('bodyDataFinished', _bodyDataFinished)

    def _bodyDataFinished_INITIAL(self, reason=None):
        if False:
            while True:
                i = 10
        "\n        Move to the C{'DEFERRED_CLOSE'} state to wait for a protocol to\n        which to deliver the response body.\n        "
        self._state = 'DEFERRED_CLOSE'
        if reason is None:
            reason = Failure(ResponseDone('Response body fully received'))
        self._reason = reason

    def _bodyDataFinished_CONNECTED(self, reason=None):
        if False:
            return 10
        "\n        Disconnect the protocol and move to the C{'FINISHED'} state.\n        "
        if reason is None:
            reason = Failure(ResponseDone('Response body fully received'))
        self._bodyProtocol.connectionLost(reason)
        self._bodyProtocol = None
        self._state = 'FINISHED'

    def _bodyDataFinished_DEFERRED_CLOSE(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        It is invalid to attempt to notify the L{Response} of the end of the\n        response body data more than once.\n        '
        raise RuntimeError('Cannot finish body data more than once')

    def _bodyDataFinished_FINISHED(self):
        if False:
            print('Hello World!')
        '\n        It is invalid to attempt to notify the L{Response} of the end of the\n        response body data more than once.\n        '
        raise RuntimeError('Cannot finish body data after protocol disconnected')

@implementer(IConsumer)
class ChunkedEncoder:
    """
    Helper object which exposes L{IConsumer} on top of L{HTTP11ClientProtocol}
    for streaming request bodies to the server.
    """

    def __init__(self, transport):
        if False:
            for i in range(10):
                print('nop')
        self.transport = transport

    def _allowNoMoreWrites(self):
        if False:
            while True:
                i = 10
        '\n        Indicate that no additional writes are allowed.  Attempts to write\n        after calling this method will be met with an exception.\n        '
        self.transport = None

    def registerProducer(self, producer, streaming):
        if False:
            return 10
        '\n        Register the given producer with C{self.transport}.\n        '
        self.transport.registerProducer(producer, streaming)

    def write(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Write the given request body bytes to the transport using chunked\n        encoding.\n\n        @type data: C{bytes}\n        '
        if self.transport is None:
            raise ExcessWrite()
        self.transport.writeSequence((networkString('%x\r\n' % len(data)), data, b'\r\n'))

    def unregisterProducer(self):
        if False:
            print('Hello World!')
        '\n        Indicate that the request body is complete and finish the request.\n        '
        self.write(b'')
        self.transport.unregisterProducer()
        self._allowNoMoreWrites()

@implementer(IPushProducer)
class TransportProxyProducer:
    """
    An L{twisted.internet.interfaces.IPushProducer} implementation which
    wraps another such thing and proxies calls to it until it is told to stop.

    @ivar _producer: The wrapped L{twisted.internet.interfaces.IPushProducer}
    provider or L{None} after this proxy has been stopped.
    """
    disconnecting = False

    def __init__(self, producer):
        if False:
            while True:
                i = 10
        self._producer = producer

    def stopProxying(self):
        if False:
            i = 10
            return i + 15
        '\n        Stop forwarding calls of L{twisted.internet.interfaces.IPushProducer}\n        methods to the underlying L{twisted.internet.interfaces.IPushProducer}\n        provider.\n        '
        self._producer = None

    def stopProducing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Proxy the stoppage to the underlying producer, unless this proxy has\n        been stopped.\n        '
        if self._producer is not None:
            self._producer.stopProducing()

    def resumeProducing(self):
        if False:
            print('Hello World!')
        '\n        Proxy the resumption to the underlying producer, unless this proxy has\n        been stopped.\n        '
        if self._producer is not None:
            self._producer.resumeProducing()

    def pauseProducing(self):
        if False:
            return 10
        '\n        Proxy the pause to the underlying producer, unless this proxy has been\n        stopped.\n        '
        if self._producer is not None:
            self._producer.pauseProducing()

    def loseConnection(self):
        if False:
            i = 10
            return i + 15
        '\n        Proxy the request to lose the connection to the underlying producer,\n        unless this proxy has been stopped.\n        '
        if self._producer is not None:
            self._producer.loseConnection()

class HTTP11ClientProtocol(Protocol):
    """
    L{HTTP11ClientProtocol} is an implementation of the HTTP 1.1 client
    protocol.  It supports as few features as possible.

    @ivar _parser: After a request is issued, the L{HTTPClientParser} to
        which received data making up the response to that request is
        delivered.

    @ivar _finishedRequest: After a request is issued, the L{Deferred} which
        will fire when a L{Response} object corresponding to that request is
        available.  This allows L{HTTP11ClientProtocol} to fail the request
        if there is a connection or parsing problem.

    @ivar _currentRequest: After a request is issued, the L{Request}
        instance used to make that request.  This allows
        L{HTTP11ClientProtocol} to stop request generation if necessary (for
        example, if the connection is lost).

    @ivar _transportProxy: After a request is issued, the
        L{TransportProxyProducer} to which C{_parser} is connected.  This
        allows C{_parser} to pause and resume the transport in a way which
        L{HTTP11ClientProtocol} can exert some control over.

    @ivar _responseDeferred: After a request is issued, the L{Deferred} from
        C{_parser} which will fire with a L{Response} when one has been
        received.  This is eventually chained with C{_finishedRequest}, but
        only in certain cases to avoid double firing that Deferred.

    @ivar _state: Indicates what state this L{HTTP11ClientProtocol} instance
        is in with respect to transmission of a request and reception of a
        response.  This may be one of the following strings:

          - QUIESCENT: This is the state L{HTTP11ClientProtocol} instances
            start in.  Nothing is happening: no request is being sent and no
            response is being received or expected.

          - TRANSMITTING: When a request is made (via L{request}), the
            instance moves to this state.  L{Request.writeTo} has been used
            to start to send a request but it has not yet finished.

          - TRANSMITTING_AFTER_RECEIVING_RESPONSE: The server has returned a
            complete response but the request has not yet been fully sent
            yet.  The instance will remain in this state until the request
            is fully sent.

          - GENERATION_FAILED: There was an error while the request.  The
            request was not fully sent to the network.

          - WAITING: The request was fully sent to the network.  The
            instance is now waiting for the response to be fully received.

          - ABORTING: Application code has requested that the HTTP connection
            be aborted.

          - CONNECTION_LOST: The connection has been lost.
    @type _state: C{str}

    @ivar _abortDeferreds: A list of C{Deferred} instances that will fire when
        the connection is lost.
    """
    _state = 'QUIESCENT'
    _parser = None
    _finishedRequest = None
    _currentRequest = None
    _transportProxy = None
    _responseDeferred = None
    _log = Logger()

    def __init__(self, quiescentCallback=lambda c: None):
        if False:
            for i in range(10):
                print('nop')
        self._quiescentCallback = quiescentCallback
        self._abortDeferreds = []

    @property
    def state(self):
        if False:
            i = 10
            return i + 15
        return self._state

    def request(self, request):
        if False:
            print('Hello World!')
        '\n        Issue C{request} over C{self.transport} and return a L{Deferred} which\n        will fire with a L{Response} instance or an error.\n\n        @param request: The object defining the parameters of the request to\n           issue.\n        @type request: L{Request}\n\n        @rtype: L{Deferred}\n        @return: The deferred may errback with L{RequestGenerationFailed} if\n            the request was not fully written to the transport due to a local\n            error.  It may errback with L{RequestTransmissionFailed} if it was\n            not fully written to the transport due to a network error.  It may\n            errback with L{ResponseFailed} if the request was sent (not\n            necessarily received) but some or all of the response was lost.  It\n            may errback with L{RequestNotSent} if it is not possible to send\n            any more requests using this L{HTTP11ClientProtocol}.\n        '
        if self._state != 'QUIESCENT':
            return fail(RequestNotSent())
        self._state = 'TRANSMITTING'
        _requestDeferred = maybeDeferred(request.writeTo, self.transport)

        def cancelRequest(ign):
            if False:
                i = 10
                return i + 15
            if self._state in ('TRANSMITTING', 'TRANSMITTING_AFTER_RECEIVING_RESPONSE'):
                _requestDeferred.cancel()
            else:
                self.transport.abortConnection()
                self._disconnectParser(Failure(CancelledError()))
        self._finishedRequest = Deferred(cancelRequest)
        self._currentRequest = request
        self._transportProxy = TransportProxyProducer(self.transport)
        self._parser = HTTPClientParser(request, self._finishResponse)
        self._parser.makeConnection(self._transportProxy)
        self._responseDeferred = self._parser._responseDeferred

        def cbRequestWritten(ignored):
            if False:
                for i in range(10):
                    print('nop')
            if self._state == 'TRANSMITTING':
                self._state = 'WAITING'
                self._responseDeferred.chainDeferred(self._finishedRequest)

        def ebRequestWriting(err):
            if False:
                while True:
                    i = 10
            if self._state == 'TRANSMITTING':
                self._state = 'GENERATION_FAILED'
                self.transport.abortConnection()
                self._finishedRequest.errback(Failure(RequestGenerationFailed([err])))
            else:
                self._log.failure('Error writing request, but not in valid state to finalize request: {state}', failure=err, state=self._state)
        _requestDeferred.addCallbacks(cbRequestWritten, ebRequestWriting)
        return self._finishedRequest

    def _finishResponse(self, rest):
        if False:
            return 10
        '\n        Called by an L{HTTPClientParser} to indicate that it has parsed a\n        complete response.\n\n        @param rest: A C{bytes} giving any trailing bytes which were given to\n            the L{HTTPClientParser} which were not part of the response it\n            was parsing.\n        '
    _finishResponse = makeStatefulDispatcher('finishResponse', _finishResponse)

    def _finishResponse_WAITING(self, rest):
        if False:
            print('Hello World!')
        if self._state == 'WAITING':
            self._state = 'QUIESCENT'
        else:
            self._state = 'TRANSMITTING_AFTER_RECEIVING_RESPONSE'
            self._responseDeferred.chainDeferred(self._finishedRequest)
        if self._parser is None:
            return
        reason = ConnectionDone('synthetic!')
        connHeaders = self._parser.connHeaders.getRawHeaders(b'connection', ())
        if b'close' in connHeaders or self._state != 'QUIESCENT' or (not self._currentRequest.persistent):
            self._giveUp(Failure(reason))
        else:
            self.transport.resumeProducing()
            try:
                self._quiescentCallback(self)
            except BaseException:
                self._log.failure('')
                self.transport.loseConnection()
            self._disconnectParser(reason)
    _finishResponse_TRANSMITTING = _finishResponse_WAITING

    def _disconnectParser(self, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        If there is still a parser, call its C{connectionLost} method with the\n        given reason.  If there is not, do nothing.\n\n        @type reason: L{Failure}\n        '
        if self._parser is not None:
            parser = self._parser
            self._parser = None
            self._currentRequest = None
            self._finishedRequest = None
            self._responseDeferred = None
            self._transportProxy.stopProxying()
            self._transportProxy = None
            parser.connectionLost(reason)

    def _giveUp(self, reason):
        if False:
            i = 10
            return i + 15
        "\n        Lose the underlying connection and disconnect the parser with the given\n        L{Failure}.\n\n        Use this method instead of calling the transport's loseConnection\n        method directly otherwise random things will break.\n        "
        self.transport.loseConnection()
        self._disconnectParser(reason)

    def dataReceived(self, bytes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handle some stuff from some place.\n        '
        try:
            self._parser.dataReceived(bytes)
        except BaseException:
            self._giveUp(Failure())

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        '\n        The underlying transport went away.  If appropriate, notify the parser\n        object.\n        '
    connectionLost = makeStatefulDispatcher('connectionLost', connectionLost)

    def _connectionLost_QUIESCENT(self, reason):
        if False:
            for i in range(10):
                print('nop')
        "\n        Nothing is currently happening.  Move to the C{'CONNECTION_LOST'}\n        state but otherwise do nothing.\n        "
        self._state = 'CONNECTION_LOST'

    def _connectionLost_GENERATION_FAILED(self, reason):
        if False:
            for i in range(10):
                print('nop')
        "\n        The connection was in an inconsistent state.  Move to the\n        C{'CONNECTION_LOST'} state but otherwise do nothing.\n        "
        self._state = 'CONNECTION_LOST'

    def _connectionLost_TRANSMITTING(self, reason):
        if False:
            while True:
                i = 10
        "\n        Fail the L{Deferred} for the current request, notify the request\n        object that it does not need to continue transmitting itself, and\n        move to the C{'CONNECTION_LOST'} state.\n        "
        self._state = 'CONNECTION_LOST'
        self._finishedRequest.errback(Failure(RequestTransmissionFailed([reason])))
        del self._finishedRequest
        self._currentRequest.stopWriting()

    def _connectionLost_TRANSMITTING_AFTER_RECEIVING_RESPONSE(self, reason):
        if False:
            for i in range(10):
                print('nop')
        "\n        Move to the C{'CONNECTION_LOST'} state.\n        "
        self._state = 'CONNECTION_LOST'

    def _connectionLost_WAITING(self, reason):
        if False:
            while True:
                i = 10
        "\n        Disconnect the response parser so that it can propagate the event as\n        necessary (for example, to call an application protocol's\n        C{connectionLost} method, or to fail a request L{Deferred}) and move\n        to the C{'CONNECTION_LOST'} state.\n        "
        self._disconnectParser(reason)
        self._state = 'CONNECTION_LOST'

    def _connectionLost_ABORTING(self, reason):
        if False:
            for i in range(10):
                print('nop')
        "\n        Disconnect the response parser with a L{ConnectionAborted} failure, and\n        move to the C{'CONNECTION_LOST'} state.\n        "
        self._disconnectParser(Failure(ConnectionAborted()))
        self._state = 'CONNECTION_LOST'
        for d in self._abortDeferreds:
            d.callback(None)
        self._abortDeferreds = []

    def abort(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close the connection and cause all outstanding L{request} L{Deferred}s\n        to fire with an error.\n        '
        if self._state == 'CONNECTION_LOST':
            return succeed(None)
        self.transport.loseConnection()
        self._state = 'ABORTING'
        d = Deferred()
        self._abortDeferreds.append(d)
        return d