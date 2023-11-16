import re
from time import time
from typing import Optional, Tuple
from urllib.parse import ParseResult, urldefrag, urlparse, urlunparse
from twisted.internet import defer
from twisted.internet.protocol import ClientFactory
from twisted.web.http import HTTPClient
from scrapy import Request
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.httpobj import urlparse_cached
from scrapy.utils.python import to_bytes, to_unicode

def _parsed_url_args(parsed: ParseResult) -> Tuple[bytes, bytes, bytes, int, bytes]:
    if False:
        return 10
    path_str = urlunparse(('', '', parsed.path or '/', parsed.params, parsed.query, ''))
    path = to_bytes(path_str, encoding='ascii')
    assert parsed.hostname is not None
    host = to_bytes(parsed.hostname, encoding='ascii')
    port = parsed.port
    scheme = to_bytes(parsed.scheme, encoding='ascii')
    netloc = to_bytes(parsed.netloc, encoding='ascii')
    if port is None:
        port = 443 if scheme == b'https' else 80
    return (scheme, netloc, host, port, path)

def _parse(url: str) -> Tuple[bytes, bytes, bytes, int, bytes]:
    if False:
        i = 10
        return i + 15
    'Return tuple of (scheme, netloc, host, port, path),\n    all in bytes except for port which is int.\n    Assume url is from Request.url, which was passed via safe_url_string\n    and is ascii-only.\n    '
    url = url.strip()
    if not re.match('^\\w+://', url):
        url = '//' + url
    parsed = urlparse(url)
    return _parsed_url_args(parsed)

class ScrapyHTTPPageGetter(HTTPClient):
    delimiter = b'\n'

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.headers = Headers()
        self.sendCommand(self.factory.method, self.factory.path)
        for (key, values) in self.factory.headers.items():
            for value in values:
                self.sendHeader(key, value)
        self.endHeaders()
        if self.factory.body is not None:
            self.transport.write(self.factory.body)

    def lineReceived(self, line):
        if False:
            for i in range(10):
                print('nop')
        return HTTPClient.lineReceived(self, line.rstrip())

    def handleHeader(self, key, value):
        if False:
            while True:
                i = 10
        self.headers.appendlist(key, value)

    def handleStatus(self, version, status, message):
        if False:
            print('Hello World!')
        self.factory.gotStatus(version, status, message)

    def handleEndHeaders(self):
        if False:
            print('Hello World!')
        self.factory.gotHeaders(self.headers)

    def connectionLost(self, reason):
        if False:
            print('Hello World!')
        self._connection_lost_reason = reason
        HTTPClient.connectionLost(self, reason)
        self.factory.noPage(reason)

    def handleResponse(self, response):
        if False:
            print('Hello World!')
        if self.factory.method.upper() == b'HEAD':
            self.factory.page(b'')
        elif self.length is not None and self.length > 0:
            self.factory.noPage(self._connection_lost_reason)
        else:
            self.factory.page(response)
        self.transport.loseConnection()

    def timeout(self):
        if False:
            print('Hello World!')
        self.transport.loseConnection()
        if self.factory.url.startswith(b'https'):
            self.transport.stopProducing()
        self.factory.noPage(defer.TimeoutError(f'Getting {self.factory.url} took longer than {self.factory.timeout} seconds.'))

class ScrapyHTTPClientFactory(ClientFactory):
    protocol = ScrapyHTTPPageGetter
    waiting = 1
    noisy = False
    followRedirect = False
    afterFoundGet = False

    def _build_response(self, body, request):
        if False:
            i = 10
            return i + 15
        request.meta['download_latency'] = self.headers_time - self.start_time
        status = int(self.status)
        headers = Headers(self.response_headers)
        respcls = responsetypes.from_args(headers=headers, url=self._url, body=body)
        return respcls(url=self._url, status=status, headers=headers, body=body, protocol=to_unicode(self.version))

    def _set_connection_attributes(self, request):
        if False:
            return 10
        parsed = urlparse_cached(request)
        (self.scheme, self.netloc, self.host, self.port, self.path) = _parsed_url_args(parsed)
        proxy = request.meta.get('proxy')
        if proxy:
            (self.scheme, _, self.host, self.port, _) = _parse(proxy)
            self.path = self.url

    def __init__(self, request: Request, timeout: float=180):
        if False:
            print('Hello World!')
        self._url: str = urldefrag(request.url)[0]
        self.url: bytes = to_bytes(self._url, encoding='ascii')
        self.method: bytes = to_bytes(request.method, encoding='ascii')
        self.body: Optional[bytes] = request.body or None
        self.headers: Headers = Headers(request.headers)
        self.response_headers: Optional[Headers] = None
        self.timeout: float = request.meta.get('download_timeout') or timeout
        self.start_time: float = time()
        self.deferred: defer.Deferred = defer.Deferred().addCallback(self._build_response, request)
        self._disconnectedDeferred: defer.Deferred = defer.Deferred()
        self._set_connection_attributes(request)
        self.headers.setdefault('Host', self.netloc)
        if self.body is not None:
            self.headers['Content-Length'] = len(self.body)
            self.headers.setdefault('Connection', 'close')
        elif self.method == b'POST':
            self.headers['Content-Length'] = 0

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'<{self.__class__.__name__}: {self._url}>'

    def _cancelTimeout(self, result, timeoutCall):
        if False:
            print('Hello World!')
        if timeoutCall.active():
            timeoutCall.cancel()
        return result

    def buildProtocol(self, addr):
        if False:
            return 10
        p = ClientFactory.buildProtocol(self, addr)
        p.followRedirect = self.followRedirect
        p.afterFoundGet = self.afterFoundGet
        if self.timeout:
            from twisted.internet import reactor
            timeoutCall = reactor.callLater(self.timeout, p.timeout)
            self.deferred.addBoth(self._cancelTimeout, timeoutCall)
        return p

    def gotHeaders(self, headers):
        if False:
            while True:
                i = 10
        self.headers_time = time()
        self.response_headers = headers

    def gotStatus(self, version, status, message):
        if False:
            return 10
        '\n        Set the status of the request on us.\n        @param version: The HTTP version.\n        @type version: L{bytes}\n        @param status: The HTTP status code, an integer represented as a\n        bytestring.\n        @type status: L{bytes}\n        @param message: The HTTP status message.\n        @type message: L{bytes}\n        '
        (self.version, self.status, self.message) = (version, status, message)

    def page(self, page):
        if False:
            while True:
                i = 10
        if self.waiting:
            self.waiting = 0
            self.deferred.callback(page)

    def noPage(self, reason):
        if False:
            print('Hello World!')
        if self.waiting:
            self.waiting = 0
            self.deferred.errback(reason)

    def clientConnectionFailed(self, _, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        When a connection attempt fails, the request cannot be issued.  If no\n        result has yet been provided to the result Deferred, provide the\n        connection failure reason as an error result.\n        '
        if self.waiting:
            self.waiting = 0
            self._disconnectedDeferred.callback(None)
            self.deferred.errback(reason)