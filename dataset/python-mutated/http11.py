"""Download handlers for http and https schemes"""
import ipaddress
import logging
import re
from contextlib import suppress
from io import BytesIO
from time import time
from urllib.parse import urldefrag, urlunparse
from twisted.internet import defer, protocol, ssl
from twisted.internet.endpoints import TCP4ClientEndpoint
from twisted.internet.error import TimeoutError
from twisted.python.failure import Failure
from twisted.web.client import URI, Agent, HTTPConnectionPool, ResponseDone, ResponseFailed
from twisted.web.http import PotentialDataLoss, _DataLoss
from twisted.web.http_headers import Headers as TxHeaders
from twisted.web.iweb import UNKNOWN_LENGTH, IBodyProducer
from zope.interface import implementer
from scrapy import signals
from scrapy.core.downloader.contextfactory import load_context_factory_from_settings
from scrapy.core.downloader.webclient import _parse
from scrapy.exceptions import StopDownload
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.python import to_bytes, to_unicode
logger = logging.getLogger(__name__)

class HTTP11DownloadHandler:
    lazy = False

    def __init__(self, settings, crawler=None):
        if False:
            print('Hello World!')
        self._crawler = crawler
        from twisted.internet import reactor
        self._pool = HTTPConnectionPool(reactor, persistent=True)
        self._pool.maxPersistentPerHost = settings.getint('CONCURRENT_REQUESTS_PER_DOMAIN')
        self._pool._factory.noisy = False
        self._contextFactory = load_context_factory_from_settings(settings, crawler)
        self._default_maxsize = settings.getint('DOWNLOAD_MAXSIZE')
        self._default_warnsize = settings.getint('DOWNLOAD_WARNSIZE')
        self._fail_on_dataloss = settings.getbool('DOWNLOAD_FAIL_ON_DATALOSS')
        self._disconnect_timeout = 1

    @classmethod
    def from_crawler(cls, crawler):
        if False:
            i = 10
            return i + 15
        return cls(crawler.settings, crawler)

    def download_request(self, request, spider):
        if False:
            while True:
                i = 10
        'Return a deferred for the HTTP download'
        agent = ScrapyAgent(contextFactory=self._contextFactory, pool=self._pool, maxsize=getattr(spider, 'download_maxsize', self._default_maxsize), warnsize=getattr(spider, 'download_warnsize', self._default_warnsize), fail_on_dataloss=self._fail_on_dataloss, crawler=self._crawler)
        return agent.download_request(request)

    def close(self):
        if False:
            print('Hello World!')
        from twisted.internet import reactor
        d = self._pool.closeCachedConnections()
        delayed_call = reactor.callLater(self._disconnect_timeout, d.callback, [])

        def cancel_delayed_call(result):
            if False:
                return 10
            if delayed_call.active():
                delayed_call.cancel()
            return result
        d.addBoth(cancel_delayed_call)
        return d

class TunnelError(Exception):
    """An HTTP CONNECT tunnel could not be established by the proxy."""

class TunnelingTCP4ClientEndpoint(TCP4ClientEndpoint):
    """An endpoint that tunnels through proxies to allow HTTPS downloads. To
    accomplish that, this endpoint sends an HTTP CONNECT to the proxy.
    The HTTP CONNECT is always sent when using this endpoint, I think this could
    be improved as the CONNECT will be redundant if the connection associated
    with this endpoint comes from the pool and a CONNECT has already been issued
    for it.
    """
    _truncatedLength = 1000
    _responseAnswer = 'HTTP/1\\.. (?P<status>\\d{3})(?P<reason>.{,' + str(_truncatedLength) + '})'
    _responseMatcher = re.compile(_responseAnswer.encode())

    def __init__(self, reactor, host, port, proxyConf, contextFactory, timeout=30, bindAddress=None):
        if False:
            i = 10
            return i + 15
        (proxyHost, proxyPort, self._proxyAuthHeader) = proxyConf
        super().__init__(reactor, proxyHost, proxyPort, timeout, bindAddress)
        self._tunnelReadyDeferred = defer.Deferred()
        self._tunneledHost = host
        self._tunneledPort = port
        self._contextFactory = contextFactory
        self._connectBuffer = bytearray()

    def requestTunnel(self, protocol):
        if False:
            return 10
        'Asks the proxy to open a tunnel.'
        tunnelReq = tunnel_request_data(self._tunneledHost, self._tunneledPort, self._proxyAuthHeader)
        protocol.transport.write(tunnelReq)
        self._protocolDataReceived = protocol.dataReceived
        protocol.dataReceived = self.processProxyResponse
        self._protocol = protocol
        return protocol

    def processProxyResponse(self, rcvd_bytes):
        if False:
            for i in range(10):
                print('nop')
        'Processes the response from the proxy. If the tunnel is successfully\n        created, notifies the client that we are ready to send requests. If not\n        raises a TunnelError.\n        '
        self._connectBuffer += rcvd_bytes
        if b'\r\n\r\n' not in self._connectBuffer:
            return
        self._protocol.dataReceived = self._protocolDataReceived
        respm = TunnelingTCP4ClientEndpoint._responseMatcher.match(self._connectBuffer)
        if respm and int(respm.group('status')) == 200:
            sslOptions = self._contextFactory.creatorForNetloc(self._tunneledHost, self._tunneledPort)
            self._protocol.transport.startTLS(sslOptions, self._protocolFactory)
            self._tunnelReadyDeferred.callback(self._protocol)
        else:
            if respm:
                extra = {'status': int(respm.group('status')), 'reason': respm.group('reason').strip()}
            else:
                extra = rcvd_bytes[:self._truncatedLength]
            self._tunnelReadyDeferred.errback(TunnelError(f'Could not open CONNECT tunnel with proxy {self._host}:{self._port} [{extra!r}]'))

    def connectFailed(self, reason):
        if False:
            i = 10
            return i + 15
        'Propagates the errback to the appropriate deferred.'
        self._tunnelReadyDeferred.errback(reason)

    def connect(self, protocolFactory):
        if False:
            for i in range(10):
                print('nop')
        self._protocolFactory = protocolFactory
        connectDeferred = super().connect(protocolFactory)
        connectDeferred.addCallback(self.requestTunnel)
        connectDeferred.addErrback(self.connectFailed)
        return self._tunnelReadyDeferred

def tunnel_request_data(host, port, proxy_auth_header=None):
    if False:
        print('Hello World!')
    '\n    Return binary content of a CONNECT request.\n\n    >>> from scrapy.utils.python import to_unicode as s\n    >>> s(tunnel_request_data("example.com", 8080))\n    \'CONNECT example.com:8080 HTTP/1.1\\r\\nHost: example.com:8080\\r\\n\\r\\n\'\n    >>> s(tunnel_request_data("example.com", 8080, b"123"))\n    \'CONNECT example.com:8080 HTTP/1.1\\r\\nHost: example.com:8080\\r\\nProxy-Authorization: 123\\r\\n\\r\\n\'\n    >>> s(tunnel_request_data(b"example.com", "8090"))\n    \'CONNECT example.com:8090 HTTP/1.1\\r\\nHost: example.com:8090\\r\\n\\r\\n\'\n    '
    host_value = to_bytes(host, encoding='ascii') + b':' + to_bytes(str(port))
    tunnel_req = b'CONNECT ' + host_value + b' HTTP/1.1\r\n'
    tunnel_req += b'Host: ' + host_value + b'\r\n'
    if proxy_auth_header:
        tunnel_req += b'Proxy-Authorization: ' + proxy_auth_header + b'\r\n'
    tunnel_req += b'\r\n'
    return tunnel_req

class TunnelingAgent(Agent):
    """An agent that uses a L{TunnelingTCP4ClientEndpoint} to make HTTPS
    downloads. It may look strange that we have chosen to subclass Agent and not
    ProxyAgent but consider that after the tunnel is opened the proxy is
    transparent to the client; thus the agent should behave like there is no
    proxy involved.
    """

    def __init__(self, reactor, proxyConf, contextFactory=None, connectTimeout=None, bindAddress=None, pool=None):
        if False:
            return 10
        super().__init__(reactor, contextFactory, connectTimeout, bindAddress, pool)
        self._proxyConf = proxyConf
        self._contextFactory = contextFactory

    def _getEndpoint(self, uri):
        if False:
            print('Hello World!')
        return TunnelingTCP4ClientEndpoint(reactor=self._reactor, host=uri.host, port=uri.port, proxyConf=self._proxyConf, contextFactory=self._contextFactory, timeout=self._endpointFactory._connectTimeout, bindAddress=self._endpointFactory._bindAddress)

    def _requestWithEndpoint(self, key, endpoint, method, parsedURI, headers, bodyProducer, requestPath):
        if False:
            return 10
        key += self._proxyConf
        return super()._requestWithEndpoint(key=key, endpoint=endpoint, method=method, parsedURI=parsedURI, headers=headers, bodyProducer=bodyProducer, requestPath=requestPath)

class ScrapyProxyAgent(Agent):

    def __init__(self, reactor, proxyURI, connectTimeout=None, bindAddress=None, pool=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(reactor=reactor, connectTimeout=connectTimeout, bindAddress=bindAddress, pool=pool)
        self._proxyURI = URI.fromBytes(proxyURI)

    def request(self, method, uri, headers=None, bodyProducer=None):
        if False:
            return 10
        '\n        Issue a new request via the configured proxy.\n        '
        return self._requestWithEndpoint(key=('http-proxy', self._proxyURI.host, self._proxyURI.port), endpoint=self._getEndpoint(self._proxyURI), method=method, parsedURI=URI.fromBytes(uri), headers=headers, bodyProducer=bodyProducer, requestPath=uri)

class ScrapyAgent:
    _Agent = Agent
    _ProxyAgent = ScrapyProxyAgent
    _TunnelingAgent = TunnelingAgent

    def __init__(self, contextFactory=None, connectTimeout=10, bindAddress=None, pool=None, maxsize=0, warnsize=0, fail_on_dataloss=True, crawler=None):
        if False:
            while True:
                i = 10
        self._contextFactory = contextFactory
        self._connectTimeout = connectTimeout
        self._bindAddress = bindAddress
        self._pool = pool
        self._maxsize = maxsize
        self._warnsize = warnsize
        self._fail_on_dataloss = fail_on_dataloss
        self._txresponse = None
        self._crawler = crawler

    def _get_agent(self, request, timeout):
        if False:
            return 10
        from twisted.internet import reactor
        bindaddress = request.meta.get('bindaddress') or self._bindAddress
        proxy = request.meta.get('proxy')
        if proxy:
            (proxyScheme, proxyNetloc, proxyHost, proxyPort, proxyParams) = _parse(proxy)
            scheme = _parse(request.url)[0]
            proxyHost = to_unicode(proxyHost)
            if scheme == b'https':
                proxyAuth = request.headers.get(b'Proxy-Authorization', None)
                proxyConf = (proxyHost, proxyPort, proxyAuth)
                return self._TunnelingAgent(reactor=reactor, proxyConf=proxyConf, contextFactory=self._contextFactory, connectTimeout=timeout, bindAddress=bindaddress, pool=self._pool)
            proxyScheme = proxyScheme or b'http'
            proxyURI = urlunparse((proxyScheme, proxyNetloc, proxyParams, '', '', ''))
            return self._ProxyAgent(reactor=reactor, proxyURI=to_bytes(proxyURI, encoding='ascii'), connectTimeout=timeout, bindAddress=bindaddress, pool=self._pool)
        return self._Agent(reactor=reactor, contextFactory=self._contextFactory, connectTimeout=timeout, bindAddress=bindaddress, pool=self._pool)

    def download_request(self, request):
        if False:
            return 10
        from twisted.internet import reactor
        timeout = request.meta.get('download_timeout') or self._connectTimeout
        agent = self._get_agent(request, timeout)
        url = urldefrag(request.url)[0]
        method = to_bytes(request.method)
        headers = TxHeaders(request.headers)
        if isinstance(agent, self._TunnelingAgent):
            headers.removeHeader(b'Proxy-Authorization')
        if request.body:
            bodyproducer = _RequestBodyProducer(request.body)
        else:
            bodyproducer = None
        start_time = time()
        d = agent.request(method, to_bytes(url, encoding='ascii'), headers, bodyproducer)
        d.addCallback(self._cb_latency, request, start_time)
        d.addCallback(self._cb_bodyready, request)
        d.addCallback(self._cb_bodydone, request, url)
        self._timeout_cl = reactor.callLater(timeout, d.cancel)
        d.addBoth(self._cb_timeout, request, url, timeout)
        return d

    def _cb_timeout(self, result, request, url, timeout):
        if False:
            return 10
        if self._timeout_cl.active():
            self._timeout_cl.cancel()
            return result
        if self._txresponse:
            self._txresponse._transport.stopProducing()
        raise TimeoutError(f'Getting {url} took longer than {timeout} seconds.')

    def _cb_latency(self, result, request, start_time):
        if False:
            print('Hello World!')
        request.meta['download_latency'] = time() - start_time
        return result

    @staticmethod
    def _headers_from_twisted_response(response):
        if False:
            return 10
        headers = Headers()
        if response.length != UNKNOWN_LENGTH:
            headers[b'Content-Length'] = str(response.length).encode()
        headers.update(response.headers.getAllRawHeaders())
        return headers

    def _cb_bodyready(self, txresponse, request):
        if False:
            for i in range(10):
                print('nop')
        headers_received_result = self._crawler.signals.send_catch_log(signal=signals.headers_received, headers=self._headers_from_twisted_response(txresponse), body_length=txresponse.length, request=request, spider=self._crawler.spider)
        for (handler, result) in headers_received_result:
            if isinstance(result, Failure) and isinstance(result.value, StopDownload):
                logger.debug('Download stopped for %(request)s from signal handler %(handler)s', {'request': request, 'handler': handler.__qualname__})
                txresponse._transport.stopProducing()
                txresponse._transport.loseConnection()
                return {'txresponse': txresponse, 'body': b'', 'flags': ['download_stopped'], 'certificate': None, 'ip_address': None, 'failure': result if result.value.fail else None}
        if txresponse.length == 0:
            return {'txresponse': txresponse, 'body': b'', 'flags': None, 'certificate': None, 'ip_address': None}
        maxsize = request.meta.get('download_maxsize', self._maxsize)
        warnsize = request.meta.get('download_warnsize', self._warnsize)
        expected_size = txresponse.length if txresponse.length != UNKNOWN_LENGTH else -1
        fail_on_dataloss = request.meta.get('download_fail_on_dataloss', self._fail_on_dataloss)
        if maxsize and expected_size > maxsize:
            warning_msg = 'Cancelling download of %(url)s: expected response size (%(size)s) larger than download max size (%(maxsize)s).'
            warning_args = {'url': request.url, 'size': expected_size, 'maxsize': maxsize}
            logger.warning(warning_msg, warning_args)
            txresponse._transport.loseConnection()
            raise defer.CancelledError(warning_msg % warning_args)
        if warnsize and expected_size > warnsize:
            logger.warning('Expected response size (%(size)s) larger than download warn size (%(warnsize)s) in request %(request)s.', {'size': expected_size, 'warnsize': warnsize, 'request': request})

        def _cancel(_):
            if False:
                for i in range(10):
                    print('nop')
            txresponse._transport._producer.abortConnection()
        d = defer.Deferred(_cancel)
        txresponse.deliverBody(_ResponseReader(finished=d, txresponse=txresponse, request=request, maxsize=maxsize, warnsize=warnsize, fail_on_dataloss=fail_on_dataloss, crawler=self._crawler))
        self._txresponse = txresponse
        return d

    def _cb_bodydone(self, result, request, url):
        if False:
            while True:
                i = 10
        headers = self._headers_from_twisted_response(result['txresponse'])
        respcls = responsetypes.from_args(headers=headers, url=url, body=result['body'])
        try:
            version = result['txresponse'].version
            protocol = f'{to_unicode(version[0])}/{version[1]}.{version[2]}'
        except (AttributeError, TypeError, IndexError):
            protocol = None
        response = respcls(url=url, status=int(result['txresponse'].code), headers=headers, body=result['body'], flags=result['flags'], certificate=result['certificate'], ip_address=result['ip_address'], protocol=protocol)
        if result.get('failure'):
            result['failure'].value.response = response
            return result['failure']
        return response

@implementer(IBodyProducer)
class _RequestBodyProducer:

    def __init__(self, body):
        if False:
            i = 10
            return i + 15
        self.body = body
        self.length = len(body)

    def startProducing(self, consumer):
        if False:
            print('Hello World!')
        consumer.write(self.body)
        return defer.succeed(None)

    def pauseProducing(self):
        if False:
            print('Hello World!')
        pass

    def stopProducing(self):
        if False:
            print('Hello World!')
        pass

class _ResponseReader(protocol.Protocol):

    def __init__(self, finished, txresponse, request, maxsize, warnsize, fail_on_dataloss, crawler):
        if False:
            return 10
        self._finished = finished
        self._txresponse = txresponse
        self._request = request
        self._bodybuf = BytesIO()
        self._maxsize = maxsize
        self._warnsize = warnsize
        self._fail_on_dataloss = fail_on_dataloss
        self._fail_on_dataloss_warned = False
        self._reached_warnsize = False
        self._bytes_received = 0
        self._certificate = None
        self._ip_address = None
        self._crawler = crawler

    def _finish_response(self, flags=None, failure=None):
        if False:
            for i in range(10):
                print('nop')
        self._finished.callback({'txresponse': self._txresponse, 'body': self._bodybuf.getvalue(), 'flags': flags, 'certificate': self._certificate, 'ip_address': self._ip_address, 'failure': failure})

    def connectionMade(self):
        if False:
            while True:
                i = 10
        if self._certificate is None:
            with suppress(AttributeError):
                self._certificate = ssl.Certificate(self.transport._producer.getPeerCertificate())
        if self._ip_address is None:
            self._ip_address = ipaddress.ip_address(self.transport._producer.getPeer().host)

    def dataReceived(self, bodyBytes):
        if False:
            while True:
                i = 10
        if self._finished.called:
            return
        self._bodybuf.write(bodyBytes)
        self._bytes_received += len(bodyBytes)
        bytes_received_result = self._crawler.signals.send_catch_log(signal=signals.bytes_received, data=bodyBytes, request=self._request, spider=self._crawler.spider)
        for (handler, result) in bytes_received_result:
            if isinstance(result, Failure) and isinstance(result.value, StopDownload):
                logger.debug('Download stopped for %(request)s from signal handler %(handler)s', {'request': self._request, 'handler': handler.__qualname__})
                self.transport.stopProducing()
                self.transport.loseConnection()
                failure = result if result.value.fail else None
                self._finish_response(flags=['download_stopped'], failure=failure)
        if self._maxsize and self._bytes_received > self._maxsize:
            logger.warning('Received (%(bytes)s) bytes larger than download max size (%(maxsize)s) in request %(request)s.', {'bytes': self._bytes_received, 'maxsize': self._maxsize, 'request': self._request})
            self._bodybuf.truncate(0)
            self._finished.cancel()
        if self._warnsize and self._bytes_received > self._warnsize and (not self._reached_warnsize):
            self._reached_warnsize = True
            logger.warning('Received more bytes than download warn size (%(warnsize)s) in request %(request)s.', {'warnsize': self._warnsize, 'request': self._request})

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        if self._finished.called:
            return
        if reason.check(ResponseDone):
            self._finish_response()
            return
        if reason.check(PotentialDataLoss):
            self._finish_response(flags=['partial'])
            return
        if reason.check(ResponseFailed) and any((r.check(_DataLoss) for r in reason.value.reasons)):
            if not self._fail_on_dataloss:
                self._finish_response(flags=['dataloss'])
                return
            if not self._fail_on_dataloss_warned:
                logger.warning("Got data loss in %s. If you want to process broken responses set the setting DOWNLOAD_FAIL_ON_DATALOSS = False -- This message won't be shown in further requests", self._txresponse.request.absoluteURI.decode())
                self._fail_on_dataloss_warned = True
        self._finished.errback(reason)