from __future__ import annotations
import functools
import http.client
import io
import ssl
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
import zlib
from urllib.request import DataHandler, FileHandler, FTPHandler, HTTPCookieProcessor, HTTPDefaultErrorHandler, HTTPErrorProcessor, UnknownHandler
from ._helper import InstanceStoreMixin, add_accept_encoding_header, create_connection, create_socks_proxy_socket, get_redirect_method, make_socks_proxy_opts, select_proxy
from .common import Features, RequestHandler, Response, register_rh
from .exceptions import CertificateVerifyError, HTTPError, IncompleteRead, ProxyError, RequestError, SSLError, TransportError
from ..dependencies import brotli
from ..socks import ProxyError as SocksProxyError
from ..utils import update_url_query
from ..utils.networking import normalize_url
SUPPORTED_ENCODINGS = ['gzip', 'deflate']
CONTENT_DECODE_ERRORS = [zlib.error, OSError]
if brotli:
    SUPPORTED_ENCODINGS.append('br')
    CONTENT_DECODE_ERRORS.append(brotli.error)

def _create_http_connection(http_class, source_address, *args, **kwargs):
    if False:
        while True:
            i = 10
    hc = http_class(*args, **kwargs)
    if hasattr(hc, '_create_connection'):
        hc._create_connection = create_connection
    if source_address is not None:
        hc.source_address = (source_address, 0)
    return hc

class HTTPHandler(urllib.request.AbstractHTTPHandler):
    """Handler for HTTP requests and responses.

    This class, when installed with an OpenerDirector, automatically adds
    the standard headers to every HTTP request and handles gzipped, deflated and
    brotli responses from web servers.

    Part of this code was copied from:

    http://techknack.net/python-urllib2-handlers/

    Andrew Rowls, the author of that code, agreed to release it to the
    public domain.
    """

    def __init__(self, context=None, source_address=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._source_address = source_address
        self._context = context

    @staticmethod
    def _make_conn_class(base, req):
        if False:
            i = 10
            return i + 15
        conn_class = base
        socks_proxy = req.headers.pop('Ytdl-socks-proxy', None)
        if socks_proxy:
            conn_class = make_socks_conn_class(conn_class, socks_proxy)
        return conn_class

    def http_open(self, req):
        if False:
            i = 10
            return i + 15
        conn_class = self._make_conn_class(http.client.HTTPConnection, req)
        return self.do_open(functools.partial(_create_http_connection, conn_class, self._source_address), req)

    def https_open(self, req):
        if False:
            while True:
                i = 10
        conn_class = self._make_conn_class(http.client.HTTPSConnection, req)
        return self.do_open(functools.partial(_create_http_connection, conn_class, self._source_address), req, context=self._context)

    @staticmethod
    def deflate(data):
        if False:
            print('Hello World!')
        if not data:
            return data
        try:
            return zlib.decompress(data, -zlib.MAX_WBITS)
        except zlib.error:
            return zlib.decompress(data)

    @staticmethod
    def brotli(data):
        if False:
            for i in range(10):
                print('nop')
        if not data:
            return data
        return brotli.decompress(data)

    @staticmethod
    def gz(data):
        if False:
            return 10
        if not data:
            return data
        return zlib.decompress(data, wbits=zlib.MAX_WBITS | 16)

    def http_request(self, req):
        if False:
            while True:
                i = 10
        url = req.get_full_url()
        url_escaped = normalize_url(url)
        if url != url_escaped:
            req = update_Request(req, url=url_escaped)
        return super().do_request_(req)

    def http_response(self, req, resp):
        if False:
            return 10
        old_resp = resp
        decoded_response = None
        for encoding in (e.strip() for e in reversed(resp.headers.get('Content-encoding', '').split(','))):
            if encoding == 'gzip':
                decoded_response = self.gz(decoded_response or resp.read())
            elif encoding == 'deflate':
                decoded_response = self.deflate(decoded_response or resp.read())
            elif encoding == 'br' and brotli:
                decoded_response = self.brotli(decoded_response or resp.read())
        if decoded_response is not None:
            resp = urllib.request.addinfourl(io.BytesIO(decoded_response), old_resp.headers, old_resp.url, old_resp.code)
            resp.msg = old_resp.msg
        if 300 <= resp.code < 400:
            location = resp.headers.get('Location')
            if location:
                location = location.encode('iso-8859-1').decode()
                location_escaped = normalize_url(location)
                if location != location_escaped:
                    del resp.headers['Location']
                    resp.headers['Location'] = location_escaped
        return resp
    https_request = http_request
    https_response = http_response

def make_socks_conn_class(base_class, socks_proxy):
    if False:
        i = 10
        return i + 15
    assert issubclass(base_class, (http.client.HTTPConnection, http.client.HTTPSConnection))
    proxy_args = make_socks_proxy_opts(socks_proxy)

    class SocksConnection(base_class):
        _create_connection = create_connection

        def connect(self):
            if False:
                for i in range(10):
                    print('nop')
            self.sock = create_connection((proxy_args['addr'], proxy_args['port']), timeout=self.timeout, source_address=self.source_address, _create_socket_func=functools.partial(create_socks_proxy_socket, (self.host, self.port), proxy_args))
            if isinstance(self, http.client.HTTPSConnection):
                self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)
    return SocksConnection

class RedirectHandler(urllib.request.HTTPRedirectHandler):
    """YoutubeDL redirect handler

    The code is based on HTTPRedirectHandler implementation from CPython [1].

    This redirect handler fixes and improves the logic to better align with RFC7261
     and what browsers tend to do [2][3]

    1. https://github.com/python/cpython/blob/master/Lib/urllib/request.py
    2. https://datatracker.ietf.org/doc/html/rfc7231
    3. https://github.com/python/cpython/issues/91306
    """
    http_error_301 = http_error_303 = http_error_307 = http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if False:
            print('Hello World!')
        if code not in (301, 302, 303, 307, 308):
            raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)
        new_data = req.data
        remove_headers = ['Cookie']
        new_method = get_redirect_method(req.get_method(), code)
        if new_method != req.get_method():
            new_data = None
            remove_headers.extend(['Content-Length', 'Content-Type'])
        new_headers = {k: v for (k, v) in req.headers.items() if k.title() not in remove_headers}
        return urllib.request.Request(newurl, headers=new_headers, origin_req_host=req.origin_req_host, unverifiable=True, method=new_method, data=new_data)

class ProxyHandler(urllib.request.BaseHandler):
    handler_order = 100

    def __init__(self, proxies=None):
        if False:
            i = 10
            return i + 15
        self.proxies = proxies
        for type in ('http', 'https', 'ftp'):
            setattr(self, '%s_open' % type, lambda r, meth=self.proxy_open: meth(r))

    def proxy_open(self, req):
        if False:
            for i in range(10):
                print('nop')
        proxy = select_proxy(req.get_full_url(), self.proxies)
        if proxy is None:
            return
        if urllib.parse.urlparse(proxy).scheme.lower() in ('socks4', 'socks4a', 'socks5', 'socks5h'):
            req.add_header('Ytdl-socks-proxy', proxy)
            return None
        return urllib.request.ProxyHandler.proxy_open(self, req, proxy, None)

class PUTRequest(urllib.request.Request):

    def get_method(self):
        if False:
            while True:
                i = 10
        return 'PUT'

class HEADRequest(urllib.request.Request):

    def get_method(self):
        if False:
            print('Hello World!')
        return 'HEAD'

def update_Request(req, url=None, data=None, headers=None, query=None):
    if False:
        return 10
    req_headers = req.headers.copy()
    req_headers.update(headers or {})
    req_data = data if data is not None else req.data
    req_url = update_url_query(url or req.get_full_url(), query)
    req_get_method = req.get_method()
    if req_get_method == 'HEAD':
        req_type = HEADRequest
    elif req_get_method == 'PUT':
        req_type = PUTRequest
    else:
        req_type = urllib.request.Request
    new_req = req_type(req_url, data=req_data, headers=req_headers, origin_req_host=req.origin_req_host, unverifiable=req.unverifiable)
    if hasattr(req, 'timeout'):
        new_req.timeout = req.timeout
    return new_req

class UrllibResponseAdapter(Response):
    """
    HTTP Response adapter class for urllib addinfourl and http.client.HTTPResponse
    """

    def __init__(self, res: http.client.HTTPResponse | urllib.response.addinfourl):
        if False:
            print('Hello World!')
        super().__init__(fp=res, headers=res.headers, url=res.url, status=getattr(res, 'status', None) or res.getcode(), reason=getattr(res, 'reason', None))

    def read(self, amt=None):
        if False:
            return 10
        try:
            return self.fp.read(amt)
        except Exception as e:
            handle_response_read_exceptions(e)
            raise e

def handle_sslerror(e: ssl.SSLError):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(e, ssl.SSLError):
        return
    if isinstance(e, ssl.SSLCertVerificationError):
        raise CertificateVerifyError(cause=e) from e
    raise SSLError(cause=e) from e

def handle_response_read_exceptions(e):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(e, http.client.IncompleteRead):
        raise IncompleteRead(partial=len(e.partial), cause=e, expected=e.expected) from e
    elif isinstance(e, ssl.SSLError):
        handle_sslerror(e)
    elif isinstance(e, (OSError, EOFError, http.client.HTTPException, *CONTENT_DECODE_ERRORS)):
        raise TransportError(cause=e) from e

@register_rh
class UrllibRH(RequestHandler, InstanceStoreMixin):
    _SUPPORTED_URL_SCHEMES = ('http', 'https', 'data', 'ftp')
    _SUPPORTED_PROXY_SCHEMES = ('http', 'socks4', 'socks4a', 'socks5', 'socks5h')
    _SUPPORTED_FEATURES = (Features.NO_PROXY, Features.ALL_PROXY)
    RH_NAME = 'urllib'

    def __init__(self, *, enable_file_urls: bool=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.enable_file_urls = enable_file_urls
        if self.enable_file_urls:
            self._SUPPORTED_URL_SCHEMES = (*self._SUPPORTED_URL_SCHEMES, 'file')

    def _check_extensions(self, extensions):
        if False:
            for i in range(10):
                print('nop')
        super()._check_extensions(extensions)
        extensions.pop('cookiejar', None)
        extensions.pop('timeout', None)

    def _create_instance(self, proxies, cookiejar):
        if False:
            return 10
        opener = urllib.request.OpenerDirector()
        handlers = [ProxyHandler(proxies), HTTPHandler(debuglevel=int(bool(self.verbose)), context=self._make_sslcontext(), source_address=self.source_address), HTTPCookieProcessor(cookiejar), DataHandler(), UnknownHandler(), HTTPDefaultErrorHandler(), FTPHandler(), HTTPErrorProcessor(), RedirectHandler()]
        if self.enable_file_urls:
            handlers.append(FileHandler())
        for handler in handlers:
            opener.add_handler(handler)
        opener.addheaders = []
        return opener

    def _send(self, request):
        if False:
            print('Hello World!')
        headers = self._merge_headers(request.headers)
        add_accept_encoding_header(headers, SUPPORTED_ENCODINGS)
        urllib_req = urllib.request.Request(url=request.url, data=request.data, headers=dict(headers), method=request.method)
        opener = self._get_instance(proxies=request.proxies or self.proxies, cookiejar=request.extensions.get('cookiejar') or self.cookiejar)
        try:
            res = opener.open(urllib_req, timeout=float(request.extensions.get('timeout') or self.timeout))
        except urllib.error.HTTPError as e:
            if isinstance(e.fp, (http.client.HTTPResponse, urllib.response.addinfourl)):
                e._closer.close_called = True
                raise HTTPError(UrllibResponseAdapter(e.fp), redirect_loop='redirect error' in str(e)) from e
            raise
        except urllib.error.URLError as e:
            cause = e.reason
            if 'tunnel connection failed' in str(cause).lower() or isinstance(cause, SocksProxyError):
                raise ProxyError(cause=e) from e
            handle_response_read_exceptions(cause)
            raise TransportError(cause=e) from e
        except (http.client.InvalidURL, ValueError) as e:
            raise RequestError(cause=e) from e
        except Exception as e:
            handle_response_read_exceptions(e)
            raise
        return UrllibResponseAdapter(res)