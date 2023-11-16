import contextlib
import functools
import http.client
import logging
import re
import socket
import warnings
from ..dependencies import brotli, requests, urllib3
from ..utils import bug_reports_message, int_or_none, variadic
if requests is None:
    raise ImportError('requests module is not installed')
if urllib3 is None:
    raise ImportError('urllib3 module is not installed')
urllib3_version = tuple((int_or_none(x, default=0) for x in urllib3.__version__.split('.')))
if urllib3_version < (1, 26, 17):
    raise ImportError('Only urllib3 >= 1.26.17 is supported')
if requests.__build__ < 143616:
    raise ImportError('Only requests >= 2.31.0 is supported')
import requests.adapters
import requests.utils
import urllib3.connection
import urllib3.exceptions
from ._helper import InstanceStoreMixin, add_accept_encoding_header, create_connection, create_socks_proxy_socket, get_redirect_method, make_socks_proxy_opts, select_proxy
from .common import Features, RequestHandler, Response, register_preference, register_rh
from .exceptions import CertificateVerifyError, HTTPError, IncompleteRead, ProxyError, RequestError, SSLError, TransportError
from ..socks import ProxyError as SocksProxyError
SUPPORTED_ENCODINGS = ['gzip', 'deflate']
if brotli is not None:
    SUPPORTED_ENCODINGS.append('br')
"\nOverride urllib3's behavior to not convert lower-case percent-encoded characters\nto upper-case during url normalization process.\n\nRFC3986 defines that the lower or upper case percent-encoded hexidecimal characters are equivalent\nand normalizers should convert them to uppercase for consistency [1].\n\nHowever, some sites may have an incorrect implementation where they provide\na percent-encoded url that is then compared case-sensitively.[2]\n\nWhile this is a very rare case, since urllib does not do this normalization step, it\nis best to avoid it in requests too for compatability reasons.\n\n1: https://tools.ietf.org/html/rfc3986#section-2.1\n2: https://github.com/streamlink/streamlink/pull/4003\n"

class Urllib3PercentREOverride:

    def __init__(self, r: re.Pattern):
        if False:
            while True:
                i = 10
        self.re = r

    def __getattr__(self, item):
        if False:
            while True:
                i = 10
        return self.re.__getattribute__(item)

    def subn(self, repl, string, *args, **kwargs):
        if False:
            return 10
        return (string, self.re.subn(repl, string, *args, **kwargs)[1])
import urllib3.util.url
if hasattr(urllib3.util.url, 'PERCENT_RE'):
    urllib3.util.url.PERCENT_RE = Urllib3PercentREOverride(urllib3.util.url.PERCENT_RE)
elif hasattr(urllib3.util.url, '_PERCENT_RE'):
    urllib3.util.url._PERCENT_RE = Urllib3PercentREOverride(urllib3.util.url._PERCENT_RE)
else:
    warnings.warn('Failed to patch PERCENT_RE in urllib3 (does the attribute exist?)' + bug_reports_message())
'\nWorkaround for issue in urllib.util.ssl_.py: ssl_wrap_context does not pass\nserver_hostname to SSLContext.wrap_socket if server_hostname is an IP,\nhowever this is an issue because we set check_hostname to True in our SSLContext.\n\nMonkey-patching IS_SECURETRANSPORT forces ssl_wrap_context to pass server_hostname regardless.\n\nThis has been fixed in urllib3 2.0+.\nSee: https://github.com/urllib3/urllib3/issues/517\n'
if urllib3_version < (2, 0, 0):
    with contextlib.suppress():
        urllib3.util.IS_SECURETRANSPORT = urllib3.util.ssl_.IS_SECURETRANSPORT = True
requests.adapters.select_proxy = select_proxy

class RequestsResponseAdapter(Response):

    def __init__(self, res: requests.models.Response):
        if False:
            print('Hello World!')
        super().__init__(fp=res.raw, headers=res.headers, url=res.url, status=res.status_code, reason=res.reason)
        self._requests_response = res

    def read(self, amt: int=None):
        if False:
            while True:
                i = 10
        try:
            return self.fp.read(amt, decode_content=True)
        except urllib3.exceptions.SSLError as e:
            raise SSLError(cause=e) from e
        except urllib3.exceptions.ProtocolError as e:
            ir_err = next((err for err in (e.__context__, e.__cause__, *variadic(e.args)) if isinstance(err, http.client.IncompleteRead)), None)
            if ir_err is not None:
                partial = ir_err.partial if isinstance(ir_err.partial, int) else len(ir_err.partial)
                raise IncompleteRead(partial=partial, expected=ir_err.expected) from e
            raise TransportError(cause=e) from e
        except urllib3.exceptions.HTTPError as e:
            raise TransportError(cause=e) from e

class RequestsHTTPAdapter(requests.adapters.HTTPAdapter):

    def __init__(self, ssl_context=None, proxy_ssl_context=None, source_address=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self._pm_args = {}
        if ssl_context:
            self._pm_args['ssl_context'] = ssl_context
        if source_address:
            self._pm_args['source_address'] = (source_address, 0)
        self._proxy_ssl_context = proxy_ssl_context or ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return super().init_poolmanager(*args, **kwargs, **self._pm_args)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        if False:
            return 10
        extra_kwargs = {}
        if not proxy.lower().startswith('socks') and self._proxy_ssl_context:
            extra_kwargs['proxy_ssl_context'] = self._proxy_ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs, **self._pm_args, **extra_kwargs)

    def cert_verify(*args, **kwargs):
        if False:
            while True:
                i = 10
        pass

class RequestsSession(requests.sessions.Session):
    """
    Ensure unified redirect method handling with our urllib redirect handler.
    """

    def rebuild_method(self, prepared_request, response):
        if False:
            print('Hello World!')
        new_method = get_redirect_method(prepared_request.method, response.status_code)
        if new_method == prepared_request.method:
            response._real_status_code = response.status_code
            response.status_code = 308
        prepared_request.method = new_method

    def rebuild_auth(self, prepared_request, response):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(response, '_real_status_code'):
            response.status_code = response._real_status_code
            del response._real_status_code
        return super().rebuild_auth(prepared_request, response)

class Urllib3LoggingFilter(logging.Filter):

    def filter(self, record):
        if False:
            i = 10
            return i + 15
        if record.msg == '%s://%s:%s "%s %s %s" %s %s':
            return False
        return True

class Urllib3LoggingHandler(logging.Handler):
    """Redirect urllib3 logs to our logger"""

    def __init__(self, logger, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self._logger = logger

    def emit(self, record):
        if False:
            print('Hello World!')
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                self._logger.error(msg)
            else:
                self._logger.stdout(msg)
        except Exception:
            self.handleError(record)

@register_rh
class RequestsRH(RequestHandler, InstanceStoreMixin):
    """Requests RequestHandler
    https://github.com/psf/requests
    """
    _SUPPORTED_URL_SCHEMES = ('http', 'https')
    _SUPPORTED_ENCODINGS = tuple(SUPPORTED_ENCODINGS)
    _SUPPORTED_PROXY_SCHEMES = ('http', 'https', 'socks4', 'socks4a', 'socks5', 'socks5h')
    _SUPPORTED_FEATURES = (Features.NO_PROXY, Features.ALL_PROXY)
    RH_NAME = 'requests'

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        logger = logging.getLogger('urllib3')
        handler = Urllib3LoggingHandler(logger=self._logger)
        handler.setFormatter(logging.Formatter('requests: %(message)s'))
        handler.addFilter(Urllib3LoggingFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        if self.verbose:
            urllib3.connection.HTTPConnection.debuglevel = 1
            logger.setLevel(logging.DEBUG)
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self._clear_instances()

    def _check_extensions(self, extensions):
        if False:
            print('Hello World!')
        super()._check_extensions(extensions)
        extensions.pop('cookiejar', None)
        extensions.pop('timeout', None)

    def _create_instance(self, cookiejar):
        if False:
            return 10
        session = RequestsSession()
        http_adapter = RequestsHTTPAdapter(ssl_context=self._make_sslcontext(), source_address=self.source_address, max_retries=urllib3.util.retry.Retry(False))
        session.adapters.clear()
        session.headers = requests.models.CaseInsensitiveDict({'Connection': 'keep-alive'})
        session.mount('https://', http_adapter)
        session.mount('http://', http_adapter)
        session.cookies = cookiejar
        session.trust_env = False
        return session

    def _send(self, request):
        if False:
            print('Hello World!')
        headers = self._merge_headers(request.headers)
        add_accept_encoding_header(headers, SUPPORTED_ENCODINGS)
        max_redirects_exceeded = False
        session = self._get_instance(cookiejar=request.extensions.get('cookiejar') or self.cookiejar)
        try:
            requests_res = session.request(method=request.method, url=request.url, data=request.data, headers=headers, timeout=float(request.extensions.get('timeout') or self.timeout), proxies=request.proxies or self.proxies, allow_redirects=True, stream=True)
        except requests.exceptions.TooManyRedirects as e:
            max_redirects_exceeded = True
            requests_res = e.response
        except requests.exceptions.SSLError as e:
            if 'CERTIFICATE_VERIFY_FAILED' in str(e):
                raise CertificateVerifyError(cause=e) from e
            raise SSLError(cause=e) from e
        except requests.exceptions.ProxyError as e:
            raise ProxyError(cause=e) from e
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise TransportError(cause=e) from e
        except urllib3.exceptions.HTTPError as e:
            raise TransportError(cause=e) from e
        except requests.exceptions.RequestException as e:
            raise RequestError(cause=e) from e
        res = RequestsResponseAdapter(requests_res)
        if not 200 <= res.status < 300:
            raise HTTPError(res, redirect_loop=max_redirects_exceeded)
        return res

@register_preference(RequestsRH)
def requests_preference(rh, request):
    if False:
        return 10
    return 100

class SocksHTTPConnection(urllib3.connection.HTTPConnection):

    def __init__(self, _socks_options, *args, **kwargs):
        if False:
            print('Hello World!')
        self._proxy_args = _socks_options
        super().__init__(*args, **kwargs)

    def _new_conn(self):
        if False:
            return 10
        try:
            return create_connection(address=(self._proxy_args['addr'], self._proxy_args['port']), timeout=self.timeout, source_address=self.source_address, _create_socket_func=functools.partial(create_socks_proxy_socket, (self.host, self.port), self._proxy_args))
        except (socket.timeout, TimeoutError) as e:
            raise urllib3.exceptions.ConnectTimeoutError(self, f'Connection to {self.host} timed out. (connect timeout={self.timeout})') from e
        except SocksProxyError as e:
            raise urllib3.exceptions.ProxyError(str(e), e) from e
        except (OSError, socket.error) as e:
            raise urllib3.exceptions.NewConnectionError(self, f'Failed to establish a new connection: {e}') from e

class SocksHTTPSConnection(SocksHTTPConnection, urllib3.connection.HTTPSConnection):
    pass

class SocksHTTPConnectionPool(urllib3.HTTPConnectionPool):
    ConnectionCls = SocksHTTPConnection

class SocksHTTPSConnectionPool(urllib3.HTTPSConnectionPool):
    ConnectionCls = SocksHTTPSConnection

class SocksProxyManager(urllib3.PoolManager):

    def __init__(self, socks_proxy, username=None, password=None, num_pools=10, headers=None, **connection_pool_kw):
        if False:
            return 10
        connection_pool_kw['_socks_options'] = make_socks_proxy_opts(socks_proxy)
        super().__init__(num_pools, headers, **connection_pool_kw)
        self.pool_classes_by_scheme = {'http': SocksHTTPConnectionPool, 'https': SocksHTTPSConnectionPool}
requests.adapters.SOCKSProxyManager = SocksProxyManager