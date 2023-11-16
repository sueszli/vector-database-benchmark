from __future__ import annotations
import re
import socket
import json
import json as unshadowed_json
from base64 import b64encode
from contextlib import contextmanager
from json.decoder import JSONDecodeError
from urllib.parse import urlparse, urlunparse
from ssl import SSLError
import time
import traceback
from typing import Callable, Optional, Tuple, Dict, Any, Generator, cast
from http.cookiejar import CookieJar
import gevent
from gevent.timeout import Timeout
from geventhttpclient._parser import HTTPParseError
from geventhttpclient.client import HTTPClientPool
from geventhttpclient.useragent import UserAgent, CompatRequest, CompatResponse, ConnectionError
from geventhttpclient.response import HTTPConnectionClosed, HTTPSocketPoolResponse
from geventhttpclient.header import Headers
from requests.utils import get_encoding_from_headers
from locust.user import User
from locust.exception import LocustError, CatchResponseError, ResponseError
from locust.env import Environment
from locust.util.deprecation import DeprecatedFastHttpLocustClass as FastHttpLocust
CompatRequest.unverifiable = False
CompatRequest.type = 'https'
absolute_http_url_regexp = re.compile('^https?://', re.I)
FAILURE_EXCEPTIONS = (ConnectionError, ConnectionRefusedError, ConnectionResetError, socket.error, SSLError, Timeout, HTTPConnectionClosed)

def _construct_basic_auth_str(username, password):
    if False:
        while True:
            i = 10
    'Construct Authorization header value to be used in HTTP Basic Auth'
    if isinstance(username, str):
        username = username.encode('latin1')
    if isinstance(password, str):
        password = password.encode('latin1')
    return 'Basic ' + b64encode(b':'.join((username, password))).strip().decode('ascii')

def insecure_ssl_context_factory():
    if False:
        return 10
    context = gevent.ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = gevent.ssl.CERT_NONE
    return context

class FastHttpSession:
    auth_header = None

    def __init__(self, environment: Environment, base_url: str, user: Optional[User], insecure=True, client_pool: Optional[HTTPClientPool]=None, ssl_context_factory: Optional[Callable]=None, **kwargs):
        if False:
            while True:
                i = 10
        self.environment = environment
        self.base_url = base_url
        self.cookiejar = CookieJar()
        self.user = user
        if not ssl_context_factory:
            if insecure:
                ssl_context_factory = insecure_ssl_context_factory
            else:
                ssl_context_factory = gevent.ssl.create_default_context
        self.client = LocustUserAgent(cookiejar=self.cookiejar, ssl_context_factory=ssl_context_factory, insecure=insecure, client_pool=client_pool, **kwargs)
        parsed_url = urlparse(self.base_url)
        if parsed_url.username and parsed_url.password:
            netloc = parsed_url.hostname or ''
            if parsed_url.port:
                netloc += ':%d' % parsed_url.port
            self.base_url = urlunparse((parsed_url.scheme, netloc, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment))
            self.auth_header = _construct_basic_auth_str(parsed_url.username, parsed_url.password)

    def _build_url(self, path):
        if False:
            for i in range(10):
                print('nop')
        "prepend url with hostname unless it's already an absolute URL"
        if absolute_http_url_regexp.match(path):
            return path
        else:
            return f'{self.base_url}{path}'

    def _send_request_safe_mode(self, method, url, **kwargs):
        if False:
            return 10
        '\n        Send an HTTP request, and catch any exception that might occur due to either\n        connection problems, or invalid HTTP status codes\n        '
        try:
            return self.client.urlopen(url, method=method, **kwargs)
        except FAILURE_EXCEPTIONS as e:
            if hasattr(e, 'response'):
                r = e.response
            else:
                req = self.client._make_request(url, method=method, headers=kwargs.get('headers'), payload=kwargs.get('payload'), params=kwargs.get('params'))
                r = ErrorResponse(url=url, request=req)
            r.error = e
            return r

    def request(self, method: str, url: str, name: str | None=None, data: str | dict | None=None, catch_response: bool=False, stream: bool=False, headers: dict | None=None, auth=None, json: dict | None=None, allow_redirects=True, context: dict={}, **kwargs) -> ResponseContextManager | FastResponse:
        if False:
            return 10
        '\n        Send and HTTP request\n        Returns :py:class:`locust.contrib.fasthttp.FastResponse` object.\n\n        :param method: method for the new :class:`Request` object.\n        :param url: path that will be concatenated with the base host URL that has been specified.\n            Can also be a full URL, in which case the full URL will be requested, and the base host\n            is ignored.\n        :param name: (optional) An argument that can be specified to use as label in Locust\'s\n            statistics instead of the URL path. This can be used to group different URL\'s\n            that are requested into a single entry in Locust\'s statistics.\n        :param catch_response: (optional) Boolean argument that, if set, can be used to make a request\n            return a context manager to work as argument to a with statement. This will allow the\n            request to be marked as a fail based on the content of the response, even if the response\n            code is ok (2xx). The opposite also works, one can use catch_response to catch a request\n            and then mark it as successful even if the response code was not (i.e 500 or 404).\n        :param data: (optional) String/bytes to send in the body of the request.\n        :param json: (optional) Dictionary to send in the body of the request.\n            Automatically sets Content-Type and Accept headers to "application/json".\n            Only used if data is not set.\n        :param headers: (optional) Dictionary of HTTP Headers to send with the request.\n        :param auth: (optional) Auth (username, password) tuple to enable Basic HTTP Auth.\n        :param stream: (optional) If set to true the response body will not be consumed immediately\n            and can instead be consumed by accessing the stream attribute on the Response object.\n            Another side effect of setting stream to True is that the time for downloading the response\n            content will not be accounted for in the request time that is reported by Locust.\n        '
        built_url = self._build_url(url)
        start_time = time.time()
        if self.user:
            context = {**self.user.context(), **context}
        headers = headers or {}
        if auth:
            headers['Authorization'] = _construct_basic_auth_str(auth[0], auth[1])
        elif self.auth_header:
            headers['Authorization'] = self.auth_header
        if 'Accept-Encoding' not in headers and 'accept-encoding' not in headers:
            headers['Accept-Encoding'] = 'gzip, deflate'
        if not data and json is not None:
            data = unshadowed_json.dumps(json)
            if 'Content-Type' not in headers and 'content-type' not in headers:
                headers['Content-Type'] = 'application/json'
            if 'Accept' not in headers and 'accept' not in headers:
                headers['Accept'] = 'application/json'
        if not allow_redirects:
            old_redirect_response_codes = self.client.redirect_resonse_codes
            self.client.redirect_resonse_codes = []
        start_perf_counter = time.perf_counter()
        response = self._send_request_safe_mode(method, built_url, payload=data, headers=headers, **kwargs)
        request_meta = {'request_type': method, 'name': name or url, 'context': context, 'response': response, 'exception': None, 'start_time': start_time, 'url': built_url}
        if not allow_redirects:
            self.client.redirect_resonse_codes = old_redirect_response_codes
        if stream:
            request_meta['response_length'] = int(response.headers.get('response_length') or 0)
        else:
            try:
                request_meta['response_length'] = len(response.content or '')
            except HTTPParseError as e:
                request_meta['response_time'] = (time.perf_counter() - start_perf_counter) * 1000
                request_meta['response_length'] = 0
                request_meta['exception'] = e
                self.environment.events.request.fire(**request_meta)
                return response
        request_meta['response_time'] = int((time.perf_counter() - start_perf_counter) * 1000)
        if catch_response:
            return ResponseContextManager(response, environment=self.environment, request_meta=request_meta)
        else:
            try:
                response.raise_for_status()
            except FAILURE_EXCEPTIONS as e:
                request_meta['exception'] = e
            self.environment.events.request.fire(**request_meta)
            return response

    def delete(self, url, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.request('DELETE', url, **kwargs)

    def get(self, url, **kwargs):
        if False:
            while True:
                i = 10
        'Sends a GET request'
        return self.request('GET', url, **kwargs)

    def head(self, url, **kwargs):
        if False:
            i = 10
            return i + 15
        'Sends a HEAD request'
        return self.request('HEAD', url, **kwargs)

    def options(self, url, **kwargs):
        if False:
            i = 10
            return i + 15
        'Sends a OPTIONS request'
        return self.request('OPTIONS', url, **kwargs)

    def patch(self, url, data=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Sends a POST request'
        return self.request('PATCH', url, data=data, **kwargs)

    def post(self, url, data=None, **kwargs):
        if False:
            print('Hello World!')
        'Sends a POST request'
        return self.request('POST', url, data=data, **kwargs)

    def put(self, url, data=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Sends a PUT request'
        return self.request('PUT', url, data=data, **kwargs)

class FastHttpUser(User):
    """
    FastHttpUser provides the same API as HttpUser, but uses geventhttpclient instead of python-requests
    as its underlying client. It uses considerably less CPU on the load generator, and should work
    as a simple drop-in-replacement in most cases.
    """
    network_timeout: float = 60.0
    'Parameter passed to FastHttpSession'
    connection_timeout: float = 60.0
    'Parameter passed to FastHttpSession'
    max_redirects: int = 5
    'Parameter passed to FastHttpSession. Default 5, meaning 4 redirects.'
    max_retries: int = 1
    'Parameter passed to FastHttpSession. Default 1, meaning zero retries.'
    insecure: bool = True
    'Parameter passed to FastHttpSession. Default True, meaning no SSL verification.'
    default_headers: Optional[dict] = None
    'Parameter passed to FastHttpSession. Adds the listed headers to every request.'
    concurrency: int = 10
    'Parameter passed to FastHttpSession. Describes number of concurrent requests allowed by the FastHttpSession. Default 10.\n    Note that setting this value has no effect when custom client_pool was given, and you need to spawn a your own gevent pool\n    to use it (as Users only have one greenlet). See test_fasthttp.py / test_client_pool_concurrency for an example.'
    client_pool: Optional[HTTPClientPool] = None
    'HTTP client pool to use. If not given, a new pool is created per single user.'
    ssl_context_factory: Optional[Callable] = None
    'A callable that return a SSLContext for overriding the default context created by the FastHttpSession.'
    abstract = True
    'Dont register this as a User class that can be run by itself'
    _callstack_regex = re.compile('  File "(\\/.[^"]*)", line (\\d*),(.*)')

    def __init__(self, environment):
        if False:
            print('Hello World!')
        super().__init__(environment)
        if self.host is None:
            raise LocustError('You must specify the base host. Either in the host attribute in the User class, or on the command line using the --host option.')
        if not re.match('^https?://[^/]+', self.host, re.I):
            raise LocustError(f'Invalid host (`{self.host}`), must be a valid base URL. E.g. http://example.com')
        self.client: FastHttpSession = FastHttpSession(self.environment, base_url=self.host, network_timeout=self.network_timeout, connection_timeout=self.connection_timeout, max_redirects=self.max_redirects, max_retries=self.max_retries, insecure=self.insecure, concurrency=self.concurrency, user=self, client_pool=self.client_pool, ssl_context_factory=self.ssl_context_factory, headers=self.default_headers)
        '\n        Instance of HttpSession that is created upon instantiation of User.\n        The client support cookies, and therefore keeps the session between HTTP requests.\n        '

    @contextmanager
    def rest(self, method, url, headers: Optional[dict]=None, **kwargs) -> Generator[RestResponseContextManager, None, None]:
        if False:
            print('Hello World!')
        '\n        A wrapper for self.client.request that:\n\n        * Parses the JSON response to a dict called ``js`` in the response object. Marks the request as failed if the response was not valid JSON.\n        * Defaults ``Content-Type`` and ``Accept`` headers to ``application/json``\n        * Sets ``catch_response=True`` (so always use a :ref:`with-block <catch-response>`)\n        * Catches any unhandled exceptions thrown inside your with-block, marking the sample as failed (instead of exiting the task immediately without even firing the request event)\n        '
        headers = headers or {}
        if not ('Content-Type' in headers or 'content-type' in headers):
            headers['Content-Type'] = 'application/json'
        if not ('Accept' in headers or 'accept' in headers):
            headers['Accept'] = 'application/json'
        with self.client.request(method, url, catch_response=True, headers=headers, **kwargs) as r:
            resp = cast(RestResponseContextManager, r)
            resp.js = None
            if resp.text is None:
                resp.failure(str(resp.error))
            elif resp.text:
                try:
                    resp.js = resp.json()
                except JSONDecodeError as e:
                    resp.failure(f'Could not parse response as JSON. {resp.text[:250]}, response code {resp.status_code}, error {e}')
            try:
                yield resp
            except AssertionError as e:
                if e.args:
                    if e.args[0].endswith(','):
                        short_resp = resp.text[:200] if resp.text else resp.text
                        resp.failure(f'{e.args[0][:-1]}, response was {short_resp}')
                    else:
                        resp.failure(e.args[0])
                else:
                    resp.failure('Assertion failed')
            except Exception as e:
                error_lines = []
                for l in traceback.format_exc().split('\n'):
                    m = self._callstack_regex.match(l)
                    if m:
                        filename = re.sub('/(home|Users/\\w*)/', '~/', m.group(1))
                        error_lines.append(filename + ':' + m.group(2) + m.group(3))
                    short_resp = resp.text[:200] if resp.text else resp.text
                    resp.failure(f"{e.__class__.__name__}: {e} at {', '.join(error_lines)}. Response was {short_resp}")

    @contextmanager
    def rest_(self, method, url, name=None, **kwargs) -> Generator[RestResponseContextManager, None, None]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Some REST api:s use a timestamp as part of their query string (mainly to break through caches).\n        This is a convenience method for that, appending a _=<timestamp> parameter automatically\n        '
        separator = '&' if '?' in url else '?'
        if name is None:
            name = url + separator + '_=...'
        with self.rest(method, f'{url}{separator}_={int(time.time() * 1000)}', name=name, **kwargs) as resp:
            yield resp

class FastRequest(CompatRequest):
    payload: Optional[str] = None

    @property
    def body(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self.payload

class FastResponse(CompatResponse):
    headers: Optional[Headers] = None
    'Dict like object containing the response headers'
    _response: Optional[HTTPSocketPoolResponse] = None
    encoding: Optional[str] = None
    'In some cases setting the encoding explicitly is needed. If so, do it before calling .text'
    request: Optional[FastRequest] = None

    def __init__(self, ghc_response: HTTPSocketPoolResponse, request: Optional[FastRequest]=None, sent_request: Optional[str]=None):
        if False:
            while True:
                i = 10
        super().__init__(ghc_response, request, sent_request)
        self.request = request

    @property
    def text(self) -> Optional[str]:
        if False:
            return 10
        '\n        Returns the text content of the response as a decoded string\n        '
        if self.content is None:
            return None
        if self.encoding is None:
            if self.headers is None:
                self.encoding = 'utf-8'
            else:
                self.encoding = get_encoding_from_headers(self.headers) or ''
        return str(self.content, self.encoding, errors='replace')

    @property
    def url(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Get "response" URL, which is the same as the request URL. This is a small deviation from HttpSession, which gets the final (possibly redirected) URL.\n        '
        if self.request is not None:
            return self.request.url
        return None

    def json(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the response as json and returns a dict\n        '
        return json.loads(self.text)

    def raise_for_status(self):
        if False:
            i = 10
            return i + 15
        'Raise any connection errors that occurred during the request'
        if hasattr(self, 'error') and self.error:
            raise self.error

    @property
    def status_code(self) -> int:
        if False:
            return 10
        '\n        We override status_code in order to return None if no valid response was\n        returned. E.g. in the case of connection errors\n        '
        return self._response.get_code() if self._response is not None else 0

    def _content(self):
        if False:
            print('Hello World!')
        if self.headers is None:
            return None
        return super()._content()

    def success(self):
        if False:
            print('Hello World!')
        raise LocustError('If you want to change the state of the request, you must pass catch_response=True. See http://docs.locust.io/en/stable/writing-a-locustfile.html#validating-responses')

    def failure(self):
        if False:
            while True:
                i = 10
        raise LocustError('If you want to change the state of the request, you must pass catch_response=True. See http://docs.locust.io/en/stable/writing-a-locustfile.html#validating-responses')

class ErrorResponse:
    """
    This is used as a dummy response object when geventhttpclient raises an error
    that doesn't have a real Response object attached. E.g. a socket error or similar
    """
    headers: Optional[Headers] = None
    content = None
    status_code = 0
    error: Optional[Exception] = None
    text: Optional[str] = None
    request: CompatRequest

    def __init__(self, url: str, request: CompatRequest):
        if False:
            for i in range(10):
                print('nop')
        self.url = url
        self.request = request

    def raise_for_status(self):
        if False:
            return 10
        raise self.error

class LocustUserAgent(UserAgent):
    response_type = FastResponse
    request_type = FastRequest
    valid_response_codes = frozenset([200, 201, 202, 203, 204, 205, 206, 207, 208, 226, 301, 302, 303, 307])

    def __init__(self, client_pool: Optional[HTTPClientPool]=None, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        if client_pool is not None:
            self.clientpool = client_pool

    def _urlopen(self, request):
        if False:
            while True:
                i = 10
        'Override _urlopen() in order to make it use the response_type attribute'
        client = self.clientpool.get_client(request.url_split)
        resp = client.request(request.method, request.url_split.request_uri, body=request.payload, headers=request.headers)
        return self.response_type(resp, request=request, sent_request=resp._sent_request)

class ResponseContextManager(FastResponse):
    """
    A Response class that also acts as a context manager that provides the ability to manually
    control if an HTTP request should be marked as successful or a failure in Locust's statistics

    This class is a subclass of :py:class:`FastResponse <locust.contrib.fasthttp.FastResponse>`
    with two additional methods: :py:meth:`success <locust.contrib.fasthttp.ResponseContextManager.success>`
    and :py:meth:`failure <locust.contrib.fasthttp.ResponseContextManager.failure>`.
    """
    _manual_result = None
    _entered = False

    def __init__(self, response, environment, request_meta):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__ = response.__dict__
        self._cached_content = response.content
        self._environment = environment
        self.request_meta = request_meta

    def __enter__(self):
        if False:
            while True:
                i = 10
        self._entered = True
        return self

    def __exit__(self, exc, value, traceback):
        if False:
            return 10
        if self._manual_result is not None:
            if self._manual_result is True:
                self._report_request()
            elif isinstance(self._manual_result, Exception):
                self.request_meta['exception'] = self._manual_result
                self._report_request()
            return exc is None
        if exc:
            if isinstance(value, ResponseError):
                self.request_meta['exception'] = value
                self._report_request()
            else:
                return False
        else:
            try:
                self.raise_for_status()
            except FAILURE_EXCEPTIONS as e:
                self.request_meta['exception'] = e
            self._report_request()
        return True

    def _report_request(self):
        if False:
            return 10
        self._environment.events.request.fire(**self.request_meta)

    def success(self):
        if False:
            i = 10
            return i + 15
        '\n        Report the response as successful\n\n        Example::\n\n            with self.client.get("/does/not/exist", catch_response=True) as response:\n                if response.status_code == 404:\n                    response.success()\n        '
        if not self._entered:
            raise LocustError('Tried to set status on a request that has not yet been made. Make sure you use a with-block, like this:\n\nwith self.client.request(..., catch_response=True) as response:\n    response.success()')
        self._manual_result = True

    def failure(self, exc):
        if False:
            i = 10
            return i + 15
        '\n        Report the response as a failure.\n\n        if exc is anything other than a python exception (like a string) it will\n        be wrapped inside a CatchResponseError.\n\n        Example::\n\n            with self.client.get("/", catch_response=True) as response:\n                if response.content == "":\n                    response.failure("No data")\n        '
        if not self._entered:
            raise LocustError('Tried to set status on a request that has not yet been made. Make sure you use a with-block, like this:\n\nwith self.client.request(..., catch_response=True) as response:\n    response.failure(...)')
        if not isinstance(exc, Exception):
            exc = CatchResponseError(exc)
        self._manual_result = exc

class RestResponseContextManager(ResponseContextManager):
    js: dict
    error: Exception
    headers: Headers