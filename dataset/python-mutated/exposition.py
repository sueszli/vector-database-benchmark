import base64
from contextlib import closing
import gzip
from http.server import BaseHTTPRequestHandler
import os
import socket
from socketserver import ThreadingMixIn
import ssl
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import BaseHandler, build_opener, HTTPHandler, HTTPRedirectHandler, HTTPSHandler, Request
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer
from .openmetrics import exposition as openmetrics
from .registry import CollectorRegistry, REGISTRY
from .utils import floatToGoString
__all__ = ('CONTENT_TYPE_LATEST', 'delete_from_gateway', 'generate_latest', 'instance_ip_grouping_key', 'make_asgi_app', 'make_wsgi_app', 'MetricsHandler', 'push_to_gateway', 'pushadd_to_gateway', 'start_http_server', 'start_wsgi_server', 'write_to_textfile')
CONTENT_TYPE_LATEST = 'text/plain; version=0.0.4; charset=utf-8'
'Content type of the latest text format'

class _PrometheusRedirectHandler(HTTPRedirectHandler):
    """
    Allow additional methods (e.g. PUT) and data forwarding in redirects.

    Use of this class constitute a user's explicit agreement to the
    redirect responses the Prometheus client will receive when using it.
    You should only use this class if you control or otherwise trust the
    redirect behavior involved and are certain it is safe to full transfer
    the original request (method and data) to the redirected URL. For
    example, if you know there is a cosmetic URL redirect in front of a
    local deployment of a Prometheus server, and all redirects are safe,
    this is the class to use to handle redirects in that case.

    The standard HTTPRedirectHandler does not forward request data nor
    does it allow redirected PUT requests (which Prometheus uses for some
    operations, for example `push_to_gateway`) because these cannot
    generically guarantee no violations of HTTP RFC 2616 requirements for
    the user to explicitly confirm redirects that could have unexpected
    side effects (such as rendering a PUT request non-idempotent or
    creating multiple resources not named in the original request).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if False:
            print('Hello World!')
        "\n        Apply redirect logic to a request.\n\n        See parent HTTPRedirectHandler.redirect_request for parameter info.\n\n        If the redirect is disallowed, this raises the corresponding HTTP error.\n        If the redirect can't be determined, return None to allow other handlers\n        to try. If the redirect is allowed, return the new request.\n\n        This method specialized for the case when (a) the user knows that the\n        redirect will not cause unacceptable side effects for any request method,\n        and (b) the user knows that any request data should be passed through to\n        the redirect. If either condition is not met, this should not be used.\n        "
        m = getattr(req, 'method', req.get_method())
        if not (code in (301, 302, 303, 307) and m in ('GET', 'HEAD') or (code in (301, 302, 303) and m in ('POST', 'PUT'))):
            raise HTTPError(req.full_url, code, msg, headers, fp)
        new_request = Request(newurl.replace(' ', '%20'), headers=req.headers, origin_req_host=req.origin_req_host, unverifiable=True, data=req.data)
        new_request.method = m
        return new_request

def _bake_output(registry, accept_header, accept_encoding_header, params, disable_compression):
    if False:
        for i in range(10):
            print('nop')
    'Bake output for metrics output.'
    (encoder, content_type) = choose_encoder(accept_header)
    if 'name[]' in params:
        registry = registry.restricted_registry(params['name[]'])
    output = encoder(registry)
    headers = [('Content-Type', content_type)]
    if not disable_compression and gzip_accepted(accept_encoding_header):
        output = gzip.compress(output)
        headers.append(('Content-Encoding', 'gzip'))
    return ('200 OK', headers, output)

def make_wsgi_app(registry: CollectorRegistry=REGISTRY, disable_compression: bool=False) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    'Create a WSGI app which serves the metrics from a registry.'

    def prometheus_app(environ, start_response):
        if False:
            print('Hello World!')
        accept_header = environ.get('HTTP_ACCEPT')
        accept_encoding_header = environ.get('HTTP_ACCEPT_ENCODING')
        params = parse_qs(environ.get('QUERY_STRING', ''))
        if environ['PATH_INFO'] == '/favicon.ico':
            status = '200 OK'
            headers = [('', '')]
            output = b''
        else:
            (status, headers, output) = _bake_output(registry, accept_header, accept_encoding_header, params, disable_compression)
        start_response(status, headers)
        return [output]
    return prometheus_app

class _SilentHandler(WSGIRequestHandler):
    """WSGI handler that does not log requests."""

    def log_message(self, format, *args):
        if False:
            for i in range(10):
                print('nop')
        'Log nothing.'

class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    """Thread per request HTTP server."""
    daemon_threads = True

def _get_best_family(address, port):
    if False:
        for i in range(10):
            print('nop')
    'Automatically select address family depending on address'
    infos = socket.getaddrinfo(address, port)
    (family, _, _, _, sockaddr) = next(iter(infos))
    return (family, sockaddr[0])

def _get_ssl_ctx(certfile: str, keyfile: str, protocol: int, cafile: Optional[str]=None, capath: Optional[str]=None, client_auth_required: bool=False) -> ssl.SSLContext:
    if False:
        print('Hello World!')
    'Load context supports SSL.'
    ssl_cxt = ssl.SSLContext(protocol=protocol)
    if cafile is not None or capath is not None:
        try:
            ssl_cxt.load_verify_locations(cafile, capath)
        except IOError as exc:
            exc_type = type(exc)
            msg = str(exc)
            raise exc_type(f'Cannot load CA certificate chain from file {cafile!r} or directory {capath!r}: {msg}')
    else:
        try:
            ssl_cxt.load_default_certs(purpose=ssl.Purpose.CLIENT_AUTH)
        except IOError as exc:
            exc_type = type(exc)
            msg = str(exc)
            raise exc_type(f'Cannot load default CA certificate chain: {msg}')
    if client_auth_required:
        ssl_cxt.verify_mode = ssl.CERT_REQUIRED
    try:
        ssl_cxt.load_cert_chain(certfile=certfile, keyfile=keyfile)
    except IOError as exc:
        exc_type = type(exc)
        msg = str(exc)
        raise exc_type(f'Cannot load server certificate file {certfile!r} or its private key file {keyfile!r}: {msg}')
    return ssl_cxt

def start_wsgi_server(port: int, addr: str='0.0.0.0', registry: CollectorRegistry=REGISTRY, certfile: Optional[str]=None, keyfile: Optional[str]=None, client_cafile: Optional[str]=None, client_capath: Optional[str]=None, protocol: int=ssl.PROTOCOL_TLS_SERVER, client_auth_required: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    'Starts a WSGI server for prometheus metrics as a daemon thread.'

    class TmpServer(ThreadingWSGIServer):
        """Copy of ThreadingWSGIServer to update address_family locally"""
    (TmpServer.address_family, addr) = _get_best_family(addr, port)
    app = make_wsgi_app(registry)
    httpd = make_server(addr, port, app, TmpServer, handler_class=_SilentHandler)
    if certfile and keyfile:
        context = _get_ssl_ctx(certfile, keyfile, protocol, client_cafile, client_capath, client_auth_required)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    t = threading.Thread(target=httpd.serve_forever)
    t.daemon = True
    t.start()
start_http_server = start_wsgi_server

def generate_latest(registry: CollectorRegistry=REGISTRY) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    'Returns the metrics from the registry in latest text format as a string.'

    def sample_line(line):
        if False:
            i = 10
            return i + 15
        if line.labels:
            labelstr = '{{{0}}}'.format(','.join(['{}="{}"'.format(k, v.replace('\\', '\\\\').replace('\n', '\\n').replace('"', '\\"')) for (k, v) in sorted(line.labels.items())]))
        else:
            labelstr = ''
        timestamp = ''
        if line.timestamp is not None:
            timestamp = f' {int(float(line.timestamp) * 1000):d}'
        return f'{line.name}{labelstr} {floatToGoString(line.value)}{timestamp}\n'
    output = []
    for metric in registry.collect():
        try:
            mname = metric.name
            mtype = metric.type
            if mtype == 'counter':
                mname = mname + '_total'
            elif mtype == 'info':
                mname = mname + '_info'
                mtype = 'gauge'
            elif mtype == 'stateset':
                mtype = 'gauge'
            elif mtype == 'gaugehistogram':
                mtype = 'histogram'
            elif mtype == 'unknown':
                mtype = 'untyped'
            output.append('# HELP {} {}\n'.format(mname, metric.documentation.replace('\\', '\\\\').replace('\n', '\\n')))
            output.append(f'# TYPE {mname} {mtype}\n')
            om_samples: Dict[str, List[str]] = {}
            for s in metric.samples:
                for suffix in ['_created', '_gsum', '_gcount']:
                    if s.name == metric.name + suffix:
                        om_samples.setdefault(suffix, []).append(sample_line(s))
                        break
                else:
                    output.append(sample_line(s))
        except Exception as exception:
            exception.args = (exception.args or ('',)) + (metric,)
            raise
        for (suffix, lines) in sorted(om_samples.items()):
            output.append('# HELP {}{} {}\n'.format(metric.name, suffix, metric.documentation.replace('\\', '\\\\').replace('\n', '\\n')))
            output.append(f'# TYPE {metric.name}{suffix} gauge\n')
            output.extend(lines)
    return ''.join(output).encode('utf-8')

def choose_encoder(accept_header: str) -> Tuple[Callable[[CollectorRegistry], bytes], str]:
    if False:
        while True:
            i = 10
    accept_header = accept_header or ''
    for accepted in accept_header.split(','):
        if accepted.split(';')[0].strip() == 'application/openmetrics-text':
            return (openmetrics.generate_latest, openmetrics.CONTENT_TYPE_LATEST)
    return (generate_latest, CONTENT_TYPE_LATEST)

def gzip_accepted(accept_encoding_header: str) -> bool:
    if False:
        return 10
    accept_encoding_header = accept_encoding_header or ''
    for accepted in accept_encoding_header.split(','):
        if accepted.split(';')[0].strip().lower() == 'gzip':
            return True
    return False

class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that gives metrics from ``REGISTRY``."""
    registry: CollectorRegistry = REGISTRY

    def do_GET(self) -> None:
        if False:
            return 10
        registry = self.registry
        accept_header = self.headers.get('Accept')
        accept_encoding_header = self.headers.get('Accept-Encoding')
        params = parse_qs(urlparse(self.path).query)
        (status, headers, output) = _bake_output(registry, accept_header, accept_encoding_header, params, False)
        self.send_response(int(status.split(' ')[0]))
        for header in headers:
            self.send_header(*header)
        self.end_headers()
        self.wfile.write(output)

    def log_message(self, format: str, *args: Any) -> None:
        if False:
            return 10
        'Log nothing.'

    @classmethod
    def factory(cls, registry: CollectorRegistry) -> type:
        if False:
            i = 10
            return i + 15
        'Returns a dynamic MetricsHandler class tied\n           to the passed registry.\n        '
        cls_name = str(cls.__name__)
        MyMetricsHandler = type(cls_name, (cls, object), {'registry': registry})
        return MyMetricsHandler

def write_to_textfile(path: str, registry: CollectorRegistry) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Write metrics to the given path.\n\n    This is intended for use with the Node exporter textfile collector.\n    The path must end in .prom for the textfile collector to process it.'
    tmppath = f'{path}.{os.getpid()}.{threading.current_thread().ident}'
    with open(tmppath, 'wb') as f:
        f.write(generate_latest(registry))
    if os.name == 'nt':
        os.replace(tmppath, path)
    else:
        os.rename(tmppath, path)

def _make_handler(url: str, method: str, timeout: Optional[float], headers: Sequence[Tuple[str, str]], data: bytes, base_handler: Union[BaseHandler, type]) -> Callable[[], None]:
    if False:
        print('Hello World!')

    def handle() -> None:
        if False:
            print('Hello World!')
        request = Request(url, data=data)
        request.get_method = lambda : method
        for (k, v) in headers:
            request.add_header(k, v)
        resp = build_opener(base_handler).open(request, timeout=timeout)
        if resp.code >= 400:
            raise OSError(f'error talking to pushgateway: {resp.code} {resp.msg}')
    return handle

def default_handler(url: str, method: str, timeout: Optional[float], headers: List[Tuple[str, str]], data: bytes) -> Callable[[], None]:
    if False:
        print('Hello World!')
    'Default handler that implements HTTP/HTTPS connections.\n\n    Used by the push_to_gateway functions. Can be re-used by other handlers.'
    return _make_handler(url, method, timeout, headers, data, HTTPHandler)

def passthrough_redirect_handler(url: str, method: str, timeout: Optional[float], headers: List[Tuple[str, str]], data: bytes) -> Callable[[], None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Handler that automatically trusts redirect responses for all HTTP methods.\n\n    Augments standard HTTPRedirectHandler capability by permitting PUT requests,\n    preserving the method upon redirect, and passing through all headers and\n    data from the original request. Only use this handler if you control or\n    trust the source of redirect responses you encounter when making requests\n    via the Prometheus client. This handler will simply repeat the identical\n    request, including same method and data, to the new redirect URL.'
    return _make_handler(url, method, timeout, headers, data, _PrometheusRedirectHandler)

def basic_auth_handler(url: str, method: str, timeout: Optional[float], headers: List[Tuple[str, str]], data: bytes, username: Optional[str]=None, password: Optional[str]=None) -> Callable[[], None]:
    if False:
        i = 10
        return i + 15
    "Handler that implements HTTP/HTTPS connections with Basic Auth.\n\n    Sets auth headers using supplied 'username' and 'password', if set.\n    Used by the push_to_gateway functions. Can be re-used by other handlers."

    def handle():
        if False:
            for i in range(10):
                print('nop')
        'Handler that implements HTTP Basic Auth.\n        '
        if username is not None and password is not None:
            auth_value = f'{username}:{password}'.encode()
            auth_token = base64.b64encode(auth_value)
            auth_header = b'Basic ' + auth_token
            headers.append(('Authorization', auth_header))
        default_handler(url, method, timeout, headers, data)()
    return handle

def tls_auth_handler(url: str, method: str, timeout: Optional[float], headers: List[Tuple[str, str]], data: bytes, certfile: str, keyfile: str, cafile: Optional[str]=None, protocol: int=ssl.PROTOCOL_TLS_CLIENT, insecure_skip_verify: bool=False) -> Callable[[], None]:
    if False:
        print('Hello World!')
    'Handler that implements an HTTPS connection with TLS Auth.\n\n    The default protocol (ssl.PROTOCOL_TLS_CLIENT) will also enable\n    ssl.CERT_REQUIRED and SSLContext.check_hostname by default. This can be\n    disabled by setting insecure_skip_verify to True.\n\n    Both this handler and the TLS feature on pushgateay are experimental.'
    context = ssl.SSLContext(protocol=protocol)
    if cafile is not None:
        context.load_verify_locations(cafile)
    else:
        context.load_default_certs()
    if insecure_skip_verify:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    handler = HTTPSHandler(context=context)
    return _make_handler(url, method, timeout, headers, data, handler)

def push_to_gateway(gateway: str, job: str, registry: CollectorRegistry, grouping_key: Optional[Dict[str, Any]]=None, timeout: Optional[float]=30, handler: Callable=default_handler) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Push metrics to the given pushgateway.\n\n    `gateway` the url for your push gateway. Either of the form\n              \'http://pushgateway.local\', or \'pushgateway.local\'.\n              Scheme defaults to \'http\' if none is provided\n    `job` is the job label to be attached to all pushed metrics\n    `registry` is an instance of CollectorRegistry\n    `grouping_key` please see the pushgateway documentation for details.\n                   Defaults to None\n    `timeout` is how long push will attempt to connect before giving up.\n              Defaults to 30s, can be set to None for no timeout.\n    `handler` is an optional function which can be provided to perform\n              requests to the \'gateway\'.\n              Defaults to None, in which case an http or https request\n              will be carried out by a default handler.\n              If not None, the argument must be a function which accepts\n              the following arguments:\n              url, method, timeout, headers, and content\n              May be used to implement additional functionality not\n              supported by the built-in default handler (such as SSL\n              client certicates, and HTTP authentication mechanisms).\n              \'url\' is the URL for the request, the \'gateway\' argument\n              described earlier will form the basis of this URL.\n              \'method\' is the HTTP method which should be used when\n              carrying out the request.\n              \'timeout\' requests not successfully completed after this\n              many seconds should be aborted.  If timeout is None, then\n              the handler should not set a timeout.\n              \'headers\' is a list of ("header-name","header-value") tuples\n              which must be passed to the pushgateway in the form of HTTP\n              request headers.\n              The function should raise an exception (e.g. IOError) on\n              failure.\n              \'content\' is the data which should be used to form the HTTP\n              Message Body.\n\n    This overwrites all metrics with the same job and grouping_key.\n    This uses the PUT HTTP method.'
    _use_gateway('PUT', gateway, job, registry, grouping_key, timeout, handler)

def pushadd_to_gateway(gateway: str, job: str, registry: Optional[CollectorRegistry], grouping_key: Optional[Dict[str, Any]]=None, timeout: Optional[float]=30, handler: Callable=default_handler) -> None:
    if False:
        print('Hello World!')
    "PushAdd metrics to the given pushgateway.\n\n    `gateway` the url for your push gateway. Either of the form\n              'http://pushgateway.local', or 'pushgateway.local'.\n              Scheme defaults to 'http' if none is provided\n    `job` is the job label to be attached to all pushed metrics\n    `registry` is an instance of CollectorRegistry\n    `grouping_key` please see the pushgateway documentation for details.\n                   Defaults to None\n    `timeout` is how long push will attempt to connect before giving up.\n              Defaults to 30s, can be set to None for no timeout.\n    `handler` is an optional function which can be provided to perform\n              requests to the 'gateway'.\n              Defaults to None, in which case an http or https request\n              will be carried out by a default handler.\n              See the 'prometheus_client.push_to_gateway' documentation\n              for implementation requirements.\n\n    This replaces metrics with the same name, job and grouping_key.\n    This uses the POST HTTP method."
    _use_gateway('POST', gateway, job, registry, grouping_key, timeout, handler)

def delete_from_gateway(gateway: str, job: str, grouping_key: Optional[Dict[str, Any]]=None, timeout: Optional[float]=30, handler: Callable=default_handler) -> None:
    if False:
        print('Hello World!')
    "Delete metrics from the given pushgateway.\n\n    `gateway` the url for your push gateway. Either of the form\n              'http://pushgateway.local', or 'pushgateway.local'.\n              Scheme defaults to 'http' if none is provided\n    `job` is the job label to be attached to all pushed metrics\n    `grouping_key` please see the pushgateway documentation for details.\n                   Defaults to None\n    `timeout` is how long delete will attempt to connect before giving up.\n              Defaults to 30s, can be set to None for no timeout.\n    `handler` is an optional function which can be provided to perform\n              requests to the 'gateway'.\n              Defaults to None, in which case an http or https request\n              will be carried out by a default handler.\n              See the 'prometheus_client.push_to_gateway' documentation\n              for implementation requirements.\n\n    This deletes metrics with the given job and grouping_key.\n    This uses the DELETE HTTP method."
    _use_gateway('DELETE', gateway, job, None, grouping_key, timeout, handler)

def _use_gateway(method: str, gateway: str, job: str, registry: Optional[CollectorRegistry], grouping_key: Optional[Dict[str, Any]], timeout: Optional[float], handler: Callable) -> None:
    if False:
        print('Hello World!')
    gateway_url = urlparse(gateway)
    if not gateway_url.scheme or gateway_url.scheme not in ['http', 'https']:
        gateway = f'http://{gateway}'
    gateway = gateway.rstrip('/')
    url = '{}/metrics/{}/{}'.format(gateway, *_escape_grouping_key('job', job))
    data = b''
    if method != 'DELETE':
        if registry is None:
            registry = REGISTRY
        data = generate_latest(registry)
    if grouping_key is None:
        grouping_key = {}
    url += ''.join(('/{}/{}'.format(*_escape_grouping_key(str(k), str(v))) for (k, v) in sorted(grouping_key.items())))
    handler(url=url, method=method, timeout=timeout, headers=[('Content-Type', CONTENT_TYPE_LATEST)], data=data)()

def _escape_grouping_key(k, v):
    if False:
        i = 10
        return i + 15
    if v == '':
        return (k + '@base64', '=')
    elif '/' in v:
        return (k + '@base64', base64.urlsafe_b64encode(v.encode('utf-8')).decode('utf-8'))
    else:
        return (k, quote_plus(v))

def instance_ip_grouping_key() -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Grouping key with instance set to the IP Address of this host.'
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
        if sys.platform == 'darwin':
            s.connect(('10.255.255.255', 1))
        else:
            s.connect(('localhost', 0))
        return {'instance': s.getsockname()[0]}
from .asgi import make_asgi_app