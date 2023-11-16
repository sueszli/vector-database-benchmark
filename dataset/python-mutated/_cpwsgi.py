"""WSGI interface (see PEP 333 and 3333).

Note that WSGI environ keys and values are 'native strings'; that is,
whatever the type of "" is. For Python 2, that's a byte string; for Python 3,
it's a unicode string. But PEP 3333 says: "even if Python's str type is
actually Unicode "under the hood", the content of native strings must
still be translatable to bytes via the Latin-1 encoding!"
"""
import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator

def downgrade_wsgi_ux_to_1x(environ):
    if False:
        i = 10
        return i + 15
    'Return a new environ dict for WSGI 1.x from the given WSGI u.x environ.\n    '
    env1x = {}
    url_encoding = environ[ntou('wsgi.url_encoding')]
    for (k, v) in environ.copy().items():
        if k in [ntou('PATH_INFO'), ntou('SCRIPT_NAME'), ntou('QUERY_STRING')]:
            v = v.encode(url_encoding)
        elif isinstance(v, str):
            v = v.encode('ISO-8859-1')
        env1x[k.encode('ISO-8859-1')] = v
    return env1x

class VirtualHost(object):
    """Select a different WSGI application based on the Host header.

    This can be useful when running multiple sites within one CP server.
    It allows several domains to point to different applications. For example::

        root = Root()
        RootApp = cherrypy.Application(root)
        Domain2App = cherrypy.Application(root)
        SecureApp = cherrypy.Application(Secure())

        vhost = cherrypy._cpwsgi.VirtualHost(
            RootApp,
            domains={
                'www.domain2.example': Domain2App,
                'www.domain2.example:443': SecureApp,
            },
        )

        cherrypy.tree.graft(vhost)
    """
    default = None
    'Required. The default WSGI application.'
    use_x_forwarded_host = True
    'If True (the default), any "X-Forwarded-Host"\n    request header will be used instead of the "Host" header. This\n    is commonly added by HTTP servers (such as Apache) when proxying.'
    domains = {}
    'A dict of {host header value: application} pairs.\n    The incoming "Host" request header is looked up in this dict,\n    and, if a match is found, the corresponding WSGI application\n    will be called instead of the default. Note that you often need\n    separate entries for "example.com" and "www.example.com".\n    In addition, "Host" headers may contain the port number.\n    '

    def __init__(self, default, domains=None, use_x_forwarded_host=True):
        if False:
            while True:
                i = 10
        self.default = default
        self.domains = domains or {}
        self.use_x_forwarded_host = use_x_forwarded_host

    def __call__(self, environ, start_response):
        if False:
            i = 10
            return i + 15
        domain = environ.get('HTTP_HOST', '')
        if self.use_x_forwarded_host:
            domain = environ.get('HTTP_X_FORWARDED_HOST', domain)
        nextapp = self.domains.get(domain)
        if nextapp is None:
            nextapp = self.default
        return nextapp(environ, start_response)

class InternalRedirector(object):
    """WSGI middleware that handles raised cherrypy.InternalRedirect."""

    def __init__(self, nextapp, recursive=False):
        if False:
            i = 10
            return i + 15
        self.nextapp = nextapp
        self.recursive = recursive

    def __call__(self, environ, start_response):
        if False:
            return 10
        redirections = []
        while True:
            environ = environ.copy()
            try:
                return self.nextapp(environ, start_response)
            except _cherrypy.InternalRedirect:
                ir = _sys.exc_info()[1]
                sn = environ.get('SCRIPT_NAME', '')
                path = environ.get('PATH_INFO', '')
                qs = environ.get('QUERY_STRING', '')
                old_uri = sn + path
                if qs:
                    old_uri += '?' + qs
                redirections.append(old_uri)
                if not self.recursive:
                    new_uri = sn + ir.path
                    if ir.query_string:
                        new_uri += '?' + ir.query_string
                    if new_uri in redirections:
                        ir.request.close()
                        tmpl = 'InternalRedirector visited the same URL twice: %r'
                        raise RuntimeError(tmpl % new_uri)
                environ['REQUEST_METHOD'] = 'GET'
                environ['PATH_INFO'] = ir.path
                environ['QUERY_STRING'] = ir.query_string
                environ['wsgi.input'] = io.BytesIO()
                environ['CONTENT_LENGTH'] = '0'
                environ['cherrypy.previous_request'] = ir.request

class ExceptionTrapper(object):
    """WSGI middleware that traps exceptions."""

    def __init__(self, nextapp, throws=(KeyboardInterrupt, SystemExit)):
        if False:
            while True:
                i = 10
        self.nextapp = nextapp
        self.throws = throws

    def __call__(self, environ, start_response):
        if False:
            print('Hello World!')
        return _TrappedResponse(self.nextapp, environ, start_response, self.throws)

class _TrappedResponse(object):
    response = iter([])

    def __init__(self, nextapp, environ, start_response, throws):
        if False:
            return 10
        self.nextapp = nextapp
        self.environ = environ
        self.start_response = start_response
        self.throws = throws
        self.started_response = False
        self.response = self.trap(self.nextapp, self.environ, self.start_response)
        self.iter_response = iter(self.response)

    def __iter__(self):
        if False:
            return 10
        self.started_response = True
        return self

    def __next__(self):
        if False:
            return 10
        return self.trap(next, self.iter_response)

    def close(self):
        if False:
            return 10
        if hasattr(self.response, 'close'):
            self.response.close()

    def trap(self, func, *args, **kwargs):
        if False:
            while True:
                i = 10
        try:
            return func(*args, **kwargs)
        except self.throws:
            raise
        except StopIteration:
            raise
        except Exception:
            tb = _cperror.format_exc()
            _cherrypy.log(tb, severity=40)
            if not _cherrypy.request.show_tracebacks:
                tb = ''
            (s, h, b) = _cperror.bare_error(tb)
            if True:
                s = s.decode('ISO-8859-1')
                h = [(k.decode('ISO-8859-1'), v.decode('ISO-8859-1')) for (k, v) in h]
            if self.started_response:
                self.iter_response = iter([])
            else:
                self.iter_response = iter(b)
            try:
                self.start_response(s, h, _sys.exc_info())
            except Exception:
                _cherrypy.log(traceback=True, severity=40)
                raise
            if self.started_response:
                return b''.join(b)
            else:
                return b

class AppResponse(object):
    """WSGI response iterable for CherryPy applications."""

    def __init__(self, environ, start_response, cpapp):
        if False:
            i = 10
            return i + 15
        self.cpapp = cpapp
        try:
            self.environ = environ
            self.run()
            r = _cherrypy.serving.response
            outstatus = r.output_status
            if not isinstance(outstatus, bytes):
                raise TypeError('response.output_status is not a byte string.')
            outheaders = []
            for (k, v) in r.header_list:
                if not isinstance(k, bytes):
                    tmpl = 'response.header_list key %r is not a byte string.'
                    raise TypeError(tmpl % k)
                if not isinstance(v, bytes):
                    tmpl = 'response.header_list value %r is not a byte string.'
                    raise TypeError(tmpl % v)
                outheaders.append((k, v))
            if True:
                outstatus = outstatus.decode('ISO-8859-1')
                outheaders = [(k.decode('ISO-8859-1'), v.decode('ISO-8859-1')) for (k, v) in outheaders]
            self.iter_response = iter(r.body)
            self.write = start_response(outstatus, outheaders)
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        return next(self.iter_response)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Close and de-reference the current request and response. (Core)'
        streaming = _cherrypy.serving.response.stream
        self.cpapp.release_serving()
        if streaming and is_closable_iterator(self.iter_response):
            iter_close = self.iter_response.close
            try:
                iter_close()
            except Exception:
                _cherrypy.log(traceback=True, severity=40)

    def run(self):
        if False:
            while True:
                i = 10
        'Create a Request object using environ.'
        env = self.environ.get
        local = httputil.Host('', int(env('SERVER_PORT', 80) or -1), env('SERVER_NAME', ''))
        remote = httputil.Host(env('REMOTE_ADDR', ''), int(env('REMOTE_PORT', -1) or -1), env('REMOTE_HOST', ''))
        scheme = env('wsgi.url_scheme')
        sproto = env('ACTUAL_SERVER_PROTOCOL', 'HTTP/1.1')
        (request, resp) = self.cpapp.get_serving(local, remote, scheme, sproto)
        request.login = env('LOGON_USER') or env('REMOTE_USER') or None
        request.multithread = self.environ['wsgi.multithread']
        request.multiprocess = self.environ['wsgi.multiprocess']
        request.wsgi_environ = self.environ
        request.prev = env('cherrypy.previous_request', None)
        meth = self.environ['REQUEST_METHOD']
        path = httputil.urljoin(self.environ.get('SCRIPT_NAME', ''), self.environ.get('PATH_INFO', ''))
        qs = self.environ.get('QUERY_STRING', '')
        (path, qs) = self.recode_path_qs(path, qs) or (path, qs)
        rproto = self.environ.get('SERVER_PROTOCOL')
        headers = self.translate_headers(self.environ)
        rfile = self.environ['wsgi.input']
        request.run(meth, path, qs, rproto, headers, rfile)
    headerNames = {'HTTP_CGI_AUTHORIZATION': 'Authorization', 'CONTENT_LENGTH': 'Content-Length', 'CONTENT_TYPE': 'Content-Type', 'REMOTE_HOST': 'Remote-Host', 'REMOTE_ADDR': 'Remote-Addr'}

    def recode_path_qs(self, path, qs):
        if False:
            return 10
        old_enc = self.environ.get('wsgi.url_encoding', 'ISO-8859-1')
        new_enc = self.cpapp.find_config(self.environ.get('PATH_INFO', ''), 'request.uri_encoding', 'utf-8')
        if new_enc.lower() == old_enc.lower():
            return
        try:
            return (path.encode(old_enc).decode(new_enc), qs.encode(old_enc).decode(new_enc))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    def translate_headers(self, environ):
        if False:
            for i in range(10):
                print('nop')
        'Translate CGI-environ header names to HTTP header names.'
        for cgiName in environ:
            if cgiName in self.headerNames:
                yield (self.headerNames[cgiName], environ[cgiName])
            elif cgiName[:5] == 'HTTP_':
                translatedHeader = cgiName[5:].replace('_', '-')
                yield (translatedHeader, environ[cgiName])

class CPWSGIApp(object):
    """A WSGI application object for a CherryPy Application."""
    pipeline = [('ExceptionTrapper', ExceptionTrapper), ('InternalRedirector', InternalRedirector)]
    "A list of (name, wsgiapp) pairs. Each 'wsgiapp' MUST be a\n    constructor that takes an initial, positional 'nextapp' argument,\n    plus optional keyword arguments, and returns a WSGI application\n    (that takes environ and start_response arguments). The 'name' can\n    be any you choose, and will correspond to keys in self.config."
    head = None
    "Rather than nest all apps in the pipeline on each call, it's only\n    done the first time, and the result is memoized into self.head. Set\n    this to None again if you change self.pipeline after calling self."
    config = {}
    'A dict whose keys match names listed in the pipeline. Each\n    value is a further dict which will be passed to the corresponding\n    named WSGI callable (from the pipeline) as keyword arguments.'
    response_class = AppResponse
    'The class to instantiate and return as the next app in the WSGI chain.\n    '

    def __init__(self, cpapp, pipeline=None):
        if False:
            for i in range(10):
                print('nop')
        self.cpapp = cpapp
        self.pipeline = self.pipeline[:]
        if pipeline:
            self.pipeline.extend(pipeline)
        self.config = self.config.copy()

    def tail(self, environ, start_response):
        if False:
            for i in range(10):
                print('nop')
        "WSGI application callable for the actual CherryPy application.\n\n        You probably shouldn't call this; call self.__call__ instead,\n        so that any WSGI middleware in self.pipeline can run first.\n        "
        return self.response_class(environ, start_response, self.cpapp)

    def __call__(self, environ, start_response):
        if False:
            for i in range(10):
                print('nop')
        head = self.head
        if head is None:
            head = self.tail
            for (name, callable) in self.pipeline[::-1]:
                conf = self.config.get(name, {})
                head = callable(head, **conf)
            self.head = head
        return head(environ, start_response)

    def namespace_handler(self, k, v):
        if False:
            for i in range(10):
                print('nop')
        "Config handler for the 'wsgi' namespace."
        if k == 'pipeline':
            self.pipeline.extend(v)
        elif k == 'response_class':
            self.response_class = v
        else:
            (name, arg) = k.split('.', 1)
            bucket = self.config.setdefault(name, {})
            bucket[arg] = v