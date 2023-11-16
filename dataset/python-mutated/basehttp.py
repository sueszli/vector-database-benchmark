"""
HTTP server that implements the Python WSGI protocol (PEP 333, rev 1.21).

Based on wsgiref.simple_server which is part of the standard library since 2.5.

This is a simple server for use in testing or debugging Django apps. It hasn't
been reviewed for security issues. DON'T USE IT FOR PRODUCTION USE!
"""
import logging
import socket
import socketserver
import sys
from collections import deque
from wsgiref import simple_server
from django.core.exceptions import ImproperlyConfigured
from django.core.handlers.wsgi import LimitedStream
from django.core.wsgi import get_wsgi_application
from django.db import connections
from django.utils.module_loading import import_string
__all__ = ('WSGIServer', 'WSGIRequestHandler')
logger = logging.getLogger('django.server')

def get_internal_wsgi_application():
    if False:
        for i in range(10):
            print('nop')
    "\n    Load and return the WSGI application as configured by the user in\n    ``settings.WSGI_APPLICATION``. With the default ``startproject`` layout,\n    this will be the ``application`` object in ``projectname/wsgi.py``.\n\n    This function, and the ``WSGI_APPLICATION`` setting itself, are only useful\n    for Django's internal server (runserver); external WSGI servers should just\n    be configured to point to the correct application object directly.\n\n    If settings.WSGI_APPLICATION is not set (is ``None``), return\n    whatever ``django.core.wsgi.get_wsgi_application`` returns.\n    "
    from django.conf import settings
    app_path = getattr(settings, 'WSGI_APPLICATION')
    if app_path is None:
        return get_wsgi_application()
    try:
        return import_string(app_path)
    except ImportError as err:
        raise ImproperlyConfigured("WSGI application '%s' could not be loaded; Error importing module." % app_path) from err

def is_broken_pipe_error():
    if False:
        while True:
            i = 10
    (exc_type, _, _) = sys.exc_info()
    return issubclass(exc_type, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError))

class WSGIServer(simple_server.WSGIServer):
    """BaseHTTPServer that implements the Python WSGI protocol"""
    request_queue_size = 10

    def __init__(self, *args, ipv6=False, allow_reuse_address=True, **kwargs):
        if False:
            return 10
        if ipv6:
            self.address_family = socket.AF_INET6
        self.allow_reuse_address = allow_reuse_address
        super().__init__(*args, **kwargs)

    def handle_error(self, request, client_address):
        if False:
            return 10
        if is_broken_pipe_error():
            logger.info('- Broken pipe from %s', client_address)
        else:
            super().handle_error(request, client_address)

class ThreadedWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
    """A threaded version of the WSGIServer"""
    daemon_threads = True

    def __init__(self, *args, connections_override=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.connections_override = connections_override

    def process_request_thread(self, request, client_address):
        if False:
            while True:
                i = 10
        if self.connections_override:
            for (alias, conn) in self.connections_override.items():
                connections[alias] = conn
        super().process_request_thread(request, client_address)

    def _close_connections(self):
        if False:
            for i in range(10):
                print('nop')
        connections.close_all()

    def close_request(self, request):
        if False:
            i = 10
            return i + 15
        self._close_connections()
        super().close_request(request)

class ServerHandler(simple_server.ServerHandler):
    http_version = '1.1'

    def __init__(self, stdin, stdout, stderr, environ, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Use a LimitedStream so that unread request data will be ignored at\n        the end of the request. WSGIRequest uses a LimitedStream but it\n        shouldn't discard the data since the upstream servers usually do this.\n        This fix applies only for testserver/runserver.\n        "
        try:
            content_length = int(environ.get('CONTENT_LENGTH'))
        except (ValueError, TypeError):
            content_length = 0
        super().__init__(LimitedStream(stdin, content_length), stdout, stderr, environ, **kwargs)

    def cleanup_headers(self):
        if False:
            return 10
        super().cleanup_headers()
        if self.environ['REQUEST_METHOD'] == 'HEAD' and 'Content-Length' in self.headers:
            del self.headers['Content-Length']
        if self.environ['REQUEST_METHOD'] != 'HEAD' and 'Content-Length' not in self.headers:
            self.headers['Connection'] = 'close'
        elif not isinstance(self.request_handler.server, socketserver.ThreadingMixIn):
            self.headers['Connection'] = 'close'
        if self.headers.get('Connection') == 'close':
            self.request_handler.close_connection = True

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_stdin().read()
        super().close()

    def finish_response(self):
        if False:
            print('Hello World!')
        if self.environ['REQUEST_METHOD'] == 'HEAD':
            try:
                deque(self.result, maxlen=0)
                if not self.headers_sent:
                    self.send_headers()
            finally:
                self.close()
        else:
            super().finish_response()

class WSGIRequestHandler(simple_server.WSGIRequestHandler):
    protocol_version = 'HTTP/1.1'

    def address_string(self):
        if False:
            print('Hello World!')
        return self.client_address[0]

    def log_message(self, format, *args):
        if False:
            for i in range(10):
                print('nop')
        extra = {'request': self.request, 'server_time': self.log_date_time_string()}
        if args[1][0] == '4':
            if args[0].startswith('\x16\x03'):
                extra['status_code'] = 500
                logger.error("You're accessing the development server over HTTPS, but it only supports HTTP.", extra=extra)
                return
        if args[1].isdigit() and len(args[1]) == 3:
            status_code = int(args[1])
            extra['status_code'] = status_code
            if status_code >= 500:
                level = logger.error
            elif status_code >= 400:
                level = logger.warning
            else:
                level = logger.info
        else:
            level = logger.info
        level(format, *args, extra=extra)

    def get_environ(self):
        if False:
            return 10
        for k in self.headers:
            if '_' in k:
                del self.headers[k]
        return super().get_environ()

    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        self.close_connection = True
        self.handle_one_request()
        while not self.close_connection:
            self.handle_one_request()
        try:
            self.connection.shutdown(socket.SHUT_WR)
        except (AttributeError, OSError):
            pass

    def handle_one_request(self):
        if False:
            while True:
                i = 10
        'Copy of WSGIRequestHandler.handle() but with different ServerHandler'
        self.raw_requestline = self.rfile.readline(65537)
        if len(self.raw_requestline) > 65536:
            self.requestline = ''
            self.request_version = ''
            self.command = ''
            self.send_error(414)
            return
        if not self.parse_request():
            return
        handler = ServerHandler(self.rfile, self.wfile, self.get_stderr(), self.get_environ())
        handler.request_handler = self
        handler.run(self.server.get_app())

def run(addr, port, wsgi_handler, ipv6=False, threading=False, on_bind=None, server_cls=WSGIServer):
    if False:
        return 10
    server_address = (addr, port)
    if threading:
        httpd_cls = type('WSGIServer', (socketserver.ThreadingMixIn, server_cls), {})
    else:
        httpd_cls = server_cls
    httpd = httpd_cls(server_address, WSGIRequestHandler, ipv6=ipv6)
    if on_bind is not None:
        on_bind(getattr(httpd, 'server_port', port))
    if threading:
        httpd.daemon_threads = True
    httpd.set_app(wsgi_handler)
    httpd.serve_forever()