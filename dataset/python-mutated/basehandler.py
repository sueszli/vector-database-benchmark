from __future__ import absolute_import, division, print_function, unicode_literals
'\n    sockjs.tornado.basehandler\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n    Various base http handlers\n'
import datetime
import socket
import logging
from tornado.web import RequestHandler
from tornado.gen import coroutine
from urllib.parse import urlparse
CACHE_TIME = 31536000
LOG = logging.getLogger('tornado.general')

class BaseHandler(RequestHandler):
    """Base request handler with set of helpers."""

    def initialize(self, server):
        if False:
            i = 10
            return i + 15
        'Initialize request\n\n        `server`\n            SockJSRouter instance.\n        '
        self.server = server
        self.logged = False

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        'Increment connection count'
        self.logged = True
        self.server.stats.on_conn_opened()

    def _log_disconnect(self):
        if False:
            i = 10
            return i + 15
        'Decrement connection count'
        if self.logged:
            self.server.stats.on_conn_closed()
            self.logged = False

    def finish(self, chunk=None):
        if False:
            for i in range(10):
                print('nop')
        'Tornado `finish` handler'
        self._log_disconnect()
        super(BaseHandler, self).finish(chunk)

    def on_connection_close(self):
        if False:
            return 10
        'Tornado `on_connection_close` handler'
        self._log_disconnect()

    def enable_cache(self):
        if False:
            while True:
                i = 10
        'Enable client-side caching for the current request'
        self.set_header('Cache-Control', 'max-age=%d, public' % CACHE_TIME)
        d = datetime.datetime.now() + datetime.timedelta(seconds=CACHE_TIME)
        self.set_header('Expires', d.strftime('%a, %d %b %Y %H:%M:%S'))
        self.set_header('access-control-max-age', CACHE_TIME)

    def disable_cache(self):
        if False:
            i = 10
            return i + 15
        'Disable client-side cache for the current request'
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')

    def handle_session_cookie(self):
        if False:
            for i in range(10):
                print('nop')
        'Handle JSESSIONID cookie logic'
        if not self.server.settings['jsessionid']:
            return
        cookie = self.cookies.get('JSESSIONID')
        if not cookie:
            cv = 'dummy'
        else:
            cv = cookie.value
        self.set_cookie('JSESSIONID', cv)

    def safe_finish(self):
        if False:
            i = 10
            return i + 15
        'Finish session. If it will blow up - connection was set to Keep-Alive and\n        client dropped connection, ignore any IOError or socket error.'
        try:
            self.finish()
        except (socket.error, IOError):
            LOG.debug('Ignoring IOError in safe_finish()')
            pass

class PreflightHandler(BaseHandler):
    """CORS preflight handler"""

    @coroutine
    def options(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'XHR cross-domain OPTIONS handler'
        self.enable_cache()
        self.handle_session_cookie()
        self.preflight()
        if self.verify_origin():
            allowed_methods = getattr(self, 'access_methods', 'OPTIONS, POST')
            self.set_header('Access-Control-Allow-Methods', allowed_methods)
            self.set_header('Allow', allowed_methods)
            self.set_status(204)
        else:
            self.set_status(403)
        self.finish()

    def preflight(self):
        if False:
            print('Hello World!')
        'Handles request authentication'
        origin = self.request.headers.get('Origin', '*')
        self.set_header('Access-Control-Allow-Origin', origin)
        headers = self.request.headers.get('Access-Control-Request-Headers')
        if headers:
            self.set_header('Access-Control-Allow-Headers', headers)
        self.set_header('Access-Control-Allow-Credentials', 'true')

    def verify_origin(self):
        if False:
            print('Hello World!')
        'Verify if request can be served'
        origin = self.request.headers.get('Origin', '*')
        same_domain = self.check_origin(origin)
        if same_domain:
            return True
        allow_origin = self.server.settings.get('websocket_allow_origin', '*')
        if allow_origin == '':
            return False
        elif allow_origin == '*':
            return True
        else:
            parsed_origin = urlparse(origin)
            origin = parsed_origin.netloc
            origin = origin.lower()
            return origin in allow_origin

    def check_origin(self, origin):
        if False:
            print('Hello World!')
        parsed_origin = urlparse(origin)
        origin = parsed_origin.netloc
        origin = origin.lower()
        host = self.request.headers.get('Host')
        return origin == host