"""
Serve web page and handle web sockets using Tornado.
"""
import json
import time
import asyncio
import socket
import mimetypes
import traceback
import threading
from urllib.parse import urlparse
import tornado
from tornado import gen, netutil
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketHandler, WebSocketClosedError
from tornado.httpserver import HTTPServer
from tornado.platform.asyncio import AsyncIOMainLoop
from ._app import manager
from ._session import get_page
from ._server import AbstractServer
from ._assetstore import assets
from ._clientcore import serializer
from . import logger
from .. import config
if tornado.version_info < (4,):
    raise RuntimeError('Flexx requires Tornado v4.0 or higher.')
IMPORT_TIME = time.time()

def is_main_thread():
    if False:
        while True:
            i = 10
    ' Get whether this is the main thread. '
    return isinstance(threading.current_thread(), threading._MainThread)

class TornadoServer(AbstractServer):
    """ Flexx Server implemented in Tornado.
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._app = None
        self._server = None
        super().__init__(*args, **kwargs)

    def _open(self, host, port, **kwargs):
        if False:
            return 10
        self._io_loop = AsyncIOMainLoop()
        if hasattr(IOLoop, '_current'):
            IOLoop._current.instance = None
        else:
            IOLoop.current().instance = None
        self._io_loop.make_current()
        if config.ssl_certfile:
            if 'ssl_options' not in kwargs:
                kwargs['ssl_options'] = {}
            if 'certfile' not in kwargs['ssl_options']:
                kwargs['ssl_options']['certfile'] = config.ssl_certfile
        if config.ssl_keyfile:
            if 'ssl_options' not in kwargs:
                kwargs['ssl_options'] = {}
            if 'keyfile' not in kwargs['ssl_options']:
                kwargs['ssl_options']['keyfile'] = config.ssl_keyfile
        if config.tornado_debug:
            app_kwargs = dict(debug=True)
        else:
            app_kwargs = dict()
        self._app = Application([('/flexx/ws/(.*)', WSHandler), ('/flexx/(.*)', MainHandler), ('/(.*)', AppHandler)], **app_kwargs)
        self._app._io_loop = self._io_loop
        if tornado.version_info < (5,):
            kwargs['io_loop'] = self._io_loop
        self._server = HTTPServer(self._app, **kwargs)
        if port:
            try:
                port = int(port)
            except ValueError:
                port = port_hash(port)
            self._server.listen(port, host)
        else:
            prefered_port = port_hash('Flexx')
            for i in range(8):
                port = prefered_port + i
                try:
                    self._server.listen(port, host)
                    break
                except OSError:
                    pass
            else:
                [sock] = netutil.bind_sockets(None, host, family=socket.AF_INET)
                self._server.add_sockets([sock])
                port = sock.getsockname()[1]
        self._serving = self._app._flexx_serving = (host, port)
        proto = 'http'
        if 'ssl_options' in kwargs:
            proto = 'https'
        logger.info('Serving apps at %s://%s:%i/' % (proto, host, port))

    def _close(self):
        if False:
            while True:
                i = 10
        self._server.stop()

    @property
    def app(self):
        if False:
            for i in range(10):
                print('nop')
        ' The Tornado Application object being used.'
        return self._app

    @property
    def server(self):
        if False:
            print('Hello World!')
        ' The Tornado HttpServer object being used.'
        return self._server

    @property
    def protocol(self):
        if False:
            for i in range(10):
                print('nop')
        ' Get a string representing served protocol.'
        if self._server.ssl_options is not None:
            return 'https'
        return 'http'

def port_hash(name):
    if False:
        while True:
            i = 10
    ' Given a string, returns a port number between 49152 and 65535\n\n    This range (of 2**14 posibilities) is the range for dynamic and/or\n    private ports (ephemeral ports) specified by iana.org. The algorithm\n    is deterministic.\n    '
    fac = 3537390177
    val = 0
    for c in name:
        val += (val >> 3) + ord(c) * fac
    val += (val >> 3) + len(name) * fac
    return 49152 + val % 2 ** 14

class FlexxHandler(RequestHandler):
    """ Base class for Flexx' Tornado request handlers.
    """

    def initialize(self, **kwargs):
        if False:
            return 10
        pass

    def write_error(self, status_code, **kwargs):
        if False:
            return 10
        if status_code == 404:
            self.write('flexx.ui wants you to connect to root (404)')
        else:
            if config.browser_stacktrace:
                msg = 'Flexx.ui encountered an error: <br /><br />'
                try:
                    (type, value, tb) = kwargs['exc_info']
                    tb_str = ''.join(traceback.format_tb(tb))
                    msg += '<pre>%s\n%s</pre>' % (tb_str, str(value))
                except Exception:
                    pass
                self.write(msg)
            super().write_error(status_code, **kwargs)

    def on_finish(self):
        if False:
            i = 10
            return i + 15
        pass

class AppHandler(FlexxHandler):
    """ Handler for http requests to get apps.
    """

    @gen.coroutine
    def get(self, full_path):
        if False:
            while True:
                i = 10
        logger.debug('Incoming request at %r' % full_path)
        ok_app_names = ('__main__', '__default__', '__index__')
        parts = [p for p in full_path.split('/') if p]
        app_name = None
        path = '/'.join(parts)
        if parts:
            if path.lower() == 'flexx':
                return self.redirect('/flexx/')
            if parts[0] in ok_app_names or manager.has_app_name(parts[0]):
                app_name = parts[0]
                path = '/'.join(parts[1:])
        if app_name is None:
            if len(parts) == 1 and '.' in full_path:
                return self.redirect('/flexx/data/' + full_path)
            app_name = '__main__'
        if app_name == '__main__':
            app_name = manager.has_app_name('__main__')
        elif '/' not in full_path:
            return self.redirect('/%s/' % app_name)
        if not app_name:
            if not parts:
                app_name = '__index__'
            else:
                name = parts[0] if parts else '__main__'
                return self.write('No app "%s" is currently hosted.' % name)
        if app_name == '__index__':
            self._get_index(app_name, path)
        else:
            self._get_app(app_name, path)

    def _get_index(self, app_name, path):
        if False:
            print('Hello World!')
        if path:
            return self.redirect('/flexx/__index__')
        all_apps = ['<li><a href="%s/">%s</a></li>' % (name, name) for name in manager.get_app_names()]
        the_list = '<ul>%s</ul>' % ''.join(all_apps) if all_apps else 'no apps'
        self.write('Index of available apps: ' + the_list)

    def _get_app(self, app_name, path):
        if False:
            for i in range(10):
                print('nop')
        if path.startswith(('flexx/data/', 'flexx/assets/')):
            return self.redirect('/' + path)
        correct_app_name = manager.has_app_name(app_name)
        if not correct_app_name:
            return self.write('No app "%s" is currently hosted.' % app_name)
        if correct_app_name != app_name:
            return self.redirect('/%s/%s' % (correct_app_name, path))
        session_id = self.get_argument('session_id', '')
        if session_id:
            session = manager.get_session_by_id(session_id)
            if session and session.status == session.STATUS.PENDING:
                self.write(get_page(session).encode())
            else:
                self.redirect('/%s/' % app_name)
        else:
            session = manager.create_session(app_name, request=self.request)
            self.write(get_page(session).encode())

class MainHandler(RequestHandler):
    """ Handler for assets, commands, etc. Basically, everything for
    which te path is clear.
    """

    def _guess_mime_type(self, fname):
        if False:
            print('Hello World!')
        ' Set the mimetype if we can guess it from the filename.\n        '
        guess = mimetypes.guess_type(fname)[0]
        if guess:
            self.set_header('Content-Type', guess)

    @gen.coroutine
    def get(self, full_path):
        if False:
            i = 10
            return i + 15
        logger.debug('Incoming request at %s' % full_path)
        parts = [p for p in full_path.split('/') if p]
        if not parts:
            return self.write('Root url for flexx: assets, assetview, data, cmd')
        selector = parts[0]
        path = '/'.join(parts[1:])
        if selector in ('assets', 'assetview', 'data'):
            self._get_asset(selector, path)
        elif selector == 'info':
            self._get_info(selector, path)
        elif selector == 'cmd':
            self._get_cmd(selector, path)
        else:
            return self.write('Invalid url path "%s".' % full_path)

    def _get_asset(self, selector, path):
        if False:
            for i in range(10):
                print('nop')
        (session_id, _, filename) = path.partition('/')
        session_id = '' if session_id == 'shared' else session_id
        asset_provider = assets
        if session_id and selector != 'data':
            return self.write('Only supports shared assets, not %s' % filename)
        elif session_id:
            asset_provider = manager.get_session_by_id(session_id)
        if asset_provider is None:
            return self.write('Invalid session %r' % session_id)
        if not filename:
            return self.write('Root dir for %s/%s' % (selector, path))
        if selector == 'assets':
            if '.js:' in filename or '.css:' in filename or filename[0] == ':':
                (fname, where) = filename.split(':')[:2]
                return self.redirect('/flexx/assetview/%s/%s#L%s' % (session_id or 'shared', fname.replace('/:', ':'), where))
            try:
                res = asset_provider.get_asset(filename)
            except KeyError:
                self.write('Could not load asset %r' % filename)
            else:
                self._guess_mime_type(filename)
                self.write(res.to_string())
        elif selector == 'assetview':
            try:
                res = asset_provider.get_asset(filename)
            except KeyError:
                return self.write('Could not load asset %r' % filename)
            else:
                res = res.to_string()
            style = 'pre {display:block; width: 100%; padding:0; margin:0;} a {text-decoration: none; color: #000; background: #ddd;} :target {background:#ada;} '
            lines = ['<html><head><style>%s</style></head><body>' % style]
            for (i, line) in enumerate(res.splitlines()):
                table = {ord('&'): '&amp;', ord('<'): '&lt;', ord('>'): '&gt;'}
                line = line.translate(table).replace('\t', '    ')
                lines.append('<pre id="L%i"><a href="#L%i">%s</a>  %s</pre>' % (i + 1, i + 1, str(i + 1).rjust(4).replace(' ', '&nbsp'), line))
            lines.append('</body></html>')
            return self.write('\n'.join(lines))
        elif selector == 'data':
            res = asset_provider.get_data(filename)
            if res is None:
                return self.send_error(404)
            else:
                self._guess_mime_type(filename)
                return self.write(res)
        else:
            raise RuntimeError('Invalid asset type %r' % selector)

    def _get_info(self, selector, info):
        if False:
            print('Hello World!')
        ' Provide some rudimentary information about the server.\n        Note that this is publicly accesible.\n        '
        runtime = time.time() - IMPORT_TIME
        napps = len(manager.get_app_names())
        nsessions = sum([len(manager.get_connections(x)) for x in manager.get_app_names()])
        info = []
        info.append('Runtime: %1.1f s' % runtime)
        info.append('Number of apps: %i' % napps)
        info.append('Number of sessions: %i' % nsessions)
        info = '\n'.join(['<li>%s</li>' % i for i in info])
        self.write('<ul>' + info + '</ul>')

    def _get_cmd(self, selector, path):
        if False:
            print('Hello World!')
        ' Allow control of the server using http, but only from localhost!\n        '
        if not self.request.host.startswith('localhost:'):
            self.write('403')
            return
        if not path:
            self.write('No command given')
        elif path == 'info':
            info = dict(address=self.application._flexx_serving, app_names=manager.get_app_names(), nsessions=sum([len(manager.get_connections(x)) for x in manager.get_app_names()]))
            self.write(json.dumps(info))
        elif path == 'stop':
            asyncio.get_event_loop().stop()
            self.write('Stopping event loop.')
        else:
            self.write('unknown command %r' % path)

class MessageCounter:
    """ Simple class to count incoming messages and periodically log
    the number of messages per second.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._collect_interval = 0.2
        self._notify_interval = 3.0
        self._window_interval = 4.0
        self._mps = [(time.time(), 0)]
        self._collect_count = 0
        self._collect_stoptime = 0
        self._stop = False
        self._notify()

    def trigger(self):
        if False:
            while True:
                i = 10
        t = time.time()
        if t < self._collect_stoptime:
            self._collect_count += 1
        else:
            self._mps.append((self._collect_stoptime, self._collect_count))
            self._collect_count = 1
            self._collect_stoptime = t + self._collect_interval

    def _notify(self):
        if False:
            print('Hello World!')
        mintime = time.time() - self._window_interval
        self._mps = [x for x in self._mps if x[0] > mintime]
        if self._mps:
            n = sum([x[1] for x in self._mps])
            T = self._mps[-1][0] - self._mps[0][0] + self._collect_interval
        else:
            (n, T) = (0, self._collect_interval)
        logger.debug('Websocket messages per second: %1.1f' % (n / T))
        if not self._stop:
            loop = asyncio.get_event_loop()
            loop.call_later(self._notify_interval, self._notify)

    def stop(self):
        if False:
            while True:
                i = 10
        self._stop = True

class WSHandler(WebSocketHandler):
    """ Handler for websocket.
    """
    known_reasons = {1000: 'client done', 1001: 'client closed', 1002: 'protocol error', 1003: 'could not accept data'}

    def open(self, path=None):
        if False:
            print('Hello World!')
        ' Called when a new connection is made.\n        '
        if not hasattr(self, 'close_code'):
            (self.close_code, self.close_reason) = (None, None)
        self._session = None
        self._mps_counter = MessageCounter()
        if isinstance(path, bytes):
            path = path.decode()
        self.app_name = path.strip('/')
        logger.debug('New websocket connection %s' % path)
        if manager.has_app_name(self.app_name):
            self.application._io_loop.spawn_callback(self.pinger1)
        else:
            self.close(1003, 'Could not associate socket with an app.')

    def on_message(self, message):
        if False:
            print('Hello World!')
        ' Called when a new message is received from JS.\n\n        This handles one message per event loop iteration.\n\n        We now have a very basic protocol for receiving messages,\n        we should at some point define a real formalized protocol.\n        '
        self._mps_counter.trigger()
        try:
            command = serializer.decode(message)
        except Exception as err:
            err.skip_tb = 1
            logger.exception(err)
        self._pongtime = time.time()
        if self._session is None:
            if command[0] == 'HI_FLEXX':
                session_id = command[1]
                try:
                    self._session = manager.connect_client(self, self.app_name, session_id, cookies=self.cookies)
                except Exception as err:
                    self.close(1003, 'Could not launch app: %r' % err)
                    raise
        else:
            try:
                self._session._receive_command(command)
            except Exception as err:
                err.skip_tb = 1
                logger.exception(err)

    def on_close(self):
        if False:
            for i in range(10):
                print('nop')
        ' Called when the connection is closed.\n        '
        self.close_code = code = self.close_code or 0
        reason = self.close_reason or self.known_reasons.get(code, '')
        logger.debug('Websocket closed: %s (%i)' % (reason, code))
        self._mps_counter.stop()
        if self._session is not None:
            manager.disconnect_client(self._session)
            self._session = None

    @gen.coroutine
    def pinger1(self):
        if False:
            i = 10
            return i + 15
        " Check for timeouts. This helps remove lingering false connections.\n\n        This uses the websocket's native ping-ping mechanism. On the\n        browser side, pongs work even if JS is busy. On the Python side\n        we perform a check whether we were really waiting or whether Python\n        was too busy to detect the pong.\n        "
        self._pongtime = time.time()
        self._pingtime = pingtime = 0
        while self.close_code is None:
            dt = config.ws_timeout
            if pingtime <= self._pongtime:
                self.ping(b'x')
                pingtime = self._pingtime = time.time()
                iters_since_ping = 0
            yield gen.sleep(dt / 5)
            iters_since_ping += 1
            if iters_since_ping < 5:
                pass
            elif time.time() - self._pongtime > dt:
                logger.warning('Closing connection due to lack of pong')
                self.close(1000, 'Conection timed out (no pong).')
                return

    def on_pong(self, data):
        if False:
            i = 10
            return i + 15
        " Implement the ws's on_pong() method. Called when our ping\n        is returned by the browser.\n        "
        self._pongtime = time.time()

    def write_command(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(cmd, tuple) and len(cmd) >= 1
        bb = serializer.encode(cmd)
        try:
            self.write_message(bb, binary=True)
        except WebSocketClosedError:
            self.close(1000, 'closed by client')

    def close(self, *args):
        if False:
            print('Hello World!')
        try:
            super().close(*args)
        except TypeError:
            super().close()

    def close_this(self):
        if False:
            i = 10
            return i + 15
        ' Call this to close the websocket\n        '
        self.close(1000, 'closed by server')

    def check_origin(self, origin):
        if False:
            return 10
        ' Handle cross-domain access; override default same origin policy.\n        '
        serving_host = self.request.headers.get('Host')
        (serving_hostname, _, serving_port) = serving_host.partition(':')
        connecting_host = urlparse(origin).netloc
        (connecting_hostname, _, connecting_port) = connecting_host.partition(':')
        serving_port = serving_port or '80'
        connecting_port = connecting_port or '80'
        if serving_hostname == 'localhost':
            return True
        elif serving_host == connecting_host:
            return True
        elif serving_hostname == '0.0.0.0' and serving_port == connecting_port:
            return True
        elif connecting_host in config.host_whitelist:
            return True
        else:
            logger.warning('Connection refused from %s' % origin)
            return False