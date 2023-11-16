import os
import sys
try:
    import tornado
except ImportError:
    raise RuntimeError('You need tornado installed to use this worker.')
import tornado.web
import tornado.httpserver
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.wsgi import WSGIContainer
from gunicorn.workers.base import Worker
from gunicorn import __version__ as gversion
from gunicorn.sock import ssl_context
TORNADO5 = tornado.version_info >= (5, 0, 0)

class TornadoWorker(Worker):

    @classmethod
    def setup(cls):
        if False:
            return 10
        web = sys.modules.pop('tornado.web')
        old_clear = web.RequestHandler.clear

        def clear(self):
            if False:
                i = 10
                return i + 15
            old_clear(self)
            if 'Gunicorn' not in self._headers['Server']:
                self._headers['Server'] += ' (Gunicorn/%s)' % gversion
        web.RequestHandler.clear = clear
        sys.modules['tornado.web'] = web

    def handle_exit(self, sig, frame):
        if False:
            print('Hello World!')
        if self.alive:
            super().handle_exit(sig, frame)

    def handle_request(self):
        if False:
            i = 10
            return i + 15
        self.nr += 1
        if self.alive and self.nr >= self.max_requests:
            self.log.info('Autorestarting worker after current request.')
            self.alive = False

    def watchdog(self):
        if False:
            while True:
                i = 10
        if self.alive:
            self.notify()
        if self.ppid != os.getppid():
            self.log.info('Parent changed, shutting down: %s', self)
            self.alive = False

    def heartbeat(self):
        if False:
            print('Hello World!')
        if not self.alive:
            if self.server_alive:
                if hasattr(self, 'server'):
                    try:
                        self.server.stop()
                    except Exception:
                        pass
                self.server_alive = False
            elif TORNADO5:
                for callback in self.callbacks:
                    callback.stop()
                self.ioloop.stop()
            elif not self.ioloop._callbacks:
                self.ioloop.stop()

    def init_process(self):
        if False:
            i = 10
            return i + 15
        IOLoop.clear_current()
        super().init_process()

    def run(self):
        if False:
            print('Hello World!')
        self.ioloop = IOLoop.instance()
        self.alive = True
        self.server_alive = False
        if TORNADO5:
            self.callbacks = []
            self.callbacks.append(PeriodicCallback(self.watchdog, 1000))
            self.callbacks.append(PeriodicCallback(self.heartbeat, 1000))
            for callback in self.callbacks:
                callback.start()
        else:
            PeriodicCallback(self.watchdog, 1000, io_loop=self.ioloop).start()
            PeriodicCallback(self.heartbeat, 1000, io_loop=self.ioloop).start()
        app = self.wsgi
        if tornado.version_info[0] < 6:
            if not isinstance(app, tornado.web.Application) or isinstance(app, tornado.wsgi.WSGIApplication):
                app = WSGIContainer(app)
        elif not isinstance(app, WSGIContainer) and (not isinstance(app, tornado.web.Application)):
            app = WSGIContainer(app)
        httpserver = sys.modules['tornado.httpserver']
        if hasattr(httpserver, 'HTTPConnection'):
            old_connection_finish = httpserver.HTTPConnection.finish

            def finish(other):
                if False:
                    while True:
                        i = 10
                self.handle_request()
                old_connection_finish(other)
            httpserver.HTTPConnection.finish = finish
            sys.modules['tornado.httpserver'] = httpserver
            server_class = tornado.httpserver.HTTPServer
        else:

            class _HTTPServer(tornado.httpserver.HTTPServer):

                def on_close(instance, server_conn):
                    if False:
                        while True:
                            i = 10
                    self.handle_request()
                    super(_HTTPServer, instance).on_close(server_conn)
            server_class = _HTTPServer
        if self.cfg.is_ssl:
            if TORNADO5:
                server = server_class(app, ssl_options=ssl_context(self.cfg))
            else:
                server = server_class(app, io_loop=self.ioloop, ssl_options=ssl_context(self.cfg))
        elif TORNADO5:
            server = server_class(app)
        else:
            server = server_class(app, io_loop=self.ioloop)
        self.server = server
        self.server_alive = True
        for s in self.sockets:
            s.setblocking(0)
            if hasattr(server, 'add_socket'):
                server.add_socket(s)
            elif hasattr(server, '_sockets'):
                server._sockets[s.fileno()] = s
        server.no_keep_alive = self.cfg.keepalive <= 0
        server.start(num_processes=1)
        self.ioloop.start()