from concurrent import futures
import errno
import os
import selectors
import socket
import ssl
import sys
import time
from collections import deque
from datetime import datetime
from functools import partial
from threading import RLock
from . import base
from .. import http
from .. import util
from .. import sock
from ..http import wsgi

class TConn(object):

    def __init__(self, cfg, sock, client, server):
        if False:
            return 10
        self.cfg = cfg
        self.sock = sock
        self.client = client
        self.server = server
        self.timeout = None
        self.parser = None
        self.initialized = False
        self.sock.setblocking(False)

    def init(self):
        if False:
            for i in range(10):
                print('nop')
        self.initialized = True
        self.sock.setblocking(True)
        if self.parser is None:
            if self.cfg.is_ssl:
                self.sock = sock.ssl_wrap_socket(self.sock, self.cfg)
            self.parser = http.RequestParser(self.cfg, self.sock, self.client)

    def set_timeout(self):
        if False:
            print('Hello World!')
        self.timeout = time.time() + self.cfg.keepalive

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        util.close(self.sock)

class ThreadWorker(base.Worker):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.worker_connections = self.cfg.worker_connections
        self.max_keepalived = self.cfg.worker_connections - self.cfg.threads
        self.tpool = None
        self.poller = None
        self._lock = None
        self.futures = deque()
        self._keep = deque()
        self.nr_conns = 0

    @classmethod
    def check_config(cls, cfg, log):
        if False:
            print('Hello World!')
        max_keepalived = cfg.worker_connections - cfg.threads
        if max_keepalived <= 0 and cfg.keepalive:
            log.warning('No keepalived connections can be handled. ' + 'Check the number of worker connections and threads.')

    def init_process(self):
        if False:
            i = 10
            return i + 15
        self.tpool = self.get_thread_pool()
        self.poller = selectors.DefaultSelector()
        self._lock = RLock()
        super().init_process()

    def get_thread_pool(self):
        if False:
            while True:
                i = 10
        'Override this method to customize how the thread pool is created'
        return futures.ThreadPoolExecutor(max_workers=self.cfg.threads)

    def handle_quit(self, sig, frame):
        if False:
            i = 10
            return i + 15
        self.alive = False
        self.cfg.worker_int(self)
        self.tpool.shutdown(False)
        time.sleep(0.1)
        sys.exit(0)

    def _wrap_future(self, fs, conn):
        if False:
            return 10
        fs.conn = conn
        self.futures.append(fs)
        fs.add_done_callback(self.finish_request)

    def enqueue_req(self, conn):
        if False:
            return 10
        conn.init()
        fs = self.tpool.submit(self.handle, conn)
        self._wrap_future(fs, conn)

    def accept(self, server, listener):
        if False:
            i = 10
            return i + 15
        try:
            (sock, client) = listener.accept()
            conn = TConn(self.cfg, sock, client, server)
            self.nr_conns += 1
            with self._lock:
                self.poller.register(conn.sock, selectors.EVENT_READ, partial(self.on_client_socket_readable, conn))
        except EnvironmentError as e:
            if e.errno not in (errno.EAGAIN, errno.ECONNABORTED, errno.EWOULDBLOCK):
                raise

    def on_client_socket_readable(self, conn, client):
        if False:
            for i in range(10):
                print('nop')
        with self._lock:
            self.poller.unregister(client)
            if conn.initialized:
                try:
                    self._keep.remove(conn)
                except ValueError:
                    return
        self.enqueue_req(conn)

    def murder_keepalived(self):
        if False:
            i = 10
            return i + 15
        now = time.time()
        while True:
            with self._lock:
                try:
                    conn = self._keep.popleft()
                except IndexError:
                    break
            delta = conn.timeout - now
            if delta > 0:
                with self._lock:
                    self._keep.appendleft(conn)
                break
            else:
                self.nr_conns -= 1
                with self._lock:
                    try:
                        self.poller.unregister(conn.sock)
                    except EnvironmentError as e:
                        if e.errno != errno.EBADF:
                            raise
                    except KeyError:
                        pass
                    except ValueError:
                        pass
                conn.close()

    def is_parent_alive(self):
        if False:
            return 10
        if self.ppid != os.getppid():
            self.log.info('Parent changed, shutting down: %s', self)
            return False
        return True

    def run(self):
        if False:
            print('Hello World!')
        for sock in self.sockets:
            sock.setblocking(False)
            server = sock.getsockname()
            acceptor = partial(self.accept, server)
            self.poller.register(sock, selectors.EVENT_READ, acceptor)
        while self.alive:
            self.notify()
            if self.nr_conns < self.worker_connections:
                events = self.poller.select(1.0)
                for (key, _) in events:
                    callback = key.data
                    callback(key.fileobj)
                result = futures.wait(self.futures, timeout=0, return_when=futures.FIRST_COMPLETED)
            else:
                result = futures.wait(self.futures, timeout=1.0, return_when=futures.FIRST_COMPLETED)
            for fut in result.done:
                self.futures.remove(fut)
            if not self.is_parent_alive():
                break
            self.murder_keepalived()
        self.tpool.shutdown(False)
        self.poller.close()
        for s in self.sockets:
            s.close()
        futures.wait(self.futures, timeout=self.cfg.graceful_timeout)

    def finish_request(self, fs):
        if False:
            for i in range(10):
                print('nop')
        if fs.cancelled():
            self.nr_conns -= 1
            fs.conn.close()
            return
        try:
            (keepalive, conn) = fs.result()
            if keepalive and self.alive:
                conn.sock.setblocking(False)
                conn.set_timeout()
                with self._lock:
                    self._keep.append(conn)
                    self.poller.register(conn.sock, selectors.EVENT_READ, partial(self.on_client_socket_readable, conn))
            else:
                self.nr_conns -= 1
                conn.close()
        except Exception:
            self.nr_conns -= 1
            fs.conn.close()

    def handle(self, conn):
        if False:
            print('Hello World!')
        keepalive = False
        req = None
        try:
            req = next(conn.parser)
            if not req:
                return (False, conn)
            keepalive = self.handle_request(req, conn)
            if keepalive:
                return (keepalive, conn)
        except http.errors.NoMoreData as e:
            self.log.debug('Ignored premature client disconnection. %s', e)
        except StopIteration as e:
            self.log.debug('Closing connection. %s', e)
        except ssl.SSLError as e:
            if e.args[0] == ssl.SSL_ERROR_EOF:
                self.log.debug('ssl connection closed')
                conn.sock.close()
            else:
                self.log.debug('Error processing SSL request.')
                self.handle_error(req, conn.sock, conn.client, e)
        except EnvironmentError as e:
            if e.errno not in (errno.EPIPE, errno.ECONNRESET, errno.ENOTCONN):
                self.log.exception('Socket error processing request.')
            elif e.errno == errno.ECONNRESET:
                self.log.debug('Ignoring connection reset')
            elif e.errno == errno.ENOTCONN:
                self.log.debug('Ignoring socket not connected')
            else:
                self.log.debug('Ignoring connection epipe')
        except Exception as e:
            self.handle_error(req, conn.sock, conn.client, e)
        return (False, conn)

    def handle_request(self, req, conn):
        if False:
            for i in range(10):
                print('nop')
        environ = {}
        resp = None
        try:
            self.cfg.pre_request(self, req)
            request_start = datetime.now()
            (resp, environ) = wsgi.create(req, conn.sock, conn.client, conn.server, self.cfg)
            environ['wsgi.multithread'] = True
            self.nr += 1
            if self.nr >= self.max_requests:
                if self.alive:
                    self.log.info('Autorestarting worker after current request.')
                    self.alive = False
                resp.force_close()
            if not self.alive or not self.cfg.keepalive:
                resp.force_close()
            elif len(self._keep) >= self.max_keepalived:
                resp.force_close()
            respiter = self.wsgi(environ, resp.start_response)
            try:
                if isinstance(respiter, environ['wsgi.file_wrapper']):
                    resp.write_file(respiter)
                else:
                    for item in respiter:
                        resp.write(item)
                resp.close()
            finally:
                request_time = datetime.now() - request_start
                self.log.access(resp, req, environ, request_time)
                if hasattr(respiter, 'close'):
                    respiter.close()
            if resp.should_close():
                self.log.debug('Closing connection.')
                return False
        except EnvironmentError:
            util.reraise(*sys.exc_info())
        except Exception:
            if resp and resp.headers_sent:
                self.log.exception('Error handling request')
                try:
                    conn.sock.shutdown(socket.SHUT_RDWR)
                    conn.sock.close()
                except EnvironmentError:
                    pass
                raise StopIteration()
            raise
        finally:
            try:
                self.cfg.post_request(self, req, environ, resp)
            except Exception:
                self.log.exception('Exception in post_request hook')
        return True