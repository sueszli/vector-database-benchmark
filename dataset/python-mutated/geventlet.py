from functools import partial
import sys
try:
    import eventlet
except ImportError:
    raise RuntimeError('eventlet worker requires eventlet 0.24.1 or higher')
else:
    from packaging.version import parse as parse_version
    if parse_version(eventlet.__version__) < parse_version('0.24.1'):
        raise RuntimeError('eventlet worker requires eventlet 0.24.1 or higher')
from eventlet import hubs, greenthread
from eventlet.greenio import GreenSocket
import eventlet.wsgi
import greenlet
from gunicorn.workers.base_async import AsyncWorker
from gunicorn.sock import ssl_wrap_socket
EVENTLET_WSGI_LOCAL = getattr(eventlet.wsgi, 'WSGI_LOCAL', None)
EVENTLET_ALREADY_HANDLED = getattr(eventlet.wsgi, 'ALREADY_HANDLED', None)

def _eventlet_socket_sendfile(self, file, offset=0, count=None):
    if False:
        print('Hello World!')
    if self.gettimeout() == 0:
        raise ValueError('non-blocking sockets are not supported')
    if offset:
        file.seek(offset)
    blocksize = min(count, 8192) if count else 8192
    total_sent = 0
    file_read = file.read
    sock_send = self.send
    try:
        while True:
            if count:
                blocksize = min(count - total_sent, blocksize)
                if blocksize <= 0:
                    break
            data = memoryview(file_read(blocksize))
            if not data:
                break
            while True:
                try:
                    sent = sock_send(data)
                except BlockingIOError:
                    continue
                else:
                    total_sent += sent
                    if sent < len(data):
                        data = data[sent:]
                    else:
                        break
        return total_sent
    finally:
        if total_sent > 0 and hasattr(file, 'seek'):
            file.seek(offset + total_sent)

def _eventlet_serve(sock, handle, concurrency):
    if False:
        while True:
            i = 10
    '\n    Serve requests forever.\n\n    This code is nearly identical to ``eventlet.convenience.serve`` except\n    that it attempts to join the pool at the end, which allows for gunicorn\n    graceful shutdowns.\n    '
    pool = eventlet.greenpool.GreenPool(concurrency)
    server_gt = eventlet.greenthread.getcurrent()
    while True:
        try:
            (conn, addr) = sock.accept()
            gt = pool.spawn(handle, conn, addr)
            gt.link(_eventlet_stop, server_gt, conn)
            (conn, addr, gt) = (None, None, None)
        except eventlet.StopServe:
            sock.close()
            pool.waitall()
            return

def _eventlet_stop(client, server, conn):
    if False:
        while True:
            i = 10
    '\n    Stop a greenlet handling a request and close its connection.\n\n    This code is lifted from eventlet so as not to depend on undocumented\n    functions in the library.\n    '
    try:
        try:
            client.wait()
        finally:
            conn.close()
    except greenlet.GreenletExit:
        pass
    except Exception:
        greenthread.kill(server, *sys.exc_info())

def patch_sendfile():
    if False:
        print('Hello World!')
    if not hasattr(GreenSocket, 'sendfile'):
        GreenSocket.sendfile = _eventlet_socket_sendfile

class EventletWorker(AsyncWorker):

    def patch(self):
        if False:
            while True:
                i = 10
        hubs.use_hub()
        eventlet.monkey_patch()
        patch_sendfile()

    def is_already_handled(self, respiter):
        if False:
            print('Hello World!')
        if getattr(EVENTLET_WSGI_LOCAL, 'already_handled', None):
            raise StopIteration()
        if respiter == EVENTLET_ALREADY_HANDLED:
            raise StopIteration()
        return super().is_already_handled(respiter)

    def init_process(self):
        if False:
            i = 10
            return i + 15
        self.patch()
        super().init_process()

    def handle_quit(self, sig, frame):
        if False:
            print('Hello World!')
        eventlet.spawn(super().handle_quit, sig, frame)

    def handle_usr1(self, sig, frame):
        if False:
            while True:
                i = 10
        eventlet.spawn(super().handle_usr1, sig, frame)

    def timeout_ctx(self):
        if False:
            return 10
        return eventlet.Timeout(self.cfg.keepalive or None, False)

    def handle(self, listener, client, addr):
        if False:
            return 10
        if self.cfg.is_ssl:
            client = ssl_wrap_socket(client, self.cfg)
        super().handle(listener, client, addr)

    def run(self):
        if False:
            return 10
        acceptors = []
        for sock in self.sockets:
            gsock = GreenSocket(sock)
            gsock.setblocking(1)
            hfun = partial(self.handle, gsock)
            acceptor = eventlet.spawn(_eventlet_serve, gsock, hfun, self.worker_connections)
            acceptors.append(acceptor)
            eventlet.sleep(0.0)
        while self.alive:
            self.notify()
            eventlet.sleep(1.0)
        self.notify()
        t = None
        try:
            with eventlet.Timeout(self.cfg.graceful_timeout) as t:
                for a in acceptors:
                    a.kill(eventlet.StopServe())
                for a in acceptors:
                    a.wait()
        except eventlet.Timeout as te:
            if te != t:
                raise
            for a in acceptors:
                a.kill()