__license__ = 'GPL v3'
__copyright__ = '2015, Kovid Goyal <kovid at kovidgoyal.net>'
import errno
import os
import signal
import socket
import ssl
import subprocess
import sys
import time
from threading import Lock, Thread
from calibre.constants import islinux, ismacos, iswindows
from calibre.srv.http_response import create_http_handler
from calibre.srv.loop import ServerLoop
from calibre.srv.opts import Options
from calibre.srv.standalone import create_option_parser
from calibre.srv.utils import create_sock_pair
from calibre.srv.web_socket import DummyHandler
from calibre.utils.monotonic import monotonic
from polyglot.builtins import error_message, itervalues, native_string_type
from polyglot.queue import Empty, Queue
MAX_RETRIES = 10

class NoAutoReload(EnvironmentError):
    pass

class WatcherBase:
    EXTENSIONS_TO_WATCH = frozenset('py pyj svg'.split())
    BOUNCE_INTERVAL = 2

    def __init__(self, worker, log):
        if False:
            print('Hello World!')
        (self.worker, self.log) = (worker, log)
        fpath = os.path.abspath(__file__)
        d = os.path.dirname
        self.base = d(d(d(d(fpath))))
        self.last_restart_time = monotonic()

    def handle_modified(self, modified):
        if False:
            i = 10
            return i + 15
        if modified:
            if monotonic() - self.last_restart_time > self.BOUNCE_INTERVAL:
                modified = {os.path.relpath(x, self.base) if x.startswith(self.base) else x for x in modified if x}
                changed = os.pathsep.join(sorted(modified))
                self.log('')
                self.log.warn('Restarting server because of changed files:', changed)
                self.log('')
                self.worker.restart()
                self.last_restart_time = monotonic()

    def force_restart(self):
        if False:
            for i in range(10):
                print('nop')
        self.worker.restart(forced=True)
        self.last_restart_time = monotonic()

    def file_is_watched(self, fname):
        if False:
            return 10
        return fname and fname.rpartition('.')[-1] in self.EXTENSIONS_TO_WATCH
if islinux:
    import select
    from calibre.utils.inotify import INotifyTreeWatcher

    class Watcher(WatcherBase):

        def __init__(self, root_dirs, worker, log):
            if False:
                i = 10
                return i + 15
            WatcherBase.__init__(self, worker, log)
            (self.client_sock, self.srv_sock) = create_sock_pair()
            self.fd_map = {}
            for d in frozenset(root_dirs):
                w = INotifyTreeWatcher(d, self.ignore_event)
                self.fd_map[w._inotify_fd] = w

        def loop(self):
            if False:
                for i in range(10):
                    print('nop')
            while True:
                r = select.select([self.srv_sock] + list(self.fd_map), [], [])[0]
                modified = set()
                for fd in r:
                    if fd is self.srv_sock:
                        self.srv_sock.recv(1000)
                        self.force_restart()
                        continue
                    w = self.fd_map[fd]
                    modified |= w()
                self.handle_modified(modified)

        def ignore_event(self, path, name):
            if False:
                while True:
                    i = 10
            return not self.file_is_watched(name)

        def wakeup(self):
            if False:
                while True:
                    i = 10
            self.client_sock.sendall(b'w')
elif iswindows:
    from calibre.srv.utils import HandleInterrupt
    from calibre_extensions import winutil

    class TreeWatcher(Thread):

        def __init__(self, path_to_watch, modified_queue):
            if False:
                return 10
            Thread.__init__(self, name='TreeWatcher', daemon=True)
            self.modified_queue = modified_queue
            self.path_to_watch = path_to_watch

        def run(self):
            if False:
                return 10
            dir_handle = winutil.create_file(self.path_to_watch, winutil.FILE_LIST_DIRECTORY, winutil.FILE_SHARE_READ, winutil.OPEN_EXISTING, winutil.FILE_FLAG_BACKUP_SEMANTICS)
            try:
                buffer = b'0' * 8192
                while True:
                    try:
                        results = winutil.read_directory_changes(dir_handle, buffer, True, winutil.FILE_NOTIFY_CHANGE_FILE_NAME | winutil.FILE_NOTIFY_CHANGE_DIR_NAME | winutil.FILE_NOTIFY_CHANGE_ATTRIBUTES | winutil.FILE_NOTIFY_CHANGE_SIZE | winutil.FILE_NOTIFY_CHANGE_LAST_WRITE | winutil.FILE_NOTIFY_CHANGE_SECURITY)
                        for (action, filename) in results:
                            if self.file_is_watched(filename):
                                self.modified_queue.put(os.path.join(self.path_to_watch, filename))
                    except OverflowError:
                        pass
            except Exception:
                import traceback
                traceback.print_exc()

    class Watcher(WatcherBase):

        def __init__(self, root_dirs, worker, log):
            if False:
                for i in range(10):
                    print('nop')
            WatcherBase.__init__(self, worker, log)
            self.watchers = []
            self.modified_queue = Queue()
            for d in frozenset(root_dirs):
                self.watchers.append(TreeWatcher(d, self.modified_queue))

        def wakeup(self):
            if False:
                for i in range(10):
                    print('nop')
            self.modified_queue.put(True)

        def loop(self):
            if False:
                while True:
                    i = 10
            for w in self.watchers:
                w.start()
            with HandleInterrupt(lambda : self.modified_queue.put(None)):
                while True:
                    path = self.modified_queue.get()
                    if path is None:
                        break
                    if path is True:
                        self.force_restart()
                    else:
                        self.handle_modified({path})
elif ismacos:
    from fsevents import Observer, Stream

    class Watcher(WatcherBase):

        def __init__(self, root_dirs, worker, log):
            if False:
                print('Hello World!')
            WatcherBase.__init__(self, worker, log)
            self.stream = Stream(self.notify, *(x.encode('utf-8') for x in root_dirs), file_events=True)
            self.wait_queue = Queue()

        def wakeup(self):
            if False:
                print('Hello World!')
            self.wait_queue.put(True)

        def loop(self):
            if False:
                return 10
            observer = Observer()
            observer.schedule(self.stream)
            observer.daemon = True
            observer.start()
            try:
                while True:
                    try:
                        if self.wait_queue.get(10000) is True:
                            self.force_restart()
                    except Empty:
                        pass
            finally:
                observer.unschedule(self.stream)
                observer.stop()

        def notify(self, ev):
            if False:
                print('Hello World!')
            name = ev.name
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            if self.file_is_watched(name):
                self.handle_modified({name})
else:
    Watcher = None

def find_dirs_to_watch(fpath, dirs, add_default_dirs):
    if False:
        i = 10
        return i + 15
    dirs = {os.path.abspath(x) for x in dirs}

    def add(x):
        if False:
            print('Hello World!')
        if os.path.isdir(x):
            dirs.add(x)
    if add_default_dirs:
        d = os.path.dirname
        srv = d(fpath)
        add(srv)
        base = d(d(d(srv)))
        add(os.path.join(base, 'resources', 'server'))
        add(os.path.join(base, 'src', 'calibre', 'db'))
        add(os.path.join(base, 'src', 'pyj'))
        add(os.path.join(base, 'imgsrc', 'srv'))
    return dirs

def join_process(p, timeout=5):
    if False:
        while True:
            i = 10
    t = Thread(target=p.wait, name='JoinProcess')
    t.daemon = True
    t.start()
    t.join(timeout)
    return p.poll()

class Worker:

    def __init__(self, cmd, log, server, timeout=5):
        if False:
            while True:
                i = 10
        self.cmd = cmd
        self.log = log
        self.server = server
        self.p = None
        self.wakeup = None
        self.timeout = timeout
        cmd = self.cmd
        if 'calibre-debug' in cmd[0].lower():
            try:
                idx = cmd.index('--')
            except ValueError:
                cmd = ['srv']
            else:
                cmd = ['srv'] + cmd[idx + 1:]
        opts = create_option_parser().parse_args(cmd)[0]
        self.port = opts.port
        self.uses_ssl = bool(opts.ssl_certfile and opts.ssl_keyfile)
        self.connection_timeout = opts.timeout
        self.retry_count = 0
        t = Thread(name='PingThread', target=self.ping_thread)
        t.daemon = True
        t.start()

    def ping_thread(self):
        if False:
            for i in range(10):
                print('nop')
        while True:
            self.server.ping()
            time.sleep(30)

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.restart()
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        if self.p and self.p.poll() is None:
            self.clean_kill(send_signal=False)

    def clean_kill(self, send_signal=True):
        if False:
            return 10
        if self.p is not None:
            if send_signal:
                self.p.send_signal(getattr(signal, 'CTRL_BREAK_EVENT', signal.SIGINT))
            if join_process(self.p) is None:
                self.p.kill()
                self.p.wait()
            self.log('Killed server process %d with return code: %d' % (self.p.pid, self.p.returncode))
            self.p = None

    def restart(self, forced=False):
        if False:
            while True:
                i = 10
        from calibre.utils.rapydscript import CompileFailure, compile_srv
        self.clean_kill()
        if forced:
            self.retry_count += 1
        else:
            self.retry_count = 0
        try:
            compile_srv()
        except OSError as e:
            if e.errno != errno.ENOENT or not getattr(e, 'filename', False):
                raise
            st = monotonic()
            while not os.path.exists(e.filename) and monotonic() - st < 3:
                time.sleep(0.01)
            compile_srv()
        except CompileFailure as e:
            self.log.error(error_message(e))
            time.sleep(0.1 * self.retry_count)
            if self.retry_count < MAX_RETRIES and self.wakeup is not None:
                self.wakeup()
            return
        self.retry_count = 0
        self.p = subprocess.Popen(self.cmd, creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0))
        self.wait_for_listen()
        self.server.notify_reload()

    def wait_for_listen(self):
        if False:
            for i in range(10):
                print('nop')
        st = monotonic()
        while monotonic() - st < 5:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            try:
                if self.uses_ssl:
                    s = ssl.wrap_socket(s)
                s.connect(('localhost', self.port))
                return
            except OSError:
                time.sleep(0.01)
            finally:
                s.close()
        self.log.error('Restarted server did not start listening on:', self.port)

class ReloadHandler(DummyHandler):

    def __init__(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        DummyHandler.__init__(self, *args, **kw)
        self.connections = {}
        self.conn_lock = Lock()

    def handle_websocket_upgrade(self, connection_id, connection_ref, inheaders):
        if False:
            for i in range(10):
                print('nop')
        with self.conn_lock:
            self.connections[connection_id] = connection_ref

    def handle_websocket_close(self, connection_id):
        if False:
            i = 10
            return i + 15
        with self.conn_lock:
            self.connections.pop(connection_id, None)

    def notify_reload(self):
        if False:
            for i in range(10):
                print('nop')
        with self.conn_lock:
            for connref in itervalues(self.connections):
                conn = connref()
                if conn is not None and conn.ready:
                    conn.send_websocket_message('reload')

    def ping(self):
        if False:
            for i in range(10):
                print('nop')
        with self.conn_lock:
            for connref in itervalues(self.connections):
                conn = connref()
                if conn is not None and conn.ready:
                    conn.send_websocket_message('ping')

class ReloadServer(Thread):
    daemon = True

    def __init__(self, listen_on):
        if False:
            i = 10
            return i + 15
        Thread.__init__(self, name='ReloadServer')
        self.reload_handler = ReloadHandler()
        self.loop = ServerLoop(create_http_handler(websocket_handler=self.reload_handler), opts=Options(shutdown_timeout=0.1, listen_on=listen_on or '127.0.0.1', port=0))
        self.loop.LISTENING_MSG = None
        self.notify_reload = self.reload_handler.notify_reload
        self.ping = self.reload_handler.ping
        self.start()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.loop.serve_forever()
        except KeyboardInterrupt:
            pass

    def __enter__(self):
        if False:
            print('Hello World!')
        while not self.loop.ready and self.is_alive():
            time.sleep(0.01)
        self.address = self.loop.bound_address[:2]
        os.environ['CALIBRE_AUTORELOAD_PORT'] = native_string_type(self.address[1])
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        self.loop.stop()
        self.join(self.loop.opts.shutdown_timeout)

def auto_reload(log, dirs=frozenset(), cmd=None, add_default_dirs=True, listen_on=None):
    if False:
        return 10
    if Watcher is None:
        raise NoAutoReload('Auto-reload is not supported on this operating system')
    fpath = os.path.abspath(__file__)
    if not os.access(fpath, os.R_OK):
        raise NoAutoReload('Auto-reload can only be used when running from source')
    if cmd is None:
        cmd = list(sys.argv)
        cmd.remove('--auto-reload')
    if os.path.basename(cmd[0]) == 'run-local':
        cmd.insert(1, 'calibre-server')
    dirs = find_dirs_to_watch(fpath, dirs, add_default_dirs)
    log('Auto-restarting server on changes press Ctrl-C to quit')
    log('Watching %d directory trees for changes' % len(dirs))
    with ReloadServer(listen_on) as server, Worker(cmd, log, server) as worker:
        w = Watcher(dirs, worker, log)
        worker.wakeup = w.wakeup
        try:
            w.loop()
        except KeyboardInterrupt:
            pass