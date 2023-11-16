"""
Python WebSocket server base with support for "wss://" encryption.
Copyright 2011 Joel Martin
Copyright 2016 Pierre Ossman
Licensed under LGPL version 3 (see docs/LICENSE.LGPL-3)

You can make a cert/key with openssl using:
openssl req -new -x509 -days 365 -nodes -out self.pem -keyout self.pem
as taken from http://docs.python.org/dev/library/ssl.html#certificates

"""
import os, sys, time, errno, signal, socket, select, logging
import multiprocessing
from http.server import SimpleHTTPRequestHandler
for (mod, msg) in [('ssl', 'TLS/SSL/wss is disabled'), ('resource', 'daemonizing is disabled')]:
    try:
        globals()[mod] = __import__(mod)
    except ImportError:
        globals()[mod] = None
        print("WARNING: no '%s' module, %s" % (mod, msg))
if sys.platform == 'win32':
    import multiprocessing.reduction
from websockify.websocket import WebSocketWantReadError, WebSocketWantWriteError
from websockify.websocketserver import WebSocketRequestHandlerMixIn

class CompatibleWebSocket(WebSocketRequestHandlerMixIn.SocketClass):

    def select_subprotocol(self, protocols):
        if False:
            print('Hello World!')
        if 'binary' in protocols:
            return 'binary'
        else:
            return ''

class WebSockifyRequestHandler(WebSocketRequestHandlerMixIn, SimpleHTTPRequestHandler):
    """
    WebSocket Request Handler Class, derived from SimpleHTTPRequestHandler.
    Must be sub-classed with new_websocket_client method definition.
    The request handler can be configured by setting optional
    attributes on the server object:

    * only_upgrade: If true, SimpleHTTPRequestHandler will not be enabled,
      only websocket is allowed.
    * verbose: If true, verbose logging is activated.
    * daemon: Running as daemon, do not write to console etc
    * record: Record raw frame data as JavaScript array into specified filename
    * run_once: Handle a single request
    * handler_id: A sequence number for this connection, appended to record filename
    """
    server_version = 'WebSockify'
    protocol_version = 'HTTP/1.1'
    SocketClass = CompatibleWebSocket

    class CClose(Exception):
        pass

    def __init__(self, req, addr, server):
        if False:
            i = 10
            return i + 15
        self.only_upgrade = getattr(server, 'only_upgrade', False)
        self.verbose = getattr(server, 'verbose', False)
        self.daemon = getattr(server, 'daemon', False)
        self.record = getattr(server, 'record', False)
        self.run_once = getattr(server, 'run_once', False)
        self.rec = None
        self.handler_id = getattr(server, 'handler_id', False)
        self.file_only = getattr(server, 'file_only', False)
        self.traffic = getattr(server, 'traffic', False)
        self.web_auth = getattr(server, 'web_auth', False)
        self.host_token = getattr(server, 'host_token', False)
        self.logger = getattr(server, 'logger', None)
        if self.logger is None:
            self.logger = WebSockifyServer.get_logger()
        super().__init__(req, addr, server)

    def log_message(self, format, *args):
        if False:
            i = 10
            return i + 15
        self.logger.info('%s - - [%s] %s' % (self.client_address[0], self.log_date_time_string(), format % args))

    def print_traffic(self, token='.'):
        if False:
            while True:
                i = 10
        ' Show traffic flow mode. '
        if self.traffic:
            sys.stdout.write(token)
            sys.stdout.flush()

    def msg(self, msg, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Output message with handler_id prefix. '
        prefix = '% 3d: ' % self.handler_id
        self.logger.log(logging.INFO, '%s%s' % (prefix, msg), *args, **kwargs)

    def vmsg(self, msg, *args, **kwargs):
        if False:
            return 10
        ' Same as msg() but as debug. '
        prefix = '% 3d: ' % self.handler_id
        self.logger.log(logging.DEBUG, '%s%s' % (prefix, msg), *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        if False:
            print('Hello World!')
        ' Same as msg() but as warning. '
        prefix = '% 3d: ' % self.handler_id
        self.logger.log(logging.WARN, '%s%s' % (prefix, msg), *args, **kwargs)

    def send_frames(self, bufs=None):
        if False:
            for i in range(10):
                print('nop')
        ' Encode and send WebSocket frames. Any frames already\n        queued will be sent first. If buf is not set then only queued\n        frames will be sent. Returns True if any frames could not be\n        fully sent, in which case the caller should call again when\n        the socket is ready. '
        tdelta = int(time.time() * 1000) - self.start_time
        if bufs:
            for buf in bufs:
                if self.rec:
                    bufstr = buf.decode('latin1').encode('unicode_escape').decode('ascii').replace("'", "\\'")
                    self.rec.write("'{{{0}{{{1}',\n".format(tdelta, bufstr))
                self.send_parts.append(buf)
        while self.send_parts:
            try:
                self.request.sendmsg(self.send_parts[0])
            except WebSocketWantWriteError:
                self.print_traffic('<.')
                return True
            self.send_parts.pop(0)
            self.print_traffic('<')
        return False

    def recv_frames(self):
        if False:
            return 10
        ' Receive and decode WebSocket frames.\n\n        Returns:\n            (bufs_list, closed_string)\n        '
        closed = False
        bufs = []
        tdelta = int(time.time() * 1000) - self.start_time
        while True:
            try:
                buf = self.request.recvmsg()
            except WebSocketWantReadError:
                self.print_traffic('}.')
                break
            if buf is None:
                closed = {'code': self.request.close_code, 'reason': self.request.close_reason}
                return (bufs, closed)
            self.print_traffic('}')
            if self.rec:
                bufstr = buf.decode('latin1').encode('unicode_escape').decode('ascii').replace("'", "\\'")
                self.rec.write("'}}{0}}}{1}',\n".format(tdelta, bufstr))
            bufs.append(buf)
            if not self.request.pending():
                break
        return (bufs, closed)

    def send_close(self, code=1000, reason=''):
        if False:
            i = 10
            return i + 15
        ' Send a WebSocket orderly close frame. '
        self.request.shutdown(socket.SHUT_RDWR, code, reason)

    def send_pong(self, data=''.encode('ascii')):
        if False:
            i = 10
            return i + 15
        ' Send a WebSocket pong frame. '
        self.request.pong(data)

    def send_ping(self, data=''.encode('ascii')):
        if False:
            i = 10
            return i + 15
        ' Send a WebSocket ping frame. '
        self.request.ping(data)

    def handle_upgrade(self):
        if False:
            i = 10
            return i + 15
        self.validate_connection()
        self.auth_connection()
        super().handle_upgrade()

    def handle_websocket(self):
        if False:
            while True:
                i = 10
        self.server.ws_connection = True
        self.send_parts = []
        self.recv_part = None
        self.start_time = int(time.time() * 1000)
        client_addr = ''
        is_ssl = False
        try:
            client_addr = self.client_address[0]
            is_ssl = self.client_address[2]
        except IndexError:
            pass
        if is_ssl:
            self.stype = 'SSL/TLS (wss://)'
        else:
            self.stype = 'Plain non-SSL (ws://)'
        self.log_message('%s: %s WebSocket connection', client_addr, self.stype)
        if self.path != '/':
            self.log_message("%s: Path: '%s'", client_addr, self.path)
        if self.record:
            fname = '%s.%s' % (self.record, self.handler_id)
            self.log_message('opening record file: %s', fname)
            self.rec = open(fname, 'w+')
            self.rec.write('var VNC_frame_data = [\n')
        try:
            self.new_websocket_client()
        except self.CClose:
            (_, exc, _) = sys.exc_info()
            self.send_close(exc.args[0], exc.args[1])

    def do_GET(self):
        if False:
            i = 10
            return i + 15
        if self.web_auth:
            self.auth_connection()
        if self.only_upgrade:
            self.send_error(405)
        else:
            super().do_GET()

    def list_directory(self, path):
        if False:
            i = 10
            return i + 15
        if self.file_only:
            self.send_error(404)
        else:
            return super().list_directory(path)

    def new_websocket_client(self):
        if False:
            i = 10
            return i + 15
        ' Do something with a WebSockets client connection. '
        raise Exception('WebSocketRequestHandler.new_websocket_client() must be overloaded')

    def validate_connection(self):
        if False:
            while True:
                i = 10
        ' Ensure that the connection has a valid token, and set the target. '
        pass

    def auth_connection(self):
        if False:
            print('Hello World!')
        ' Ensure that the connection is authorized. '
        pass

    def do_HEAD(self):
        if False:
            print('Hello World!')
        if self.web_auth:
            self.auth_connection()
        if self.only_upgrade:
            self.send_error(405)
        else:
            super().do_HEAD()

    def finish(self):
        if False:
            i = 10
            return i + 15
        if self.rec:
            self.rec.write("'EOF'];\n")
            self.rec.close()
        super().finish()

    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        if self.run_once:
            self.handle_one_request()
        else:
            super().handle()

    def log_request(self, code='-', size='-'):
        if False:
            print('Hello World!')
        if self.verbose:
            super().log_request(code, size)

class WebSockifyServer:
    """
    WebSockets server class.
    As an alternative, the standard library SocketServer can be used
    """
    policy_response = '<cross-domain-policy><allow-access-from domain="*" to-ports="*" /></cross-domain-policy>\n'
    log_prefix = 'websocket'

    class EClose(Exception):
        pass

    class Terminate(Exception):
        pass

    def __init__(self, RequestHandlerClass, listen_fd=None, listen_host='', listen_port=None, source_is_ipv6=False, verbose=False, cert='', key='', key_password=None, ssl_only=None, verify_client=False, cafile=None, daemon=False, record='', web='', web_auth=False, file_only=False, run_once=False, timeout=0, idle_timeout=0, traffic=False, tcp_keepalive=True, tcp_keepcnt=None, tcp_keepidle=None, tcp_keepintvl=None, ssl_ciphers=None, ssl_options=0, unix_listen=None, unix_listen_mode=None):
        if False:
            i = 10
            return i + 15
        self.RequestHandlerClass = RequestHandlerClass
        self.verbose = verbose
        self.listen_fd = listen_fd
        self.unix_listen = unix_listen
        self.unix_listen_mode = unix_listen_mode
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.prefer_ipv6 = source_is_ipv6
        self.ssl_only = ssl_only
        self.ssl_ciphers = ssl_ciphers
        self.ssl_options = ssl_options
        self.verify_client = verify_client
        self.daemon = daemon
        self.run_once = run_once
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.traffic = traffic
        self.file_only = file_only
        self.web_auth = web_auth
        self.launch_time = time.time()
        self.ws_connection = False
        self.handler_id = 1
        self.terminating = False
        self.logger = self.get_logger()
        self.tcp_keepalive = tcp_keepalive
        self.tcp_keepcnt = tcp_keepcnt
        self.tcp_keepidle = tcp_keepidle
        self.tcp_keepintvl = tcp_keepintvl
        self.key = None
        self.key_password = key_password
        self.cert = os.path.abspath(cert)
        self.web = self.record = self.cafile = ''
        if key:
            self.key = os.path.abspath(key)
        if web:
            self.web = os.path.abspath(web)
        if record:
            self.record = os.path.abspath(record)
        if cafile:
            self.cafile = os.path.abspath(cafile)
        if self.web:
            os.chdir(self.web)
        self.only_upgrade = not self.web
        if not ssl and self.ssl_only:
            raise Exception("No 'ssl' module and SSL-only specified")
        if self.daemon and (not resource):
            raise Exception("Module 'resource' required to daemonize")
        self.msg('WebSocket server settings:')
        if self.listen_fd != None:
            self.msg('  - Listen for inetd connections')
        elif self.unix_listen != None:
            self.msg('  - Listen on unix socket %s', self.unix_listen)
        else:
            self.msg('  - Listen on %s:%s', self.listen_host, self.listen_port)
        if self.web:
            if self.file_only:
                self.msg('  - Web server (no directory listings). Web root: %s', self.web)
            else:
                self.msg('  - Web server. Web root: %s', self.web)
        if ssl:
            if os.path.exists(self.cert):
                self.msg('  - SSL/TLS support')
                if self.ssl_only:
                    self.msg('  - Deny non-SSL/TLS connections')
            else:
                self.msg('  - No SSL/TLS support (no cert file)')
        else:
            self.msg("  - No SSL/TLS support (no 'ssl' module)")
        if self.daemon:
            self.msg('  - Backgrounding (daemon)')
        if self.record:
            self.msg("  - Recording to '%s.*'", self.record)

    @staticmethod
    def get_logger():
        if False:
            return 10
        return logging.getLogger('%s.%s' % (WebSockifyServer.log_prefix, WebSockifyServer.__class__.__name__))

    @staticmethod
    def socket(host, port=None, connect=False, prefer_ipv6=False, unix_socket=None, unix_socket_mode=None, unix_socket_listen=False, use_ssl=False, tcp_keepalive=True, tcp_keepcnt=None, tcp_keepidle=None, tcp_keepintvl=None):
        if False:
            for i in range(10):
                print('nop')
        ' Resolve a host (and optional port) to an IPv4 or IPv6\n        address. Create a socket. Bind to it if listen is set,\n        otherwise connect to it. Return the socket.\n        '
        flags = 0
        if host == '':
            host = None
        if connect and (not (port or unix_socket)):
            raise Exception('Connect mode requires a port')
        if use_ssl and (not ssl):
            raise Exception('SSL socket requested but Python SSL module not loaded.')
        if not connect and use_ssl:
            raise Exception('SSL only supported in connect mode (for now)')
        if not connect:
            flags = flags | socket.AI_PASSIVE
        if not unix_socket:
            addrs = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM, socket.IPPROTO_TCP, flags)
            if not addrs:
                raise Exception("Could not resolve host '%s'" % host)
            addrs.sort(key=lambda x: x[0])
            if prefer_ipv6:
                addrs.reverse()
            sock = socket.socket(addrs[0][0], addrs[0][1])
            if tcp_keepalive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if tcp_keepcnt:
                    sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT, tcp_keepcnt)
                if tcp_keepidle:
                    sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE, tcp_keepidle)
                if tcp_keepintvl:
                    sock.setsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL, tcp_keepintvl)
            if connect:
                sock.connect(addrs[0][4])
                if use_ssl:
                    sock = ssl.wrap_socket(sock)
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(addrs[0][4])
                sock.listen(100)
        elif unix_socket_listen:
            try:
                os.unlink(unix_socket)
            except FileNotFoundError:
                pass
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            oldmask = os.umask(511 ^ unix_socket_mode)
            try:
                sock.bind(unix_socket)
            finally:
                os.umask(oldmask)
            sock.listen(100)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(unix_socket)
        return sock

    @staticmethod
    def daemonize(keepfd=None, chdir='/'):
        if False:
            for i in range(10):
                print('nop')
        if keepfd is None:
            keepfd = []
        os.umask(0)
        if chdir:
            os.chdir(chdir)
        else:
            os.chdir('/')
        os.setgid(os.getgid())
        os.setuid(os.getuid())
        if os.fork() > 0:
            os._exit(0)
        os.setsid()
        if os.fork() > 0:
            os._exit(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        if maxfd == resource.RLIM_INFINITY:
            maxfd = 256
        for fd in reversed(range(maxfd)):
            try:
                if fd not in keepfd:
                    os.close(fd)
            except OSError:
                (_, exc, _) = sys.exc_info()
                if exc.errno != errno.EBADF:
                    raise
        os.dup2(os.open(os.devnull, os.O_RDWR), sys.stdin.fileno())
        os.dup2(os.open(os.devnull, os.O_RDWR), sys.stdout.fileno())
        os.dup2(os.open(os.devnull, os.O_RDWR), sys.stderr.fileno())

    def do_handshake(self, sock, address):
        if False:
            print('Hello World!')
        '\n        do_handshake does the following:\n        - Peek at the first few bytes from the socket.\n        - If the connection is an HTTPS/SSL/TLS connection then SSL\n          wrap the socket.\n        - Read from the (possibly wrapped) socket.\n        - If we have received a HTTP GET request and the webserver\n          functionality is enabled, answer it, close the socket and\n          return.\n        - Assume we have a WebSockets connection, parse the client\n          handshake data.\n        - Send a WebSockets handshake server response.\n        - Return the socket for this WebSocket client.\n        '
        ready = select.select([sock], [], [], 3)[0]
        if not ready:
            raise self.EClose('')
        handshake = sock.recv(1024, socket.MSG_PEEK)
        if not handshake:
            raise self.EClose('')
        elif handshake[0] in (22, 128):
            if not ssl:
                raise self.EClose("SSL connection but no 'ssl' module")
            if not os.path.exists(self.cert):
                raise self.EClose("SSL connection but '%s' not found" % self.cert)
            retsock = None
            try:
                context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                if self.ssl_ciphers is not None:
                    context.set_ciphers(self.ssl_ciphers)
                context.options = self.ssl_options
                context.load_cert_chain(certfile=self.cert, keyfile=self.key, password=self.key_password)
                if self.verify_client:
                    context.verify_mode = ssl.CERT_REQUIRED
                    if self.cafile:
                        context.load_verify_locations(cafile=self.cafile)
                    else:
                        context.set_default_verify_paths()
                retsock = context.wrap_socket(sock, server_side=True)
            except ssl.SSLError:
                (_, x, _) = sys.exc_info()
                if x.args[0] == ssl.SSL_ERROR_EOF:
                    if len(x.args) > 1:
                        raise self.EClose(x.args[1])
                    else:
                        raise self.EClose('Got SSL_ERROR_EOF')
                else:
                    raise
        elif self.ssl_only:
            raise self.EClose('non-SSL connection received but disallowed')
        else:
            retsock = sock
        if len(address) == 2:
            address = (address[0], address[1], retsock != sock)
        self.RequestHandlerClass(retsock, address, self)
        return retsock

    def msg(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ' Output message as info '
        self.logger.log(logging.INFO, *args, **kwargs)

    def vmsg(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        ' Same as msg() but as debug. '
        self.logger.log(logging.DEBUG, *args, **kwargs)

    def warn(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Same as msg() but as warning. '
        self.logger.log(logging.WARN, *args, **kwargs)

    def started(self):
        if False:
            for i in range(10):
                print('nop')
        ' Called after WebSockets startup '
        self.vmsg('WebSockets server started')

    def poll(self):
        if False:
            i = 10
            return i + 15
        ' Run periodically while waiting for connections. '
        pass

    def terminate(self):
        if False:
            while True:
                i = 10
        if not self.terminating:
            self.terminating = True
            raise self.Terminate()

    def multiprocessing_SIGCHLD(self, sig, stack):
        if False:
            i = 10
            return i + 15
        multiprocessing.active_children()

    def fallback_SIGCHLD(self, sig, stack):
        if False:
            while True:
                i = 10
        try:
            result = os.waitpid(-1, os.WNOHANG)
            while result[0]:
                self.vmsg('Reaped child process %s' % result[0])
                result = os.waitpid(-1, os.WNOHANG)
        except OSError:
            pass

    def do_SIGINT(self, sig, stack):
        if False:
            for i in range(10):
                print('nop')
        self.terminate()

    def do_SIGTERM(self, sig, stack):
        if False:
            while True:
                i = 10
        self.terminate()

    def top_new_client(self, startsock, address):
        if False:
            for i in range(10):
                print('nop')
        ' Do something with a WebSockets client connection. '
        client = None
        try:
            try:
                client = self.do_handshake(startsock, address)
            except self.EClose:
                (_, exc, _) = sys.exc_info()
                if exc.args[0]:
                    self.msg('%s: %s' % (address[0], exc.args[0]))
            except WebSockifyServer.Terminate:
                raise
            except Exception:
                (_, exc, _) = sys.exc_info()
                self.msg('handler exception: %s' % str(exc))
                self.vmsg('exception', exc_info=True)
        finally:
            if client and client != startsock:
                client.close()

    def get_log_fd(self):
        if False:
            while True:
                i = 10
        '\n        Get file descriptors for the loggers.\n        They should not be closed when the process is forked.\n        '
        descriptors = []
        for handler in self.logger.parent.handlers:
            if isinstance(handler, logging.FileHandler):
                descriptors.append(handler.stream.fileno())
        return descriptors

    def start_server(self):
        if False:
            i = 10
            return i + 15
        '\n        Daemonize if requested. Listen for for connections. Run\n        do_handshake() method for each connection. If the connection\n        is a WebSockets client then call new_websocket_client() method (which must\n        be overridden) for each new client connection.\n        '
        if self.listen_fd != None:
            lsock = socket.fromfd(self.listen_fd, socket.AF_INET, socket.SOCK_STREAM)
        elif self.unix_listen != None:
            lsock = self.socket(host=None, unix_socket=self.unix_listen, unix_socket_mode=self.unix_listen_mode, unix_socket_listen=True)
        else:
            lsock = self.socket(self.listen_host, self.listen_port, False, self.prefer_ipv6, tcp_keepalive=self.tcp_keepalive, tcp_keepcnt=self.tcp_keepcnt, tcp_keepidle=self.tcp_keepidle, tcp_keepintvl=self.tcp_keepintvl)
        if self.daemon:
            keepfd = self.get_log_fd()
            keepfd.append(lsock.fileno())
            self.daemonize(keepfd=keepfd, chdir=self.web)
        self.started()
        original_signals = {signal.SIGINT: signal.getsignal(signal.SIGINT), signal.SIGTERM: signal.getsignal(signal.SIGTERM)}
        if getattr(signal, 'SIGCHLD', None) is not None:
            original_signals[signal.SIGCHLD] = signal.getsignal(signal.SIGCHLD)
        signal.signal(signal.SIGINT, self.do_SIGINT)
        signal.signal(signal.SIGTERM, self.do_SIGTERM)
        if getattr(signal, 'SIGCHLD', None) is not None:
            signal.signal(signal.SIGCHLD, self.multiprocessing_SIGCHLD)
        last_active_time = self.launch_time
        try:
            while True:
                try:
                    try:
                        startsock = None
                        pid = err = 0
                        child_count = 0
                        child_count = len(multiprocessing.active_children())
                        time_elapsed = time.time() - self.launch_time
                        if self.timeout and time_elapsed > self.timeout:
                            self.msg('listener exit due to --timeout %s' % self.timeout)
                            break
                        if self.idle_timeout:
                            idle_time = 0
                            if child_count == 0:
                                idle_time = time.time() - last_active_time
                            else:
                                idle_time = 0
                                last_active_time = time.time()
                            if idle_time > self.idle_timeout and child_count == 0:
                                self.msg('listener exit due to --idle-timeout %s' % self.idle_timeout)
                                break
                        try:
                            self.poll()
                            ready = select.select([lsock], [], [], 1)[0]
                            if lsock in ready:
                                (startsock, address) = lsock.accept()
                                if self.unix_listen != None:
                                    address = [self.unix_listen]
                            else:
                                continue
                        except self.Terminate:
                            raise
                        except Exception:
                            (_, exc, _) = sys.exc_info()
                            if hasattr(exc, 'errno'):
                                err = exc.errno
                            elif hasattr(exc, 'args'):
                                err = exc.args[0]
                            else:
                                err = exc[0]
                            if err == errno.EINTR:
                                self.vmsg('Ignoring interrupted syscall')
                                continue
                            else:
                                raise
                        if self.run_once:
                            self.top_new_client(startsock, address)
                            if self.ws_connection:
                                self.msg('%s: exiting due to --run-once' % address[0])
                                break
                        else:
                            self.vmsg('%s: new handler Process' % address[0])
                            p = multiprocessing.Process(target=self.top_new_client, args=(startsock, address))
                            p.start()
                        self.handler_id += 1
                    except (self.Terminate, SystemExit, KeyboardInterrupt):
                        self.msg('In exit')
                        if not self.run_once:
                            children = multiprocessing.active_children()
                            for child in children:
                                self.msg('Terminating child %s' % child.pid)
                                child.terminate()
                        break
                    except Exception:
                        exc = sys.exc_info()[1]
                        self.msg('handler exception: %s', str(exc))
                        self.vmsg('exception', exc_info=True)
                finally:
                    if startsock:
                        startsock.close()
        finally:
            self.vmsg('Closing socket listening at %s:%s', self.listen_host, self.listen_port)
            lsock.close()
            for (sig, func) in original_signals.items():
                signal.signal(sig, func)