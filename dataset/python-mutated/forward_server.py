"""TCP/IP forwarding/echo service for testing."""
from __future__ import print_function
import array
from datetime import datetime
import errno
from functools import partial
import logging
import multiprocessing
import os
import socket
import struct
import sys
import threading
import traceback
import pika.compat
if pika.compat.PY3:

    def buffer(object, offset, size):
        if False:
            while True:
                i = 10
        'array etc. have the buffer protocol'
        return object[offset:offset + size]
try:
    import SocketServer
except ImportError:
    import socketserver as SocketServer

def _trace(fmt, *args):
    if False:
        return 10
    'Format and output the text to stderr'
    print(fmt % args + '\n', end='', file=sys.stderr)

class ForwardServer(object):
    """ Implement a TCP/IP forwarding/echo service for testing. Listens for
    an incoming TCP/IP connection, accepts it, then connects to the given
    remote address and forwards data back and forth between the two
    endpoints.

    This is similar to a subset of `netcat` functionality, but without
    dependency on any specific flavor of netcat

    Connection forwarding example; forward local connection to default
      rabbitmq addr, connect to rabbit via forwarder, then disconnect
      forwarder, then attempt another pika operation to see what happens

        with ForwardServer(("localhost", 5672)) as fwd:
            params = pika.ConnectionParameters(
                host=fwd.server_address[0],
                port=fwd.server_address[1])
            conn = pika.BlockingConnection(params)

        # Once outside the context, the forwarder is disconnected

        # Let's see what happens in pika with a disconnected server
        channel = conn.channel()

    Echo server example
        def produce(sock):
            sock.sendall("12345")
            sock.shutdown(socket.SHUT_WR)

        with ForwardServer(None) as echo:
            sock = socket.socket()
            sock.connect(echo.server_address)

            worker = threading.Thread(target=produce,
                                      args=[sock])
            worker.start()

            data = sock.makefile().read()
            assert data == "12345", data

        worker.join()

    """
    _SUBPROC_TIMEOUT = 10

    def __init__(self, remote_addr, remote_addr_family=socket.AF_INET, remote_socket_type=socket.SOCK_STREAM, server_addr=('127.0.0.1', 0), server_addr_family=socket.AF_INET, server_socket_type=socket.SOCK_STREAM, local_linger_args=None):
        if False:
            i = 10
            return i + 15
        '\n        :param tuple remote_addr: remote server\'s IP address, whose structure\n          depends on remote_addr_family; pair (host-or-ip-addr, port-number).\n          Pass None to have ForwardServer behave as echo server.\n        :param remote_addr_family: socket.AF_INET (the default), socket.AF_INET6\n          or socket.AF_UNIX.\n        :param remote_socket_type: only socket.SOCK_STREAM is supported at this\n          time\n        :param server_addr: optional address for binding this server\'s listening\n          socket; the format depends on server_addr_family; defaults to\n          ("127.0.0.1", 0)\n        :param server_addr_family: Address family for this server\'s listening\n          socket; socket.AF_INET (the default), socket.AF_INET6 or\n          socket.AF_UNIX; defaults to socket.AF_INET\n        :param server_socket_type: only socket.SOCK_STREAM is supported at this\n          time\n        :param tuple local_linger_args: SO_LINGER sockoverride for the local\n          connection sockets, to be configured after connection is accepted.\n          None for default, which is to not change the SO_LINGER option.\n          Otherwise, its a two-tuple, where the first element is the `l_onoff`\n          switch, and the second element is the `l_linger` value, in seconds\n        '
        self._logger = logging.getLogger(__name__)
        self._remote_addr = remote_addr
        self._remote_addr_family = remote_addr_family
        assert remote_socket_type == socket.SOCK_STREAM, remote_socket_type
        self._remote_socket_type = remote_socket_type
        assert server_addr is not None
        self._server_addr = server_addr
        assert server_addr_family is not None
        self._server_addr_family = server_addr_family
        assert server_socket_type == socket.SOCK_STREAM, server_socket_type
        self._server_socket_type = server_socket_type
        self._local_linger_args = local_linger_args
        self._subproc = None

    @property
    def running(self):
        if False:
            print('Hello World!')
        'Property: True if ForwardServer is active'
        return self._subproc is not None

    @property
    def server_address_family(self):
        if False:
            while True:
                i = 10
        "Property: Get listening socket's address family\n\n        NOTE: undefined before server starts and after it shuts down\n        "
        assert self._server_addr_family is not None, 'Not in context'
        return self._server_addr_family

    @property
    def server_address(self):
        if False:
            for i in range(10):
                print('nop')
        " Property: Get listening socket's address; the returned value\n        depends on the listening socket's address family\n\n        NOTE: undefined before server starts and after it shuts down\n        "
        assert self._server_addr is not None, 'Not in context'
        return self._server_addr

    def __enter__(self):
        if False:
            print('Hello World!')
        ' Context manager entry. Starts the forwarding server\n\n        :returns: self\n        '
        return self.start()

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        ' Context manager exit; stops the forwarding server\n        '
        self.stop()

    def start(self):
        if False:
            print('Hello World!')
        ' Start the server\n\n        NOTE: The context manager is the recommended way to use\n        ForwardServer. start()/stop() are alternatives to the context manager\n        use case and are mutually exclusive with it.\n\n        :returns: self\n        '
        queue = multiprocessing.Queue()
        self._subproc = multiprocessing.Process(target=_run_server, kwargs=dict(local_addr=self._server_addr, local_addr_family=self._server_addr_family, local_socket_type=self._server_socket_type, local_linger_args=self._local_linger_args, remote_addr=self._remote_addr, remote_addr_family=self._remote_addr_family, remote_socket_type=self._remote_socket_type, queue=queue))
        self._subproc.daemon = True
        self._subproc.start()
        try:
            (self._server_addr_family, self._server_addr) = queue.get(block=True, timeout=self._SUBPROC_TIMEOUT)
            queue.close()
        except Exception:
            try:
                self._logger.exception('Failed while waiting for local socket info')
                raise
            finally:
                try:
                    self.stop()
                except Exception:
                    self._logger.exception('Emergency subprocess shutdown failed')
        return self

    def stop(self):
        if False:
            print('Hello World!')
        'Stop the server\n\n        NOTE: The context manager is the recommended way to use\n        ForwardServer. start()/stop() are alternatives to the context manager\n        use case and are mutually exclusive with it.\n        '
        self._logger.info('ForwardServer STOPPING')
        try:
            self._subproc.terminate()
            self._subproc.join(timeout=self._SUBPROC_TIMEOUT)
            if self._subproc.is_alive():
                self._logger.error('ForwardServer failed to terminate, killing it')
                os.kill(self._subproc.pid)
                self._subproc.join(timeout=self._SUBPROC_TIMEOUT)
                assert not self._subproc.is_alive(), self._subproc
            exit_code = self._subproc.exitcode
            self._logger.info('ForwardServer terminated with exitcode=%s', exit_code)
        finally:
            self._subproc = None

def _run_server(local_addr, local_addr_family, local_socket_type, local_linger_args, remote_addr, remote_addr_family, remote_socket_type, queue):
    if False:
        return 10
    " Run the server; executed in the subprocess\n\n    :param local_addr: listening address\n    :param local_addr_family: listening address family; one of socket.AF_*\n    :param local_socket_type: listening socket type; typically\n        socket.SOCK_STREAM\n    :param tuple local_linger_args: SO_LINGER sockoverride for the local\n        connection sockets, to be configured after connection is accepted.\n        Pass None to not change SO_LINGER. Otherwise, its a two-tuple, where the\n        first element is the `l_onoff` switch, and the second element is the\n        `l_linger` value in seconds\n    :param remote_addr: address of the target server. Pass None to have\n        ForwardServer behave as echo server\n    :param remote_addr_family: address family for connecting to target server;\n        one of socket.AF_*\n    :param remote_socket_type: socket type for connecting to target server;\n        typically socket.SOCK_STREAM\n    :param multiprocessing.Queue queue: queue for depositing the forwarding\n        server's actual listening socket address family and bound address. The\n        parent process waits for this.\n    "

    class _ThreadedTCPServer(SocketServer.ThreadingMixIn, SocketServer.TCPServer, object):
        """Threaded streaming server for forwarding"""
        address_family = local_addr_family
        socket_type = local_socket_type
        allow_reuse_address = True

        def __init__(self):
            if False:
                return 10
            handler_class_factory = partial(_TCPHandler, local_linger_args=local_linger_args, remote_addr=remote_addr, remote_addr_family=remote_addr_family, remote_socket_type=remote_socket_type)
            super(_ThreadedTCPServer, self).__init__(local_addr, handler_class_factory, bind_and_activate=True)
    server = _ThreadedTCPServer()
    queue.put([server.socket.family, server.server_address])
    queue.close()
    server.serve_forever()

class _TCPHandler(SocketServer.StreamRequestHandler, object):
    """TCP/IP session handler instantiated by TCPServer upon incoming
    connection. Implements forwarding/echo of the incoming connection.
    """
    _SOCK_RX_BUF_SIZE = 16 * 1024

    def __init__(self, request, client_address, server, local_linger_args, remote_addr, remote_addr_family, remote_socket_type):
        if False:
            print('Hello World!')
        '\n        :param request: for super\n        :param client_address: for super\n        "paarm server:  for super\n        :param tuple local_linger_args: SO_LINGER sockoverride for the local\n            connection sockets, to be configured after connection is accepted.\n            Pass None to not change SO_LINGER. Otherwise, its a two-tuple, where\n            the first element is the `l_onoff` switch, and the second element is\n            the `l_linger` value in seconds\n        :param remote_addr: address of the target server. Pass None to have\n            ForwardServer behave as echo server.\n        :param remote_addr_family: address family for connecting to target\n            server; one of socket.AF_*\n        :param remote_socket_type: socket type for connecting to target server;\n            typically socket.SOCK_STREAM\n        :param **kwargs: kwargs for super class\n        '
        self._local_linger_args = local_linger_args
        self._remote_addr = remote_addr
        self._remote_addr_family = remote_addr_family
        self._remote_socket_type = remote_socket_type
        super(_TCPHandler, self).__init__(request=request, client_address=client_address, server=server)

    def handle(self):
        if False:
            while True:
                i = 10
        'Connect to remote and forward data between local and remote'
        local_sock = self.connection
        if self._local_linger_args is not None:
            (l_onoff, l_linger) = self._local_linger_args
            local_sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', l_onoff, l_linger))
        if self._remote_addr is not None:
            remote_dest_sock = remote_src_sock = socket.socket(family=self._remote_addr_family, type=self._remote_socket_type, proto=socket.IPPROTO_IP)
            remote_dest_sock.connect(self._remote_addr)
            _trace('%s _TCPHandler connected to remote %s', datetime.utcnow(), remote_dest_sock.getpeername())
        else:
            (remote_dest_sock, remote_src_sock) = pika.compat._nonblocking_socketpair()
            remote_dest_sock.setblocking(True)
            remote_src_sock.setblocking(True)
        try:
            local_forwarder = threading.Thread(target=self._forward, args=(local_sock, remote_dest_sock))
            local_forwarder.setDaemon(True)
            local_forwarder.start()
            try:
                self._forward(remote_src_sock, local_sock)
            finally:
                local_forwarder.join()
        finally:
            try:
                try:
                    _safe_shutdown_socket(remote_dest_sock, socket.SHUT_RDWR)
                finally:
                    if remote_src_sock is not remote_dest_sock:
                        _safe_shutdown_socket(remote_src_sock, socket.SHUT_RDWR)
            finally:
                remote_dest_sock.close()
                if remote_src_sock is not remote_dest_sock:
                    remote_src_sock.close()

    def _forward(self, src_sock, dest_sock):
        if False:
            i = 10
            return i + 15
        'Forward from src_sock to dest_sock'
        src_peername = src_sock.getpeername()
        _trace('%s forwarding from %s to %s', datetime.utcnow(), src_peername, dest_sock.getpeername())
        try:
            rx_buf = array.array('B', [0] * self._SOCK_RX_BUF_SIZE)
            while True:
                try:
                    nbytes = src_sock.recv_into(rx_buf)
                except pika.compat.SOCKET_ERROR as exc:
                    if exc.errno == errno.EINTR:
                        continue
                    elif exc.errno == errno.ECONNRESET:
                        _trace('%s errno.ECONNRESET from %s', datetime.utcnow(), src_peername)
                        break
                    else:
                        _trace('%s Unexpected errno=%s from %s\n%s', datetime.utcnow(), exc.errno, src_peername, ''.join(traceback.format_stack()))
                        raise
                if not nbytes:
                    _trace('%s EOF on %s', datetime.utcnow(), src_peername)
                    break
                try:
                    dest_sock.sendall(buffer(rx_buf, 0, nbytes))
                except pika.compat.SOCKET_ERROR as exc:
                    if exc.errno == errno.EPIPE:
                        _trace('%s Destination peer %s closed its end of the connection: errno.EPIPE', datetime.utcnow(), dest_sock.getpeername())
                        break
                    elif exc.errno == errno.ECONNRESET:
                        _trace('%s Destination peer %s forcibly closed connection: errno.ECONNRESET', datetime.utcnow(), dest_sock.getpeername())
                        break
                    else:
                        _trace('%s Unexpected errno=%s in sendall to %s\n%s', datetime.utcnow(), exc.errno, dest_sock.getpeername(), ''.join(traceback.format_stack()))
                        raise
        except:
            _trace('forward failed\n%s', ''.join(traceback.format_exc()))
            raise
        finally:
            _trace('%s done forwarding from %s', datetime.utcnow(), src_peername)
            try:
                _safe_shutdown_socket(src_sock, socket.SHUT_RD)
            finally:
                _safe_shutdown_socket(dest_sock, socket.SHUT_WR)

def echo(port=0):
    if False:
        i = 10
        return i + 15
    ' This function implements a simple echo server for testing the\n    Forwarder class.\n\n    :param int port: port number on which to listen\n\n    We run this function and it prints out the listening socket binding.\n    Then, we run Forwarder and point it at this echo "server".\n    Then, we run telnet and point it at forwarder and see if whatever we\n    type gets echoed back to us.\n\n    This function waits for the client to connect and exits after the client\n    closes the connection\n    '
    lsock = socket.socket()
    lsock.bind(('', port))
    lsock.listen(1)
    _trace('Listening on sockname=%s', lsock.getsockname())
    (sock, remote_addr) = lsock.accept()
    try:
        _trace('Connection from peer=%s', remote_addr)
        while True:
            try:
                data = sock.recv(4 * 1024)
            except pika.compat.SOCKET_ERROR as exc:
                if exc.errno == errno.EINTR:
                    continue
                else:
                    raise
            if not data:
                break
            sock.sendall(data)
    finally:
        try:
            _safe_shutdown_socket(sock, socket.SHUT_RDWR)
        finally:
            sock.close()

def _safe_shutdown_socket(sock, how=socket.SHUT_RDWR):
    if False:
        while True:
            i = 10
    ' Shutdown a socket, suppressing ENOTCONN\n    '
    try:
        sock.shutdown(how)
    except pika.compat.SOCKET_ERROR as exc:
        if exc.errno != errno.ENOTCONN:
            raise