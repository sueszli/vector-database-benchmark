from contextlib import closing
import time
import socket
import socketserver
from struct import pack, unpack
import sys
import threading
import traceback
from typing import Generator
import warnings
_SERVER_POLL_INTERVAL = 0.1
_TRUNCATE_MSG_LEN = 4000
_log_print_lock = threading.Lock()

def _get_log_print_lock() -> threading.Lock:
    if False:
        for i in range(10):
            print('nop')
    return _log_print_lock

class WriteLogToStdout(socketserver.StreamRequestHandler):

    def _read_bline(self) -> Generator[bytes, None, None]:
        if False:
            i = 10
            return i + 15
        while self.server.is_active:
            packed_number_bytes = self.rfile.read(4)
            if not packed_number_bytes:
                time.sleep(_SERVER_POLL_INTERVAL)
                continue
            number_bytes = unpack('>i', packed_number_bytes)[0]
            message = self.rfile.read(number_bytes)
            yield message

    def handle(self) -> None:
        if False:
            print('Hello World!')
        self.request.setblocking(0)
        for bline in self._read_bline():
            with _get_log_print_lock():
                sys.stderr.write(bline.decode('utf-8') + '\n')
                sys.stderr.flush()

class LogStreamingServer:

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.server = None
        self.serve_thread = None
        self.port = None

    @staticmethod
    def _get_free_port(spark_host_address: str='') -> int:
        if False:
            print('Hello World!')
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as tcp:
            tcp.bind((spark_host_address, 0))
            (_, port) = tcp.getsockname()
        return port

    def start(self, spark_host_address: str='') -> None:
        if False:
            i = 10
            return i + 15
        if self.server:
            raise RuntimeError('Cannot start the server twice.')

        def serve_task(port: int) -> None:
            if False:
                while True:
                    i = 10
            with socketserver.ThreadingTCPServer(('0.0.0.0', port), WriteLogToStdout) as server:
                self.server = server
                server.is_active = True
                server.serve_forever(poll_interval=_SERVER_POLL_INTERVAL)
        self.port = LogStreamingServer._get_free_port(spark_host_address)
        self.serve_thread = threading.Thread(target=serve_task, args=(self.port,))
        self.serve_thread.setDaemon(True)
        self.serve_thread.start()

    def shutdown(self) -> None:
        if False:
            print('Hello World!')
        if self.server:
            time.sleep(_SERVER_POLL_INTERVAL * 2)
            sys.stdout.flush()
            self.server.is_active = False
            self.server.shutdown()
            self.serve_thread.join()
            self.server = None
            self.serve_thread = None

class LogStreamingClientBase:

    @staticmethod
    def _maybe_truncate_msg(message: str) -> str:
        if False:
            i = 10
            return i + 15
        if len(message) > _TRUNCATE_MSG_LEN:
            message = message[:_TRUNCATE_MSG_LEN]
            return message + '...(truncated)'
        else:
            return message

    def send(self, message: str) -> None:
        if False:
            while True:
                i = 10
        pass

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

class LogStreamingClient(LogStreamingClientBase):
    """
    A client that streams log messages to :class:`LogStreamingServer`.
    In case of failures, the client will skip messages instead of raising an error.
    """
    _log_callback_client = None
    _server_address = None
    _singleton_lock = threading.Lock()

    @staticmethod
    def _init(address: str, port: int) -> None:
        if False:
            while True:
                i = 10
        LogStreamingClient._server_address = (address, port)

    @staticmethod
    def _destroy() -> None:
        if False:
            i = 10
            return i + 15
        LogStreamingClient._server_address = None
        if LogStreamingClient._log_callback_client is not None:
            LogStreamingClient._log_callback_client.close()

    def __init__(self, address: str, port: int, timeout: int=10):
        if False:
            print('Hello World!')
        '\n        Creates a connection to the logging server and authenticates.This client is best effort,\n        if authentication or sending a message  fails, the client will be marked as not alive and\n        stop trying to send message.\n\n        :param address: Address where the service is running.\n        :param port: Port where the service is listening for new connections.\n        '
        self.address = address
        self.port = port
        self.timeout = timeout
        self.sock = None
        self.failed = True
        self._lock = threading.RLock()

    def _fail(self, error_msg: str) -> None:
        if False:
            return 10
        self.failed = True
        warnings.warn(f'{error_msg}: {traceback.format_exc()}\n')

    def _connect(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.port == -1:
            self._fail('Log streaming server is not available.')
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.address, self.port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = sock
            self.failed = False
        except (OSError, IOError):
            self._fail('Error connecting log streaming server')

    def send(self, message: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends a message.\n        '
        with self._lock:
            if self.sock is None:
                self._connect()
            if not self.failed:
                try:
                    message = LogStreamingClientBase._maybe_truncate_msg(message)
                    binary_message = message.encode('utf-8')
                    packed_number_bytes = pack('>i', len(binary_message))
                    self.sock.sendall(packed_number_bytes + binary_message)
                except Exception:
                    self._fail('Error sending logs to driver, stopping log streaming')

    def close(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Closes the connection.\n        '
        if self.sock:
            self.sock.close()
            self.sock = None