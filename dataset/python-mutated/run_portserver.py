"""A server to hand out network ports to applications running on one host.
Typical usage:
 1) Run one instance of this process on each of your unittest farm hosts.
 2) Set the PORTSERVER_ADDRESS environment variable in your test runner
    environment to let the portpicker library know to use a port server
    rather than attempt to find ports on its own.
$ /path/to/portserver.py &
$ export PORTSERVER_ADDRESS=portserver.sock
$ # ... launch a bunch of tests that use portpicker ...
"""
from __future__ import annotations
import argparse
import collections
import logging
import os
import socket
import sys
import threading
from typing import Callable, Deque, Final, List, Optional, Sequence
from scripts import common
from core import utils
_PROTOCOLS: Final = [(socket.SOCK_STREAM, socket.IPPROTO_TCP), (socket.SOCK_DGRAM, socket.IPPROTO_UDP)]

def get_process_command_line(pid: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the command for a process.\n\n    Args:\n        pid: int. The process ID.\n\n    Returns:\n        str. The command that started the process.\n    '
    try:
        with utils.open_file('/proc/{}/cmdline'.format(pid), 'r') as f:
            return f.read()
    except IOError:
        return ''

def get_process_start_time(pid: int) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Get the start time for a process.\n\n    Args:\n        pid: int. The process ID.\n\n    Returns:\n        int. The time when the process started.\n    '
    try:
        with utils.open_file('/proc/{}/stat'.format(pid), 'r') as f:
            return int(f.readline().split()[21])
    except IOError:
        return 0

def sock_bind(port: int, socket_type: int, socket_protocol: int) -> Optional[int]:
    if False:
        while True:
            i = 10
    'Try to bind to a socket of the specified type, protocol, and port.\n    For the port to be considered available, the kernel must support at least\n    one of (IPv6, IPv4), and the port must be available on each supported\n    family.\n\n    Args:\n        port: int. The port number to bind to, or 0 to have the OS pick\n            a free port.\n        socket_type: int. The type of the socket (e.g.:\n            socket.SOCK_STREAM).\n        socket_protocol: int. The protocol of the socket (e.g.:\n            socket.IPPROTO_TCP).\n\n    Returns:\n        int|None. The port number on success or None on failure.\n    '
    got_socket = False
    for family in (socket.AF_INET6, socket.AF_INET):
        try:
            sock = socket.socket(family, socket_type, socket_protocol)
            got_socket = True
        except socket.error:
            continue
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', port))
            if socket_type == socket.SOCK_STREAM:
                sock.listen(1)
            port = sock.getsockname()[1]
        except socket.error:
            return None
        finally:
            sock.close()
    return port if got_socket else None

def is_port_free(port: int) -> bool:
    if False:
        while True:
            i = 10
    'Check if specified port is free.\n\n    Args:\n        port: int. Port to check.\n\n    Returns:\n        bool. Whether the port is free to use for both TCP and UDP.\n    '
    return bool(sock_bind(port, *_PROTOCOLS[0]) and sock_bind(port, *_PROTOCOLS[1]))

def should_allocate_port(pid: int) -> bool:
    if False:
        print('Hello World!')
    'Determine whether to allocate a port for a process id.\n\n    Args:\n        pid: int. The process ID.\n\n    Returns:\n        bool. Whether or not to allocate a port to the process.\n    '
    if pid <= 0:
        logging.info('Not allocating a port to invalid pid')
        return False
    if pid == 1:
        logging.info('Not allocating a port to init.')
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        logging.info('Not allocating a port to a non-existent process')
        return False
    return True

class _PortInfo:
    """Container class for information about a given port assignment.

    Attributes:
      port: int. Port number.
      pid: int. Process id or 0 if unassigned.
      start_time: int. Time in seconds since the epoch that the process
          started.
    """
    __slots__ = ('port', 'pid', 'start_time')

    def __init__(self, port: int) -> None:
        if False:
            return 10
        self.port = port
        self.pid = 0
        self.start_time = 0

class PortPool:
    """Manage available ports for processes.

    Ports are reclaimed when the reserving process exits and the reserved port
    is no longer in use.  Only ports which are free for both TCP and UDP will be
    handed out.  It is easier to not differentiate between protocols.
    The pool must be pre-seeded with add_port_to_free_pool() calls
    after which get_port_for_process() will allocate and reclaim ports.
    The len() of a PortPool returns the total number of ports being managed.

    Attributes:
      ports_checked_for_last_request: int. The number of ports examined
          in order to return from the most recent get_port_for_process()
          request.  A high number here likely means the number of
          available ports with no active process using them is getting
          low.
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._port_queue: Deque[_PortInfo] = collections.deque()
        self.ports_checked_for_last_request = 0

    def num_ports(self) -> int:
        if False:
            print('Hello World!')
        'Get the number of ports in the pool.\n\n        Returns:\n            int. The number of ports in the pool.\n        '
        return len(self._port_queue)

    def get_port_for_process(self, pid: int) -> int:
        if False:
            return 10
        'Allocates a port for the given process.\n\n        Args:\n            pid: int. ID for process to allocate port to.\n\n        Returns:\n            int. Allocated port or 0 if none could be allocated.\n\n        Raises:\n            RuntimeError. No ports being managed.\n        '
        if not self._port_queue:
            raise RuntimeError('No ports being managed.')
        check_count = 0
        max_ports_to_test = len(self._port_queue)
        while check_count < max_ports_to_test:
            candidate = self._port_queue.pop()
            self._port_queue.appendleft(candidate)
            check_count += 1
            if candidate.start_time == 0 or candidate.start_time != get_process_start_time(candidate.pid):
                if is_port_free(candidate.port):
                    candidate.pid = pid
                    candidate.start_time = get_process_start_time(pid)
                    if not candidate.start_time:
                        logging.info("Can't read start time for pid %d.", pid)
                    self.ports_checked_for_last_request = check_count
                    return candidate.port
                else:
                    logging.info('Port %d unexpectedly in use, last owning pid %d.', candidate.port, candidate.pid)
        logging.info('All ports in use.')
        self.ports_checked_for_last_request = check_count
        return 0

    def add_port_to_free_pool(self, port: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a new port to the free pool for allocation.\n\n        Args:\n            port: int. The port number to add to the pool.\n\n        Raises:\n            ValueError. The given port not in [1, 65535] range.\n        '
        if port < 1 or port > 65535:
            raise ValueError('Port must be in the [1, 65535] range, not %d.' % port)
        port_info = _PortInfo(port=port)
        self._port_queue.append(port_info)

class PortServerRequestHandler:
    """A class to handle port allocation and status requests.

    Allocates ports to process ids via the dead simple port server protocol
    when the handle_port_request asyncio.coroutine handler has been registered.
    Statistics can be logged using the dump_stats method.
    """

    def __init__(self, ports_to_serve: Sequence[int]) -> None:
        if False:
            return 10
        'Initialize a new port server.\n\n        Args:\n            ports_to_serve: Sequence[int]. A sequence of unique port numbers\n                to test and offer up to clients.\n        '
        self._port_pool = PortPool()
        self._total_allocations = 0
        self._denied_allocations = 0
        self._client_request_errors = 0
        for port in ports_to_serve:
            self._port_pool.add_port_to_free_pool(port)

    def handle_port_request(self, client_data: bytes) -> Optional[bytes]:
        if False:
            i = 10
            return i + 15
        'Given a port request body, parse it and respond appropriately.\n\n        Args:\n            client_data: bytes. The request bytes from the client.\n\n        Returns:\n            Optional[bytes]. The response to return to the client.\n        '
        try:
            pid = int(client_data)
        except ValueError as error:
            self._client_request_errors += 1
            logging.warning('Could not parse request: %s', error)
            return None
        logging.info('Request on behalf of pid %d.', pid)
        logging.info('cmdline: %s', get_process_command_line(pid))
        if not should_allocate_port(pid):
            self._denied_allocations += 1
            return None
        port = self._port_pool.get_port_for_process(pid)
        if port > 0:
            self._total_allocations += 1
            logging.debug('Allocated port %d to pid %d', port, pid)
            return '{:d}\n'.format(port).encode(encoding='utf-8')
        else:
            self._denied_allocations += 1
            logging.info('Denied allocation to pid %d', pid)
            return b''

    def dump_stats(self) -> None:
        if False:
            while True:
                i = 10
        'Logs statistics of our operation.'
        logging.info('Dumping statistics:')
        stats = []
        stats.append('client-request-errors {}'.format(self._client_request_errors))
        stats.append('denied-allocations {}'.format(self._denied_allocations))
        stats.append('num-ports-managed {}'.format(self._port_pool.num_ports()))
        stats.append('num-ports-checked-for-last-request {}'.format(self._port_pool.ports_checked_for_last_request))
        stats.append('total-allocations {}'.format(self._total_allocations))
        for stat in stats:
            logging.info(stat)

def _parse_command_line(args: Optional[List[str]]=None) -> argparse.Namespace:
    if False:
        while True:
            i = 10
    'Configure and parse our command line flags.\n\n    Returns:\n        Namespace. The parsed arguments.\n    '
    parser = argparse.ArgumentParser()
    parser.add_argument('--portserver_static_pool', type=str, default='15000-24999', help='Comma separated N-P Range(s) of ports to manage (inclusive).')
    parser.add_argument('--portserver_unix_socket_address', type=str, default='portserver.sock', help='Address of AF_UNIX socket on which to listen (first @ is a NUL).')
    if not args:
        args = sys.argv[1:]
    return parser.parse_args(args=args)

def _parse_port_ranges(pool_str: str) -> List[int]:
    if False:
        while True:
            i = 10
    "Given a 'N-P,X-Y' description of port ranges, return a set of ints.\n\n    Args:\n        pool_str: str. The N-P,X-Y description of port ranges.\n\n    Returns:\n        List[int]. The port numbers in the port ranges.\n    "
    ports = set()
    for range_str in pool_str.split(','):
        try:
            (a, b) = range_str.split('-', 1)
            (start, end) = (int(a), int(b))
        except ValueError:
            logging.info('Ignoring unparsable port range %r.', range_str)
            continue
        if start < 1 or end > 65535:
            logging.info('Ignoring out of bounds port range %r.', range_str)
            continue
        ports.update(set(range(start, end + 1)))
    return list(ports)

class Server:
    """Manages the portserver server.

    Attributes:
        max_backlog: int. The maximum number of pending requests to hold
            at a time.
        message_size: int. Maximum number of bytes to read from each
            connection to read the request.
    """
    max_backlog = 5
    message_size = 1024

    def __init__(self, handler: Callable[[bytes], Optional[bytes]], socket_path: str) -> None:
        if False:
            while True:
                i = 10
        'Runs the portserver\n\n        Args:\n            handler: Callable. Function that accepts a port allocation\n                request string and returns the allocated port number.\n            socket_path: str. Path to socket file.\n        '
        self.socket_path = socket_path
        self.socket = self._start_server(self.socket_path)
        self.handler = handler

    def run(self) -> None:
        if False:
            while True:
                i = 10
        'Run the server in an infinite loop.\n\n        Spawns a thread to handle each connection to the socket. Uses\n        the handle_connection function to handle each connection.\n        '
        while True:
            (connection, _) = self.socket.accept()
            thread = threading.Thread(target=Server.handle_connection, args=(connection, self.handler))
            thread.start()

    def close(self) -> None:
        if False:
            print('Hello World!')
        'Gracefully shut down the server.\n\n        Shutting down the server involves closing the socket and\n        removing the socket file.\n        '
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except socket.error:
            pass
        finally:
            try:
                self.socket.close()
            finally:
                if not self.socket_path.startswith('\x00'):
                    os.remove(self.socket_path)

    @staticmethod
    def handle_connection(connection: socket.SocketType, handler: Callable[[bytes], socket.SocketType]) -> None:
        if False:
            while True:
                i = 10
        'Handle a socket connection.\n\n        Reads the request from the socket connection and passes it to\n        the handler.\n\n        Args:\n            connection: Socket. The connection socket to read the\n                request from.\n            handler: Callable. The handler function that will handle the\n                request. Should accept a string with the PID of the\n                requesting process and return allocated socket.\n        '
        request = connection.recv(Server.message_size)
        response = handler(request)
        connection.sendall(response)
        connection.close()

    def _start_server(self, path: str) -> socket.SocketType:
        if False:
            while True:
                i = 10
        'Start the server bound to a socket file.\n\n        Args:\n            path: str. Path to socket file. No such file should exist,\n                and a new one will be created.\n\n        Returns:\n            Socket. A new socket object bound to the socket file.\n\n        Raises:\n            RuntimeError. Failed to bind socket to the given path.\n        '
        sock = self._get_socket()
        try:
            sock.bind(path)
        except socket.error as err:
            raise RuntimeError('Failed to bind socket {}. Error: {}'.format(path, err)) from err
        sock.listen(self.max_backlog)
        return sock

    def _get_socket(self) -> socket.SocketType:
        if False:
            while True:
                i = 10
        'Get a new socket.\n\n        Returns:\n            Socket. A new socket object. If UNIX sockets are supported,\n            such a socket will be created. Otherwise, an AF_INET socket\n            is created.\n        '
        if hasattr(socket, 'AF_UNIX'):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock

def main(args: Optional[List[str]]=None) -> None:
    if False:
        print('Hello World!')
    'Runs the portserver until ctrl-C, then shuts it down.'
    config = _parse_command_line(args)
    ports_to_serve = _parse_port_ranges(config.portserver_static_pool)
    if not ports_to_serve:
        logging.error('No ports. Invalid port ranges in --portserver_static_pool?')
        sys.exit(1)
    request_handler = PortServerRequestHandler(ports_to_serve)
    server = Server(request_handler.handle_port_request, config.portserver_unix_socket_address.replace('@', '\x00', 1))
    logging.info('Serving portserver on %s' % config.portserver_unix_socket_address)
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info('Stopping portserver due to ^C.')
    finally:
        server.close()
        request_handler.dump_stats()
        logging.info('Shutting down portserver.')
        sys.exit(0)
if __name__ == '__main__':
    main()