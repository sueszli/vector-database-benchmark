"""Unit tests for scripts/run_portserver.py"""
from __future__ import annotations
import builtins
import io
import logging
import os
import socket
import sys
from core import utils
from core.tests import test_utils
from scripts import run_portserver
from typing import List, Union

class MockSocket:
    server_closed = False
    port: int = 8181

    def setsockopt(self, *unused_args: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def bind(self, *unused_args: str) -> None:
        if False:
            return 10
        pass

    def listen(self, *unused_args: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def getsockname(self, *unused_args: str) -> List[Union[str, int]]:
        if False:
            i = 10
            return i + 15
        return ['Address', self.port]

    def recv(self, *unused_args: str) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def sendall(self, *unused_args: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def shutdown(self, *unused_args: str) -> None:
        if False:
            print('Hello World!')
        raise socket.error('Some error occurred.')

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self.server_closed = True

class MockServer:

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def close(self) -> None:
        if False:
            return 10
        pass

class RunPortserverTests(test_utils.GenericTestBase):
    """Unit tests for scripts/run_portserver.py"""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.terminal_logs: list[str] = []

        def mock_logging(*msgs: str) -> None:
            if False:
                return 10
            all_messages = [*msgs]
            for msg in all_messages:
                self.terminal_logs.append(msg)
        self.swap_log = self.swap(logging, 'info', mock_logging)
        self.terminal_err_logs: list[str] = []

        def mock_logging_err(*msgs: str) -> None:
            if False:
                return 10
            all_messages = [*msgs]
            for msg in all_messages:
                self.terminal_err_logs.append(msg)
        self.swap_log_err = self.swap(logging, 'error', mock_logging_err)

    def test_get_process_start_time_handles_ioerror(self) -> None:
        if False:
            i = 10
            return i + 15

        def mock_open(*unused_args: str, **unused_kwargs: str) -> None:
            if False:
                print('Hello World!')
            raise IOError('File not found.')
        pid = 12345
        swap_open = self.swap_with_checks(utils, 'open_file', mock_open, expected_args=(('/proc/{}/stat'.format(pid), 'r'),))
        with swap_open:
            returned_time = run_portserver.get_process_start_time(pid)
        self.assertEqual(returned_time, 0)

    def test_get_process_start_time(self) -> None:
        if False:
            print('Hello World!')
        dummy_file_object = io.StringIO('A B C D E F G H I J K L M N O P Q R S T U 11 V')
        pid = 12345
        swap_open = self.swap_with_checks(utils, 'open_file', lambda *unused_args, **unused_kwargs: dummy_file_object, expected_args=(('/proc/{}/stat'.format(pid), 'r'),))
        with swap_open:
            returned_time = run_portserver.get_process_start_time(pid)
        self.assertEqual(returned_time, 11)
        dummy_file_object.close()

    def test_get_process_command_line_handles_ioerror(self) -> None:
        if False:
            print('Hello World!')

        def mock_open(*unused_args: str, **unused_kwargs: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            raise IOError('File not found.')
        pid = 12345
        swap_open = self.swap_with_checks(utils, 'open_file', mock_open, expected_args=(('/proc/{}/cmdline'.format(pid), 'r'),))
        with swap_open:
            returned_text = run_portserver.get_process_command_line(pid)
        self.assertEqual(returned_text, '')

    def test_get_process_command_line(self) -> None:
        if False:
            i = 10
            return i + 15
        dummy_file_object = io.StringIO('')
        expected_text = dummy_file_object.read()
        pid = 12345
        swap_open = self.swap_with_checks(utils, 'open_file', lambda *unused_args, **unused_kwargs: dummy_file_object, expected_args=(('/proc/{}/cmdline'.format(pid), 'r'),))
        with swap_open:
            returned_text = run_portserver.get_process_command_line(pid)
        self.assertEqual(returned_text, expected_text)
        dummy_file_object.close()

    def test_sock_bind_handles_error_while_creating_socket(self) -> None:
        if False:
            i = 10
            return i + 15
        port = 8181

        def mock_socket(*unused_args: str) -> None:
            if False:
                while True:
                    i = 10
            raise socket.error('Some error occurred.')
        swap_socket = self.swap(socket, 'socket', mock_socket)
        with swap_socket:
            returned_port = run_portserver.sock_bind(port, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.assertIsNone(returned_port)

    def test_socket_gets_bind_to_a_port(self) -> None:
        if False:
            while True:
                i = 10
        swap_socket = self.swap(socket, 'socket', lambda *unused_args: MockSocket())
        with swap_socket:
            returned_port = run_portserver.sock_bind(8181, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        self.assertEqual(returned_port, 8181)

    def test_sock_bind_handles_error_while_getting_port_name(self) -> None:
        if False:
            while True:
                i = 10

        class FailingMockSocket(MockSocket):
            """Socket that fails while invoking getsockname()."""

            def getsockname(self, *unused_args: str) -> None:
                if False:
                    return 10
                raise socket.error('Some error occurred.')
        swap_socket = self.swap(socket, 'socket', lambda *unused_args: FailingMockSocket())
        with swap_socket:
            returned_port = run_portserver.sock_bind(8181, socket.SOCK_DGRAM, socket.IPPROTO_TCP)
        self.assertIsNone(returned_port)

    def test_is_port_free(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        swap_sock_bind = self.swap(run_portserver, 'sock_bind', lambda *unused_args: True)
        with swap_sock_bind:
            result = run_portserver.is_port_free(8181)
        self.assertTrue(result)

    def test_should_allocate_port(self) -> None:
        if False:
            while True:
                i = 10
        pid = 12345
        swap_os_kill = self.swap_with_checks(os, 'kill', lambda *unused_args: None, expected_args=((pid, 0),))
        with swap_os_kill:
            result = run_portserver.should_allocate_port(pid)
        self.assertTrue(result)

    def test_should_allocate_port_handles_invalid_pid(self) -> None:
        if False:
            print('Hello World!')
        pid = 0
        with self.swap_log:
            result = run_portserver.should_allocate_port(pid)
        self.assertFalse(result)
        self.assertIn('Not allocating a port to invalid pid', self.terminal_logs)

    def test_should_allocate_port_handles_init_pid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pid = 1
        with self.swap_log:
            result = run_portserver.should_allocate_port(pid)
        self.assertFalse(result)
        self.assertIn('Not allocating a port to init.', self.terminal_logs)

    def test_should_allocate_port_handles_oserror(self) -> None:
        if False:
            i = 10
            return i + 15
        pid = 12345

        def mock_kill(*unused_args: str) -> None:
            if False:
                print('Hello World!')
            raise OSError('Some XYZ error occurred.')
        swap_os_kill = self.swap_with_checks(os, 'kill', mock_kill, expected_args=((pid, 0),))
        with swap_os_kill, self.swap_log:
            result = run_portserver.should_allocate_port(pid)
        self.assertFalse(result)
        self.assertIn('Not allocating a port to a non-existent process', self.terminal_logs)

    def test_port_pool_handles_invalid_port_request(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        port = -1
        port_pool = run_portserver.PortPool()
        error_msg = 'Port must be in the \\[1, 65535\\] range, not -1.'
        with self.assertRaisesRegex(ValueError, error_msg):
            port_pool.add_port_to_free_pool(port)

    def test_port_pool_handles_empty_port_queue(self) -> None:
        if False:
            return 10
        port_pool = run_portserver.PortPool()
        error_msg = 'No ports being managed.'
        with self.assertRaisesRegex(RuntimeError, error_msg):
            port_pool.get_port_for_process(12345)

    def test_get_port_for_process_successfully(self) -> None:
        if False:
            return 10
        port = 8181
        swap_get_process_start_time = self.swap(run_portserver, 'get_process_start_time', lambda _: 0)
        swap_is_port_free = self.swap(run_portserver, 'is_port_free', lambda _: True)
        port_pool = run_portserver.PortPool()
        port_pool.add_port_to_free_pool(port)
        self.assertEqual(port_pool.num_ports(), 1)
        with swap_get_process_start_time, swap_is_port_free:
            returned_port = port_pool.get_port_for_process(12345)
        self.assertEqual(returned_port, port)

    def test_get_port_for_process_looks_for_free_port(self) -> None:
        if False:
            return 10
        port1 = 8181
        port2 = 8182
        swap_get_process_start_time = self.swap(run_portserver, 'get_process_start_time', lambda _: 1)
        swap_is_port_free = self.swap(run_portserver, 'is_port_free', lambda _: True)
        port_pool = run_portserver.PortPool()
        port_pool.add_port_to_free_pool(port1)
        port_pool.add_port_to_free_pool(port2)
        port = port_pool._port_queue.pop()
        port.start_time = 1
        port_pool._port_queue.append(port)
        self.assertEqual(port_pool.num_ports(), 2)
        with swap_get_process_start_time, swap_is_port_free:
            returned_port = port_pool.get_port_for_process(12345)
        self.assertEqual(returned_port, port1)

    def test_get_port_for_process_handles_no_free_port(self) -> None:
        if False:
            return 10
        port = 8181
        swap_get_process_start_time = self.swap(run_portserver, 'get_process_start_time', lambda _: 0)
        swap_is_port_free = self.swap(run_portserver, 'is_port_free', lambda _: False)
        port_pool = run_portserver.PortPool()
        port_pool.add_port_to_free_pool(port)
        self.assertEqual(port_pool.num_ports(), 1)
        with swap_get_process_start_time, swap_is_port_free, self.swap_log:
            returned_port = port_pool.get_port_for_process(12345)
        self.assertEqual(returned_port, 0)
        self.assertIn('All ports in use.', self.terminal_logs)

    def test_port_server_request_handler_handles_invalid_request(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        request_handler = run_portserver.PortServerRequestHandler((8181,))
        response = request_handler.handle_port_request(b'abcd')
        with self.swap_log:
            request_handler.dump_stats()
        self.assertIsNone(response)
        self.assertIn('client-request-errors 1', self.terminal_logs)

    def test_port_server_request_handler_handles_denied_allocations(self) -> None:
        if False:
            print('Hello World!')
        request_handler = run_portserver.PortServerRequestHandler((8181,))
        response = request_handler.handle_port_request(b'0')
        with self.swap_log:
            request_handler.dump_stats()
        self.assertIsNone(response)
        self.assertIn('denied-allocations 1', self.terminal_logs)

    def test_port_server_request_handler_handles_no_free_ports(self) -> None:
        if False:
            while True:
                i = 10
        request_handler = run_portserver.PortServerRequestHandler((8181,))
        swap_get_port = self.swap(run_portserver.PortPool, 'get_port_for_process', lambda *unused_args: 0)
        swap_should_allocate_port = self.swap(run_portserver, 'should_allocate_port', lambda _: True)
        with self.swap_log, swap_get_port, swap_should_allocate_port:
            response = request_handler.handle_port_request(b'1010')
            request_handler.dump_stats()
        self.assertEqual(response, b'')
        self.assertIn('denied-allocations 1', self.terminal_logs)

    def test_port_server_request_handler_allocates_port_to_client(self) -> None:
        if False:
            while True:
                i = 10
        request_handler = run_portserver.PortServerRequestHandler((8181,))
        swap_get_port = self.swap(run_portserver.PortPool, 'get_port_for_process', lambda *unused_args: 8080)
        swap_should_allocate_port = self.swap(run_portserver, 'should_allocate_port', lambda _: True)
        with self.swap_log, swap_get_port, swap_should_allocate_port:
            response = request_handler.handle_port_request(b'1010')
            request_handler.dump_stats()
        self.assertEqual(response, b'8080\n')
        self.assertIn('total-allocations 1', self.terminal_logs)

    def test_failure_to_start_server_throws_error(self) -> None:
        if False:
            print('Hello World!')

        class FailingMockSocket(MockSocket):
            """Socket that fails while invoking bind()."""

            def bind(self, *unused_args: str) -> None:
                if False:
                    return 10
                raise socket.error('Some error occurred.')

        def dummy_handler(data: bytes) -> bytes:
            if False:
                for i in range(10):
                    print('nop')
            return data
        swap_socket = self.swap(socket, 'socket', lambda *unused_args: FailingMockSocket())
        error_msg = 'Failed to bind socket {}. Error: {}'.format(8181, socket.error('Some error occurred.'))
        with swap_socket, self.assertRaisesRegex(RuntimeError, error_msg):
            run_portserver.Server(dummy_handler, '8181')

    def test_server_closes_gracefully(self) -> None:
        if False:
            print('Hello World!')
        mock_socket = MockSocket()
        mock_socket.port = 8181

        def dummy_handler(data: bytes) -> bytes:
            if False:
                print('Hello World!')
            return data
        swap_hasattr = self.swap_with_checks(builtins, 'hasattr', lambda *unused_args: False, expected_args=((socket, 'AF_UNIX'),))
        swap_socket = self.swap(socket, 'socket', lambda *unused_args: mock_socket)
        with swap_socket, swap_hasattr:
            server = run_portserver.Server(dummy_handler, '\x008181')
            run_portserver.Server.handle_connection(MockSocket(), dummy_handler)
            self.assertFalse(server.socket.server_closed)
            server.close()
        self.assertTrue(server.socket.server_closed)

    def test_server_on_close_removes_the_socket_file(self) -> None:
        if False:
            i = 10
            return i + 15
        path = '8181'

        def dummy_handler(data: bytes) -> bytes:
            if False:
                print('Hello World!')
            return data
        swap_hasattr = self.swap_with_checks(builtins, 'hasattr', lambda *unused_args: False, expected_args=((socket, 'AF_UNIX'),))
        swap_socket = self.swap(socket, 'socket', lambda *unused_args: MockSocket())
        swap_remove = self.swap_with_checks(os, 'remove', lambda _: None, expected_args=((path,),))
        with swap_socket, swap_hasattr, swap_remove:
            server = run_portserver.Server(dummy_handler, path)
            self.assertFalse(server.socket.server_closed)
            server.close()
        self.assertTrue(server.socket.server_closed)

    def test_null_port_ranges_while_calling_script_throws_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        swap_server = self.swap(run_portserver, 'Server', lambda *unused_args: MockServer())
        swap_sys_exit = self.swap(sys, 'exit', lambda _: None)
        with self.swap_log_err, swap_sys_exit, swap_server:
            run_portserver.main(args=['--portserver_static_pool', 'abc-efgh'])
        self.assertIn('No ports. Invalid port ranges in --portserver_static_pool?', self.terminal_err_logs)

    def test_out_of_bound_port_ranges_while_calling_script_throws_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        swap_server = self.swap(run_portserver, 'Server', lambda *unused_args: MockServer())
        swap_sys_exit = self.swap(sys, 'exit', lambda _: None)
        with self.swap_log_err, swap_sys_exit, swap_server:
            run_portserver.main(args=['--portserver_static_pool', '0-8182'])
        self.assertIn('No ports. Invalid port ranges in --portserver_static_pool?', self.terminal_err_logs)

    def test_server_starts_on_calling_script_successfully(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        swap_server = self.swap(run_portserver, 'Server', lambda *unused_args: MockServer())
        swap_sys_exit = self.swap(sys, 'exit', lambda _: None)
        with self.swap_log, swap_sys_exit, swap_server:
            run_portserver.main()
        self.assertIn('Serving portserver on portserver.sock', self.terminal_logs)

    def test_server_closes_on_keyboard_interrupt(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        class InterruptedMockServer(MockServer):
            """Server that gets interrupted while invoking run()."""

            def run(self) -> None:
                if False:
                    return 10
                raise KeyboardInterrupt('^C pressed.')
        swap_server = self.swap(run_portserver, 'Server', lambda *unused_args: InterruptedMockServer())
        swap_sys_exit = self.swap(sys, 'exit', lambda _: None)
        with self.swap_log, swap_sys_exit, swap_server:
            run_portserver.main(['--portserver_unix_socket_address', '8181'])
        self.assertIn('Stopping portserver due to ^C.', self.terminal_logs)
        self.assertIn('Shutting down portserver.', self.terminal_logs)