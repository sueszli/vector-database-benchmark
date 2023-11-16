from test import support
from test.support import socket_helper
from test.support import threading_helper
import errno
import socket
import sys
import threading
import time
import unittest
import unittest.mock
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import asynchat
    import asyncore
HOST = socket_helper.HOST
SERVER_QUIT = b'QUIT\n'

class echo_server(threading.Thread):
    chunk_size = 1

    def __init__(self, event):
        if False:
            return 10
        threading.Thread.__init__(self)
        self.event = event
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = socket_helper.bind_port(self.sock)
        self.start_resend_event = None

    def run(self):
        if False:
            print('Hello World!')
        self.sock.listen()
        self.event.set()
        (conn, client) = self.sock.accept()
        self.buffer = b''
        while SERVER_QUIT not in self.buffer:
            data = conn.recv(1)
            if not data:
                break
            self.buffer = self.buffer + data
        self.buffer = self.buffer.replace(SERVER_QUIT, b'')
        if self.start_resend_event:
            self.start_resend_event.wait()
        try:
            while self.buffer:
                n = conn.send(self.buffer[:self.chunk_size])
                time.sleep(0.001)
                self.buffer = self.buffer[n:]
        except:
            pass
        conn.close()
        self.sock.close()

class echo_client(asynchat.async_chat):

    def __init__(self, terminator, server_port):
        if False:
            while True:
                i = 10
        asynchat.async_chat.__init__(self)
        self.contents = []
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect((HOST, server_port))
        self.set_terminator(terminator)
        self.buffer = b''

    def handle_connect(self):
        if False:
            return 10
        pass
    if sys.platform == 'darwin':

        def handle_expt(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def collect_incoming_data(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.buffer += data

    def found_terminator(self):
        if False:
            for i in range(10):
                print('nop')
        self.contents.append(self.buffer)
        self.buffer = b''

def start_echo_server():
    if False:
        for i in range(10):
            print('nop')
    event = threading.Event()
    s = echo_server(event)
    s.start()
    event.wait()
    event.clear()
    time.sleep(0.01)
    return (s, event)

class TestAsynchat(unittest.TestCase):
    usepoll = False

    def setUp(self):
        if False:
            return 10
        self._threads = threading_helper.threading_setup()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        threading_helper.threading_cleanup(*self._threads)

    def line_terminator_check(self, term, server_chunk):
        if False:
            print('Hello World!')
        event = threading.Event()
        s = echo_server(event)
        s.chunk_size = server_chunk
        s.start()
        event.wait()
        event.clear()
        time.sleep(0.01)
        c = echo_client(term, s.port)
        c.push(b'hello ')
        c.push(b'world' + term)
        c.push(b"I'm not dead yet!" + term)
        c.push(SERVER_QUIT)
        asyncore.loop(use_poll=self.usepoll, count=300, timeout=0.01)
        threading_helper.join_thread(s)
        self.assertEqual(c.contents, [b'hello world', b"I'm not dead yet!"])

    def test_line_terminator1(self):
        if False:
            for i in range(10):
                print('nop')
        for l in (1, 2, 3):
            self.line_terminator_check(b'\n', l)

    def test_line_terminator2(self):
        if False:
            return 10
        for l in (1, 2, 3):
            self.line_terminator_check(b'\r\n', l)

    def test_line_terminator3(self):
        if False:
            while True:
                i = 10
        for l in (1, 2, 3):
            self.line_terminator_check(b'qqq', l)

    def numeric_terminator_check(self, termlen):
        if False:
            while True:
                i = 10
        (s, event) = start_echo_server()
        c = echo_client(termlen, s.port)
        data = b"hello world, I'm not dead yet!\n"
        c.push(data)
        c.push(SERVER_QUIT)
        asyncore.loop(use_poll=self.usepoll, count=300, timeout=0.01)
        threading_helper.join_thread(s)
        self.assertEqual(c.contents, [data[:termlen]])

    def test_numeric_terminator1(self):
        if False:
            for i in range(10):
                print('nop')
        self.numeric_terminator_check(1)

    def test_numeric_terminator2(self):
        if False:
            while True:
                i = 10
        self.numeric_terminator_check(6)

    def test_none_terminator(self):
        if False:
            while True:
                i = 10
        (s, event) = start_echo_server()
        c = echo_client(None, s.port)
        data = b"hello world, I'm not dead yet!\n"
        c.push(data)
        c.push(SERVER_QUIT)
        asyncore.loop(use_poll=self.usepoll, count=300, timeout=0.01)
        threading_helper.join_thread(s)
        self.assertEqual(c.contents, [])
        self.assertEqual(c.buffer, data)

    def test_simple_producer(self):
        if False:
            return 10
        (s, event) = start_echo_server()
        c = echo_client(b'\n', s.port)
        data = b"hello world\nI'm not dead yet!\n"
        p = asynchat.simple_producer(data + SERVER_QUIT, buffer_size=8)
        c.push_with_producer(p)
        asyncore.loop(use_poll=self.usepoll, count=300, timeout=0.01)
        threading_helper.join_thread(s)
        self.assertEqual(c.contents, [b'hello world', b"I'm not dead yet!"])

    def test_string_producer(self):
        if False:
            i = 10
            return i + 15
        (s, event) = start_echo_server()
        c = echo_client(b'\n', s.port)
        data = b"hello world\nI'm not dead yet!\n"
        c.push_with_producer(data + SERVER_QUIT)
        asyncore.loop(use_poll=self.usepoll, count=300, timeout=0.01)
        threading_helper.join_thread(s)
        self.assertEqual(c.contents, [b'hello world', b"I'm not dead yet!"])

    def test_empty_line(self):
        if False:
            for i in range(10):
                print('nop')
        (s, event) = start_echo_server()
        c = echo_client(b'\n', s.port)
        c.push(b"hello world\n\nI'm not dead yet!\n")
        c.push(SERVER_QUIT)
        asyncore.loop(use_poll=self.usepoll, count=300, timeout=0.01)
        threading_helper.join_thread(s)
        self.assertEqual(c.contents, [b'hello world', b'', b"I'm not dead yet!"])

    def test_close_when_done(self):
        if False:
            return 10
        (s, event) = start_echo_server()
        s.start_resend_event = threading.Event()
        c = echo_client(b'\n', s.port)
        c.push(b"hello world\nI'm not dead yet!\n")
        c.push(SERVER_QUIT)
        c.close_when_done()
        asyncore.loop(use_poll=self.usepoll, count=300, timeout=0.01)
        s.start_resend_event.set()
        threading_helper.join_thread(s)
        self.assertEqual(c.contents, [])
        self.assertGreater(len(s.buffer), 0)

    def test_push(self):
        if False:
            while True:
                i = 10
        (s, event) = start_echo_server()
        c = echo_client(b'\n', s.port)
        data = b'bytes\n'
        c.push(data)
        c.push(bytearray(data))
        c.push(memoryview(data))
        self.assertRaises(TypeError, c.push, 10)
        self.assertRaises(TypeError, c.push, 'unicode')
        c.push(SERVER_QUIT)
        asyncore.loop(use_poll=self.usepoll, count=300, timeout=0.01)
        threading_helper.join_thread(s)
        self.assertEqual(c.contents, [b'bytes', b'bytes', b'bytes'])

class TestAsynchat_WithPoll(TestAsynchat):
    usepoll = True

class TestAsynchatMocked(unittest.TestCase):

    def test_blockingioerror(self):
        if False:
            i = 10
            return i + 15
        sock = unittest.mock.Mock()
        sock.recv.side_effect = BlockingIOError(errno.EAGAIN)
        dispatcher = asynchat.async_chat()
        dispatcher.set_socket(sock)
        self.addCleanup(dispatcher.del_channel)
        with unittest.mock.patch.object(dispatcher, 'handle_error') as error:
            dispatcher.handle_read()
        self.assertFalse(error.called)

class TestHelperFunctions(unittest.TestCase):

    def test_find_prefix_at_end(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(asynchat.find_prefix_at_end('qwerty\r', '\r\n'), 1)
        self.assertEqual(asynchat.find_prefix_at_end('qwertydkjf', '\r\n'), 0)

class TestNotConnected(unittest.TestCase):

    def test_disallow_negative_terminator(self):
        if False:
            print('Hello World!')
        client = asynchat.async_chat()
        self.assertRaises(ValueError, client.set_terminator, -1)
if __name__ == '__main__':
    unittest.main()