"""
Tests of nbio_interface.AbstractIOServices adaptations

"""
import collections
import errno
import logging
import os
import platform
import socket
import time
import unittest
import pika.compat
from pika.adapters.utils import nbio_interface
from pika.compat import ON_WINDOWS
from tests.misc.forward_server import ForwardServer
from tests.stubs.io_services_test_stubs import IOServicesTestStubs

class AsyncServicesTestBase(unittest.TestCase):

    @property
    def logger(self):
        if False:
            print('Hello World!')
        'Return the logger for tests to use\n\n        '
        return logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

    def create_nonblocking_tcp_socket(self):
        if False:
            print('Hello World!')
        'Create a TCP stream socket and schedule cleanup to close it\n\n        '
        sock = socket.socket()
        sock.setblocking(False)
        self.addCleanup(sock.close)
        return sock

    def create_nonblocking_socketpair(self):
        if False:
            while True:
                i = 10
        'Creates a non-blocking socket pair and schedules cleanup to close\n        them\n\n        :returns: two-tuple of connected non-blocking sockets\n\n        '
        pair = pika.compat._nonblocking_socketpair()
        self.addCleanup(pair[0].close)
        self.addCleanup(pair[1].close)
        return pair

    def create_blocking_socketpair(self):
        if False:
            while True:
                i = 10
        'Creates a blocking socket pair and schedules cleanup to close\n        them\n\n        :returns: two-tuple of connected non-blocking sockets\n\n        '
        pair = self.create_nonblocking_socketpair()
        pair[0].setblocking(True)
        pair[1].setblocking(True)
        return pair

    @staticmethod
    def safe_connect_nonblocking_socket(sock, addr_pair):
        if False:
            print('Hello World!')
        'Initiate socket connection, suppressing EINPROGRESS/EWOULDBLOCK\n        :param socket.socket sock\n        :param addr_pair: two tuple of address string and port integer\n        '
        try:
            sock.connect(addr_pair)
        except pika.compat.SOCKET_ERROR as error:
            if error.errno not in (errno.EINPROGRESS, errno.EWOULDBLOCK):
                raise

    def get_dead_socket_address(self):
        if False:
            print('Hello World!')
        '\n\n        :return: socket address pair (ip-addr, port) that will refuse connection\n\n        '
        (s1, s2) = pika.compat._nonblocking_socketpair()
        s2.close()
        self.addCleanup(s1.close)
        return s1.getsockname()

class TestGetNativeIOLoop(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        native_loop = self.create_nbio().get_native_ioloop()
        self.assertIsNotNone(self._native_loop)
        self.assertIs(native_loop, self._native_loop)

class TestRunWithStopFromThreadsafeCallback(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        loop = self.create_nbio()
        bucket = []

        def callback():
            if False:
                print('Hello World!')
            loop.stop()
            bucket.append('I was called')
        loop.add_callback_threadsafe(callback)
        loop.run()
        self.assertEqual(bucket, ['I was called'])

@unittest.skipIf(pika.compat.ON_WINDOWS, 'Windows timing is too precise')
class TestCallLaterDoesNotCallAheadOfTime(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        loop = self.create_nbio()
        bucket = []

        def callback():
            if False:
                while True:
                    i = 10
            loop.stop()
            bucket.append('I was here')
        start_time = pika.compat.time_now()
        loop.call_later(0.1, callback)
        loop.run()
        self.assertGreaterEqual(round(pika.compat.time_now() - start_time, 3), 0.1)
        self.assertEqual(bucket, ['I was here'])

class TestCallLaterCancelReturnsNone(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        loop = self.create_nbio()
        self.assertIsNone(loop.call_later(0, lambda : None).cancel())

class TestCallLaterCancelTwiceFromOwnCallback(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        loop = self.create_nbio()
        bucket = []

        def callback():
            if False:
                for i in range(10):
                    print('nop')
            timer.cancel()
            timer.cancel()
            loop.stop()
            bucket.append('I was here')
        timer = loop.call_later(0.1, callback)
        loop.run()
        self.assertEqual(bucket, ['I was here'])

class TestCallLaterCallInOrder(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        loop = self.create_nbio()
        bucket = []
        loop.call_later(0.3, lambda : bucket.append(3) or loop.stop())
        loop.call_later(0, lambda : bucket.append(1))
        loop.call_later(0.15, lambda : bucket.append(2))
        loop.run()
        self.assertEqual(bucket, [1, 2, 3])

class TestCallLaterCancelledDoesNotCallBack(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        loop = self.create_nbio()
        bucket = []
        timer1 = loop.call_later(0, lambda : bucket.append(1))
        timer1.cancel()
        loop.call_later(0.15, lambda : bucket.append(2) or loop.stop())
        loop.run()
        self.assertEqual(bucket, [2])

class SocketWatcherTestBase(AsyncServicesTestBase):
    WatcherActivity = collections.namedtuple('io_services_test_WatcherActivity', ['readable', 'writable'])

    def _check_socket_watchers_fired(self, sock, expected):
        if False:
            i = 10
            return i + 15
        "Registers reader and writer for the given socket, runs the event loop\n        until either one fires and asserts against expectation.\n\n        :param AsyncServicesTestBase | IOServicesTestStubs self:\n        :param socket.socket sock:\n        :param WatcherActivity expected: What's expected by caller\n        "
        nbio = self.create_nbio()
        stops_requested = []

        def stop_loop():
            if False:
                i = 10
                return i + 15
            if not stops_requested:
                nbio.stop()
            stops_requested.append(1)
        reader_bucket = [False]

        def on_readable():
            if False:
                print('Hello World!')
            self.logger.debug('on_readable() called.')
            reader_bucket.append(True)
            stop_loop()
        writer_bucket = [False]

        def on_writable():
            if False:
                for i in range(10):
                    print('nop')
            self.logger.debug('on_writable() called.')
            writer_bucket.append(True)
            stop_loop()
        timeout_bucket = []

        def on_timeout():
            if False:
                print('Hello World!')
            timeout_bucket.append(True)
            stop_loop()
        timeout_timer = nbio.call_later(5, on_timeout)
        nbio.set_reader(sock.fileno(), on_readable)
        nbio.set_writer(sock.fileno(), on_writable)
        try:
            nbio.run()
        finally:
            timeout_timer.cancel()
            nbio.remove_reader(sock.fileno())
            nbio.remove_writer(sock.fileno())
        if timeout_bucket:
            raise AssertionError('which_socket_watchers_fired() timed out.')
        readable = reader_bucket[-1]
        writable = writer_bucket[-1]
        if readable != expected.readable:
            raise AssertionError('Expected readable={!r}, but got {!r} (writable={!r})'.format(expected.readable, readable, writable))
        if writable != expected.writable:
            raise AssertionError('Expected writable={!r}, but got {!r} (readable={!r})'.format(expected.writable, writable, readable))

class TestSocketWatchersUponConnectionAndNoIncomingData(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        (s1, _s2) = self.create_blocking_socketpair()
        expected = self.WatcherActivity(readable=False, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersUponConnectionAndIncomingData(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        (s1, s2) = self.create_blocking_socketpair()
        s2.send(b'abc')
        expected = self.WatcherActivity(readable=True, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersWhenFailsToConnect(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        sock = self.create_nonblocking_tcp_socket()
        self.safe_connect_nonblocking_socket(sock, self.get_dead_socket_address())
        expected = self.WatcherActivity(readable=False if ON_WINDOWS else True, writable=True)
        self._check_socket_watchers_fired(sock, expected)

class TestSocketWatchersAfterRemotePeerCloses(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        (s1, s2) = self.create_blocking_socketpair()
        s2.close()
        expected = self.WatcherActivity(readable=True, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersAfterRemotePeerClosesWithIncomingData(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        (s1, s2) = self.create_blocking_socketpair()
        s2.send(b'abc')
        s2.close()
        expected = self.WatcherActivity(readable=True, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersAfterRemotePeerShutsRead(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            return 10
        (s1, s2) = self.create_blocking_socketpair()
        s2.shutdown(socket.SHUT_RD)
        expected = self.WatcherActivity(readable=False, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersAfterRemotePeerShutsWrite(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        (s1, s2) = self.create_blocking_socketpair()
        s2.shutdown(socket.SHUT_WR)
        expected = self.WatcherActivity(readable=True, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersAfterRemotePeerShutsWriteWithIncomingData(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        (s1, s2) = self.create_blocking_socketpair()
        s2.send(b'abc')
        s2.shutdown(socket.SHUT_WR)
        expected = self.WatcherActivity(readable=True, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersAfterRemotePeerShutsReadWrite(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        (s1, s2) = self.create_blocking_socketpair()
        s2.shutdown(socket.SHUT_RDWR)
        expected = self.WatcherActivity(readable=True, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersAfterLocalPeerShutsRead(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        (s1, _s2) = self.create_blocking_socketpair()
        s1.shutdown(socket.SHUT_RD)
        expected = self.WatcherActivity(readable=False if ON_WINDOWS else True, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersAfterLocalPeerShutsWrite(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        (s1, _s2) = self.create_blocking_socketpair()
        s1.shutdown(socket.SHUT_WR)
        expected = self.WatcherActivity(readable=False, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestSocketWatchersAfterLocalPeerShutsReadWrite(SocketWatcherTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        (s1, _s2) = self.create_blocking_socketpair()
        s1.shutdown(socket.SHUT_RDWR)
        expected = self.WatcherActivity(readable=False if ON_WINDOWS else True, writable=True)
        self._check_socket_watchers_fired(s1, expected)

class TestGetaddrinfoWWWGoogleDotComPort80(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        nbio = self.create_nbio()
        result_bucket = []

        def on_done(result):
            if False:
                return 10
            result_bucket.append(result)
            nbio.stop()
        ref = nbio.getaddrinfo('www.google.com', 80, socktype=socket.SOCK_STREAM, on_done=on_done)
        nbio.run()
        self.assertEqual(len(result_bucket), 1)
        result = result_bucket[0]
        self.logger.debug('TestGetaddrinfoWWWGoogleDotComPort80: result=%r', result)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result[0]), 5)
        for (family, socktype, proto, canonname, sockaddr) in result:
            self.assertIn(family, [socket.AF_INET, socket.AF_INET6])
            self.assertEqual(socktype, socket.SOCK_STREAM)
            if pika.compat.ON_WINDOWS:
                self.assertEqual(proto, socket.IPPROTO_IP)
            else:
                self.assertEqual(proto, socket.IPPROTO_TCP)
            self.assertEqual(canonname, '')
            (ipaddr, port) = sockaddr[:2]
            self.assertIsInstance(ipaddr, str)
            self.assertGreater(len(ipaddr), 0)
            socket.inet_pton(family, ipaddr)
            self.assertEqual(port, 80)
        self.assertEqual(ref.cancel(), False)

class TestGetaddrinfoNonExistentHost(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        nbio = self.create_nbio()
        result_bucket = []

        def on_done(result):
            if False:
                while True:
                    i = 10
            result_bucket.append(result)
            nbio.stop()
        ref = nbio.getaddrinfo('www.google.comSSS', 80, socktype=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP, on_done=on_done)
        nbio.run()
        self.assertEqual(len(result_bucket), 1)
        result = result_bucket[0]
        self.assertIsInstance(result, socket.gaierror)
        self.assertEqual(ref.cancel(), False)

class TestGetaddrinfoCancelBeforeLoopRun(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        nbio = self.create_nbio()
        on_done_bucket = []

        def on_done(result):
            if False:
                while True:
                    i = 10
            on_done_bucket.append(result)
        ref = nbio.getaddrinfo('www.google.com', 80, socktype=socket.SOCK_STREAM, on_done=on_done)
        self.assertEqual(ref.cancel(), True)
        nbio.add_callback_threadsafe(nbio.stop)
        nbio.run()
        self.assertFalse(on_done_bucket)

class TestGetaddrinfoCancelAfterLoopRun(AsyncServicesTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            return 10
        nbio = self.create_nbio()
        on_done_bucket = []

        def on_done(result):
            if False:
                i = 10
                return i + 15
            self.logger.error('Unexpected completion of cancelled getaddrinfo()')
            on_done_bucket.append(result)
        cancel_result_bucket = []

        def cancel_and_stop_from_loop():
            if False:
                i = 10
                return i + 15
            self.logger.debug('Cancelling getaddrinfo() from loop callback.')
            cancel_result_bucket.append(getaddr_ref.cancel())
            nbio.stop()
        nbio.add_callback_threadsafe(cancel_and_stop_from_loop)
        getaddr_ref = nbio.getaddrinfo('www.google.com', 80, socktype=socket.SOCK_STREAM, on_done=on_done)
        nbio.run()
        self.assertEqual(cancel_result_bucket, [True])
        self.assertFalse(on_done_bucket)

class SocketConnectorTestBase(AsyncServicesTestBase):

    def set_up_sockets_for_connect(self, family):
        if False:
            return 10
        "\n        :param IOServicesTestStubs | SocketConnectorTestBase self:\n\n        :return: two-tuple (lsock, csock), where lscok is the listening sock and\n            csock is the socket that's can be connected to the listening socket.\n        :rtype: tuple\n        "
        lsock = socket.socket(family, socket.SOCK_STREAM)
        self.addCleanup(lsock.close)
        ipaddr = pika.compat._LOCALHOST_V6 if family == socket.AF_INET6 else pika.compat._LOCALHOST
        lsock.bind((ipaddr, 0))
        lsock.listen(1)
        csock = socket.socket(family, socket.SOCK_STREAM)
        self.addCleanup(csock.close)
        csock.setblocking(False)
        return (lsock, csock)

    def check_successful_connect(self, family):
        if False:
            while True:
                i = 10
        '\n        :param IOServicesTestStubs | SocketConnectorTestBase self:\n        '
        nbio = self.create_nbio()
        (lsock, csock) = self.set_up_sockets_for_connect(family)
        on_done_result_bucket = []

        def on_done(result):
            if False:
                for i in range(10):
                    print('nop')
            on_done_result_bucket.append(result)
            nbio.stop()
        connect_ref = nbio.connect_socket(csock, lsock.getsockname(), on_done)
        nbio.run()
        self.assertEqual(on_done_result_bucket, [None])
        self.assertEqual(csock.getpeername(), lsock.getsockname())
        self.assertEqual(connect_ref.cancel(), False)

    def check_failed_connect(self, family):
        if False:
            return 10
        '\n        :param IOServicesTestStubs | SocketConnectorTestBase self:\n        '
        nbio = self.create_nbio()
        (lsock, csock) = self.set_up_sockets_for_connect(family)
        laddr = lsock.getsockname()
        lsock.close()
        on_done_result_bucket = []

        def on_done(result):
            if False:
                for i in range(10):
                    print('nop')
            on_done_result_bucket.append(result)
            nbio.stop()
        connect_ref = nbio.connect_socket(csock, laddr, on_done)
        nbio.run()
        self.assertEqual(len(on_done_result_bucket), 1)
        self.assertIsInstance(on_done_result_bucket[0], Exception)
        with self.assertRaises(Exception):
            csock.getpeername()
        self.assertEqual(connect_ref.cancel(), False)

    def check_cancel_connect(self, family):
        if False:
            i = 10
            return i + 15
        '\n        :param IOServicesTestStubs | SocketConnectorTestBase self:\n        '
        nbio = self.create_nbio()
        (lsock, csock) = self.set_up_sockets_for_connect(family)
        on_done_result_bucket = []

        def on_done(result):
            if False:
                for i in range(10):
                    print('nop')
            on_done_result_bucket.append(result)
            self.fail('Got done callacks on cancelled connection request.')
        connect_ref = nbio.connect_socket(csock, lsock.getsockname(), on_done)
        self.assertEqual(connect_ref.cancel(), True)
        nbio.add_callback_threadsafe(nbio.stop)
        nbio.run()
        self.assertFalse(on_done_result_bucket)
        with self.assertRaises(Exception):
            csock.getpeername()
        self.assertEqual(connect_ref.cancel(), False)

class TestConnectSocketIPv4Success(SocketConnectorTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        self.check_successful_connect(family=socket.AF_INET)

class TestConnectSocketIPv4Fail(SocketConnectorTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        self.check_failed_connect(socket.AF_INET)

class TestConnectSocketToDisconnectedPeer(SocketConnectorTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        "Differs from `TestConnectSocketIPV4Fail` in that this test attempts\n        to connect to the address of a socket whose peer had disconnected from\n        it. `TestConnectSocketIPv4Fail` attempts to connect to a closed socket\n        that was previously listening. We want to see what happens in this case\n        because we're seeing strange behavior in TestConnectSocketIPv4Fail when\n        testing with Twisted on Linux, such that the reactor calls the\n        descriptors's `connectionLost()` method, but not its `write()` method.\n        "
        nbio = self.create_nbio()
        csock = self.create_nonblocking_tcp_socket()
        badaddr = self.get_dead_socket_address()
        on_done_result_bucket = []

        def on_done(result):
            if False:
                print('Hello World!')
            on_done_result_bucket.append(result)
            nbio.stop()
        connect_ref = nbio.connect_socket(csock, badaddr, on_done)
        nbio.run()
        self.assertEqual(len(on_done_result_bucket), 1)
        self.assertIsInstance(on_done_result_bucket[0], Exception)
        with self.assertRaises(Exception):
            csock.getpeername()
        self.assertEqual(connect_ref.cancel(), False)

class TestConnectSocketIPv4Cancel(SocketConnectorTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        self.check_cancel_connect(socket.AF_INET)

class TestConnectSocketIPv6Success(SocketConnectorTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        self.check_successful_connect(family=socket.AF_INET6)

class TestConnectSocketIPv6Fail(SocketConnectorTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        self.check_failed_connect(socket.AF_INET6)

class StreamingTestBase(AsyncServicesTestBase):
    pass

class TestStreamConnectorTxRx(StreamingTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        nbio = self.create_nbio()
        original_data = tuple((os.urandom(1000) for _ in pika.compat.xrange(1000)))
        original_data_length = sum((len(s) for s in original_data))
        my_protocol_bucket = []
        logger = self.logger

        class TestStreamConnectorTxRxStreamProtocol(nbio_interface.AbstractStreamProtocol):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.transport = None
                self.connection_lost_error_bucket = []
                self.eof_rx = False
                self.all_rx_data = b''
                my_protocol_bucket.append(self)

            def connection_made(self, transport):
                if False:
                    i = 10
                    return i + 15
                logger.info('connection_made(%r)', transport)
                self.transport = transport
                for chunk in original_data:
                    self.transport.write(chunk)

            def connection_lost(self, error):
                if False:
                    print('Hello World!')
                logger.info('connection_lost(%r)', error)
                self.connection_lost_error_bucket.append(error)
                nbio.stop()

            def eof_received(self):
                if False:
                    for i in range(10):
                        print('nop')
                logger.info('eof_received()')
                self.eof_rx = True
                return False

            def data_received(self, data):
                if False:
                    i = 10
                    return i + 15
                self.all_rx_data += data
                if self.transport.get_write_buffer_size() == 0 and len(self.all_rx_data) >= original_data_length:
                    self.transport.abort()
        streaming_connection_result_bucket = []
        socket_connect_done_result_bucket = []
        with ForwardServer(remote_addr=None) as echo:
            sock = self.create_nonblocking_tcp_socket()
            logger.info('created sock=%s', sock)

            def on_streaming_creation_done(result):
                if False:
                    i = 10
                    return i + 15
                logger.info('on_streaming_creation_done(%r)', result)
                streaming_connection_result_bucket.append(result)

            def on_socket_connect_done(result):
                if False:
                    for i in range(10):
                        print('nop')
                logger.info('on_socket_connect_done(%r)', result)
                socket_connect_done_result_bucket.append(result)
                nbio.create_streaming_connection(TestStreamConnectorTxRxStreamProtocol, sock, on_streaming_creation_done)
            nbio.connect_socket(sock, echo.server_address, on_socket_connect_done)
            nbio.run()
        self.assertEqual(socket_connect_done_result_bucket, [None])
        my_proto = my_protocol_bucket[0]
        (transport, protocol) = streaming_connection_result_bucket[0]
        self.assertIsInstance(transport, nbio_interface.AbstractStreamTransport)
        self.assertIs(protocol, my_proto)
        self.assertIs(transport, my_proto.transport)
        self.assertEqual(my_proto.connection_lost_error_bucket, [None])
        self.assertFalse(my_proto.eof_rx)
        self.assertEqual(len(my_proto.all_rx_data), original_data_length)
        self.assertEqual(my_proto.all_rx_data, b''.join(original_data))

class TestStreamConnectorRaisesValueErrorFromUnconnectedSocket(StreamingTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            return 10
        nbio = self.create_nbio()
        with self.assertRaises(ValueError) as exc_ctx:
            nbio.create_streaming_connection(lambda : None, self.create_nonblocking_tcp_socket(), lambda result: None)
        self.assertIn('getpeername() failed', exc_ctx.exception.args[0])

class TestStreamConnectorBrokenPipe(StreamingTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            return 10
        nbio = self.create_nbio()
        my_protocol_bucket = []
        logger = self.logger
        streaming_connection_result_bucket = []
        socket_connect_done_result_bucket = []
        echo = ForwardServer(remote_addr=None)
        echo.start()
        self.addCleanup(lambda : echo.stop() if echo.running else None)

        class TestStreamConnectorTxRxStreamProtocol(nbio_interface.AbstractStreamProtocol):

            def __init__(self):
                if False:
                    return 10
                self.transport = None
                self.connection_lost_error_bucket = []
                self.eof_rx = False
                self.all_rx_data = b''
                my_protocol_bucket.append(self)
                self._timer_ref = None

            def connection_made(self, transport):
                if False:
                    print('Hello World!')
                logger.info('connection_made(%r)', transport)
                self.transport = transport
                echo.stop()
                self._on_write_timer()

            def connection_lost(self, error):
                if False:
                    for i in range(10):
                        print('nop')
                logger.info('connection_lost(%r)', error)
                self.connection_lost_error_bucket.append(error)
                self._timer_ref.cancel()
                nbio.stop()

            def eof_received(self):
                if False:
                    i = 10
                    return i + 15
                logger.info('eof_received()')
                self.eof_rx = True
                self.transport.write(b'eof_received')
                return True

            def data_received(self, data):
                if False:
                    for i in range(10):
                        print('nop')
                logger.info('data_received: len=%s', len(data))
                self.all_rx_data += data

            def _on_write_timer(self):
                if False:
                    i = 10
                    return i + 15
                self.transport.write(b'_on_write_timer')
                self._timer_ref = nbio.call_later(0.01, self._on_write_timer)
        sock = self.create_nonblocking_tcp_socket()
        logger.info('created sock=%s', sock)

        def on_streaming_creation_done(result):
            if False:
                print('Hello World!')
            logger.info('on_streaming_creation_done(%r)', result)
            streaming_connection_result_bucket.append(result)

        def on_socket_connect_done(result):
            if False:
                while True:
                    i = 10
            logger.info('on_socket_connect_done(%r)', result)
            socket_connect_done_result_bucket.append(result)
            nbio.create_streaming_connection(TestStreamConnectorTxRxStreamProtocol, sock, on_streaming_creation_done)
        nbio.connect_socket(sock, echo.server_address, on_socket_connect_done)
        nbio.run()
        self.assertEqual(socket_connect_done_result_bucket, [None])
        my_proto = my_protocol_bucket[0]
        error = my_proto.connection_lost_error_bucket[0]
        self.assertIsInstance(error, pika.compat.SOCKET_ERROR)
        self.assertIn(error.errno, [errno.EPIPE, errno.ECONNRESET, errno.EPROTOTYPE])

class TestStreamConnectorEOFReceived(StreamingTestBase, IOServicesTestStubs):

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        nbio = self.create_nbio()
        original_data = [b'A' * 1000]
        my_protocol_bucket = []
        logger = self.logger
        streaming_connection_result_bucket = []

        class TestStreamConnectorTxRxStreamProtocol(nbio_interface.AbstractStreamProtocol):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.transport = None
                self.connection_lost_error_bucket = []
                self.eof_rx = False
                self.all_rx_data = b''
                my_protocol_bucket.append(self)

            def connection_made(self, transport):
                if False:
                    while True:
                        i = 10
                logger.info('connection_made(%r)', transport)
                self.transport = transport
                for chunk in original_data:
                    self.transport.write(chunk)

            def connection_lost(self, error):
                if False:
                    while True:
                        i = 10
                logger.info('connection_lost(%r)', error)
                self.connection_lost_error_bucket.append(error)
                nbio.stop()

            def eof_received(self):
                if False:
                    return 10
                logger.info('eof_received()')
                self.eof_rx = True
                return False

            def data_received(self, data):
                if False:
                    for i in range(10):
                        print('nop')
                self.all_rx_data += data
        (local_sock, remote_sock) = self.create_nonblocking_socketpair()
        remote_sock.settimeout(10)
        logger.info('created local_sock=%s, remote_sock=%s', local_sock, remote_sock)

        def on_streaming_creation_done(result):
            if False:
                i = 10
                return i + 15
            logger.info('on_streaming_creation_done(%r)', result)
            streaming_connection_result_bucket.append(result)
            remote_sock.shutdown(socket.SHUT_WR)
        nbio.create_streaming_connection(TestStreamConnectorTxRxStreamProtocol, local_sock, on_streaming_creation_done)
        nbio.run()
        my_proto = my_protocol_bucket[0]
        self.assertTrue(my_proto.eof_rx)
        self.assertEqual(my_proto.connection_lost_error_bucket, [None])
        remote_sock.recv(sum((len(chunk) for chunk in original_data)))
        self.assertEqual(remote_sock.recv(1), b'')

class TestStreamConnectorProtocolInterfaceFailsBase(StreamingTestBase):
    """Base test class for streaming protocol method fails"""

    def linkup_streaming_connection(self, nbio, sock, on_create_done, proto_constructor_exc=None, proto_connection_made_exc=None, proto_eof_received_exc=None, proto_data_received_exc=None):
        if False:
            return 10
        'Links up transport and protocol. On protocol.connection_lost(),\n        requests stop of ioloop.\n\n        :param nbio_interface.AbstractIOServices nbio:\n        :param socket.socket sock: connected socket\n        :param on_create_done: `create_streaming_connection()` completion\n            function.\n        :param proto_constructor_exc: None or exception to raise in constructor\n        :param proto_connection_made_exc: None or exception to raise in\n            `connection_made()`\n        :param proto_eof_received_exc:  None or exception to raise in\n            `eof_received()`\n        :param proto_data_received_exc:  None or exception to raise in\n            `data_received()`\n        :return: return value of `create_streaming_connection()`\n        :rtype: nbio_interface.AbstractIOReference\n\n        '
        logger = self.logger

        class TestStreamConnectorProtocol(nbio_interface.AbstractStreamProtocol):

            def __init__(self):
                if False:
                    print('Hello World!')
                self.transport = None
                self.connection_lost_error_bucket = []
                self.eof_rx = False
                self.all_rx_data = b''
                if proto_constructor_exc is not None:
                    logger.info('Raising proto_constructor_exc=%r', proto_constructor_exc)
                    raise proto_constructor_exc

            def connection_made(self, transport):
                if False:
                    return 10
                logger.info('connection_made(%r)', transport)
                self.transport = transport
                if proto_connection_made_exc is not None:
                    logger.info('Raising proto_connection_made_exc=%r', proto_connection_made_exc)
                    raise proto_connection_made_exc

            def connection_lost(self, error):
                if False:
                    while True:
                        i = 10
                logger.info('connection_lost(%r), stopping ioloop', error)
                self.connection_lost_error_bucket.append(error)
                nbio.stop()

            def eof_received(self):
                if False:
                    return 10
                logger.info('eof_received()')
                self.eof_rx = True
                if proto_eof_received_exc is not None:
                    logger.info('Raising proto_eof_received_exc=%r', proto_eof_received_exc)
                    raise proto_eof_received_exc
                return False

            def data_received(self, data):
                if False:
                    for i in range(10):
                        print('nop')
                logger.info('data_received: len=%s', len(data))
                self.all_rx_data += data
                if proto_data_received_exc is not None:
                    logger.info('Raising proto_data_received_exc=%r', proto_data_received_exc)
                    raise proto_data_received_exc
        return nbio.create_streaming_connection(TestStreamConnectorProtocol, sock, on_create_done)

class TestStreamConnectorProtocolConstructorFails(TestStreamConnectorProtocolInterfaceFailsBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        nbio = self.create_nbio()

        class ProtocolConstructorError(Exception):
            pass
        result_bucket = []

        def on_completed(result):
            if False:
                i = 10
                return i + 15
            result_bucket.append(result)
            nbio.stop()
        (local_sock, remote_sock) = self.create_nonblocking_socketpair()
        remote_sock.settimeout(10)
        self.linkup_streaming_connection(nbio, local_sock, on_completed, proto_constructor_exc=ProtocolConstructorError)
        nbio.run()
        self.assertIsInstance(result_bucket[0], ProtocolConstructorError)
        self.assertEqual(remote_sock.recv(1), b'')

class TestStreamConnectorConnectionMadeFails(TestStreamConnectorProtocolInterfaceFailsBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        nbio = self.create_nbio()

        class ConnectionMadeError(Exception):
            pass
        result_bucket = []

        def on_completed(result):
            if False:
                print('Hello World!')
            result_bucket.append(result)
            nbio.stop()
        (local_sock, remote_sock) = self.create_nonblocking_socketpair()
        remote_sock.settimeout(10)
        self.linkup_streaming_connection(nbio, local_sock, on_completed, proto_connection_made_exc=ConnectionMadeError)
        nbio.run()
        self.assertIsInstance(result_bucket[0], ConnectionMadeError)
        self.assertEqual(remote_sock.recv(1), b'')

class TestStreamConnectorEOFReceivedFails(TestStreamConnectorProtocolInterfaceFailsBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        nbio = self.create_nbio()

        class EOFReceivedError(Exception):
            pass
        (local_sock, remote_sock) = self.create_nonblocking_socketpair()
        remote_sock.settimeout(10)
        linkup_result_bucket = []

        def on_linkup_completed(result):
            if False:
                return 10
            linkup_result_bucket.append(result)
            remote_sock.shutdown(socket.SHUT_WR)
        self.linkup_streaming_connection(nbio, local_sock, on_linkup_completed, proto_eof_received_exc=EOFReceivedError)
        nbio.run()
        (_transport, proto) = linkup_result_bucket[0]
        self.assertTrue(proto.eof_rx)
        self.assertIsInstance(proto.connection_lost_error_bucket[0], EOFReceivedError)
        self.assertEqual(remote_sock.recv(1), b'')

class TestStreamConnectorDataReceivedFails(TestStreamConnectorProtocolInterfaceFailsBase, IOServicesTestStubs):

    def start(self):
        if False:
            while True:
                i = 10
        nbio = self.create_nbio()

        class DataReceivedError(Exception):
            pass
        (local_sock, remote_sock) = self.create_nonblocking_socketpair()
        remote_sock.settimeout(10)
        linkup_result_bucket = []

        def on_linkup_completed(result):
            if False:
                return 10
            linkup_result_bucket.append(result)
            remote_sock.shutdown(socket.SHUT_WR)
        self.linkup_streaming_connection(nbio, local_sock, on_linkup_completed, proto_data_received_exc=DataReceivedError)
        remote_sock.send(b'abc')
        nbio.run()
        (_transport, proto) = linkup_result_bucket[0]
        self.assertFalse(proto.eof_rx)
        self.assertIsInstance(proto.connection_lost_error_bucket[0], DataReceivedError)
        self.assertEqual(remote_sock.recv(1), b'')

class TestStreamConnectorAbortTransport(TestStreamConnectorProtocolInterfaceFailsBase, IOServicesTestStubs):

    def start(self):
        if False:
            i = 10
            return i + 15
        nbio = self.create_nbio()
        (local_sock, remote_sock) = self.create_nonblocking_socketpair()
        remote_sock.settimeout(10)
        linkup_result_bucket = []

        def on_linkup_completed(result):
            if False:
                for i in range(10):
                    print('nop')
            linkup_result_bucket.append(result)
            result[0].abort()
        self.linkup_streaming_connection(nbio, local_sock, on_linkup_completed)
        nbio.run()
        (_transport, proto) = linkup_result_bucket[0]
        self.assertFalse(proto.eof_rx)
        self.assertIsNone(proto.connection_lost_error_bucket[0])
        self.assertEqual(remote_sock.recv(1), b'')

class TestStreamConnectorCancelLinkup(TestStreamConnectorProtocolInterfaceFailsBase, IOServicesTestStubs):

    def start(self):
        if False:
            print('Hello World!')
        nbio = self.create_nbio()
        (local_sock, remote_sock) = self.create_nonblocking_socketpair()
        remote_sock.settimeout(10)
        linkup_result_bucket = []

        def on_linkup_completed(result):
            if False:
                print('Hello World!')
            linkup_result_bucket.append(result)
        ref = self.linkup_streaming_connection(nbio, local_sock, on_linkup_completed)
        ref.cancel()
        nbio.add_callback_threadsafe(nbio.stop)
        nbio.run()
        self.assertEqual(linkup_result_bucket, [])
        self.assertEqual(remote_sock.recv(1), b'')