"""blocking adapter test"""
from datetime import datetime
import functools
import logging
import socket
import threading
import unittest
import uuid
import pika
from pika.adapters import blocking_connection
from pika.compat import as_bytes, time_now
import pika.connection
import pika.exceptions
from pika.exchange_type import ExchangeType
from tests.misc.forward_server import ForwardServer
from tests.misc.test_utils import retry_assertion
LOGGER = logging.getLogger(__name__)
PARAMS_URL_TEMPLATE = 'amqp://guest:guest@127.0.0.1:%(port)s/%%2f?socket_timeout=1'
DEFAULT_URL = PARAMS_URL_TEMPLATE % {'port': 5672}
DEFAULT_PARAMS = pika.URLParameters(DEFAULT_URL)
DEFAULT_TIMEOUT = 15

def setUpModule():
    if False:
        i = 10
        return i + 15
    logging.basicConfig(level=logging.DEBUG)

class BlockingTestCaseBase(unittest.TestCase):
    TIMEOUT = DEFAULT_TIMEOUT

    def _connect(self, url=DEFAULT_URL, connection_class=pika.BlockingConnection, impl_class=None):
        if False:
            print('Hello World!')
        parameters = pika.URLParameters(url)
        return self._connect_params(parameters, connection_class, impl_class)

    def _connect_params(self, parameters, connection_class=pika.BlockingConnection, impl_class=None):
        if False:
            i = 10
            return i + 15
        connection = connection_class(parameters, _impl_class=impl_class)
        self.addCleanup(lambda : connection.close() if connection.is_open else None)
        connection._impl._adapter_call_later(self.TIMEOUT, self._on_test_timeout)
        self._instrument_io_loop_exception_leak_detection(connection)
        return connection

    def _instrument_io_loop_exception_leak_detection(self, connection):
        if False:
            return 10
        "Instrument the given connection to detect and fail test when\n        an exception is leaked through the I/O loop\n\n        NOTE: BlockingConnection's underlying asynchronous connection adapter\n        (SelectConnection) uses callbacks to communicate with its user (\n        BlockingConnection in this case). If BlockingConnection leaks\n        exceptions back into the I/O loop or the asynchronous connection\n        adapter, we interrupt their normal workflow and introduce a high\n        likelihood of state inconsistency.\n        "
        real_poll = connection._impl.ioloop.poll

        def my_poll(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            try:
                return real_poll(*args, **kwargs)
            except BaseException as exc:
                self.fail('Unwanted exception leaked into asynchronous layer via ioloop.poll(): {!r}'.format(exc))
        connection._impl.ioloop.poll = my_poll
        self.addCleanup(setattr, connection._impl.ioloop, 'poll', real_poll)
        real_process_timeouts = connection._impl.ioloop.process_timeouts

        def my_process_timeouts(*args, **kwargs):
            if False:
                while True:
                    i = 10
            try:
                return real_process_timeouts(*args, **kwargs)
            except AssertionError:
                raise
            except BaseException as exc:
                self.fail('Unwanted exception leaked into asynchronous layer via ioloop.process_timeouts(): {!r}'.format(exc))
        connection._impl.ioloop.process_timeouts = my_process_timeouts
        self.addCleanup(setattr, connection._impl.ioloop, 'process_timeouts', real_process_timeouts)

    def _on_test_timeout(self):
        if False:
            return 10
        'Called when test times out'
        LOGGER.info('%s TIMED OUT (%s)', datetime.utcnow(), self)
        self.fail('Test timed out')

    @retry_assertion(TIMEOUT / 2)
    def _assert_exact_message_count_with_retries(self, channel, queue, expected_count):
        if False:
            for i in range(10):
                print('nop')
        frame = channel.queue_declare(queue, passive=True)
        self.assertEqual(frame.method.message_count, expected_count)

class TestCreateAndCloseConnection(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingConnection: Create and close connection'
        connection = self._connect()
        self.assertIsInstance(connection, pika.BlockingConnection)
        self.assertTrue(connection.is_open)
        self.assertFalse(connection.is_closed)
        self.assertFalse(connection._impl.is_closing)
        connection.close()
        self.assertTrue(connection.is_closed)
        self.assertFalse(connection.is_open)
        self.assertFalse(connection._impl.is_closing)

class TestCreateConnectionWithNoneSocketAndStackTimeouts(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        ' BlockingConnection: create a connection with socket and stack timeouts both None\n\n        '
        params = pika.URLParameters(DEFAULT_URL)
        params.socket_timeout = None
        params.stack_timeout = None
        with self._connect_params(params) as connection:
            self.assertTrue(connection.is_open)

class TestCreateConnectionFromTwoConfigsFirstUnreachable(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        ' BlockingConnection: create a connection from two configs, first unreachable\n\n        '
        sock = socket.socket()
        self.addCleanup(sock.close)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()
        bad_params = pika.URLParameters(PARAMS_URL_TEMPLATE % {'port': port})
        good_params = pika.URLParameters(DEFAULT_URL)
        with self._connect_params([bad_params, good_params]) as connection:
            self.assertNotEqual(connection._impl.params.port, bad_params.port)
            self.assertEqual(connection._impl.params.port, good_params.port)

class TestCreateConnectionFromTwoUnreachableConfigs(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        ' BlockingConnection: creating a connection from two unreachable         configs raises AMQPConnectionError\n\n        '
        sock = socket.socket()
        self.addCleanup(sock.close)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()
        bad_params = pika.URLParameters(PARAMS_URL_TEMPLATE % {'port': port})
        with self.assertRaises(pika.exceptions.AMQPConnectionError):
            self._connect_params([bad_params, bad_params])

class TestMultiCloseConnectionRaisesWrongState(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection: Close connection twice raises ConnectionWrongStateError'
        connection = self._connect()
        self.assertIsInstance(connection, pika.BlockingConnection)
        self.assertTrue(connection.is_open)
        self.assertFalse(connection.is_closed)
        self.assertFalse(connection._impl.is_closing)
        connection.close()
        self.assertTrue(connection.is_closed)
        self.assertFalse(connection.is_open)
        self.assertFalse(connection._impl.is_closing)
        with self.assertRaises(pika.exceptions.ConnectionWrongStateError):
            connection.close()

class TestConnectionContextManagerClosesConnection(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection: connection context manager closes connection'
        with self._connect() as connection:
            self.assertIsInstance(connection, pika.BlockingConnection)
            self.assertTrue(connection.is_open)
        self.assertTrue(connection.is_closed)

class TestConnectionContextManagerExitSurvivesClosedConnection(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingConnection: connection context manager exit survives closed connection'
        with self._connect() as connection:
            self.assertTrue(connection.is_open)
            connection.close()
            self.assertTrue(connection.is_closed)
        self.assertTrue(connection.is_closed)

class TestConnectionContextManagerClosesConnectionAndPassesOriginalException(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection: connection context manager closes connection and passes original exception'

        class MyException(Exception):
            pass
        with self.assertRaises(MyException):
            with self._connect() as connection:
                self.assertTrue(connection.is_open)
                raise MyException()
        self.assertTrue(connection.is_closed)

class TestConnectionContextManagerClosesConnectionAndPassesSystemException(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingConnection: connection context manager closes connection and passes system exception'
        with self.assertRaises(SystemExit):
            with self._connect() as connection:
                self.assertTrue(connection.is_open)
                raise SystemExit()
        self.assertTrue(connection.is_closed)

class TestLostConnectionResultsInIsClosedConnectionAndChannel(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        connection = self._connect()
        channel = connection.channel()
        connection._impl._transport._sock.shutdown(socket.SHUT_RDWR)
        with self.assertRaises(pika.exceptions.StreamLostError):
            channel.basic_qos()
        self.assertFalse(channel.is_open)
        self.assertTrue(channel.is_closed)
        self.assertFalse(connection.is_open)
        self.assertTrue(connection.is_closed)

class TestUpdateSecret(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        connection = self._connect()
        channel = connection.channel()
        connection.update_secret('new_secret', 'reason')
        self.assertTrue(channel.is_open)
        self.assertFalse(channel.is_closed)
        self.assertTrue(connection.is_open)
        self.assertFalse(connection.is_closed)

class TestUpdateSecretOnClosedRaisesWrongState(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        connection = self._connect()
        connection.close()
        with self.assertRaises(pika.exceptions.ConnectionWrongStateError):
            connection.update_secret('new_secret', 'reason')
        self.assertFalse(connection.is_open)
        self.assertTrue(connection.is_closed)

class TestUpdateSecretExpectsStrings(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        connection = self._connect()
        with self.assertRaises(AssertionError):
            connection.update_secret(1, 'reason')
        with self.assertRaises(AssertionError):
            connection.update_secret('new_secret', 1)

class TestInvalidExchangeTypeRaisesConnectionClosed(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection: ConnectionClosed raised when creating exchange with invalid type'
        connection = self._connect()
        ch = connection.channel()
        exg_name = 'TestInvalidExchangeTypeRaisesConnectionClosed_' + uuid.uuid1().hex
        with self.assertRaises(pika.exceptions.ConnectionClosed) as ex_cm:
            ch.exchange_declare(exg_name, exchange_type='ZZwwInvalid')
        self.assertEqual(ex_cm.exception.args[0], 503)

class TestCreateAndCloseConnectionWithChannelAndConsumer(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection: Create and close connection with channel and consumer'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestCreateAndCloseConnectionWithChannelAndConsumer_q' + uuid.uuid1().hex
        body1 = 'a' * 1024
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body=body1)
        ch.basic_consume(q_name, lambda *x: None, auto_ack=True, exclusive=False, arguments=None)
        connection.close()
        self.assertTrue(connection.is_closed)
        self.assertFalse(connection.is_open)
        self.assertFalse(connection._impl.is_closing)
        self.assertFalse(connection._impl._channels)
        self.assertFalse(ch._consumer_infos)
        self.assertFalse(ch._impl._consumers)

class TestUsingInvalidQueueArgument(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection raises expected exception when invalid queue parameter is used\n        '
        connection = self._connect()
        ch = connection.channel()
        with self.assertRaises(TypeError):
            ch.queue_declare(queue=[1, 2, 3])

class TestSuddenBrokerDisconnectBeforeChannel(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingConnection resets properly on TCP/IP drop during channel()\n        '
        with ForwardServer(remote_addr=(DEFAULT_PARAMS.host, DEFAULT_PARAMS.port), local_linger_args=(1, 0)) as fwd:
            self.connection = self._connect(PARAMS_URL_TEMPLATE % {'port': fwd.server_address[1]})
        with self.assertRaises(pika.exceptions.StreamLostError):
            self.connection.channel()
        self.assertTrue(self.connection.is_closed)
        self.assertFalse(self.connection.is_open)
        self.assertIsNone(self.connection._impl._transport)

class TestNoAccessToConnectionAfterConnectionLost(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection no access file descriptor after StreamLostError\n        '
        with ForwardServer(remote_addr=(DEFAULT_PARAMS.host, DEFAULT_PARAMS.port), local_linger_args=(1, 0)) as fwd:
            self.connection = self._connect(PARAMS_URL_TEMPLATE % {'port': fwd.server_address[1]})
        with self.assertRaises(pika.exceptions.StreamLostError):
            self.connection.channel()
        self.assertTrue(self.connection.is_closed)
        self.assertFalse(self.connection.is_open)
        self.assertIsNone(self.connection._impl._transport)
        with self.assertRaises(pika.exceptions.ConnectionWrongStateError):
            self.connection.channel()

class TestConnectWithDownedBroker(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        ' BlockingConnection to downed broker results in AMQPConnectionError\n\n        '
        sock = socket.socket()
        self.addCleanup(sock.close)
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
        sock.close()
        with self.assertRaises(pika.exceptions.AMQPConnectionError):
            self.connection = self._connect(PARAMS_URL_TEMPLATE % {'port': port})

class TestDisconnectDuringConnectionStart(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        ' BlockingConnection TCP/IP connection loss in CONNECTION_START\n        '
        fwd = ForwardServer(remote_addr=(DEFAULT_PARAMS.host, DEFAULT_PARAMS.port), local_linger_args=(1, 0))
        fwd.start()
        self.addCleanup(lambda : fwd.stop() if fwd.running else None)

        class MySelectConnection(pika.SelectConnection):
            assert hasattr(pika.SelectConnection, '_on_connection_start')

            def _on_connection_start(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                fwd.stop()
                return super(MySelectConnection, self)._on_connection_start(*args, **kwargs)
        with self.assertRaises(pika.exceptions.ProbableAuthenticationError):
            self._connect(PARAMS_URL_TEMPLATE % {'port': fwd.server_address[1]}, impl_class=MySelectConnection)

class TestDisconnectDuringConnectionTune(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        ' BlockingConnection TCP/IP connection loss in CONNECTION_TUNE\n        '
        fwd = ForwardServer(remote_addr=(DEFAULT_PARAMS.host, DEFAULT_PARAMS.port), local_linger_args=(1, 0))
        fwd.start()
        self.addCleanup(lambda : fwd.stop() if fwd.running else None)

        class MySelectConnection(pika.SelectConnection):
            assert hasattr(pika.SelectConnection, '_on_connection_tune')

            def _on_connection_tune(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                fwd.stop()
                return super(MySelectConnection, self)._on_connection_tune(*args, **kwargs)
        with self.assertRaises(pika.exceptions.ProbableAccessDeniedError):
            self._connect(PARAMS_URL_TEMPLATE % {'port': fwd.server_address[1]}, impl_class=MySelectConnection)

class TestDisconnectDuringConnectionProtocol(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        ' BlockingConnection TCP/IP connection loss in CONNECTION_PROTOCOL\n        '
        fwd = ForwardServer(remote_addr=(DEFAULT_PARAMS.host, DEFAULT_PARAMS.port), local_linger_args=(1, 0))
        fwd.start()
        self.addCleanup(lambda : fwd.stop() if fwd.running else None)

        class MySelectConnection(pika.SelectConnection):
            assert hasattr(pika.SelectConnection, '_on_stream_connected')

            def _on_stream_connected(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                fwd.stop()
                return super(MySelectConnection, self)._on_stream_connected(*args, **kwargs)
        with self.assertRaises(pika.exceptions.IncompatibleProtocolError):
            self._connect(PARAMS_URL_TEMPLATE % {'port': fwd.server_address[1]}, impl_class=MySelectConnection)

class TestProcessDataEvents(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingConnection.process_data_events'
        connection = self._connect()
        start_time = time_now()
        connection.process_data_events(time_limit=0)
        elapsed = time_now() - start_time
        self.assertLess(elapsed, 0.25)
        start_time = time_now()
        connection.process_data_events(time_limit=0.005)
        elapsed = time_now() - start_time
        self.assertGreaterEqual(elapsed, 0.005)
        self.assertLess(elapsed, 0.25)

class TestConnectionRegisterForBlockAndUnblock(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingConnection register for Connection.Blocked/Unblocked'
        connection = self._connect()
        blocked_buffer = []
        connection.add_on_connection_blocked_callback(lambda conn, frame: blocked_buffer.append((conn, frame)))
        blocked_frame = pika.frame.Method(0, pika.spec.Connection.Blocked('reason'))
        connection._impl._process_frame(blocked_frame)
        connection.sleep(0)
        self.assertEqual(len(blocked_buffer), 1)
        (conn, frame) = blocked_buffer[0]
        self.assertIs(conn, connection)
        self.assertIs(frame, blocked_frame)
        unblocked_buffer = []
        connection.add_on_connection_unblocked_callback(lambda conn, frame: unblocked_buffer.append((conn, frame)))
        unblocked_frame = pika.frame.Method(0, pika.spec.Connection.Unblocked())
        connection._impl._process_frame(unblocked_frame)
        connection.sleep(0)
        self.assertEqual(len(unblocked_buffer), 1)
        (conn, frame) = unblocked_buffer[0]
        self.assertIs(conn, connection)
        self.assertIs(frame, unblocked_frame)

class TestBlockedConnectionTimeout(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection Connection.Blocked timeout '
        url = DEFAULT_URL + '&blocked_connection_timeout=0.001'
        conn = self._connect(url=url)
        conn._impl._on_connection_blocked(conn._impl, pika.frame.Method(0, pika.spec.Connection.Blocked('TestBlockedConnectionTimeout')))
        with self.assertRaises(pika.exceptions.ConnectionBlockedTimeout):
            while True:
                conn.process_data_events(time_limit=1)

class TestAddCallbackThreadsafeFromSameThread(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingConnection.add_callback_threadsafe from same thread'
        connection = self._connect()
        start_time = time_now()
        rx_callback = []
        connection.add_callback_threadsafe(lambda : rx_callback.append(time_now()))
        while not rx_callback:
            connection.process_data_events(time_limit=None)
        self.assertEqual(len(rx_callback), 1)
        elapsed = time_now() - start_time
        self.assertLess(elapsed, 0.25)

class TestAddCallbackThreadsafeFromAnotherThread(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingConnection.add_callback_threadsafe from another thread'
        connection = self._connect()
        start_time = time_now()
        rx_callback = []
        timer = threading.Timer(0, functools.partial(connection.add_callback_threadsafe, lambda : rx_callback.append(time_now())))
        self.addCleanup(timer.cancel)
        timer.start()
        while not rx_callback:
            connection.process_data_events(time_limit=None)
        self.assertEqual(len(rx_callback), 1)
        elapsed = time_now() - start_time
        self.assertLess(elapsed, 0.25)

class TestAddCallbackThreadsafeOnClosedConnectionRaisesWrongState(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingConnection.add_callback_threadsafe on closed connection raises ConnectionWrongStateError'
        connection = self._connect()
        connection.close()
        with self.assertRaises(pika.exceptions.ConnectionWrongStateError):
            connection.add_callback_threadsafe(lambda : None)

class TestAddTimeoutRemoveTimeout(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection.call_later and remove_timeout'
        connection = self._connect()
        start_time = time_now()
        rx_callback = []
        timer_id = connection.call_later(0.005, lambda : rx_callback.append(time_now()))
        while not rx_callback:
            connection.process_data_events(time_limit=None)
        self.assertEqual(len(rx_callback), 1)
        elapsed = time_now() - start_time
        self.assertLess(elapsed, 0.25)
        connection.remove_timeout(timer_id)
        rx_callback = []
        timer_id = connection.call_later(0.001, lambda : rx_callback.append(time_now()))
        connection.remove_timeout(timer_id)
        connection.process_data_events(time_limit=0.1)
        self.assertFalse(rx_callback)
        evt = blocking_connection._TimerEvt(lambda : None)
        repr(evt)

class TestViabilityOfMultipleTimeoutsWithSameDeadlineAndCallback(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingConnection viability of multiple timeouts with same deadline and callback'
        connection = self._connect()
        rx_callback = []

        def callback():
            if False:
                return 10
            rx_callback.append(1)
        timer1 = connection.call_later(0, callback)
        timer2 = connection.call_later(0, callback)
        self.assertIsNot(timer1, timer2)
        connection.remove_timeout(timer1)
        start_wait_time = time_now()
        while not rx_callback and time_now() - start_wait_time < 0.25:
            connection.process_data_events(time_limit=0.001)
        self.assertListEqual(rx_callback, [1])

class TestRemoveTimeoutFromTimeoutCallback(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingConnection.remove_timeout from timeout callback'
        connection = self._connect()
        timer_id1 = connection.call_later(5, lambda : 0 / 0)
        rx_timer2 = []

        def on_timer2():
            if False:
                i = 10
                return i + 15
            connection.remove_timeout(timer_id1)
            connection.remove_timeout(timer_id2)
            rx_timer2.append(1)
        timer_id2 = connection.call_later(0, on_timer2)
        while not rx_timer2:
            connection.process_data_events(time_limit=None)
        self.assertFalse(connection._ready_events)

class TestSleep(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingConnection.sleep'
        connection = self._connect()
        start_time = time_now()
        connection.sleep(duration=0)
        elapsed = time_now() - start_time
        self.assertLess(elapsed, 0.25)
        start_time = time_now()
        connection.sleep(duration=0.005)
        elapsed = time_now() - start_time
        self.assertGreaterEqual(elapsed, 0.005)
        self.assertLess(elapsed, 0.25)

class TestConnectionProperties(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'Test BlockingConnection properties'
        connection = self._connect()
        self.assertTrue(connection.is_open)
        self.assertFalse(connection._impl.is_closing)
        self.assertFalse(connection.is_closed)
        self.assertTrue(connection.basic_nack_supported)
        self.assertTrue(connection.consumer_cancel_notify_supported)
        self.assertTrue(connection.exchange_exchange_bindings_supported)
        self.assertTrue(connection.publisher_confirms_supported)
        connection.close()
        self.assertFalse(connection.is_open)
        self.assertFalse(connection._impl.is_closing)
        self.assertTrue(connection.is_closed)

class TestCreateAndCloseChannel(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel: Create and close channel'
        connection = self._connect()
        ch = connection.channel()
        self.assertIsInstance(ch, blocking_connection.BlockingChannel)
        self.assertTrue(ch.is_open)
        self.assertFalse(ch.is_closed)
        self.assertFalse(ch._impl.is_closing)
        self.assertIs(ch.connection, connection)
        ch.close()
        self.assertTrue(ch.is_closed)
        self.assertFalse(ch.is_open)
        self.assertFalse(ch._impl.is_closing)

class TestExchangeDeclareAndDelete(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel: Test exchange_declare and exchange_delete'
        connection = self._connect()
        ch = connection.channel()
        name = 'TestExchangeDeclareAndDelete_' + uuid.uuid1().hex
        frame = ch.exchange_declare(name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, name)
        self.assertIsInstance(frame.method, pika.spec.Exchange.DeclareOk)
        frame = ch.exchange_declare(name, passive=True)
        self.assertIsInstance(frame.method, pika.spec.Exchange.DeclareOk)
        frame = ch.exchange_delete(name)
        self.assertIsInstance(frame.method, pika.spec.Exchange.DeleteOk)
        with self.assertRaises(pika.exceptions.ChannelClosedByBroker) as cm:
            ch.exchange_declare(name, passive=True)
        self.assertEqual(cm.exception.args[0], 404)

class TestExchangeBindAndUnbind(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingChannel: Test exchange_bind and exchange_unbind'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestExchangeBindAndUnbind_q' + uuid.uuid1().hex
        src_exg_name = 'TestExchangeBindAndUnbind_src_exg_' + uuid.uuid1().hex
        dest_exg_name = 'TestExchangeBindAndUnbind_dest_exg_' + uuid.uuid1().hex
        routing_key = 'TestExchangeBindAndUnbind'
        res = ch.confirm_delivery()
        self.assertIsNone(res)
        ch.exchange_declare(src_exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, src_exg_name)
        ch.exchange_declare(dest_exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, dest_exg_name)
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.queue_bind(q_name, exchange=dest_exg_name, routing_key=routing_key)
        with self.assertRaises(pika.exceptions.UnroutableError):
            ch.basic_publish(src_exg_name, routing_key, body='', mandatory=True)
        frame = ch.exchange_bind(destination=dest_exg_name, source=src_exg_name, routing_key=routing_key)
        self.assertIsInstance(frame.method, pika.spec.Exchange.BindOk)
        ch.basic_publish(src_exg_name, routing_key, body='TestExchangeBindAndUnbind', mandatory=True)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=1)
        frame = ch.exchange_unbind(destination=dest_exg_name, source=src_exg_name, routing_key=routing_key)
        self.assertIsInstance(frame.method, pika.spec.Exchange.UnbindOk)
        with self.assertRaises(pika.exceptions.UnroutableError):
            ch.basic_publish(src_exg_name, routing_key, body='', mandatory=True)

class TestQueueDeclareAndDelete(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel: Test queue_declare and queue_delete'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestQueueDeclareAndDelete_' + uuid.uuid1().hex
        frame = ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        self.assertIsInstance(frame.method, pika.spec.Queue.DeclareOk)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertIsInstance(frame.method, pika.spec.Queue.DeclareOk)
        frame = ch.queue_delete(q_name)
        self.assertIsInstance(frame.method, pika.spec.Queue.DeleteOk)
        with self.assertRaises(pika.exceptions.ChannelClosedByBroker) as cm:
            ch.queue_declare(q_name, passive=True)
        self.assertEqual(cm.exception.args[0], 404)

class TestPassiveQueueDeclareOfUnknownQueueRaisesChannelClosed(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel: ChannelClosed raised when passive-declaring unknown queue'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestPassiveQueueDeclareOfUnknownQueueRaisesChannelClosed_q_' + uuid.uuid1().hex
        with self.assertRaises(pika.exceptions.ChannelClosedByBroker) as ex_cm:
            ch.queue_declare(q_name, passive=True)
        self.assertEqual(ex_cm.exception.args[0], 404)

class TestQueueBindAndUnbindAndPurge(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingChannel: Test queue_bind and queue_unbind'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestQueueBindAndUnbindAndPurge_q' + uuid.uuid1().hex
        exg_name = 'TestQueueBindAndUnbindAndPurge_exg_' + uuid.uuid1().hex
        routing_key = 'TestQueueBindAndUnbindAndPurge'
        res = ch.confirm_delivery()
        self.assertIsNone(res)
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        frame = ch.queue_bind(q_name, exchange=exg_name, routing_key=routing_key)
        self.assertIsInstance(frame.method, pika.spec.Queue.BindOk)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)
        ch.basic_publish(exg_name, routing_key, body='TestQueueBindAndUnbindAndPurge', mandatory=True)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 1)
        frame = ch.queue_unbind(queue=q_name, exchange=exg_name, routing_key=routing_key)
        self.assertIsInstance(frame.method, pika.spec.Queue.UnbindOk)
        with self.assertRaises(pika.exceptions.UnroutableError):
            ch.basic_publish(exg_name, routing_key, body='TestQueueBindAndUnbindAndPurge-2', mandatory=True)
        frame = ch.queue_purge(q_name)
        self.assertIsInstance(frame.method, pika.spec.Queue.PurgeOk)
        self.assertEqual(frame.method.message_count, 1)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)

class TestBasicGet(BlockingTestCaseBase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        LOGGER.info('%s TEARING DOWN (%s)', datetime.utcnow(), self)

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingChannel.basic_get'
        LOGGER.info('%s STARTED (%s)', datetime.utcnow(), self)
        connection = self._connect()
        LOGGER.info('%s CONNECTED (%s)', datetime.utcnow(), self)
        ch = connection.channel()
        LOGGER.info('%s CREATED CHANNEL (%s)', datetime.utcnow(), self)
        q_name = 'TestBasicGet_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        LOGGER.info('%s ENABLED PUB-ACKS (%s)', datetime.utcnow(), self)
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        LOGGER.info('%s DECLARED QUEUE (%s)', datetime.utcnow(), self)
        msg = ch.basic_get(q_name, auto_ack=False)
        self.assertTupleEqual(msg, (None, None, None))
        LOGGER.info('%s GOT FROM EMPTY QUEUE (%s)', datetime.utcnow(), self)
        body = 'TestBasicGet'
        ch.basic_publish(exchange='', routing_key=q_name, body=body, mandatory=True)
        LOGGER.info('%s PUBLISHED (%s)', datetime.utcnow(), self)
        (method, properties, body) = ch.basic_get(q_name, auto_ack=False)
        LOGGER.info('%s GOT FROM NON-EMPTY QUEUE (%s)', datetime.utcnow(), self)
        self.assertIsInstance(method, pika.spec.Basic.GetOk)
        self.assertEqual(method.delivery_tag, 1)
        self.assertFalse(method.redelivered)
        self.assertEqual(method.exchange, '')
        self.assertEqual(method.routing_key, q_name)
        self.assertEqual(method.message_count, 0)
        self.assertIsInstance(properties, pika.BasicProperties)
        self.assertIsNone(properties.headers)
        self.assertEqual(body, as_bytes(body))
        ch.basic_ack(delivery_tag=method.delivery_tag)
        LOGGER.info('%s ACKED (%s)', datetime.utcnow(), self)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)

class TestBasicReject(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingChannel.basic_reject'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicReject_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicReject1', mandatory=True)
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicReject2', mandatory=True)
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicReject1'))
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicReject2'))
        ch.basic_reject(rx_method.delivery_tag, requeue=True)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=1)
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicReject2'))

class TestBasicRejectNoRequeue(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingChannel.basic_reject with requeue=False'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicRejectNoRequeue_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicRejectNoRequeue1', mandatory=True)
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicRejectNoRequeue2', mandatory=True)
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicRejectNoRequeue1'))
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicRejectNoRequeue2'))
        ch.basic_reject(rx_method.delivery_tag, requeue=False)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)

class TestBasicNack(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingChannel.basic_nack single message'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicNack_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicNack1', mandatory=True)
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicNack2', mandatory=True)
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNack1'))
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNack2'))
        ch.basic_nack(rx_method.delivery_tag, multiple=False, requeue=True)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=1)
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNack2'))

class TestBasicNackNoRequeue(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel.basic_nack with requeue=False'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicNackNoRequeue_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicNackNoRequeue1', mandatory=True)
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicNackNoRequeue2', mandatory=True)
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNackNoRequeue1'))
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNackNoRequeue2'))
        ch.basic_nack(rx_method.delivery_tag, requeue=False)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)

class TestBasicNackMultiple(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.basic_nack multiple messages'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicNackMultiple_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicNackMultiple1', mandatory=True)
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicNackMultiple2', mandatory=True)
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNackMultiple1'))
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNackMultiple2'))
        ch.basic_nack(rx_method.delivery_tag, multiple=True, requeue=True)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=2)
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNackMultiple1'))
        (rx_method, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestBasicNackMultiple2'))

class TestBasicRecoverWithRequeue(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.basic_recover with requeue=True.\n\n        NOTE: the requeue=False option is not supported by RabbitMQ broker as\n        of this writing (using RabbitMQ 3.5.1)\n        '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicRecoverWithRequeue_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicRecoverWithRequeue1', mandatory=True)
        ch.basic_publish(exchange='', routing_key=q_name, body='TestBasicRecoverWithRequeue2', mandatory=True)
        rx_messages = []
        num_messages = 0
        for msg in ch.consume(q_name, auto_ack=False):
            num_messages += 1
            if num_messages == 2:
                ch.basic_recover(requeue=True)
            if num_messages > 2:
                rx_messages.append(msg)
            if num_messages == 4:
                break
        else:
            self.fail('consumer aborted prematurely')
        (_, _, rx_body) = rx_messages[0]
        self.assertEqual(rx_body, as_bytes('TestBasicRecoverWithRequeue1'))
        (_, _, rx_body) = rx_messages[1]
        self.assertEqual(rx_body, as_bytes('TestBasicRecoverWithRequeue2'))

class TestTxCommit(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingChannel.tx_commit'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestTxCommit_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        frame = ch.tx_select()
        self.assertIsInstance(frame.method, pika.spec.Tx.SelectOk)
        ch.basic_publish(exchange='', routing_key=q_name, body='TestTxCommit1', mandatory=True)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)
        ch.tx_commit()
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 1)
        (_, _, rx_body) = ch.basic_get(q_name, auto_ack=False)
        self.assertEqual(rx_body, as_bytes('TestTxCommit1'))

class TestTxRollback(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.tx_commit'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestTxRollback_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        frame = ch.tx_select()
        self.assertIsInstance(frame.method, pika.spec.Tx.SelectOk)
        ch.basic_publish(exchange='', routing_key=q_name, body='TestTxRollback1', mandatory=True)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)
        ch.tx_rollback()
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)

class TestBasicConsumeFromUnknownQueueRaisesChannelClosed(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'ChannelClosed raised when consuming from unknown queue'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicConsumeFromUnknownQueueRaisesChannelClosed_q_' + uuid.uuid1().hex
        with self.assertRaises(pika.exceptions.ChannelClosedByBroker) as ex_cm:
            ch.basic_consume(q_name, lambda *args: None)
        self.assertEqual(ex_cm.exception.args[0], 404)

class TestPublishAndBasicPublishWithPubacksUnroutable(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingChannel.publish amd basic_publish unroutable message with pubacks'
        connection = self._connect()
        ch = connection.channel()
        exg_name = 'TestPublishAndBasicPublishUnroutable_exg_' + uuid.uuid1().hex
        routing_key = 'TestPublishAndBasicPublishUnroutable'
        res = ch.confirm_delivery()
        self.assertIsNone(res)
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        msg2_headers = dict(test_name='TestPublishAndBasicPublishWithPubacksUnroutable')
        msg2_properties = pika.spec.BasicProperties(headers=msg2_headers)
        with self.assertRaises(pika.exceptions.UnroutableError) as cm:
            ch.basic_publish(exg_name, routing_key=routing_key, body='', properties=msg2_properties, mandatory=True)
        (msg,) = cm.exception.messages
        self.assertIsInstance(msg, blocking_connection.ReturnedMessage)
        self.assertIsInstance(msg.method, pika.spec.Basic.Return)
        self.assertEqual(msg.method.reply_code, 312)
        self.assertEqual(msg.method.exchange, exg_name)
        self.assertEqual(msg.method.routing_key, routing_key)
        self.assertIsInstance(msg.properties, pika.BasicProperties)
        self.assertEqual(msg.properties.headers, msg2_headers)
        self.assertEqual(msg.body, as_bytes(''))

class TestConfirmDeliveryAfterUnroutableMessage(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.confirm_delivery following unroutable message'
        connection = self._connect()
        ch = connection.channel()
        exg_name = 'TestConfirmDeliveryAfterUnroutableMessage_exg_' + uuid.uuid1().hex
        routing_key = 'TestConfirmDeliveryAfterUnroutableMessage'
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        returned_messages = []
        ch.add_on_return_callback(lambda *args: returned_messages.append(args))
        ch.basic_publish(exg_name, routing_key=routing_key, body='', mandatory=True)
        ch.confirm_delivery()
        self.assertEqual(len(ch._pending_events), 1)
        self.assertIsInstance(ch._pending_events[0], blocking_connection._ReturnedMessageEvt)
        repr(ch._pending_events[0])
        connection.process_data_events()
        self.assertEqual(len(ch._pending_events), 0)
        ((channel, method, properties, body),) = returned_messages
        self.assertIs(channel, ch)
        self.assertIsInstance(method, pika.spec.Basic.Return)
        self.assertEqual(method.reply_code, 312)
        self.assertEqual(method.exchange, exg_name)
        self.assertEqual(method.routing_key, routing_key)
        self.assertIsInstance(properties, pika.BasicProperties)
        self.assertEqual(body, as_bytes(''))

class TestUnroutableMessagesReturnedInNonPubackMode(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingChannel: unroutable messages is returned in non-puback mode'
        connection = self._connect()
        ch = connection.channel()
        exg_name = 'TestUnroutableMessageReturnedInNonPubackMode_exg_' + uuid.uuid1().hex
        routing_key = 'TestUnroutableMessageReturnedInNonPubackMode'
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        returned_messages = []
        ch.add_on_return_callback(lambda *args: returned_messages.append(args))
        ch.basic_publish(exg_name, routing_key=routing_key, body='msg1', mandatory=True)
        ch.basic_publish(exg_name, routing_key=routing_key, body='msg2', mandatory=True)
        while len(returned_messages) < 2:
            connection.process_data_events()
        self.assertEqual(len(returned_messages), 2)
        self.assertEqual(len(ch._pending_events), 0)
        (channel, method, properties, body) = returned_messages[0]
        self.assertIs(channel, ch)
        self.assertIsInstance(method, pika.spec.Basic.Return)
        self.assertEqual(method.reply_code, 312)
        self.assertEqual(method.exchange, exg_name)
        self.assertEqual(method.routing_key, routing_key)
        self.assertIsInstance(properties, pika.BasicProperties)
        self.assertEqual(body, as_bytes('msg1'))
        (channel, method, properties, body) = returned_messages[1]
        self.assertIs(channel, ch)
        self.assertIsInstance(method, pika.spec.Basic.Return)
        self.assertEqual(method.reply_code, 312)
        self.assertEqual(method.exchange, exg_name)
        self.assertEqual(method.routing_key, routing_key)
        self.assertIsInstance(properties, pika.BasicProperties)
        self.assertEqual(body, as_bytes('msg2'))

class TestUnroutableMessageReturnedInPubackMode(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel: unroutable messages is returned in puback mode'
        connection = self._connect()
        ch = connection.channel()
        exg_name = 'TestUnroutableMessageReturnedInPubackMode_exg_' + uuid.uuid1().hex
        routing_key = 'TestUnroutableMessageReturnedInPubackMode'
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        ch.confirm_delivery()
        returned_messages = []
        ch.add_on_return_callback(lambda *args: returned_messages.append(args))
        with self.assertRaises(pika.exceptions.UnroutableError):
            ch.basic_publish(exg_name, routing_key=routing_key, body='msg1', mandatory=True)
        with self.assertRaises(pika.exceptions.UnroutableError):
            ch.basic_publish(exg_name, routing_key=routing_key, body='msg2', mandatory=True)
        self.assertEqual(len(ch._pending_events), 2)
        self.assertIsInstance(ch._pending_events[0], blocking_connection._ReturnedMessageEvt)
        self.assertIsInstance(ch._pending_events[1], blocking_connection._ReturnedMessageEvt)
        repr(ch._pending_events[0])
        repr(ch._pending_events[1])
        connection.process_data_events()
        self.assertEqual(len(ch._pending_events), 0)
        (channel, method, properties, body) = returned_messages[0]
        self.assertIs(channel, ch)
        self.assertIsInstance(method, pika.spec.Basic.Return)
        self.assertEqual(method.reply_code, 312)
        self.assertEqual(method.exchange, exg_name)
        self.assertEqual(method.routing_key, routing_key)
        self.assertIsInstance(properties, pika.BasicProperties)
        self.assertEqual(body, as_bytes('msg1'))
        (channel, method, properties, body) = returned_messages[1]
        self.assertIs(channel, ch)
        self.assertIsInstance(method, pika.spec.Basic.Return)
        self.assertEqual(method.reply_code, 312)
        self.assertEqual(method.exchange, exg_name)
        self.assertEqual(method.routing_key, routing_key)
        self.assertIsInstance(properties, pika.BasicProperties)
        self.assertEqual(body, as_bytes('msg2'))

class TestBasicPublishDeliveredWhenPendingUnroutable(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingChannel.basic_publish msg delivered despite pending unroutable message'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicPublishDeliveredWhenPendingUnroutable_q' + uuid.uuid1().hex
        exg_name = 'TestBasicPublishDeliveredWhenPendingUnroutable_exg_' + uuid.uuid1().hex
        routing_key = 'TestBasicPublishDeliveredWhenPendingUnroutable'
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.queue_bind(q_name, exchange=exg_name, routing_key=routing_key)
        ch.basic_publish(exg_name, routing_key='', body='unroutable-message', mandatory=True)
        connection.channel().close()
        ch.basic_publish(exg_name, routing_key=routing_key, body='routable-message', mandatory=True)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=1)
        msg = ch.basic_get(q_name)
        self.assertIsInstance(msg, tuple)
        (rx_method, rx_properties, rx_body) = msg
        self.assertIsInstance(rx_method, pika.spec.Basic.GetOk)
        self.assertEqual(rx_method.delivery_tag, 1)
        self.assertFalse(rx_method.redelivered)
        self.assertEqual(rx_method.exchange, exg_name)
        self.assertEqual(rx_method.routing_key, routing_key)
        self.assertIsInstance(rx_properties, pika.BasicProperties)
        self.assertEqual(rx_body, as_bytes('routable-message'))
        self.assertFalse(ch._pending_events)
        ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)

class TestPublishAndConsumeWithPubacksAndQosOfOne(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel.basic_publish, publish, basic_consume, QoS,         Basic.Cancel from broker\n        '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestPublishAndConsumeAndQos_q' + uuid.uuid1().hex
        exg_name = 'TestPublishAndConsumeAndQos_exg_' + uuid.uuid1().hex
        routing_key = 'TestPublishAndConsumeAndQos'
        res = ch.confirm_delivery()
        self.assertIsNone(res)
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.queue_bind(q_name, exchange=exg_name, routing_key=routing_key)
        msg1_headers = dict(test_name='TestPublishAndConsumeWithPubacksAndQosOfOne')
        msg1_properties = pika.spec.BasicProperties(headers=msg1_headers)
        ch.basic_publish(exg_name, routing_key=routing_key, body='via-basic_publish', properties=msg1_properties, mandatory=True)
        ch.basic_publish(exg_name, routing_key, body='via-publish', mandatory=True)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 2)
        ch.basic_qos(prefetch_size=0, prefetch_count=1, global_qos=False)
        rx_messages = []
        consumer_tag = ch.basic_consume(q_name, lambda *args: rx_messages.append(args), auto_ack=False, exclusive=False, arguments=None)
        while not rx_messages:
            connection.process_data_events(time_limit=None)
        self.assertEqual(len(rx_messages), 1)
        msg = rx_messages[0]
        self.assertIsInstance(msg, tuple)
        (rx_ch, rx_method, rx_properties, rx_body) = msg
        self.assertIs(rx_ch, ch)
        self.assertIsInstance(rx_method, pika.spec.Basic.Deliver)
        self.assertEqual(rx_method.consumer_tag, consumer_tag)
        self.assertEqual(rx_method.delivery_tag, 1)
        self.assertFalse(rx_method.redelivered)
        self.assertEqual(rx_method.exchange, exg_name)
        self.assertEqual(rx_method.routing_key, routing_key)
        self.assertIsInstance(rx_properties, pika.BasicProperties)
        self.assertEqual(rx_properties.headers, msg1_headers)
        self.assertEqual(rx_body, as_bytes('via-basic_publish'))
        self.assertFalse(ch._pending_events)
        ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
        while len(rx_messages) < 2:
            connection.process_data_events(time_limit=None)
        self.assertEqual(len(rx_messages), 2)
        msg = rx_messages[1]
        self.assertIsInstance(msg, tuple)
        (rx_ch, rx_method, rx_properties, rx_body) = msg
        self.assertIs(rx_ch, ch)
        self.assertIsInstance(rx_method, pika.spec.Basic.Deliver)
        self.assertEqual(rx_method.consumer_tag, consumer_tag)
        self.assertEqual(rx_method.delivery_tag, 2)
        self.assertFalse(rx_method.redelivered)
        self.assertEqual(rx_method.exchange, exg_name)
        self.assertEqual(rx_method.routing_key, routing_key)
        self.assertIsInstance(rx_properties, pika.BasicProperties)
        self.assertEqual(rx_body, as_bytes('via-publish'))
        self.assertFalse(ch._pending_events)
        ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)
        connection.process_data_events(time_limit=0.005)
        self.assertEqual(len(rx_messages), 2)
        rx_cancellations = []
        ch.add_on_cancel_callback(rx_cancellations.append)
        ch.queue_delete(q_name)
        ch.start_consuming()
        self.assertEqual(len(rx_cancellations), 1)
        (frame,) = rx_cancellations
        self.assertEqual(frame.method.consumer_tag, consumer_tag)

class TestBasicConsumeWithAckFromAnotherThread(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.basic_consume with ack from another thread and         requesting basic_ack via add_callback_threadsafe\n        '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicConsumeWithAckFromAnotherThread_q' + uuid.uuid1().hex
        exg_name = 'TestBasicConsumeWithAckFromAnotherThread_exg' + uuid.uuid1().hex
        routing_key = 'TestBasicConsumeWithAckFromAnotherThread'
        res = ch.confirm_delivery()
        self.assertIsNone(res)
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.queue_bind(q_name, exchange=exg_name, routing_key=routing_key)
        ch.basic_publish(exg_name, routing_key, body='msg1', mandatory=True)
        ch.basic_publish(exg_name, routing_key, body='last-msg', mandatory=True)
        ch.basic_qos(prefetch_size=0, prefetch_count=1, global_qos=False)
        rx_messages = []

        def ackAndEnqueueMessageViaAnotherThread(rx_ch, rx_method, rx_properties, rx_body):
            if False:
                print('Hello World!')
            LOGGER.debug('%s: Got message body=%r; delivery-tag=%r', datetime.now(), rx_body, rx_method.delivery_tag)

            def processOnConnectionThread():
                if False:
                    while True:
                        i = 10
                LOGGER.debug('%s: ACKing message body=%r; delivery-tag=%r', datetime.now(), rx_body, rx_method.delivery_tag)
                ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
                rx_messages.append(rx_body)
                if rx_body == b'last-msg':
                    LOGGER.debug('%s: Canceling consumer consumer-tag=%r', datetime.now(), rx_method.consumer_tag)
                    rx_ch.basic_cancel(rx_method.consumer_tag)
            timer = threading.Timer(0, lambda : connection.add_callback_threadsafe(processOnConnectionThread))
            self.addCleanup(timer.cancel)
            timer.start()
        consumer_tag = ch.basic_consume(q_name, ackAndEnqueueMessageViaAnotherThread, auto_ack=False, exclusive=False, arguments=None)
        LOGGER.debug('%s: calling start_consuming(); consumer tag=%r', datetime.now(), consumer_tag)
        ch.start_consuming()
        LOGGER.debug('%s: Returned from start_consuming(); consumer tag=%r', datetime.now(), consumer_tag)
        self.assertEqual(len(rx_messages), 2)
        self.assertEqual(rx_messages[0], b'msg1')
        self.assertEqual(rx_messages[1], b'last-msg')

class TestConsumeGeneratorWithAckFromAnotherThread(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel.consume and requesting basic_ack from another         thread via add_callback_threadsafe\n        '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestConsumeGeneratorWithAckFromAnotherThread_q' + uuid.uuid1().hex
        exg_name = 'TestConsumeGeneratorWithAckFromAnotherThread_exg' + uuid.uuid1().hex
        routing_key = 'TestConsumeGeneratorWithAckFromAnotherThread'
        res = ch.confirm_delivery()
        self.assertIsNone(res)
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.queue_bind(q_name, exchange=exg_name, routing_key=routing_key)
        ch.basic_publish(exg_name, routing_key, body='msg1', mandatory=True)
        ch.basic_publish(exg_name, routing_key, body='last-msg', mandatory=True)
        ch.basic_qos(prefetch_size=0, prefetch_count=1, global_qos=False)
        rx_messages = []

        def ackAndEnqueueMessageViaAnotherThread(rx_ch, rx_method, rx_properties, rx_body):
            if False:
                while True:
                    i = 10
            LOGGER.debug('%s: Got message body=%r; delivery-tag=%r', datetime.now(), rx_body, rx_method.delivery_tag)

            def processOnConnectionThread():
                if False:
                    i = 10
                    return i + 15
                LOGGER.debug('%s: ACKing message body=%r; delivery-tag=%r', datetime.now(), rx_body, rx_method.delivery_tag)
                ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
                rx_messages.append(rx_body)
                if rx_body == b'last-msg':
                    LOGGER.debug('%s: Canceling consumer consumer-tag=%r', datetime.now(), rx_method.consumer_tag)
                    rx_ch.cancel()
            timer = threading.Timer(0, lambda : connection.add_callback_threadsafe(processOnConnectionThread))
            self.addCleanup(timer.cancel)
            timer.start()
        for (method, properties, body) in ch.consume(q_name, auto_ack=False):
            ackAndEnqueueMessageViaAnotherThread(rx_ch=ch, rx_method=method, rx_properties=properties, rx_body=body)
        self.assertEqual(len(rx_messages), 2)
        self.assertEqual(rx_messages[0], b'msg1')
        self.assertEqual(rx_messages[1], b'last-msg')

class TestTwoBasicConsumersOnSameChannel(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingChannel: two basic_consume consumers on same channel\n        '
        connection = self._connect()
        ch = connection.channel()
        exg_name = 'TestPublishAndConsumeAndQos_exg_' + uuid.uuid1().hex
        q1_name = 'TestTwoBasicConsumersOnSameChannel_q1' + uuid.uuid1().hex
        q2_name = 'TestTwoBasicConsumersOnSameChannel_q2' + uuid.uuid1().hex
        q1_routing_key = 'TestTwoBasicConsumersOnSameChannel1'
        q2_routing_key = 'TestTwoBasicConsumersOnSameChannel2'
        ch.confirm_delivery()
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        ch.queue_declare(q1_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q1_name))
        ch.queue_bind(q1_name, exchange=exg_name, routing_key=q1_routing_key)
        ch.queue_declare(q2_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q2_name))
        ch.queue_bind(q2_name, exchange=exg_name, routing_key=q2_routing_key)
        q1_tx_message_bodies = ['q1_message+%s' % (i,) for i in pika.compat.xrange(100)]
        for message_body in q1_tx_message_bodies:
            ch.basic_publish(exg_name, q1_routing_key, body=message_body, mandatory=True)
        q2_tx_message_bodies = ['q2_message+%s' % (i,) for i in pika.compat.xrange(150)]
        for message_body in q2_tx_message_bodies:
            ch.basic_publish(exg_name, q2_routing_key, body=message_body, mandatory=True)
        q1_rx_messages = []
        q1_consumer_tag = ch.basic_consume(q1_name, lambda *args: q1_rx_messages.append(args), auto_ack=False, exclusive=False, arguments=None)
        q2_rx_messages = []
        q2_consumer_tag = ch.basic_consume(q2_name, lambda *args: q2_rx_messages.append(args), auto_ack=False, exclusive=False, arguments=None)
        while len(q1_rx_messages) < len(q1_tx_message_bodies) or len(q2_rx_messages) < len(q2_tx_message_bodies):
            connection.process_data_events(time_limit=None)
        self.assertEqual(len(q2_rx_messages), len(q2_tx_message_bodies))

        def validate_messages(rx_messages, routing_key, consumer_tag, tx_message_bodies):
            if False:
                i = 10
                return i + 15
            self.assertEqual(len(rx_messages), len(tx_message_bodies))
            for (msg, expected_body) in zip(rx_messages, tx_message_bodies):
                self.assertIsInstance(msg, tuple)
                (rx_ch, rx_method, rx_properties, rx_body) = msg
                self.assertIs(rx_ch, ch)
                self.assertIsInstance(rx_method, pika.spec.Basic.Deliver)
                self.assertEqual(rx_method.consumer_tag, consumer_tag)
                self.assertFalse(rx_method.redelivered)
                self.assertEqual(rx_method.exchange, exg_name)
                self.assertEqual(rx_method.routing_key, routing_key)
                self.assertIsInstance(rx_properties, pika.BasicProperties)
                self.assertEqual(rx_body, as_bytes(expected_body))
        validate_messages(rx_messages=q1_rx_messages, routing_key=q1_routing_key, consumer_tag=q1_consumer_tag, tx_message_bodies=q1_tx_message_bodies)
        validate_messages(rx_messages=q2_rx_messages, routing_key=q2_routing_key, consumer_tag=q2_consumer_tag, tx_message_bodies=q2_tx_message_bodies)
        self.assertFalse(ch._pending_events)

class TestBasicCancelPurgesPendingConsumerCancellationEvt(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.basic_cancel purges pending _ConsumerCancellationEvt'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicCancelPurgesPendingConsumerCancellationEvt_q' + uuid.uuid1().hex
        ch.queue_declare(q_name)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish('', routing_key=q_name, body='via-publish', mandatory=True)
        rx_messages = []
        consumer_tag = ch.basic_consume(q_name, lambda *args: rx_messages.append(args))
        while not ch._pending_events:
            connection.channel().close()
        self.assertEqual(len(ch._pending_events), 1)
        self.assertIsInstance(ch._pending_events[0], blocking_connection._ConsumerDeliveryEvt)
        ch.queue_delete(q_name)
        while len(ch._pending_events) < 2:
            connection.channel().close()
        self.assertEqual(len(ch._pending_events), 2)
        self.assertIsInstance(ch._pending_events[1], blocking_connection._ConsumerCancellationEvt)
        messages = ch.basic_cancel(consumer_tag)
        self.assertEqual(messages, [])
        self.assertEqual(len(ch._pending_events), 0)

class TestBasicPublishWithoutPubacks(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel.basic_publish without pubacks'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicPublishWithoutPubacks_q' + uuid.uuid1().hex
        exg_name = 'TestBasicPublishWithoutPubacks_exg_' + uuid.uuid1().hex
        routing_key = 'TestBasicPublishWithoutPubacks'
        ch.exchange_declare(exg_name, exchange_type=ExchangeType.direct)
        self.addCleanup(connection.channel().exchange_delete, exg_name)
        ch.queue_declare(q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.queue_bind(q_name, exchange=exg_name, routing_key=routing_key)
        msg1_headers = dict(test_name='TestBasicPublishWithoutPubacks')
        msg1_properties = pika.spec.BasicProperties(headers=msg1_headers)
        ch.basic_publish(exg_name, routing_key=routing_key, body='via-basic_publish_mandatory=True', properties=msg1_properties, mandatory=True)
        ch.basic_publish(exg_name, routing_key=routing_key, body='via-basic_publish_mandatory=False', mandatory=False)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=2)
        rx_messages = []
        consumer_tag = ch.basic_consume(q_name, lambda *args: rx_messages.append(args), auto_ack=False, exclusive=False, arguments=None)
        while not rx_messages:
            connection.process_data_events(time_limit=None)
        self.assertGreaterEqual(len(rx_messages), 1)
        msg = rx_messages[0]
        self.assertIsInstance(msg, tuple)
        (rx_ch, rx_method, rx_properties, rx_body) = msg
        self.assertIs(rx_ch, ch)
        self.assertIsInstance(rx_method, pika.spec.Basic.Deliver)
        self.assertEqual(rx_method.consumer_tag, consumer_tag)
        self.assertEqual(rx_method.delivery_tag, 1)
        self.assertFalse(rx_method.redelivered)
        self.assertEqual(rx_method.exchange, exg_name)
        self.assertEqual(rx_method.routing_key, routing_key)
        self.assertIsInstance(rx_properties, pika.BasicProperties)
        self.assertEqual(rx_properties.headers, msg1_headers)
        self.assertEqual(rx_body, as_bytes('via-basic_publish_mandatory=True'))
        self.assertFalse(ch._pending_events)
        ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
        while len(rx_messages) < 2:
            connection.process_data_events(time_limit=None)
        self.assertEqual(len(rx_messages), 2)
        msg = rx_messages[1]
        self.assertIsInstance(msg, tuple)
        (rx_ch, rx_method, rx_properties, rx_body) = msg
        self.assertIs(rx_ch, ch)
        self.assertIsInstance(rx_method, pika.spec.Basic.Deliver)
        self.assertEqual(rx_method.consumer_tag, consumer_tag)
        self.assertEqual(rx_method.delivery_tag, 2)
        self.assertFalse(rx_method.redelivered)
        self.assertEqual(rx_method.exchange, exg_name)
        self.assertEqual(rx_method.routing_key, routing_key)
        self.assertIsInstance(rx_properties, pika.BasicProperties)
        self.assertEqual(rx_body, as_bytes('via-basic_publish_mandatory=False'))
        self.assertFalse(ch._pending_events)
        ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)
        connection.process_data_events(time_limit=0.005)
        self.assertEqual(len(rx_messages), 2)

class TestPublishFromBasicConsumeCallback(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingChannel.basic_publish from basic_consume callback\n        '
        connection = self._connect()
        ch = connection.channel()
        src_q_name = 'TestPublishFromBasicConsumeCallback_src_q' + uuid.uuid1().hex
        dest_q_name = 'TestPublishFromBasicConsumeCallback_dest_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(src_q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(src_q_name))
        ch.queue_declare(dest_q_name, auto_delete=True)
        self.addCleanup(lambda : self._connect().channel().queue_delete(dest_q_name))
        ch.basic_publish('', routing_key=src_q_name, body='via-publish', mandatory=True)

        def on_consume(channel, method, props, body):
            if False:
                while True:
                    i = 10
            channel.basic_publish('', routing_key=dest_q_name, body=body, properties=props, mandatory=True)
            channel.basic_ack(method.delivery_tag)
        ch.basic_consume(src_q_name, on_consume, auto_ack=False, exclusive=False, arguments=None)
        for (_, _, rx_body) in ch.consume(dest_q_name, auto_ack=True):
            self.assertEqual(rx_body, as_bytes('via-publish'))
            break
        else:
            self.fail('failed to consume a messages from destination q')

class TestStopConsumingFromBasicConsumeCallback(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.stop_consuming from basic_consume callback\n        '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestStopConsumingFromBasicConsumeCallback_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(connection.channel().queue_delete, q_name)
        ch.basic_publish('', routing_key=q_name, body='via-publish1', mandatory=True)
        ch.basic_publish('', routing_key=q_name, body='via-publish2', mandatory=True)

        def on_consume(channel, method, props, body):
            if False:
                while True:
                    i = 10
            channel.stop_consuming()
            channel.basic_ack(method.delivery_tag)
        ch.basic_consume(q_name, on_consume, auto_ack=False, exclusive=False, arguments=None)
        ch.start_consuming()
        ch.close()
        ch = connection.channel()
        (_, _, rx_body) = ch.basic_get(q_name)
        self.assertEqual(rx_body, as_bytes('via-publish2'))
        msg = ch.basic_get(q_name)
        self.assertTupleEqual(msg, (None, None, None))

class TestCloseChannelFromBasicConsumeCallback(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.close from basic_consume callback\n        '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestCloseChannelFromBasicConsumeCallback_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(connection.channel().queue_delete, q_name)
        ch.basic_publish('', routing_key=q_name, body='via-publish1', mandatory=True)
        ch.basic_publish('', routing_key=q_name, body='via-publish2', mandatory=True)

        def on_consume(channel, method, props, body):
            if False:
                i = 10
                return i + 15
            channel.close()
        ch.basic_consume(q_name, on_consume, auto_ack=False, exclusive=False, arguments=None)
        ch.start_consuming()
        self.assertTrue(ch.is_closed)
        ch = connection.channel()
        (_, _, rx_body) = ch.basic_get(q_name)
        self.assertEqual(rx_body, as_bytes('via-publish1'))
        (_, _, rx_body) = ch.basic_get(q_name)
        self.assertEqual(rx_body, as_bytes('via-publish2'))

class TestCloseConnectionFromBasicConsumeCallback(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        'BlockingConnection.close from basic_consume callback\n        '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestCloseConnectionFromBasicConsumeCallback_q' + uuid.uuid1().hex
        ch.confirm_delivery()
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish('', routing_key=q_name, body='via-publish1', mandatory=True)
        ch.basic_publish('', routing_key=q_name, body='via-publish2', mandatory=True)

        def on_consume(channel, method, props, body):
            if False:
                print('Hello World!')
            connection.close()
        ch.basic_consume(q_name, on_consume, auto_ack=False, exclusive=False, arguments=None)
        ch.start_consuming()
        self.assertTrue(ch.is_closed)
        self.assertTrue(connection.is_closed)
        ch = self._connect().channel()
        (_, _, rx_body) = ch.basic_get(q_name)
        self.assertEqual(rx_body, as_bytes('via-publish1'))
        (_, _, rx_body) = ch.basic_get(q_name)
        self.assertEqual(rx_body, as_bytes('via-publish2'))

class TestStartConsumingRaisesChannelClosedOnSameChannelFailure(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'start_consuming() exits with ChannelClosed exception on same channel failure\n        '
        connection = self._connect()
        self._instrument_io_loop_exception_leak_detection(connection)
        ch = connection.channel()
        q_name = 'TestStartConsumingPassesChannelClosedOnSameChannelFailure_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_consume(q_name, lambda *args, **kwargs: None, auto_ack=False, exclusive=False, arguments=None)
        connection.add_callback_threadsafe(lambda : ch.basic_publish(exchange=q_name, routing_key='123', body=b'Nope this is wrong'))
        with self.assertRaises(pika.exceptions.ChannelClosedByBroker):
            ch.start_consuming()

class TestStartConsumingReturnsAfterCancelFromBroker(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'start_consuming() returns after Cancel from broker\n        '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestStartConsumingExitsOnCancelFromBroker_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        consumer_tag = ch.basic_consume(q_name, lambda *args, **kwargs: None, auto_ack=False, exclusive=False, arguments=None)
        connection.add_callback_threadsafe(lambda : self._connect().channel().queue_delete(q_name))
        ch.start_consuming()
        self.assertNotIn(consumer_tag, ch._consumer_infos)

class TestNonPubAckPublishAndConsumeHugeMessage(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel.publish/consume huge message'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestPublishAndConsumeHugeMessage_q' + uuid.uuid1().hex
        body = 'a' * 1000000
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body=body)
        LOGGER.info('Published message body size=%s', len(body))
        for (rx_method, rx_props, rx_body) in ch.consume(q_name, auto_ack=False, exclusive=False, arguments=None):
            self.assertIsInstance(rx_method, pika.spec.Basic.Deliver)
            self.assertEqual(rx_method.delivery_tag, 1)
            self.assertFalse(rx_method.redelivered)
            self.assertEqual(rx_method.exchange, '')
            self.assertEqual(rx_method.routing_key, q_name)
            self.assertIsInstance(rx_props, pika.BasicProperties)
            self.assertEqual(rx_body, as_bytes(body))
            ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
            break
        self.assertFalse(ch._queue_consumer_generator.pending_events)
        ch.close()
        ch = connection.channel()
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)

class TestNonPubAckPublishAndConsumeManyMessages(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel non-pub-ack publish/consume many messages'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestNonPubackPublishAndConsumeManyMessages_q' + uuid.uuid1().hex
        body = 'b' * 1024
        num_messages_to_publish = 500
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        for _ in pika.compat.xrange(num_messages_to_publish):
            ch.basic_publish(exchange='', routing_key=q_name, body=body)
        num_consumed = 0
        for (rx_method, rx_props, rx_body) in ch.consume(q_name, auto_ack=False, exclusive=False, arguments=None):
            num_consumed += 1
            self.assertIsInstance(rx_method, pika.spec.Basic.Deliver)
            self.assertEqual(rx_method.delivery_tag, num_consumed)
            self.assertFalse(rx_method.redelivered)
            self.assertEqual(rx_method.exchange, '')
            self.assertEqual(rx_method.routing_key, q_name)
            self.assertIsInstance(rx_props, pika.BasicProperties)
            self.assertEqual(rx_body, as_bytes(body))
            ch.basic_ack(delivery_tag=rx_method.delivery_tag, multiple=False)
            if num_consumed >= num_messages_to_publish:
                break
        self.assertFalse(ch._queue_consumer_generator.pending_events)
        ch.close()
        self.assertIsNone(ch._queue_consumer_generator)
        ch = connection.channel()
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)

class TestBasicCancelWithNonAckableConsumer(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingChannel user cancels non-ackable consumer via basic_cancel'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicCancelWithNonAckableConsumer_q' + uuid.uuid1().hex
        body1 = 'a' * 1024
        body2 = 'b' * 2048
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body=body1)
        ch.basic_publish(exchange='', routing_key=q_name, body=body2)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=2)
        consumer_tag = ch.basic_consume(q_name, lambda *x: None, auto_ack=True, exclusive=False, arguments=None)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)
        messages = ch.basic_cancel(consumer_tag)
        self.assertEqual(len(messages), 2)
        (_, _, rx_body1) = messages[0]
        self.assertEqual(rx_body1, as_bytes(body1))
        (_, _, rx_body2) = messages[1]
        self.assertEqual(rx_body2, as_bytes(body2))
        ch.close()
        ch = connection.channel()
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)

class TestBasicCancelWithAckableConsumer(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingChannel user cancels ackable consumer via basic_cancel'
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestBasicCancelWithAckableConsumer_q' + uuid.uuid1().hex
        body1 = 'a' * 1024
        body2 = 'b' * 2048
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body=body1)
        ch.basic_publish(exchange='', routing_key=q_name, body=body2)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=2)
        consumer_tag = ch.basic_consume(q_name, lambda *x: None, auto_ack=False, exclusive=False, arguments=None)
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=0)
        messages = ch.basic_cancel(consumer_tag)
        self.assertEqual(len(messages), 0)
        ch.close()
        ch = connection.channel()
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=2)

class TestUnackedMessageAutoRestoredToQueueOnChannelClose(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingChannel unacked message restored to q on channel close '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestUnackedMessageAutoRestoredToQueueOnChannelClose_q' + uuid.uuid1().hex
        body1 = 'a' * 1024
        body2 = 'b' * 2048
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body=body1)
        ch.basic_publish(exchange='', routing_key=q_name, body=body2)
        rx_messages = []
        ch.basic_consume(q_name, lambda *args: rx_messages.append(args), auto_ack=False, exclusive=False, arguments=None)
        while len(rx_messages) != 2:
            connection.process_data_events(time_limit=None)
        self.assertEqual(rx_messages[0][1].delivery_tag, 1)
        self.assertEqual(rx_messages[1][1].delivery_tag, 2)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)
        ch.close()
        ch = connection.channel()
        self._assert_exact_message_count_with_retries(channel=ch, queue=q_name, expected_count=2)

class TestNoAckMessageNotRestoredToQueueOnChannelClose(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        'BlockingChannel unacked message restored to q on channel close '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestNoAckMessageNotRestoredToQueueOnChannelClose_q' + uuid.uuid1().hex
        body1 = 'a' * 1024
        body2 = 'b' * 2048
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        ch.basic_publish(exchange='', routing_key=q_name, body=body1)
        ch.basic_publish(exchange='', routing_key=q_name, body=body2)
        num_messages = 0
        for (rx_method, _, _) in ch.consume(q_name, auto_ack=True, exclusive=False):
            num_messages += 1
            self.assertEqual(rx_method.delivery_tag, num_messages)
            if num_messages == 2:
                break
        else:
            self.fail('expected 2 messages, but consumed %i' % (num_messages,))
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)
        ch.close()
        ch = connection.channel()
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.message_count, 0)

class TestConsumeGeneratorInactivityTimeout(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingChannel consume returns 3-tuple of None values on inactivity timeout '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestConsumeGeneratorInactivityTimeout_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=True)
        for msg in ch.consume(q_name, inactivity_timeout=0.1):
            self.assertEqual(msg, (None, None, None))
            break
        else:
            self.fail('expected (None, None, None), but iterator stopped')

class TestConsumeGeneratorInterruptedByCancelFromBroker(BlockingTestCaseBase):

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        "BlockingChannel consume generator is interrupted broker's Cancel "
        connection = self._connect()
        self.assertTrue(connection.consumer_cancel_notify_supported)
        ch = connection.channel()
        q_name = 'TestConsumeGeneratorInterruptedByCancelFromBroker_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=True)
        queue_deleted = False
        for _ in ch.consume(q_name, auto_ack=False, inactivity_timeout=0.001):
            if not queue_deleted:
                ch.queue_delete(q_name)
                queue_deleted = True
        self.assertIsNone(ch._queue_consumer_generator)

class TestConsumeGeneratorCancelEncountersCancelFromBroker(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        "BlockingChannel consume generator cancel called when broker's Cancel is enqueued "
        connection = self._connect()
        self.assertTrue(connection.consumer_cancel_notify_supported)
        ch = connection.channel()
        q_name = 'TestConsumeGeneratorCancelEncountersCancelFromBroker_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=True)
        for _ in ch.consume(q_name, auto_ack=False, inactivity_timeout=0.001):
            ch.queue_delete(q_name)
            while not ch._queue_consumer_generator.pending_events:
                connection.process_data_events()
            self.assertIsInstance(ch._queue_consumer_generator.pending_events[0], blocking_connection._ConsumerCancellationEvt)
            ch.cancel()
            self.assertIsNone(ch._queue_consumer_generator)

class TestConsumeGeneratorPassesChannelClosedOnSameChannelFailure(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'consume() exits with ChannelClosed exception on same channel failure\n        '
        connection = self._connect()
        self._instrument_io_loop_exception_leak_detection(connection)
        ch = connection.channel()
        q_name = 'TestConsumeGeneratorPassesChannelClosedOnSameChannelFailure_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        connection.add_callback_threadsafe(lambda : ch.basic_publish(exchange=q_name, routing_key='123', body=b'Nope this is wrong'))
        with self.assertRaises(pika.exceptions.ChannelClosedByBroker):
            for _ in ch.consume(q_name):
                pass

class TestChannelFlow(BlockingTestCaseBase):

    def test(self):
        if False:
            print('Hello World!')
        'BlockingChannel Channel.Flow activate and deactivate '
        connection = self._connect()
        ch = connection.channel()
        q_name = 'TestChannelFlow_q' + uuid.uuid1().hex
        ch.queue_declare(q_name, auto_delete=False)
        self.addCleanup(lambda : self._connect().channel().queue_delete(q_name))
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.consumer_count, 0)
        ch.basic_consume(q_name, lambda *args: None)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.consumer_count, 1)
        active = ch.flow(True)
        self.assertEqual(active, True)
        frame = ch.queue_declare(q_name, passive=True)
        self.assertEqual(frame.method.consumer_count, 1)

class TestChannelRaisesWrongStateWhenDeclaringQueueOnClosedChannel(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingConnection: Declaring queue on closed channel raises ChannelWrongStateError'
        q_name = 'TestChannelRaisesWrongStateWhenDeclaringQueueOnClosedChannel_q' + uuid.uuid1().hex
        channel = self._connect().channel()
        channel.close()
        with self.assertRaises(pika.exceptions.ChannelWrongStateError):
            channel.queue_declare(q_name)

class TestChannelRaisesWrongStateWhenClosingClosedChannel(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingConnection: Closing closed channel raises ChannelWrongStateError'
        channel = self._connect().channel()
        channel.close()
        with self.assertRaises(pika.exceptions.ChannelWrongStateError):
            channel.close()

class TestChannelContextManagerClosesChannel(BlockingTestCaseBase):

    def test(self):
        if False:
            return 10
        'BlockingConnection: chanel context manager exit survives closed channel'
        with self._connect().channel() as channel:
            self.assertTrue(channel.is_open)
        self.assertTrue(channel.is_closed)

class TestChannelContextManagerExitSurvivesClosedChannel(BlockingTestCaseBase):

    def test(self):
        if False:
            i = 10
            return i + 15
        'BlockingConnection: chanel context manager exit survives closed channel'
        with self._connect().channel() as channel:
            self.assertTrue(channel.is_open)
            channel.close()
            self.assertTrue(channel.is_closed)
        self.assertTrue(channel.is_closed)

class TestChannelContextManagerDoesNotSuppressChannelClosedByBroker(BlockingTestCaseBase):

    def test(self):
        if False:
            while True:
                i = 10
        "BlockingConnection: chanel context manager doesn't suppress ChannelClosedByBroker exception"
        exg_name = 'TestChannelContextManagerDoesNotSuppressChannelClosedByBroker' + uuid.uuid1().hex
        with self.assertRaises(pika.exceptions.ChannelClosedByBroker):
            with self._connect().channel() as channel:
                channel.exchange_declare(exg_name, passive=True)
        self.assertTrue(channel.is_closed)
if __name__ == '__main__':
    unittest.main()