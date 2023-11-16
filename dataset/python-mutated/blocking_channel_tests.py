"""
Tests for pika.adapters.blocking_connection.BlockingChannel

"""
from collections import deque
import unittest
from unittest import mock
from pika.adapters import blocking_connection
from pika import channel
BLOCKING_CHANNEL = 'pika.adapters.blocking_connection.BlockingChannel'
BLOCKING_CONNECTION = 'pika.adapters.blocking_connection.BlockingConnection'

class ChannelTemplate(channel.Channel):
    channel_number = 1

class BlockingChannelTests(unittest.TestCase):

    @mock.patch(BLOCKING_CONNECTION)
    def _create_connection(self, connection=None):
        if False:
            while True:
                i = 10
        return connection

    def setUp(self):
        if False:
            while True:
                i = 10
        self.connection = self._create_connection()
        channel_impl_mock = mock.Mock(spec=ChannelTemplate, is_closing=False, is_closed=False, is_open=True)
        self.obj = blocking_connection.BlockingChannel(channel_impl_mock, self.connection)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        del self.connection
        del self.obj

    def test_init_initial_value_confirmation(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.obj._delivery_confirmation)

    def test_init_initial_value_pending_events(self):
        if False:
            return 10
        self.assertEqual(self.obj._pending_events, deque())

    def test_init_initial_value_buback_return(self):
        if False:
            print('Hello World!')
        self.assertIsNone(self.obj._puback_return)

    def test_basic_consume_legacy_parameter_queue(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            self.obj.basic_consume('queue', 'whoops this should be a callback')

    def test_basic_consume_legacy_parameter_callback(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            self.obj.basic_consume(mock.Mock(), 'queue')

    def test_queue_declare_legacy_parameter_callback(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self.obj.queue_declare(mock.Mock(), 'queue')

    def test_exchange_declare_legacy_parameter_callback(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self.obj.exchange_declare(mock.Mock(), 'exchange')

    def test_queue_bind_legacy_parameter_callback(self):
        if False:
            print('Hello World!')
        with self.assertRaises(TypeError):
            self.obj.queue_bind(mock.Mock(), 'queue', 'exchange')

    def test_basic_cancel_legacy_parameter(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            self.obj.basic_cancel(mock.Mock(), 'tag')

    def test_basic_get_legacy_parameter(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypeError):
            self.obj.basic_get(mock.Mock())

    def test_basic_consume(self):
        if False:
            i = 10
            return i + 15
        with mock.patch.object(self.obj._impl, '_generate_consumer_tag'):
            self.obj._impl._generate_consumer_tag.return_value = 'ctag0'
            self.obj._impl.basic_consume.return_value = 'ctag0'
            self.obj.basic_consume('queue', mock.Mock())
            self.assertEqual(self.obj._consumer_infos['ctag0'].state, blocking_connection._ConsumerInfo.ACTIVE)

    def test_context_manager(self):
        if False:
            return 10
        with self.obj as chan:
            self.assertFalse(chan._impl.close.called)
        chan._impl.close.assert_called_once_with(reply_code=0, reply_text='Normal shutdown')

    def test_context_manager_does_not_suppress_exception(self):
        if False:
            while True:
                i = 10

        class TestException(Exception):
            pass
        with self.assertRaises(TestException):
            with self.obj as chan:
                self.assertFalse(chan._impl.close.called)
                raise TestException()
        chan._impl.close.assert_called_once_with(reply_code=0, reply_text='Normal shutdown')

    def test_context_manager_exit_with_closed_channel(self):
        if False:
            while True:
                i = 10
        with self.obj as chan:
            self.assertFalse(chan._impl.close.called)
            chan.close()
        chan._impl.close.assert_called_with(reply_code=0, reply_text='Normal shutdown')

    def test_consumer_tags_property(self):
        if False:
            print('Hello World!')
        with mock.patch.object(self.obj._impl, '_generate_consumer_tag'):
            self.assertEqual(0, len(self.obj.consumer_tags))
            self.obj._impl._generate_consumer_tag.return_value = 'ctag0'
            self.obj._impl.basic_consume.return_value = 'ctag0'
            self.obj.basic_consume('queue', mock.Mock())
            self.assertEqual(1, len(self.obj.consumer_tags))
            self.assertIn('ctag0', self.obj.consumer_tags)