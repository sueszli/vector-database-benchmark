"""twisted adapter test"""
import functools
import unittest
from unittest import mock
from nose.twistedtools import reactor, deferred
from twisted.internet import defer, error as twisted_error
from twisted.python.failure import Failure
from pika.adapters.twisted_connection import ClosableDeferredQueue, ReceivedMessage, TwistedChannel, _TwistedConnectionAdapter, TwistedProtocolConnection, _TimerHandle
from pika import spec
from pika.exceptions import AMQPConnectionError, ConsumerCancelled, DuplicateGetOkCallback, NackError, UnroutableError, ChannelClosedByBroker
from pika.frame import Method

class TestCase(unittest.TestCase):
    """Imported from twisted.trial.unittest.TestCase

    We only want the assertFailure implementation, using the class directly
    hides some assertion errors.
    """

    def assertFailure(self, d, *expectedFailures):
        if False:
            return 10
        '\n        Fail if C{deferred} does not errback with one of C{expectedFailures}.\n        Returns the original Deferred with callbacks added. You will need\n        to return this Deferred from your test case.\n        '

        def _cb(ignore):
            if False:
                while True:
                    i = 10
            raise self.failureException('did not catch an error, instead got %r' % (ignore,))

        def _eb(failure):
            if False:
                for i in range(10):
                    print('nop')
            if failure.check(*expectedFailures):
                return failure.value
            else:
                output = '\nExpected: %r\nGot:\n%s' % (expectedFailures, str(failure))
                raise self.failureException(output)
        return d.addCallbacks(_cb, _eb)

class ClosableDeferredQueueTestCase(TestCase):

    @deferred(timeout=5.0)
    def test_put_closed(self):
        if False:
            i = 10
            return i + 15
        q = ClosableDeferredQueue()
        q.closed = RuntimeError('testing')
        d = self.assertFailure(q.put(None), RuntimeError)
        d.addCallback(lambda e: self.assertEqual(e.args[0], 'testing'))
        return d

    @deferred(timeout=5.0)
    def test_get_closed(self):
        if False:
            for i in range(10):
                print('nop')
        q = ClosableDeferredQueue()
        q.closed = RuntimeError('testing')
        d = self.assertFailure(q.get(), RuntimeError)
        d.addCallback(lambda e: self.assertEqual(e.args[0], 'testing'))
        return d

    def test_close(self):
        if False:
            i = 10
            return i + 15
        q = ClosableDeferredQueue()
        q.close('testing')
        self.assertEqual(q.closed, 'testing')
        self.assertEqual(q.waiting, [])
        self.assertEqual(q.pending, [])

    def test_close_waiting(self):
        if False:
            i = 10
            return i + 15
        q = ClosableDeferredQueue()
        d = q.get()
        q.close(RuntimeError('testing'))
        self.assertTrue(q.closed)
        self.assertEqual(q.waiting, [])
        self.assertEqual(q.pending, [])
        return self.assertFailure(d, RuntimeError)

    def test_close_twice(self):
        if False:
            i = 10
            return i + 15
        q = ClosableDeferredQueue()
        q.close('testing')
        self.assertEqual(q.closed, 'testing')
        q.close('testing')
        self.assertEqual(q.closed, 'testing')

class TwistedChannelTestCase(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.pika_channel = mock.Mock()
        self.channel = TwistedChannel(self.pika_channel)
        wrapped = ('basic_cancel', 'basic_get', 'basic_qos', 'basic_recover', 'exchange_bind', 'exchange_unbind', 'exchange_declare', 'exchange_delete', 'confirm_delivery', 'flow', 'queue_bind', 'queue_declare', 'queue_delete', 'queue_purge', 'queue_unbind', 'tx_commit', 'tx_rollback', 'tx_select')
        for meth_name in wrapped:
            getattr(self.pika_channel, meth_name).__name__ = meth_name

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        self.pika_channel.__repr__ = lambda _s: '<TestChannel>'
        self.assertEqual(repr(self.channel), '<TwistedChannel channel=<TestChannel>>')

    @deferred(timeout=5.0)
    def test_on_close(self):
        if False:
            for i in range(10):
                print('nop')
        self.pika_channel.add_on_close_callback.assert_called_with(self.channel._on_channel_closed)
        calls = self.channel._calls = [defer.Deferred()]
        consumers = self.channel._consumers = {'test-delivery-tag': mock.Mock()}
        error = RuntimeError('testing')
        self.channel._on_channel_closed(None, error)
        consumers['test-delivery-tag'].close.assert_called_once_with(error)
        self.assertEqual(len(self.channel._calls), 0)
        self.assertEqual(len(self.channel._consumers), 0)
        return self.assertFailure(calls[0], RuntimeError)

    @deferred(timeout=5.0)
    def test_basic_consume(self):
        if False:
            while True:
                i = 10
        d = self.channel.basic_consume(queue='testqueue')
        self.pika_channel.basic_consume.assert_called_once()
        kwargs = self.pika_channel.basic_consume.call_args_list[0][1]
        self.assertEqual(kwargs['queue'], 'testqueue')
        on_message = kwargs['on_message_callback']

        def check_cb(result):
            if False:
                i = 10
                return i + 15
            (queue, _consumer_tag) = result
            queue_get_d = queue.get()
            queue_get_d.addCallback(self.assertEqual, (self.channel, 'testmethod', 'testprops', 'testbody'))
            on_message('testchan', 'testmethod', 'testprops', 'testbody')
            return queue_get_d
        d.addCallback(check_cb)
        frame = Method(1, spec.Basic.ConsumeOk(consumer_tag='testconsumertag'))
        kwargs['callback'](frame)
        return d

    @deferred(timeout=5.0)
    def test_basic_consume_while_closed(self):
        if False:
            while True:
                i = 10
        error = RuntimeError('testing')
        self.channel._on_channel_closed(None, error)
        d = self.channel.basic_consume(queue='testqueue')
        return self.assertFailure(d, RuntimeError)

    @deferred(timeout=5.0)
    def test_basic_consume_failure(self):
        if False:
            for i in range(10):
                print('nop')
        self.pika_channel.basic_consume.side_effect = RuntimeError()
        d = self.channel.basic_consume(queue='testqueue')
        return self.assertFailure(d, RuntimeError)

    def test_basic_consume_errback_on_close(self):
        if False:
            while True:
                i = 10
        d = self.channel.basic_consume(queue='testqueue')
        self.channel._on_channel_closed(self, ChannelClosedByBroker(404, 'NOT FOUND'))
        return self.assertFailure(d, ChannelClosedByBroker)

    @deferred(timeout=5.0)
    def test_queue_delete(self):
        if False:
            i = 10
            return i + 15
        queue_obj = mock.Mock()
        self.channel._consumers = {'test-delivery-tag': queue_obj}
        self.channel._queue_name_to_consumer_tags['testqueue'] = set(['test-delivery-tag'])
        self.channel._calls = set()
        self.pika_channel.queue_delete.__name__ = 'queue_delete'
        d = self.channel.queue_delete(queue='testqueue')
        self.pika_channel.queue_delete.assert_called_once()
        call_kw = self.pika_channel.queue_delete.call_args_list[0][1]
        self.assertEqual(call_kw['queue'], 'testqueue')

        def check(_):
            if False:
                i = 10
                return i + 15
            self.assertEqual(len(self.channel._consumers), 0)
            queue_obj.close.assert_called_once()
            close_call_args = queue_obj.close.call_args_list[0][0]
            self.assertEqual(len(close_call_args), 1)
            self.assertTrue(isinstance(close_call_args[0], ConsumerCancelled))
        d.addCallback(check)
        self.assertEqual(len(self.channel._calls), 1)
        list(self.channel._calls)[0].callback(None)
        return d

    @deferred(timeout=5.0)
    def test_wrapped_method(self):
        if False:
            i = 10
            return i + 15
        self.pika_channel.queue_declare.__name__ = 'queue_declare'
        d = self.channel.queue_declare(queue='testqueue')
        self.pika_channel.queue_declare.assert_called_once()
        call_kw = self.pika_channel.queue_declare.call_args_list[0][1]
        self.assertIn('queue', call_kw)
        self.assertEqual(call_kw['queue'], 'testqueue')
        self.assertIn('callback', call_kw)
        self.assertTrue(callable(call_kw['callback']))
        call_kw['callback']('testresult')
        d.addCallback(self.assertEqual, 'testresult')
        return d

    @deferred(timeout=5.0)
    def test_wrapped_method_while_closed(self):
        if False:
            return 10
        error = RuntimeError('testing')
        self.channel._on_channel_closed(None, error)
        self.pika_channel.queue_declare.__name__ = 'queue_declare'
        d = self.channel.queue_declare(queue='testqueue')
        return self.assertFailure(d, RuntimeError)

    @deferred(timeout=5.0)
    def test_wrapped_method_multiple_args(self):
        if False:
            return 10
        self.pika_channel.queue_declare.__name__ = 'queue_declare'
        d = self.channel.queue_declare(queue='testqueue')
        call_kw = self.pika_channel.queue_declare.call_args_list[0][1]
        call_kw['callback']('testresult-1', 'testresult-2')
        d.addCallback(self.assertEqual, ('testresult-1', 'testresult-2'))
        return d

    @deferred(timeout=5.0)
    def test_wrapped_method_failure(self):
        if False:
            for i in range(10):
                print('nop')
        error = RuntimeError('testing')
        self.pika_channel.queue_declare.__name__ = 'queue_declare'
        self.pika_channel.queue_declare.side_effect = error
        d = self.channel.queue_declare(queue='testqueue')
        return self.assertFailure(d, RuntimeError)

    def test_method_not_wrapped(self):
        if False:
            return 10
        result = self.channel.basic_ack()
        self.assertFalse(isinstance(result, defer.Deferred))
        self.pika_channel.basic_ack.assert_called_once()

    def test_passthrough(self):
        if False:
            i = 10
            return i + 15
        attributes = ('channel_number', 'connection', 'is_closed', 'is_closing', 'is_open', 'flow_active', 'consumer_tags')
        for name in attributes:
            value = 'testvalue-{}'.format(name)
            setattr(self.pika_channel, name, value)
            self.assertEqual(getattr(self.channel, name), value)

    def test_callback_deferred(self):
        if False:
            for i in range(10):
                print('nop')
        d = defer.Deferred()
        replies = [spec.Basic.CancelOk]
        self.channel.callback_deferred(d, replies)
        self.pika_channel.add_callback.assert_called_with(d.callback, replies)

    def test_add_on_return_callback(self):
        if False:
            for i in range(10):
                print('nop')
        cb = mock.Mock()
        self.channel.add_on_return_callback(cb)
        self.pika_channel.add_on_return_callback.assert_called_once()
        self.pika_channel.add_on_return_callback.call_args[0][0]('testchannel', 'testmethod', 'testprops', 'testbody')
        cb.assert_called_once()
        self.assertEqual(len(cb.call_args[0]), 1)
        self.assertEqual(cb.call_args[0][0], (self.channel, 'testmethod', 'testprops', 'testbody'))

    @deferred(timeout=5.0)
    def test_basic_cancel(self):
        if False:
            for i in range(10):
                print('nop')
        queue_obj = mock.Mock()
        queue_obj_2 = mock.Mock()
        self.channel._consumers['test-consumer'] = queue_obj
        self.channel._consumers['test-consumer-2'] = queue_obj_2
        self.channel._queue_name_to_consumer_tags.update({'testqueue': set(['test-consumer']), 'testqueue-2': set(['test-consumer-2'])})
        d = self.channel.basic_cancel('test-consumer')

        def check(result):
            if False:
                while True:
                    i = 10
            self.assertTrue(isinstance(result, Method))
            queue_obj.close.assert_called_once()
            self.assertTrue(isinstance(queue_obj.close.call_args[0][0], ConsumerCancelled))
            self.assertEqual(len(self.channel._consumers), 1)
            queue_obj_2.close.assert_not_called()
            self.assertEqual(self.channel._queue_name_to_consumer_tags['testqueue'], set())
        d.addCallback(check)
        self.pika_channel.basic_cancel.assert_called_once()
        self.pika_channel.basic_cancel.call_args[1]['callback'](Method(1, spec.Basic.CancelOk(consumer_tag='test-consumer')))
        return d

    @deferred(timeout=5.0)
    def test_basic_cancel_no_consumer(self):
        if False:
            while True:
                i = 10
        d = self.channel.basic_cancel('test-consumer')

        def check(result):
            if False:
                while True:
                    i = 10
            self.assertTrue(isinstance(result, Method))
        d.addCallback(check)
        self.pika_channel.basic_cancel.assert_called_once()
        self.pika_channel.basic_cancel.call_args[1]['callback'](Method(1, spec.Basic.CancelOk(consumer_tag='test-consumer')))
        return d

    def test_consumer_cancelled_by_broker(self):
        if False:
            i = 10
            return i + 15
        self.pika_channel.add_on_cancel_callback.assert_called_with(self.channel._on_consumer_cancelled_by_broker)
        queue_obj = mock.Mock()
        self.channel._consumers['test-consumer'] = queue_obj
        self.channel._queue_name_to_consumer_tags['testqueue'] = set(['test-consumer'])
        self.channel._on_consumer_cancelled_by_broker(Method(1, spec.Basic.Cancel(consumer_tag='test-consumer')))
        queue_obj.close.assert_called_once()
        self.assertTrue(isinstance(queue_obj.close.call_args[0][0], ConsumerCancelled))
        self.assertEqual(self.channel._consumers, {})
        self.assertEqual(self.channel._queue_name_to_consumer_tags['testqueue'], set())

    @deferred(timeout=5.0)
    def test_basic_get(self):
        if False:
            i = 10
            return i + 15
        d = self.channel.basic_get(queue='testqueue')
        self.pika_channel.basic_get.assert_called_once()
        kwargs = self.pika_channel.basic_get.call_args_list[0][1]
        self.assertEqual(kwargs['queue'], 'testqueue')

        def check_cb(result):
            if False:
                print('Hello World!')
            self.assertEqual(result, (self.channel, 'testmethod', 'testprops', 'testbody'))
        d.addCallback(check_cb)
        kwargs['callback']('testchannel', 'testmethod', 'testprops', 'testbody')
        return d

    def test_basic_get_twice(self):
        if False:
            for i in range(10):
                print('nop')
        self.channel.basic_get(queue='testqueue')
        self.assertRaises(DuplicateGetOkCallback, self.channel.basic_get, 'testqueue')

    @deferred(timeout=5.0)
    def test_basic_get_empty(self):
        if False:
            while True:
                i = 10
        self.pika_channel.add_callback.assert_called_with(self.channel._on_getempty, [spec.Basic.GetEmpty], False)
        d = self.channel.basic_get(queue='testqueue')
        self.channel._on_getempty('testmethod')
        d.addCallback(self.assertIsNone)
        return d

    def test_basic_nack(self):
        if False:
            print('Hello World!')
        self.channel.basic_nack('testdeliverytag')
        self.pika_channel.basic_nack.assert_called_once_with(delivery_tag='testdeliverytag', multiple=False, requeue=True)

    @deferred(timeout=5.0)
    def test_basic_publish(self):
        if False:
            i = 10
            return i + 15
        args = [object()]
        kwargs = {'routing_key': object(), 'body': object()}
        d = self.channel.basic_publish(*args, **kwargs)
        kwargs.update(dict(exchange=args[0], mandatory=False, properties=None))
        self.pika_channel.basic_publish.assert_called_once_with(**kwargs)
        return d

    @deferred(timeout=5.0)
    def test_basic_publish_closed(self):
        if False:
            print('Hello World!')
        self.channel._on_channel_closed(None, RuntimeError('testing'))
        d = self.channel.basic_publish(None, None, None)
        self.pika_channel.basic_publish.assert_not_called()
        d = self.assertFailure(d, RuntimeError)
        d.addCallback(lambda e: self.assertEqual(e.args[0], 'testing'))
        return d

    def _test_wrapped_func(self, func, kwargs, do_callback=False):
        if False:
            while True:
                i = 10
        func.assert_called_once()
        call_kw = dict(((key, value) for (key, value) in func.call_args[1].items() if key != 'callback'))
        self.assertEqual(kwargs, call_kw)
        if do_callback:
            func.call_args[1]['callback'](do_callback)

    @deferred(timeout=5.0)
    def test_basic_qos(self):
        if False:
            return 10
        kwargs = {'prefetch_size': 2}
        d = self.channel.basic_qos(**kwargs)
        kwargs.update(dict(prefetch_count=0, global_qos=False))
        self._test_wrapped_func(self.pika_channel.basic_qos, kwargs, True)
        return d

    def test_basic_reject(self):
        if False:
            for i in range(10):
                print('nop')
        self.channel.basic_reject('testdeliverytag')
        self.pika_channel.basic_reject.assert_called_once_with(delivery_tag='testdeliverytag', requeue=True)

    @deferred(timeout=5.0)
    def test_basic_recover(self):
        if False:
            while True:
                i = 10
        d = self.channel.basic_recover()
        self._test_wrapped_func(self.pika_channel.basic_recover, {'requeue': False}, True)
        return d

    def test_close(self):
        if False:
            return 10
        self.channel.close()
        self.pika_channel.close.assert_called_once_with(reply_code=0, reply_text='Normal shutdown')

    @deferred(timeout=5.0)
    def test_confirm_delivery(self):
        if False:
            while True:
                i = 10
        d = self.channel.confirm_delivery()
        self.pika_channel.confirm_delivery.assert_called_once()
        self.assertEqual(self.pika_channel.confirm_delivery.call_args[1]['ack_nack_callback'], self.channel._on_delivery_confirmation)

        def send_message(_result):
            if False:
                i = 10
                return i + 15
            d = self.channel.basic_publish('testexch', 'testrk', 'testbody')
            frame = Method(1, spec.Basic.Ack(delivery_tag=1))
            self.channel._on_delivery_confirmation(frame)
            return d

        def check_response(frame_method):
            if False:
                i = 10
                return i + 15
            self.assertTrue(isinstance(frame_method, spec.Basic.Ack))
        d.addCallback(send_message)
        d.addCallback(check_response)
        self.pika_channel.confirm_delivery.call_args[1]['callback'](None)
        return d

    @deferred(timeout=5.0)
    def test_confirm_delivery_nacked(self):
        if False:
            print('Hello World!')
        d = self.channel.confirm_delivery()

        def send_message(_result):
            if False:
                return 10
            d = self.channel.basic_publish('testexch', 'testrk', 'testbody')
            frame = Method(1, spec.Basic.Nack(delivery_tag=1))
            self.channel._on_delivery_confirmation(frame)
            return d

        def check_response(error):
            if False:
                return 10
            self.assertIsInstance(error.value, NackError)
            self.assertEqual(len(error.value.messages), 0)
        d.addCallback(send_message)
        d.addCallbacks(self.fail, check_response)
        self.pika_channel.confirm_delivery.call_args[1]['callback'](None)
        return d

    @deferred(timeout=5.0)
    def test_confirm_delivery_returned(self):
        if False:
            return 10
        d = self.channel.confirm_delivery()
        self.pika_channel.add_on_return_callback.assert_called_once()
        return_cb = self.pika_channel.add_on_return_callback.call_args[0][0]

        def send_message(_result):
            if False:
                return 10
            d = self.channel.basic_publish('testexch', 'testrk', 'testbody')
            method = spec.Basic.Return(exchange='testexch', routing_key='testrk')
            return_cb(self.channel, method, spec.BasicProperties(), 'testbody')
            frame = Method(1, spec.Basic.Ack(delivery_tag=1))
            self.channel._on_delivery_confirmation(frame)
            return d

        def check_response(error):
            if False:
                print('Hello World!')
            self.assertIsInstance(error.value, UnroutableError)
            self.assertEqual(len(error.value.messages), 1)
            msg = error.value.messages[0]
            self.assertEqual(msg.body, 'testbody')
        d.addCallbacks(send_message, self.fail)
        d.addCallbacks(self.fail, check_response)
        self.pika_channel.confirm_delivery.call_args[1]['callback'](None)
        return d

    @deferred(timeout=5.0)
    def test_confirm_delivery_returned_nacked(self):
        if False:
            while True:
                i = 10
        d = self.channel.confirm_delivery()
        self.pika_channel.add_on_return_callback.assert_called_once()
        return_cb = self.pika_channel.add_on_return_callback.call_args[0][0]

        def send_message(_result):
            if False:
                print('Hello World!')
            d = self.channel.basic_publish('testexch', 'testrk', 'testbody')
            method = spec.Basic.Return(exchange='testexch', routing_key='testrk')
            return_cb(self.channel, method, spec.BasicProperties(), 'testbody')
            frame = Method(1, spec.Basic.Nack(delivery_tag=1))
            self.channel._on_delivery_confirmation(frame)
            return d

        def check_response(error):
            if False:
                while True:
                    i = 10
            self.assertTrue(isinstance(error.value, NackError))
            self.assertEqual(len(error.value.messages), 1)
            msg = error.value.messages[0]
            self.assertEqual(msg.body, 'testbody')
        d.addCallback(send_message)
        d.addCallbacks(self.fail, check_response)
        self.pika_channel.confirm_delivery.call_args[1]['callback'](None)
        return d

    @deferred(timeout=5.0)
    def test_confirm_delivery_multiple(self):
        if False:
            print('Hello World!')
        d = self.channel.confirm_delivery()

        def send_message(_result):
            if False:
                for i in range(10):
                    print('nop')
            d1 = self.channel.basic_publish('testexch', 'testrk', 'testbody1')
            d2 = self.channel.basic_publish('testexch', 'testrk', 'testbody2')
            frame = Method(1, spec.Basic.Ack(delivery_tag=2, multiple=True))
            self.channel._on_delivery_confirmation(frame)
            return defer.DeferredList([d1, d2])

        def check_response(results):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(len(results), 2)
            for (is_ok, result) in results:
                self.assertTrue(is_ok)
                self.assertTrue(isinstance(result, spec.Basic.Ack))
        d.addCallback(send_message)
        d.addCallback(check_response)
        self.pika_channel.confirm_delivery.call_args[1]['callback'](None)
        return d

    @deferred(timeout=5.0)
    def test_delivery_confirmation_errback_on_close(self):
        if False:
            print('Hello World!')
        d = self.channel.confirm_delivery()
        self.pika_channel.confirm_delivery.call_args[1]['callback'](None)

        def send_message_and_close_channel(_result):
            if False:
                for i in range(10):
                    print('nop')
            d = self.channel.basic_publish('testexch', 'testrk', 'testbody')
            self.channel._on_channel_closed(None, RuntimeError('testing'))
            self.assertEqual(len(self.channel._deliveries), 0)
            return d
        d.addCallback(send_message_and_close_channel)
        return self.assertFailure(d, RuntimeError)

class TwistedProtocolConnectionTestCase(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.conn = TwistedProtocolConnection()
        self.conn._impl = mock.Mock()

    @deferred(timeout=5.0)
    def test_connection(self):
        if False:
            return 10
        transport = mock.Mock()
        self.conn.connectionMade = mock.Mock()
        self.conn.makeConnection(transport)
        self.conn._impl.connection_made.assert_called_once_with(transport)
        self.conn.connectionMade.assert_called_once()
        d = self.conn.ready
        self.conn._on_connection_ready(None)
        return d

    @deferred(timeout=5.0)
    def test_channel(self):
        if False:
            print('Hello World!')
        channel = mock.Mock()
        self.conn._impl.channel.side_effect = lambda n, cb: cb(channel)
        d = self.conn.channel()
        self.conn._impl.channel.assert_called_once()

        def check(result):
            if False:
                while True:
                    i = 10
            self.assertTrue(isinstance(result, TwistedChannel))
        d.addCallback(check)
        return d

    @deferred(timeout=5.0)
    def test_channel_errback_if_connection_closed(self):
        if False:
            i = 10
            return i + 15
        self.conn._on_connection_ready('dummy')
        d = self.conn.channel()
        self.conn._on_connection_closed('test conn', RuntimeError('testing'))
        self.assertEqual(len(self.conn._calls), 0)
        return self.assertFailure(d, RuntimeError)

    def test_dataReceived(self):
        if False:
            while True:
                i = 10
        self.conn.dataReceived('testdata')
        self.conn._impl.data_received.assert_called_once_with('testdata')

    @deferred(timeout=5.0)
    def test_connectionLost(self):
        if False:
            return 10
        ready_d = self.conn.ready
        error = RuntimeError('testreason')
        self.conn.connectionLost(error)
        self.conn._impl.connection_lost.assert_called_with(error)
        self.assertIsNone(self.conn.ready)
        return self.assertFailure(ready_d, RuntimeError)

    def test_connectionLost_twice(self):
        if False:
            return 10
        ready_d = self.conn.ready
        error = RuntimeError('testreason')
        self.conn.connectionLost(error)
        self.assertTrue(ready_d.called)
        ready_d.addErrback(lambda f: None)
        self.assertIsNone(self.conn.ready)
        self.conn.connectionLost(error)

    @deferred(timeout=5.0)
    def test_on_connection_ready(self):
        if False:
            return 10
        d = self.conn.ready
        self.conn._on_connection_ready('testresult')
        self.assertTrue(d.called)
        d.addCallback(functools.partial(self.assertIsInstance, cls=TwistedProtocolConnection))
        return d

    def test_on_connection_ready_twice(self):
        if False:
            while True:
                i = 10
        d = self.conn.ready
        self.conn._on_connection_ready('testresult')
        self.assertTrue(d.called)
        self.conn._on_connection_ready('testresult')

    @deferred(timeout=5.0)
    def test_on_connection_ready_method(self):
        if False:
            while True:
                i = 10
        d = self.conn.ready
        self.conn.connectionReady = mock.Mock()
        self.conn._on_connection_ready('testresult')
        self.conn.connectionReady.assert_called_once()
        return d

    @deferred(timeout=5.0)
    def test_on_connection_failed(self):
        if False:
            i = 10
            return i + 15
        d = self.conn.ready
        self.conn._on_connection_failed(None)
        return self.assertFailure(d, AMQPConnectionError)

    def test_on_connection_failed_twice(self):
        if False:
            while True:
                i = 10
        d = self.conn.ready
        self.conn._on_connection_failed(None)
        self.assertTrue(d.called)
        d.addErrback(lambda f: None)
        self.conn._on_connection_failed(None)

    @deferred(timeout=5.0)
    def test_on_connection_closed(self):
        if False:
            i = 10
            return i + 15
        self.conn._on_connection_ready('dummy')
        d = self.conn.closed
        self.conn._on_connection_closed('test conn', 'test reason')
        self.assertTrue(d.called)
        d.addCallback(self.assertEqual, 'test reason')
        return d

    def test_on_connection_closed_twice(self):
        if False:
            print('Hello World!')
        self.conn._on_connection_ready('dummy')
        d = self.conn.closed
        self.conn._on_connection_closed('test conn', 'test reason')
        self.assertTrue(d.called)
        self.conn._on_connection_closed('test conn', 'test reason')

    @deferred(timeout=5.0)
    def test_on_connection_closed_Failure(self):
        if False:
            while True:
                i = 10
        self.conn._on_connection_ready('dummy')
        error = RuntimeError()
        d = self.conn.closed
        self.conn._on_connection_closed('test conn', Failure(error))
        self.assertTrue(d.called)

        def _check_cb(result):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(result, error)

        def _check_eb(_failure):
            if False:
                print('Hello World!')
            self.fail('The errback path should not have been triggered')
        d.addCallbacks(_check_cb, _check_eb)
        return d

    def test_close(self):
        if False:
            print('Hello World!')
        self.conn._impl.is_closed = False
        self.conn.closed = 'TESTING'
        value = self.conn.close()
        self.assertEqual(value, 'TESTING')
        self.conn._impl.close.assert_called_once_with(200, 'Normal shutdown')

    def test_close_twice(self):
        if False:
            i = 10
            return i + 15
        self.conn._impl.is_closed = True
        self.conn.close()
        self.conn._impl.close.assert_not_called()

class TwistedConnectionAdapterTestCase(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.conn = _TwistedConnectionAdapter(None, None, None, None, None)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if self.conn._transport is None:
            self.conn._transport = mock.Mock()
        self.conn.close()

    def test_adapter_disconnect_stream(self):
        if False:
            i = 10
            return i + 15
        transport = mock.Mock()
        self.conn.connection_made(transport)
        self.conn._adapter_disconnect_stream()
        transport.loseConnection.assert_called_once()

    def test_adapter_emit_data(self):
        if False:
            for i in range(10):
                print('nop')
        transport = mock.Mock()
        self.conn.connection_made(transport)
        self.conn._adapter_emit_data('testdata')
        transport.write.assert_called_with('testdata')

    def test_timeout(self):
        if False:
            print('Hello World!')
        callback = mock.Mock()
        timer_id = self.conn._adapter_call_later(5, callback)
        self.assertEqual(len(reactor.getDelayedCalls()), 1)
        self.conn._adapter_remove_timeout(timer_id)
        self.assertEqual(len(reactor.getDelayedCalls()), 0)
        callback.assert_not_called()

    @deferred(timeout=5.0)
    def test_call_threadsafe(self):
        if False:
            return 10
        callback = mock.Mock()
        self.conn._adapter_add_callback_threadsafe(callback)
        d = defer.Deferred()

        def check():
            if False:
                print('Hello World!')
            callback.assert_called_once()
            d.callback(None)
        reactor.callLater(0.1, check)
        return d

    def test_connection_made(self):
        if False:
            print('Hello World!')
        transport = mock.Mock()
        self.conn.connection_made(transport)
        self.assertEqual(self.conn._transport, transport)
        self.assertEqual(self.conn.connection_state, self.conn.CONNECTION_PROTOCOL)

    def test_connection_lost(self):
        if False:
            for i in range(10):
                print('nop')
        self.conn._on_stream_terminated = mock.Mock()
        error = Failure(RuntimeError('testreason'))
        self.conn.connection_lost(error)
        self.conn._on_stream_terminated.assert_called_with(error.value)
        self.assertIsNone(self.conn._transport)

    def test_connection_lost_connectiondone(self):
        if False:
            while True:
                i = 10
        self.conn._on_stream_terminated = mock.Mock()
        error = Failure(twisted_error.ConnectionDone())
        self.conn.connection_lost(error)
        self.assertEqual(self.conn._error, error.value)
        self.conn._on_stream_terminated.assert_called_with(None)
        self.assertIsNone(self.conn._transport)

    def test_data_received(self):
        if False:
            while True:
                i = 10
        data = b'test data'
        self.conn._on_data_available = mock.Mock()
        self.conn.data_received(data)
        self.conn._on_data_available.assert_called_once_with(data)

class TimerHandleTestCase(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.handle = mock.Mock()
        self.timer = _TimerHandle(self.handle)

    def test_cancel(self):
        if False:
            return 10
        self.timer.cancel()
        self.handle.cancel.assert_called_once()
        self.assertIsNone(self.timer._handle)

    def test_cancel_twice(self):
        if False:
            print('Hello World!')
        self.timer.cancel()
        self.timer.cancel()

    def test_cancel_already_called(self):
        if False:
            return 10
        self.handle.cancel.side_effect = twisted_error.AlreadyCalled()
        self.timer.cancel()
        self.handle.cancel.assert_called_once()

    def test_cancel_already_cancelled(self):
        if False:
            for i in range(10):
                print('nop')
        self.handle.cancel.side_effect = twisted_error.AlreadyCancelled()
        self.timer.cancel()
        self.handle.cancel.assert_called_once()