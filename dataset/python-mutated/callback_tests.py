"""
Tests for pika.callback

"""
import logging
import unittest
from unittest import mock
from pika import callback, frame, spec

class CallbackTests(unittest.TestCase):
    KEY = 'Test Key'
    ARGUMENTS = callback.CallbackManager.ARGUMENTS
    CALLS = callback.CallbackManager.CALLS
    CALLBACK = callback.CallbackManager.CALLBACK
    ONE_SHOT = callback.CallbackManager.ONE_SHOT
    ONLY_CALLER = callback.CallbackManager.ONLY_CALLER
    PREFIX_CLASS = spec.Basic.Consume
    PREFIX = 'Basic.Consume'
    ARGUMENTS_VALUE = {'foo': 'bar'}

    @property
    def _callback_dict(self):
        if False:
            i = 10
            return i + 15
        return {self.CALLBACK: self.callback_mock, self.ONE_SHOT: True, self.ONLY_CALLER: self.mock_caller, self.ARGUMENTS: self.ARGUMENTS_VALUE, self.CALLS: 1}

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.obj = callback.CallbackManager()
        self.callback_mock = mock.Mock()
        self.mock_caller = mock.Mock()

    def tearDown(self):
        if False:
            print('Hello World!')
        del self.obj
        del self.callback_mock
        del self.mock_caller

    def test_initialization(self):
        if False:
            i = 10
            return i + 15
        obj = callback.CallbackManager()
        self.assertDictEqual(obj._stack, {})

    def test_name_or_value_method_object(self):
        if False:
            for i in range(10):
                print('nop')
        value = spec.Basic.Consume()
        self.assertEqual(callback.name_or_value(value), self.PREFIX)

    def test_name_or_value_basic_consume_object(self):
        if False:
            while True:
                i = 10
        self.assertEqual(callback.name_or_value(spec.Basic.Consume()), self.PREFIX)

    def test_name_or_value_amqpobject_class(self):
        if False:
            while True:
                i = 10
        self.assertEqual(callback.name_or_value(self.PREFIX_CLASS), self.PREFIX)

    def test_name_or_value_protocol_header(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(callback.name_or_value(frame.ProtocolHeader()), 'ProtocolHeader')

    def test_name_or_value_method_frame(self):
        if False:
            print('Hello World!')
        value = frame.Method(1, self.PREFIX_CLASS())
        self.assertEqual(callback.name_or_value(value), self.PREFIX)

    def test_name_or_value_str(self):
        if False:
            while True:
                i = 10
        value = 'Test String Value'
        expectation = value
        self.assertEqual(callback.name_or_value(value), expectation)

    def test_name_or_value_unicode(self):
        if False:
            i = 10
            return i + 15
        value = u'Это тест значения'
        expectation = 'Это тест значения'
        self.assertEqual(callback.name_or_value(value), expectation)

    def test_empty_callbacks_on_init(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.obj._stack)

    def test_sanitize_decorator_with_args_only(self):
        if False:
            return 10
        self.obj.add(self.PREFIX_CLASS, self.KEY, None)
        self.assertIn(self.PREFIX, self.obj._stack.keys())

    def test_sanitize_decorator_with_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add(prefix=self.PREFIX_CLASS, key=self.KEY, callback=None)
        self.assertIn(self.PREFIX, self.obj._stack.keys())

    def test_sanitize_decorator_with_mixed_args_and_kwargs(self):
        if False:
            return 10
        self.obj.add(self.PREFIX_CLASS, key=self.KEY, callback=None)
        self.assertIn(self.PREFIX, self.obj._stack.keys())

    def test_add_first_time_prefix_added(self):
        if False:
            print('Hello World!')
        self.obj.add(self.PREFIX, self.KEY, None)
        self.assertIn(self.PREFIX, self.obj._stack)

    def test_add_first_time_key_added(self):
        if False:
            print('Hello World!')
        self.obj.add(self.PREFIX, self.KEY, None)
        self.assertIn(self.KEY, self.obj._stack[self.PREFIX])

    def test_add_first_time_callback_added(self):
        if False:
            while True:
                i = 10
        self.obj.add(self.PREFIX, self.KEY, self.callback_mock)
        self.assertEqual(self.callback_mock, self.obj._stack[self.PREFIX][self.KEY][0][self.CALLBACK])

    def test_add_oneshot_default_is_true(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add(self.PREFIX, self.KEY, None)
        self.assertTrue(self.obj._stack[self.PREFIX][self.KEY][0][self.ONE_SHOT])

    def test_add_oneshot_is_false(self):
        if False:
            i = 10
            return i + 15
        self.obj.add(self.PREFIX, self.KEY, None, False)
        self.assertFalse(self.obj._stack[self.PREFIX][self.KEY][0][self.ONE_SHOT])

    def test_add_only_caller_default_is_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add(self.PREFIX, self.KEY, None)
        self.assertFalse(self.obj._stack[self.PREFIX][self.KEY][0][self.ONLY_CALLER])

    def test_add_only_caller_true(self):
        if False:
            i = 10
            return i + 15
        self.obj.add(self.PREFIX, self.KEY, None, only_caller=True)
        self.assertTrue(self.obj._stack[self.PREFIX][self.KEY][0][self.ONLY_CALLER])

    def test_add_returns_prefix_value_and_key(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.obj.add(self.PREFIX, self.KEY, None), (self.PREFIX, self.KEY))

    def test_add_duplicate_callback(self):
        if False:
            return 10
        mock_callback = mock.Mock()

        def add_callback():
            if False:
                while True:
                    i = 10
            self.obj.add(self.PREFIX, self.KEY, mock_callback, False)
        with mock.patch('pika.callback.LOGGER', spec=logging.Logger) as logger:
            logger.warning = mock.Mock()
            add_callback()
            add_callback()
            logger.warning.assert_called_once_with(callback.CallbackManager.DUPLICATE_WARNING, self.PREFIX, self.KEY)

    def test_add_duplicate_callback_returns_prefix_value_and_key(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add(self.PREFIX, self.KEY, None)
        self.assertEqual(self.obj.add(self.PREFIX, self.KEY, None), (self.PREFIX, self.KEY))

    def test_clear(self):
        if False:
            i = 10
            return i + 15
        self.obj.add(self.PREFIX, self.KEY, None)
        self.obj.clear()
        self.assertDictEqual(self.obj._stack, dict())

    def test_cleanup_removes_prefix(self):
        if False:
            while True:
                i = 10
        other_prefix = 'Foo'
        self.obj.add(self.PREFIX, self.KEY, None)
        self.obj.add(other_prefix, 'Bar', None)
        self.obj.cleanup(self.PREFIX)
        self.assertNotIn(self.PREFIX, self.obj._stack)

    def test_cleanup_keeps_other_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        other_prefix = 'Foo'
        self.obj.add(self.PREFIX, self.KEY, None)
        self.obj.add(other_prefix, 'Bar', None)
        self.obj.cleanup(self.PREFIX)
        self.assertIn(other_prefix, self.obj._stack)

    def test_cleanup_returns_true(self):
        if False:
            while True:
                i = 10
        self.obj.add(self.PREFIX, self.KEY, None)
        self.assertTrue(self.obj.cleanup(self.PREFIX))

    def test_missing_prefix(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.obj.cleanup(self.PREFIX))

    def test_pending_none(self):
        if False:
            print('Hello World!')
        self.assertIsNone(self.obj.pending(self.PREFIX_CLASS, self.KEY))

    def test_pending_one(self):
        if False:
            return 10
        self.obj.add(self.PREFIX, self.KEY, None)
        self.assertEqual(self.obj.pending(self.PREFIX_CLASS, self.KEY), 1)

    def test_pending_two(self):
        if False:
            print('Hello World!')
        self.obj.add(self.PREFIX, self.KEY, None)
        self.obj.add(self.PREFIX, self.KEY, lambda x: True)
        self.assertEqual(self.obj.pending(self.PREFIX_CLASS, self.KEY), 2)

    def test_process_callback_false(self):
        if False:
            while True:
                i = 10
        self.obj._stack = dict()
        self.assertFalse(self.obj.process('FAIL', 'False', 'Empty', self.mock_caller, []))

    def test_process_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.obj.process(self.PREFIX_CLASS, self.KEY, self))

    def test_process_true(self):
        if False:
            print('Hello World!')
        self.obj.add(self.PREFIX, self.KEY, self.callback_mock)
        self.assertTrue(self.obj.process(self.PREFIX_CLASS, self.KEY, self))

    def test_process_mock_called(self):
        if False:
            for i in range(10):
                print('nop')
        args = (1, None, 'Hi')
        self.obj.add(self.PREFIX, self.KEY, self.callback_mock)
        self.obj.process(self.PREFIX, self.KEY, self, args)
        self.callback_mock.assert_called_once_with(args)

    def test_process_one_shot_removed(self):
        if False:
            while True:
                i = 10
        args = (1, None, 'Hi')
        self.obj.add(self.PREFIX, self.KEY, self.callback_mock)
        self.obj.process(self.PREFIX, self.KEY, self, args)
        self.assertNotIn(self.PREFIX, self.obj._stack)

    def test_process_non_one_shot_prefix_not_removed(self):
        if False:
            i = 10
            return i + 15
        self.obj.add(self.PREFIX, self.KEY, self.callback_mock, one_shot=False)
        self.obj.process(self.PREFIX, self.KEY, self)
        self.assertIn(self.PREFIX, self.obj._stack)

    def test_process_non_one_shot_key_not_removed(self):
        if False:
            return 10
        self.obj.add(self.PREFIX, self.KEY, self.callback_mock, one_shot=False)
        self.obj.process(self.PREFIX, self.KEY, self)
        self.assertIn(self.KEY, self.obj._stack[self.PREFIX])

    def test_process_non_one_shot_callback_not_removed(self):
        if False:
            return 10
        self.obj.add(self.PREFIX, self.KEY, self.callback_mock, one_shot=False)
        self.obj.process(self.PREFIX, self.KEY, self)
        self.assertEqual(self.obj._stack[self.PREFIX][self.KEY][0][self.CALLBACK], self.callback_mock)

    def test_process_only_caller_fails(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock, only_caller=self.mock_caller)
        self.obj.process(self.PREFIX_CLASS, self.KEY, self)
        self.assertFalse(self.callback_mock.called)

    def test_process_only_caller_fails_no_removal(self):
        if False:
            while True:
                i = 10
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock, only_caller=self.mock_caller)
        self.obj.process(self.PREFIX_CLASS, self.KEY, self)
        self.assertEqual(self.obj._stack[self.PREFIX][self.KEY][0][self.CALLBACK], self.callback_mock)

    def test_remove_with_no_callbacks_pending(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj = callback.CallbackManager()
        self.assertFalse(self.obj.remove(self.PREFIX, self.KEY, self.callback_mock))

    def test_remove_with_callback_true(self):
        if False:
            i = 10
            return i + 15
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        self.assertTrue(self.obj.remove(self.PREFIX, self.KEY, self.callback_mock))

    def test_remove_with_callback_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add(self.PREFIX_CLASS, self.KEY, None)
        self.assertTrue(self.obj.remove(self.PREFIX, self.KEY, self.callback_mock))

    def test_remove_with_callback_true_empty_stack(self):
        if False:
            i = 10
            return i + 15
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        self.obj.remove(prefix=self.PREFIX, key=self.KEY, callback_value=self.callback_mock)
        self.assertDictEqual(self.obj._stack, dict())

    def test_remove_with_callback_true_non_empty_stack(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.mock_caller)
        self.obj.remove(self.PREFIX, self.KEY, self.callback_mock)
        self.assertEqual(self.mock_caller, self.obj._stack[self.PREFIX][self.KEY][0][self.CALLBACK])

    def test_remove_prefix_key_with_other_key_prefix_remains(self):
        if False:
            while True:
                i = 10
        other_key = 'Other Key'
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        self.obj.add(self.PREFIX_CLASS, other_key, self.mock_caller)
        self.obj.remove(self.PREFIX, self.KEY, self.callback_mock)
        self.assertIn(self.PREFIX, self.obj._stack)

    def test_remove_prefix_key_with_other_key_remains(self):
        if False:
            return 10
        other_key = 'Other Key'
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        self.obj.add(prefix=self.PREFIX_CLASS, key=other_key, callback=self.mock_caller)
        self.obj.remove(self.PREFIX, self.KEY)
        self.assertIn(other_key, self.obj._stack[self.PREFIX])

    def test_remove_prefix_key_with_other_key_callback_remains(self):
        if False:
            print('Hello World!')
        other_key = 'Other Key'
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        self.obj.add(self.PREFIX_CLASS, other_key, self.mock_caller)
        self.obj.remove(self.PREFIX, self.KEY)
        self.assertEqual(self.mock_caller, self.obj._stack[self.PREFIX][other_key][0][self.CALLBACK])

    def test_remove_all(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        self.obj.remove_all(self.PREFIX, self.KEY)
        self.assertNotIn(self.PREFIX, self.obj._stack)

    def test_should_process_callback_true(self):
        if False:
            print('Hello World!')
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        value = self.obj._callback_dict(self.callback_mock, False, None, None)
        self.assertTrue(self.obj._should_process_callback(value, self.mock_caller, []))

    def test_should_process_callback_false_argument_fail(self):
        if False:
            return 10
        self.obj.clear()
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock, arguments={'foo': 'baz'})
        self.assertFalse(self.obj._should_process_callback(self._callback_dict, self.mock_caller, [{'foo': 'baz'}]))

    def test_should_process_callback_false_only_caller_failure(self):
        if False:
            return 10
        self.obj.add(self.PREFIX_CLASS, self.KEY, self.callback_mock)
        value = self.obj._callback_dict(self.callback_mock, False, self.mock_caller, None)
        self.assertTrue(self.obj._should_process_callback(value, self.mock_caller, []))

    def test_dict(self):
        if False:
            return 10
        self.assertDictEqual(self.obj._callback_dict(self.callback_mock, True, self.mock_caller, self.ARGUMENTS_VALUE), self._callback_dict)

    def test_arguments_match_no_arguments(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(self.obj._arguments_match(self._callback_dict, []))

    def test_arguments_match_dict_argument(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.obj._arguments_match(self._callback_dict, [self.ARGUMENTS_VALUE]))

    def test_arguments_match_dict_argument_no_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(self.obj._arguments_match(self._callback_dict, [{}]))

    def test_arguments_match_dict_argument_no_match(self):
        if False:
            return 10
        self.assertFalse(self.obj._arguments_match(self._callback_dict, [{'foo': 'baz'}]))

    def test_arguments_match_obj_argument(self):
        if False:
            for i in range(10):
                print('nop')

        class TestObj(object):
            foo = 'bar'
        test_instance = TestObj()
        self.assertTrue(self.obj._arguments_match(self._callback_dict, [test_instance]))

    def test_arguments_match_obj_no_attribute(self):
        if False:
            while True:
                i = 10

        class TestObj(object):
            qux = 'bar'
        test_instance = TestObj()
        self.assertFalse(self.obj._arguments_match(self._callback_dict, [test_instance]))

    def test_arguments_match_obj_argument_no_match(self):
        if False:
            for i in range(10):
                print('nop')

        class TestObj(object):
            foo = 'baz'
        test_instance = TestObj()
        self.assertFalse(self.obj._arguments_match(self._callback_dict, [test_instance]))

    def test_arguments_match_obj_argument_with_method(self):
        if False:
            return 10

        class TestFrame(object):
            method = None

        class MethodObj(object):
            foo = 'bar'
        test_instance = TestFrame()
        test_instance.method = MethodObj()
        self.assertTrue(self.obj._arguments_match(self._callback_dict, [test_instance]))

    def test_arguments_match_obj_argument_with_method_no_match(self):
        if False:
            while True:
                i = 10

        class TestFrame(object):
            method = None

        class MethodObj(object):
            foo = 'baz'
        test_instance = TestFrame()
        test_instance.method = MethodObj()
        self.assertFalse(self.obj._arguments_match(self._callback_dict, [test_instance]))