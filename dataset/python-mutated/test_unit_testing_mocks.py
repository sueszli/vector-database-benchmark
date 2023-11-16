from __future__ import absolute_import
import unittest2
from st2tests.base import BaseSensorTestCase
from st2tests.mocks.sensor import MockSensorWrapper
from st2tests.mocks.sensor import MockSensorService
from st2tests.mocks.action import MockActionWrapper
from st2tests.mocks.action import MockActionService
__all__ = ['BaseSensorTestCaseTestCase', 'MockSensorServiceTestCase', 'MockActionServiceTestCase']

class MockSensorClass(object):
    pass

class BaseMockResourceServiceTestCase(object):

    class TestCase(unittest2.TestCase):

        def test_get_user_info(self):
            if False:
                i = 10
                return i + 15
            result = self.mock_service.get_user_info()
            self.assertEqual(result['username'], 'admin')
            self.assertEqual(result['rbac']['roles'], ['admin'])

        def test_list_set_get_delete_values(self):
            if False:
                i = 10
                return i + 15
            result = self.mock_service.list_values()
            self.assertSequenceEqual(result, [])
            self.mock_service.set_value(name='t1.local', value='test1', local=True)
            self.mock_service.set_value(name='t1.global', value='test1', local=False)
            result = self.mock_service.list_values(local=True)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, 'dummy.test:t1.local')
            result = self.mock_service.list_values(local=False)
            self.assertEqual(result[0].name, 'dummy.test:t1.local')
            self.assertEqual(result[1].name, 't1.global')
            self.assertEqual(len(result), 2)
            self.assertEqual(self.mock_service.get_value('inexistent'), None)
            self.assertEqual(self.mock_service.get_value(name='t1.local', local=True), 'test1')
            self.assertEqual(len(self.mock_service.list_values(local=True)), 1)
            self.assertEqual(self.mock_service.delete_value('inexistent'), False)
            self.assertEqual(len(self.mock_service.list_values(local=True)), 1)
            self.assertEqual(self.mock_service.delete_value('t1.local'), True)
            self.assertEqual(len(self.mock_service.list_values(local=True)), 0)

class BaseSensorTestCaseTestCase(BaseSensorTestCase):
    sensor_cls = MockSensorClass

    def test_dispatch_and_assertTriggerDispatched(self):
        if False:
            for i in range(10):
                print('nop')
        sensor_service = self.sensor_service
        expected_msg = 'Trigger "nope" hasn\'t been dispatched'
        self.assertRaisesRegexp(AssertionError, expected_msg, self.assertTriggerDispatched, trigger='nope')
        sensor_service.dispatch(trigger='test1', payload={'a': 'b'})
        result = self.assertTriggerDispatched(trigger='test1')
        self.assertTrue(result)
        result = self.assertTriggerDispatched(trigger='test1', payload={'a': 'b'})
        self.assertTrue(result)
        expected_msg = 'Trigger "test1" hasn\'t been dispatched'
        self.assertRaisesRegexp(AssertionError, expected_msg, self.assertTriggerDispatched, trigger='test1', payload={'a': 'c'})

class MockSensorServiceTestCase(BaseMockResourceServiceTestCase.TestCase):

    def setUp(self):
        if False:
            return 10
        mock_sensor_wrapper = MockSensorWrapper(pack='dummy', class_name='test')
        self.mock_service = MockSensorService(sensor_wrapper=mock_sensor_wrapper)

    def test_get_logger(self):
        if False:
            i = 10
            return i + 15
        sensor_service = self.mock_service
        logger = sensor_service.get_logger('test')
        logger.info('test info')
        logger.debug('test debug')
        self.assertEqual(len(logger.method_calls), 2)
        (method_name, method_args, method_kwargs) = tuple(logger.method_calls[0])
        self.assertEqual(method_name, 'info')
        self.assertEqual(method_args, ('test info',))
        self.assertEqual(method_kwargs, {})
        (method_name, method_args, method_kwargs) = tuple(logger.method_calls[1])
        self.assertEqual(method_name, 'debug')
        self.assertEqual(method_args, ('test debug',))
        self.assertEqual(method_kwargs, {})

class MockActionServiceTestCase(BaseMockResourceServiceTestCase.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        mock_action_wrapper = MockActionWrapper(pack='dummy', class_name='test')
        self.mock_service = MockActionService(action_wrapper=mock_action_wrapper)