from __future__ import absolute_import
import mock
import unittest2
from oslo_config import cfg
from st2reactor.container.sensor_wrapper import SensorService
from st2common.constants.keyvalue import SYSTEM_SCOPE
from st2common.constants.keyvalue import USER_SCOPE
__all__ = ['SensorServiceTestCase']
TEST_SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {'age': {'type': 'integer'}, 'name': {'type': 'string', 'required': True}, 'address': {'type': 'string', 'default': '-'}, 'career': {'type': 'array'}, 'married': {'type': 'boolean'}, 'awards': {'type': 'object'}, 'income': {'anyOf': [{'type': 'integer'}, {'type': 'string'}]}}}

class TriggerTypeDBMock(object):

    def __init__(self, schema=None):
        if False:
            print('Hello World!')
        self.payload_schema = schema or {}

class TriggerDBMock(object):

    def __init__(self, type=None):
        if False:
            while True:
                i = 10
        self.type = type

class SensorServiceTestCase(unittest2.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')

        def side_effect(trigger, payload, trace_context):
            if False:
                return 10
            self._dispatched_count += 1
        self.sensor_service = SensorService(mock.MagicMock())
        self.sensor_service._trigger_dispatcher_service._dispatcher = mock.Mock()
        self.sensor_service._trigger_dispatcher_service._dispatcher.dispatch = mock.MagicMock(side_effect=side_effect)
        self._dispatched_count = 0
        self.validate_trigger_payload = cfg.CONF.system.validate_trigger_payload

    def tearDown(self):
        if False:
            return 10
        cfg.CONF.system.validate_trigger_payload = self.validate_trigger_payload

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock(TEST_SCHEMA)))
    def test_dispatch_success_valid_payload_validation_enabled(self):
        if False:
            print('Hello World!')
        cfg.CONF.system.validate_trigger_payload = True
        payload = {'name': 'John Doe', 'age': 25, 'career': ['foo, Inc.', 'bar, Inc.'], 'married': True, 'awards': {'2016': ['hoge prize', 'fuga prize']}, 'income': 50000}
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 1)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock(TEST_SCHEMA)))
    @mock.patch('st2common.services.triggers.get_trigger_db_by_ref', mock.MagicMock(return_value=TriggerDBMock(type='trigger-type-ref')))
    def test_dispatch_success_with_validation_enabled_trigger_reference(self):
        if False:
            for i in range(10):
                print('nop')
        cfg.CONF.system.validate_trigger_payload = True
        payload = {'name': 'John Doe', 'age': 25, 'career': ['foo, Inc.', 'bar, Inc.'], 'married': True, 'awards': {'2016': ['hoge prize', 'fuga prize']}, 'income': 50000}
        self.assertEqual(self._dispatched_count, 0)
        self.sensor_service.dispatch('pack.86582f21-1fbc-44ea-88cb-0cd2b610e93b', payload)
        self.assertEqual(self._dispatched_count, 1)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock(TEST_SCHEMA)))
    def test_dispatch_success_with_validation_disabled_and_invalid_payload(self):
        if False:
            print('Hello World!')
        "\n        Tests that an invalid payload still results in dispatch success with default config\n\n        The previous config defition used StrOpt instead of BoolOpt for\n        cfg.CONF.system.validate_trigger_payload. This meant that even though the intention\n        was to bypass validation, the fact that this option was a string, meant it always\n        resulted in True during conditionals.the\n\n        However, the other unit tests directly modified\n        cfg.CONF.system.validate_trigger_payload before running, which\n        obscured this bug during testing.\n\n        This test (as well as resetting cfg.CONF.system.validate_trigger_payload\n        to it's original value during tearDown) will test validation does\n        NOT take place with the default configuration.\n        "
        cfg.CONF.system.validate_trigger_payload = False
        payload = {'name': 'John Doe', 'age': '25'}
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 1)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock(TEST_SCHEMA)))
    def test_dispatch_failure_caused_by_incorrect_type(self):
        if False:
            print('Hello World!')
        payload = {'name': 'John Doe', 'age': '25'}
        cfg.CONF.system.validate_trigger_payload = True
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 0)
        cfg.CONF.system.validate_trigger_payload = False
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 1)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock(TEST_SCHEMA)))
    def test_dispatch_failure_caused_by_lack_of_required_parameter(self):
        if False:
            while True:
                i = 10
        payload = {'age': 25}
        cfg.CONF.system.validate_trigger_payload = True
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 0)
        cfg.CONF.system.validate_trigger_payload = False
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 1)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock(TEST_SCHEMA)))
    def test_dispatch_failure_caused_by_extra_parameter(self):
        if False:
            i = 10
            return i + 15
        payload = {'name': 'John Doe', 'hobby': 'programming'}
        cfg.CONF.system.validate_trigger_payload = True
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 0)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock(TEST_SCHEMA)))
    def test_dispatch_success_with_multiple_type_value(self):
        if False:
            return 10
        payload = {'name': 'John Doe', 'income': 1234}
        cfg.CONF.system.validate_trigger_payload = True
        self.sensor_service.dispatch('trigger-name', payload)
        payload['income'] = 'secret'
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 2)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock(TEST_SCHEMA)))
    def test_dispatch_success_with_null(self):
        if False:
            for i in range(10):
                print('nop')
        payload = {'name': 'John Doe', 'age': None}
        cfg.CONF.system.validate_trigger_payload = True
        self.sensor_service.dispatch('trigger-name', payload)
        self.assertEqual(self._dispatched_count, 1)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=TriggerTypeDBMock()))
    def test_dispatch_success_without_payload_schema(self):
        if False:
            i = 10
            return i + 15
        self.sensor_service.dispatch('trigger-name', {})
        self.assertEqual(self._dispatched_count, 1)

    @mock.patch('st2common.services.triggers.get_trigger_type_db', mock.MagicMock(return_value=None))
    def test_dispatch_trigger_type_not_in_db_should_not_dispatch(self):
        if False:
            return 10
        cfg.CONF.system.validate_trigger_payload = True
        self.sensor_service.dispatch('not-in-database-ref', {})
        self.assertEqual(self._dispatched_count, 0)

    def test_datastore_methods(self):
        if False:
            print('Hello World!')
        self.sensor_service._datastore_service = mock.Mock()
        self.sensor_service.get_value(name='foo1', scope=SYSTEM_SCOPE, decrypt=True)
        call_kwargs = self.sensor_service.datastore_service.get_value.call_args[1]
        expected_kwargs = {'name': 'foo1', 'local': True, 'scope': SYSTEM_SCOPE, 'decrypt': True}
        self.assertEqual(call_kwargs, expected_kwargs)
        self.sensor_service.set_value(name='foo2', value='bar', scope=USER_SCOPE, encrypt=True)
        call_kwargs = self.sensor_service.datastore_service.set_value.call_args[1]
        expected_kwargs = {'name': 'foo2', 'value': 'bar', 'ttl': None, 'local': True, 'scope': USER_SCOPE, 'encrypt': True}
        self.assertEqual(call_kwargs, expected_kwargs)
        self.sensor_service.delete_value(name='foo3', scope=USER_SCOPE)
        call_kwargs = self.sensor_service.datastore_service.delete_value.call_args[1]
        expected_kwargs = {'name': 'foo3', 'local': True, 'scope': USER_SCOPE}
        self.assertEqual(call_kwargs, expected_kwargs)