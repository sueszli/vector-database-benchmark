from __future__ import absolute_import
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
import os
import unittest2
import six
import mock
import eventlet
import st2tests.config as tests_config
from st2tests.base import TESTS_CONFIG_PATH
from st2common.models.db.trigger import TriggerDB
from st2reactor.container.sensor_wrapper import SensorWrapper
from st2reactor.sensor.base import Sensor, PollingSensor
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
RESOURCES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../resources'))
__all__ = ['SensorWrapperTestCase']

class SensorWrapperTestCase(unittest2.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(SensorWrapperTestCase, cls).setUpClass()
        tests_config.parse_args()

    def test_sensor_instance_has_sensor_service(self):
        if False:
            i = 10
            return i + 15
        file_path = os.path.join(RESOURCES_DIR, 'test_sensor.py')
        trigger_types = ['trigger1', 'trigger2']
        parent_args = ['--config-file', TESTS_CONFIG_PATH]
        wrapper = SensorWrapper(pack='core', file_path=file_path, class_name='TestSensor', trigger_types=trigger_types, parent_args=parent_args)
        self.assertIsNotNone(getattr(wrapper._sensor_instance, 'sensor_service', None))
        self.assertIsNotNone(getattr(wrapper._sensor_instance, 'config', None))

    def test_trigger_cud_event_handlers(self):
        if False:
            return 10
        trigger_id = '57861fcb0640fd1524e577c0'
        file_path = os.path.join(RESOURCES_DIR, 'test_sensor.py')
        trigger_types = ['trigger1', 'trigger2']
        parent_args = ['--config-file', TESTS_CONFIG_PATH]
        wrapper = SensorWrapper(pack='core', file_path=file_path, class_name='TestSensor', trigger_types=trigger_types, parent_args=parent_args)
        self.assertEqual(wrapper._trigger_names, {})
        wrapper._sensor_instance.add_trigger = mock.Mock()
        wrapper._sensor_instance.update_trigger = mock.Mock()
        wrapper._sensor_instance.remove_trigger = mock.Mock()
        self.assertEqual(wrapper._sensor_instance.add_trigger.call_count, 0)
        trigger = TriggerDB(id=trigger_id, name='test', pack='dummy', type=trigger_types[0])
        wrapper._handle_create_trigger(trigger=trigger)
        self.assertEqual(wrapper._trigger_names, {trigger_id: trigger})
        self.assertEqual(wrapper._sensor_instance.add_trigger.call_count, 1)
        self.assertEqual(wrapper._sensor_instance.update_trigger.call_count, 0)
        trigger = TriggerDB(id=trigger_id, name='test', pack='dummy', type=trigger_types[0])
        wrapper._handle_update_trigger(trigger=trigger)
        self.assertEqual(wrapper._trigger_names, {trigger_id: trigger})
        self.assertEqual(wrapper._sensor_instance.update_trigger.call_count, 1)
        self.assertEqual(wrapper._sensor_instance.remove_trigger.call_count, 0)
        trigger = TriggerDB(id=trigger_id, name='test', pack='dummy', type=trigger_types[0])
        wrapper._handle_delete_trigger(trigger=trigger)
        self.assertEqual(wrapper._trigger_names, {})
        self.assertEqual(wrapper._sensor_instance.remove_trigger.call_count, 1)

    def test_sensor_creation_passive(self):
        if False:
            print('Hello World!')
        file_path = os.path.join(RESOURCES_DIR, 'test_sensor.py')
        trigger_types = ['trigger1', 'trigger2']
        parent_args = ['--config-file', TESTS_CONFIG_PATH]
        wrapper = SensorWrapper(pack='core', file_path=file_path, class_name='TestSensor', trigger_types=trigger_types, parent_args=parent_args, db_ensure_indexes=False)
        self.assertIsInstance(wrapper._sensor_instance, Sensor)
        self.assertIsNotNone(wrapper._sensor_instance)

    def test_sensor_creation_active(self):
        if False:
            while True:
                i = 10
        file_path = os.path.join(RESOURCES_DIR, 'test_sensor.py')
        trigger_types = ['trigger1', 'trigger2']
        parent_args = ['--config-file', TESTS_CONFIG_PATH]
        poll_interval = 10
        wrapper = SensorWrapper(pack='core', file_path=file_path, class_name='TestPollingSensor', trigger_types=trigger_types, parent_args=parent_args, poll_interval=poll_interval, db_ensure_indexes=False)
        self.assertIsNotNone(wrapper._sensor_instance)
        self.assertIsInstance(wrapper._sensor_instance, PollingSensor)
        self.assertEqual(wrapper._sensor_instance._poll_interval, poll_interval)

    def test_sensor_init_fails_file_doesnt_exist(self):
        if False:
            for i in range(10):
                print('nop')
        file_path = os.path.join(RESOURCES_DIR, 'test_sensor_doesnt_exist.py')
        trigger_types = ['trigger1', 'trigger2']
        parent_args = ['--config-file', TESTS_CONFIG_PATH]
        expected_msg = 'Failed to load sensor class from file.*? No such file or directory'
        self.assertRaisesRegexp(IOError, expected_msg, SensorWrapper, pack='core', file_path=file_path, class_name='TestSensor', trigger_types=trigger_types, parent_args=parent_args)

    def test_sensor_init_fails_sensor_code_contains_typo(self):
        if False:
            print('Hello World!')
        file_path = os.path.join(RESOURCES_DIR, 'test_sensor_with_typo.py')
        trigger_types = ['trigger1', 'trigger2']
        parent_args = ['--config-file', TESTS_CONFIG_PATH]
        expected_msg = "Failed to load sensor class from file.*? 'typobar' is not defined"
        self.assertRaisesRegexp(NameError, expected_msg, SensorWrapper, pack='core', file_path=file_path, class_name='TestSensor', trigger_types=trigger_types, parent_args=parent_args)
        try:
            SensorWrapper(pack='core', file_path=file_path, class_name='TestSensor', trigger_types=trigger_types, parent_args=parent_args)
        except NameError as e:
            self.assertIn('Traceback (most recent call last)', six.text_type(e))
            self.assertIn('line 20, in <module>', six.text_type(e))
        else:
            self.fail('NameError not thrown')

    def test_sensor_wrapper_poll_method_still_works(self):
        if False:
            while True:
                i = 10
        import select
        self.assertTrue(eventlet.patcher.is_monkey_patched(select))
        self.assertTrue(select != eventlet.patcher.original('select'))
        self.assertTrue(select.poll())