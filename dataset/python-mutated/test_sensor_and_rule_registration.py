from __future__ import absolute_import
import mock
from st2tests import DbTestCase
from st2common.persistence.rule import Rule
from st2common.persistence.sensor import SensorType
from st2common.persistence.trigger import Trigger
from st2common.persistence.trigger import TriggerType
from st2common.transport.publishers import PoolPublisher
from st2common.bootstrap.sensorsregistrar import SensorsRegistrar
from st2common.bootstrap.rulesregistrar import RulesRegistrar
from tests.fixtures.packs import PACKS_DIR
from tests.fixtures.packs.pack_with_rules.fixture import PACK_PATH as PACK_WITH_RULES_PATH
from tests.fixtures.packs.pack_with_sensor.fixture import PACK_PATH as PACK_WITH_SENSOR_PATH
__all__ = ['SensorRegistrationTestCase', 'RuleRegistrationTestCase']

@mock.patch('st2common.content.utils.get_pack_base_path', mock.Mock(return_value=PACK_WITH_SENSOR_PATH))
class SensorRegistrationTestCase(DbTestCase):

    @mock.patch.object(PoolPublisher, 'publish', mock.MagicMock())
    def test_register_sensors(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(SensorType.get_all()), 0)
        self.assertEqual(len(TriggerType.get_all()), 0)
        self.assertEqual(len(Trigger.get_all()), 0)
        registrar = SensorsRegistrar()
        registrar.register_from_packs(base_dirs=[PACKS_DIR])
        sensor_dbs = SensorType.get_all()
        trigger_type_dbs = TriggerType.get_all()
        trigger_dbs = Trigger.get_all()
        self.assertEqual(len(sensor_dbs), 2)
        self.assertEqual(len(trigger_type_dbs), 2)
        self.assertEqual(len(trigger_dbs), 2)
        self.assertEqual(sensor_dbs[0].name, 'TestSensor')
        self.assertEqual(sensor_dbs[0].poll_interval, 10)
        self.assertTrue(sensor_dbs[0].enabled)
        self.assertEqual(sensor_dbs[0].metadata_file, 'sensors/test_sensor_1.yaml')
        self.assertEqual(sensor_dbs[1].name, 'TestSensorDisabled')
        self.assertEqual(sensor_dbs[1].poll_interval, 10)
        self.assertFalse(sensor_dbs[1].enabled)
        self.assertEqual(sensor_dbs[1].metadata_file, 'sensors/test_sensor_2.yaml')
        self.assertEqual(trigger_type_dbs[0].name, 'trigger_type_1')
        self.assertEqual(trigger_type_dbs[0].pack, 'pack_with_sensor')
        self.assertEqual(len(trigger_type_dbs[0].tags), 0)
        self.assertEqual(trigger_type_dbs[1].name, 'trigger_type_2')
        self.assertEqual(trigger_type_dbs[1].pack, 'pack_with_sensor')
        self.assertEqual(len(trigger_type_dbs[1].tags), 2)
        self.assertEqual(trigger_type_dbs[1].tags[0].name, 'tag1name')
        self.assertEqual(trigger_type_dbs[1].tags[0].value, 'tag1 value')
        self.assertEqual(trigger_type_dbs[0].metadata_file, 'sensors/test_sensor_1.yaml')
        self.assertEqual(trigger_type_dbs[1].metadata_file, 'sensors/test_sensor_1.yaml')
        registrar.register_from_packs(base_dirs=[PACKS_DIR])
        sensor_dbs = SensorType.get_all()
        trigger_type_dbs = TriggerType.get_all()
        trigger_dbs = Trigger.get_all()
        self.assertEqual(len(sensor_dbs), 2)
        self.assertEqual(len(trigger_type_dbs), 2)
        self.assertEqual(len(trigger_dbs), 2)
        self.assertEqual(sensor_dbs[0].name, 'TestSensor')
        self.assertEqual(sensor_dbs[0].poll_interval, 10)
        self.assertEqual(trigger_type_dbs[0].name, 'trigger_type_1')
        self.assertEqual(trigger_type_dbs[0].pack, 'pack_with_sensor')
        self.assertEqual(trigger_type_dbs[1].name, 'trigger_type_2')
        self.assertEqual(trigger_type_dbs[1].pack, 'pack_with_sensor')
        original_load = registrar._meta_loader.load

        def mock_load(*args, **kwargs):
            if False:
                return 10
            data = original_load(*args, **kwargs)
            data['poll_interval'] = 50
            data['trigger_types'][1]['description'] = 'test 2'
            return data
        registrar._meta_loader.load = mock_load
        registrar.register_from_packs(base_dirs=[PACKS_DIR])
        sensor_dbs = SensorType.get_all()
        trigger_type_dbs = TriggerType.get_all()
        trigger_dbs = Trigger.get_all()
        self.assertEqual(len(sensor_dbs), 2)
        self.assertEqual(len(trigger_type_dbs), 2)
        self.assertEqual(len(trigger_dbs), 2)
        self.assertEqual(sensor_dbs[0].name, 'TestSensor')
        self.assertEqual(sensor_dbs[0].poll_interval, 50)
        self.assertEqual(trigger_type_dbs[0].name, 'trigger_type_1')
        self.assertEqual(trigger_type_dbs[0].pack, 'pack_with_sensor')
        self.assertEqual(trigger_type_dbs[1].name, 'trigger_type_2')
        self.assertEqual(trigger_type_dbs[1].pack, 'pack_with_sensor')
        self.assertEqual(trigger_type_dbs[1].description, 'test 2')

@mock.patch('st2common.content.utils.get_pack_base_path', mock.Mock(return_value=PACK_WITH_RULES_PATH))
class RuleRegistrationTestCase(DbTestCase):

    def test_register_rules(self):
        if False:
            return 10
        self.assertEqual(len(Rule.get_all()), 0)
        self.assertEqual(len(Trigger.get_all()), 0)
        registrar = RulesRegistrar()
        registrar.register_from_packs(base_dirs=[PACKS_DIR])
        rule_dbs = Rule.get_all()
        trigger_dbs = Trigger.get_all()
        self.assertEqual(len(rule_dbs), 2)
        self.assertEqual(len(trigger_dbs), 1)
        self.assertEqual(rule_dbs[0].name, 'sample.with_the_same_timer')
        self.assertEqual(rule_dbs[1].name, 'sample.with_timer')
        self.assertIsNotNone(trigger_dbs[0].name)
        registrar.register_from_packs(base_dirs=[PACKS_DIR])
        rule_dbs = Rule.get_all()
        trigger_dbs = Trigger.get_all()
        self.assertEqual(len(rule_dbs), 2)
        self.assertEqual(len(trigger_dbs), 1)