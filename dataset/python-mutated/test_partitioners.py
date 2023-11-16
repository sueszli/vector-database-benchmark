from __future__ import absolute_import
from oslo_config import cfg
from st2common.constants.sensors import KVSTORE_PARTITION_LOADER, FILE_PARTITION_LOADER, HASH_PARTITION_LOADER
from st2common.models.db.keyvalue import KeyValuePairDB
from st2common.persistence.keyvalue import KeyValuePair
from st2reactor.container.partitioner_lookup import get_sensors_partitioner
from st2reactor.container.hash_partitioner import Range
from st2tests import config
from st2tests import DbTestCase
from st2tests.fixtures.generic.fixture import PACK_NAME as PACK
from st2tests.fixturesloader import FixturesLoader
FIXTURES_1 = {'sensors': ['sensor1.yaml', 'sensor2.yaml', 'sensor3.yaml']}

class PartitionerTest(DbTestCase):
    models = None

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super(PartitionerTest, cls).setUpClass()
        cls.models = FixturesLoader().save_fixtures_to_db(fixtures_pack=PACK, fixtures_dict=FIXTURES_1)
        config.parse_args()

    def test_default_partitioner(self):
        if False:
            while True:
                i = 10
        provider = get_sensors_partitioner()
        sensors = provider.get_sensors()
        self.assertEqual(len(sensors), len(FIXTURES_1['sensors']), 'Failed to provider all sensors')
        sensor1 = self.models['sensors']['sensor1.yaml']
        self.assertTrue(provider.is_sensor_owner(sensor1))

    def test_kvstore_partitioner(self):
        if False:
            while True:
                i = 10
        cfg.CONF.set_override(name='partition_provider', override={'name': KVSTORE_PARTITION_LOADER}, group='sensorcontainer')
        kvp = KeyValuePairDB(**{'name': 'sensornode1.sensor_partition', 'value': 'generic.Sensor1, generic.Sensor2'})
        KeyValuePair.add_or_update(kvp, publish=False, dispatch_trigger=False)
        provider = get_sensors_partitioner()
        sensors = provider.get_sensors()
        self.assertEqual(len(sensors), len(kvp.value.split(',')))
        sensor1 = self.models['sensors']['sensor1.yaml']
        self.assertTrue(provider.is_sensor_owner(sensor1))
        sensor3 = self.models['sensors']['sensor3.yaml']
        self.assertFalse(provider.is_sensor_owner(sensor3))

    def test_file_partitioner(self):
        if False:
            for i in range(10):
                print('nop')
        partition_file = FixturesLoader().get_fixture_file_path_abs(fixtures_pack=PACK, fixtures_type='sensors', fixture_name='partition_file.yaml')
        cfg.CONF.set_override(name='partition_provider', override={'name': FILE_PARTITION_LOADER, 'partition_file': partition_file}, group='sensorcontainer')
        provider = get_sensors_partitioner()
        sensors = provider.get_sensors()
        self.assertEqual(len(sensors), 2)
        sensor1 = self.models['sensors']['sensor1.yaml']
        self.assertTrue(provider.is_sensor_owner(sensor1))
        sensor3 = self.models['sensors']['sensor3.yaml']
        self.assertFalse(provider.is_sensor_owner(sensor3))

    def test_hash_partitioner(self):
        if False:
            i = 10
            return i + 15
        cfg.CONF.set_override(name='partition_provider', override={'name': HASH_PARTITION_LOADER, 'hash_ranges': '%s..%s' % (Range.RANGE_MIN_ENUM, Range.RANGE_MAX_ENUM)}, group='sensorcontainer')
        provider = get_sensors_partitioner()
        sensors = provider.get_sensors()
        self.assertEqual(len(sensors), 3)
        sensor1 = self.models['sensors']['sensor1.yaml']
        self.assertTrue(provider.is_sensor_owner(sensor1))
        sensor2 = self.models['sensors']['sensor2.yaml']
        self.assertTrue(provider.is_sensor_owner(sensor2))
        sensor3 = self.models['sensors']['sensor3.yaml']
        self.assertTrue(provider.is_sensor_owner(sensor3))