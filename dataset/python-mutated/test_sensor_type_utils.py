from __future__ import absolute_import
import mock
import unittest2
from st2common.models.api.sensor import SensorTypeAPI
from st2common.models.utils import sensor_type_utils

class SensorTypeUtilsTestCase(unittest2.TestCase):

    def test_to_sensor_db_model_no_trigger_types(self):
        if False:
            i = 10
            return i + 15
        sensor_meta = {'artifact_uri': 'file:///data/st2contrib/packs/jira/sensors/jira_sensor.py', 'class_name': 'JIRASensor', 'pack': 'jira'}
        sensor_api = SensorTypeAPI(**sensor_meta)
        sensor_model = SensorTypeAPI.to_model(sensor_api)
        self.assertEqual(sensor_model.name, sensor_meta['class_name'])
        self.assertEqual(sensor_model.pack, sensor_meta['pack'])
        self.assertEqual(sensor_model.artifact_uri, sensor_meta['artifact_uri'])
        self.assertListEqual(sensor_model.trigger_types, [])

    @mock.patch.object(sensor_type_utils, 'create_trigger_types', mock.MagicMock(return_value=['mock.trigger_ref']))
    def test_to_sensor_db_model_with_trigger_types(self):
        if False:
            i = 10
            return i + 15
        sensor_meta = {'artifact_uri': 'file:///data/st2contrib/packs/jira/sensors/jira_sensor.py', 'class_name': 'JIRASensor', 'pack': 'jira', 'trigger_types': [{'pack': 'jira', 'name': 'issue_created', 'parameters': {}}]}
        sensor_api = SensorTypeAPI(**sensor_meta)
        sensor_model = SensorTypeAPI.to_model(sensor_api)
        self.assertListEqual(sensor_model.trigger_types, ['mock.trigger_ref'])

    def test_get_sensor_entry_point(self):
        if False:
            while True:
                i = 10
        file_path = 'file:///data/st/st2reactor/st2reactor/' + 'contrib/sensors/st2_generic_webhook_sensor.py'
        class_name = 'St2GenericWebhooksSensor'
        sensor = {'artifact_uri': file_path, 'class_name': class_name, 'pack': 'core'}
        sensor_api = SensorTypeAPI(**sensor)
        entry_point = sensor_type_utils.get_sensor_entry_point(sensor_api)
        self.assertEqual(entry_point, class_name)
        file_path = 'file:///data/st2contrib/packs/jira/sensors/jira_sensor.py'
        class_name = 'JIRASensor'
        sensor = {'artifact_uri': file_path, 'class_name': class_name, 'pack': 'jira'}
        sensor_api = SensorTypeAPI(**sensor)
        entry_point = sensor_type_utils.get_sensor_entry_point(sensor_api)
        self.assertEqual(entry_point, 'sensors.jira_sensor.JIRASensor')
        file_path = 'file:///data/st2contrib/packs/docker/sensors/docker_container_sensor.py'
        class_name = 'DockerSensor'
        sensor = {'artifact_uri': file_path, 'class_name': class_name, 'pack': 'docker'}
        sensor_api = SensorTypeAPI(**sensor)
        entry_point = sensor_type_utils.get_sensor_entry_point(sensor_api)
        self.assertEqual(entry_point, 'sensors.docker_container_sensor.DockerSensor')