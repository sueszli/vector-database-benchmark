import copy
import six
import st2common.bootstrap.sensorsregistrar as sensors_registrar
from st2api.controllers.v1.sensors import SensorTypeController
from st2tests.api import FunctionalTest
from st2tests.api import APIControllerWithIncludeAndExcludeFilterTestCase
http_client = six.moves.http_client
__all__ = ['SensorTypeControllerTestCase']

class SensorTypeControllerTestCase(FunctionalTest, APIControllerWithIncludeAndExcludeFilterTestCase):
    get_all_path = '/v1/sensortypes'
    controller_cls = SensorTypeController
    include_attribute_field_name = 'entry_point'
    exclude_attribute_field_name = 'artifact_uri'
    test_exact_object_count = False

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(SensorTypeControllerTestCase, cls).setUpClass()
        sensors_registrar.register_sensors(use_pack_cache=False)

    def test_get_all_and_minus_one(self):
        if False:
            print('Hello World!')
        resp = self.app.get('/v1/sensortypes')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertEqual(len(resp.json), 3)
        self.assertEqual(resp.json[0]['name'], 'SampleSensor')
        resp = self.app.get('/v1/sensortypes/?limit=-1')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertEqual(len(resp.json), 3)
        self.assertEqual(resp.json[0]['name'], 'SampleSensor')

    def test_get_all_negative_limit(self):
        if False:
            return 10
        resp = self.app.get('/v1/sensortypes/?limit=-22', expect_errors=True)
        self.assertEqual(resp.status_int, 400)
        self.assertEqual(resp.json['faultstring'], 'Limit, "-22" specified, must be a positive number.')

    def test_get_all_filters(self):
        if False:
            print('Hello World!')
        resp = self.app.get('/v1/sensortypes')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertEqual(len(resp.json), 3)
        resp = self.app.get('/v1/sensortypes?name=foobar')
        self.assertEqual(len(resp.json), 0)
        resp = self.app.get('/v1/sensortypes?name=SampleSensor2')
        self.assertEqual(len(resp.json), 1)
        self.assertEqual(resp.json[0]['name'], 'SampleSensor2')
        self.assertEqual(resp.json[0]['ref'], 'dummy_pack_1.SampleSensor2')
        resp = self.app.get('/v1/sensortypes?name=SampleSensor3')
        self.assertEqual(len(resp.json), 1)
        self.assertEqual(resp.json[0]['name'], 'SampleSensor3')
        resp = self.app.get('/v1/sensortypes?pack=foobar')
        self.assertEqual(len(resp.json), 0)
        resp = self.app.get('/v1/sensortypes?pack=dummy_pack_1')
        self.assertEqual(len(resp.json), 3)
        resp = self.app.get('/v1/sensortypes?enabled=False')
        self.assertEqual(len(resp.json), 1)
        self.assertEqual(resp.json[0]['enabled'], False)
        resp = self.app.get('/v1/sensortypes?enabled=True')
        self.assertEqual(len(resp.json), 2)
        self.assertEqual(resp.json[0]['enabled'], True)
        self.assertEqual(resp.json[1]['enabled'], True)
        resp = self.app.get('/v1/sensortypes?trigger=dummy_pack_1.event3')
        self.assertEqual(len(resp.json), 1)
        self.assertEqual(resp.json[0]['trigger_types'], ['dummy_pack_1.event3'])
        resp = self.app.get('/v1/sensortypes?trigger=dummy_pack_1.event')
        self.assertEqual(len(resp.json), 2)
        self.assertEqual(resp.json[0]['trigger_types'], ['dummy_pack_1.event'])
        self.assertEqual(resp.json[1]['trigger_types'], ['dummy_pack_1.event'])

    def test_get_one_success(self):
        if False:
            return 10
        resp = self.app.get('/v1/sensortypes/dummy_pack_1.SampleSensor')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertEqual(resp.json['name'], 'SampleSensor')
        self.assertEqual(resp.json['ref'], 'dummy_pack_1.SampleSensor')

    def test_get_one_doesnt_exist(self):
        if False:
            return 10
        resp = self.app.get('/v1/sensortypes/1', expect_errors=True)
        self.assertEqual(resp.status_int, http_client.NOT_FOUND)

    def test_disable_and_enable_sensor(self):
        if False:
            print('Hello World!')
        resp = self.app.get('/v1/sensortypes/dummy_pack_1.SampleSensor')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertTrue(resp.json['enabled'])
        sensor_data = resp.json
        data = copy.deepcopy(sensor_data)
        data['enabled'] = False
        put_resp = self.app.put_json('/v1/sensortypes/dummy_pack_1.SampleSensor', data)
        self.assertEqual(put_resp.status_int, http_client.OK)
        self.assertEqual(put_resp.json['ref'], 'dummy_pack_1.SampleSensor')
        self.assertFalse(put_resp.json['enabled'])
        resp = self.app.get('/v1/sensortypes/dummy_pack_1.SampleSensor')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertFalse(resp.json['enabled'])
        data = copy.deepcopy(sensor_data)
        data['enabled'] = True
        put_resp = self.app.put_json('/v1/sensortypes/dummy_pack_1.SampleSensor', data)
        self.assertEqual(put_resp.status_int, http_client.OK)
        self.assertTrue(put_resp.json['enabled'])
        resp = self.app.get('/v1/sensortypes/dummy_pack_1.SampleSensor')
        self.assertEqual(resp.status_int, http_client.OK)
        self.assertTrue(resp.json['enabled'])