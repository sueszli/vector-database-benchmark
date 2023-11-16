import glob
from st2tests.api import FunctionalTest
from st2tests.fixtures.packs.all_packs_glob import PACKS_PATH
__all__ = ['PackConfigSchemasControllerTestCase']
CONFIG_SCHEMA_COUNT = len(glob.glob(f'{PACKS_PATH}/*/config.schema.yaml'))
assert CONFIG_SCHEMA_COUNT > 1

class PackConfigSchemasControllerTestCase(FunctionalTest):
    register_packs = True

    def test_get_all(self):
        if False:
            while True:
                i = 10
        resp = self.app.get('/v1/config_schemas')
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(len(resp.json), CONFIG_SCHEMA_COUNT, '/v1/config_schemas did not return all schemas.')

    def test_get_one_success(self):
        if False:
            print('Hello World!')
        resp = self.app.get('/v1/config_schemas/dummy_pack_1')
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.json['pack'], 'dummy_pack_1')
        self.assertIn('api_key', resp.json['attributes'])

    def test_get_one_doesnt_exist(self):
        if False:
            i = 10
            return i + 15
        resp = self.app.get('/v1/config_schemas/dummy_pack_2', expect_errors=True)
        self.assertEqual(resp.status_int, 404)
        self.assertIn('Unable to identify resource with pack_ref ', resp.json['faultstring'])
        ref_or_id = 'pack_doesnt_exist'
        resp = self.app.get('/v1/config_schemas/%s' % ref_or_id, expect_errors=True)
        self.assertEqual(resp.status_int, 404)
        self.assertTrue('Resource with a ref or id "%s" not found' % ref_or_id in resp.json['faultstring'])