import mock
from oslo_config import cfg
from six.moves import http_client
from st2common.constants.secrets import MASKED_ATTRIBUTE_VALUE
from st2common.persistence.auth import ApiKey
from st2tests.fixtures.generic.fixture import PACK_NAME as FIXTURES_PACK
from st2tests.fixturesloader import FixturesLoader
from st2tests.api import FunctionalTest
TEST_MODELS = {'apikeys': ['apikey1.yaml', 'apikey2.yaml', 'apikey3.yaml', 'apikey_disabled.yaml', 'apikey_malformed.yaml']}
KEY1_KEY = '1234'
KEY2_KEY = '5678'
KEY3_KEY = '9012'

class TestApiKeyController(FunctionalTest):
    apikey1 = None
    apikey2 = None
    apikey3 = None

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(TestApiKeyController, cls).setUpClass()
        cfg.CONF.set_override(name='mask_secrets', override=True, group='api')
        cfg.CONF.set_override(name='mask_secrets', override=True, group='log')
        models = FixturesLoader().save_fixtures_to_db(fixtures_pack=FIXTURES_PACK, fixtures_dict=TEST_MODELS)
        cls.apikey1 = models['apikeys']['apikey1.yaml']
        cls.apikey2 = models['apikeys']['apikey2.yaml']
        cls.apikey3 = models['apikeys']['apikey3.yaml']
        cls.apikey4 = models['apikeys']['apikey_disabled.yaml']
        cls.apikey5 = models['apikeys']['apikey_malformed.yaml']

    def test_get_all_and_minus_one(self):
        if False:
            i = 10
            return i + 15
        resp = self.app.get('/v1/apikeys')
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.headers['X-Total-Count'], '5')
        self.assertEqual(resp.headers['X-Limit'], '50')
        self.assertEqual(len(resp.json), 5, '/v1/apikeys did not return all apikeys.')
        retrieved_ids = [apikey['id'] for apikey in resp.json]
        self.assertEqual(retrieved_ids, [str(self.apikey1.id), str(self.apikey2.id), str(self.apikey3.id), str(self.apikey4.id), str(self.apikey5.id)], 'Incorrect api keys retrieved.')
        resp = self.app.get('/v1/apikeys/?limit=-1')
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.headers['X-Total-Count'], '5')
        self.assertEqual(len(resp.json), 5, '/v1/apikeys did not return all apikeys.')

    def test_get_all_with_pagnination_with_offset_and_limit(self):
        if False:
            while True:
                i = 10
        resp = self.app.get('/v1/apikeys?offset=2&limit=1')
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.headers['X-Total-Count'], '5')
        self.assertEqual(resp.headers['X-Limit'], '1')
        self.assertEqual(len(resp.json), 1)
        retrieved_ids = [apikey['id'] for apikey in resp.json]
        self.assertEqual(retrieved_ids, [str(self.apikey3.id)])

    def test_get_all_with_pagnination_with_only_offset(self):
        if False:
            return 10
        resp = self.app.get('/v1/apikeys?offset=3')
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.headers['X-Total-Count'], '5')
        self.assertEqual(resp.headers['X-Limit'], '50')
        self.assertEqual(len(resp.json), 2)
        retrieved_ids = [apikey['id'] for apikey in resp.json]
        self.assertEqual(retrieved_ids, [str(self.apikey4.id), str(self.apikey5.id)])

    def test_get_all_with_pagnination_with_only_limit(self):
        if False:
            return 10
        resp = self.app.get('/v1/apikeys?limit=2')
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.headers['X-Total-Count'], '5')
        self.assertEqual(resp.headers['X-Limit'], '2')
        self.assertEqual(len(resp.json), 2)
        retrieved_ids = [apikey['id'] for apikey in resp.json]
        self.assertEqual(retrieved_ids, [str(self.apikey1.id), str(self.apikey2.id)])

    @mock.patch('st2common.rbac.backends.noop.NoOpRBACUtils.user_is_admin', mock.Mock(return_value=False))
    def test_get_all_invalid_limit_too_large_none_admin(self):
        if False:
            return 10
        resp = self.app.get('/v1/apikeys?offset=2&limit=1000', expect_errors=True)
        self.assertEqual(resp.status_int, http_client.FORBIDDEN)
        self.assertEqual(resp.json['faultstring'], 'Limit "1000" specified, maximum value is "100"')

    def test_get_all_invalid_limit_negative_integer(self):
        if False:
            print('Hello World!')
        resp = self.app.get('/v1/apikeys?offset=2&limit=-22', expect_errors=True)
        self.assertEqual(resp.status_int, 400)
        self.assertEqual(resp.json['faultstring'], 'Limit, "-22" specified, must be a positive number.')

    def test_get_all_invalid_offset_too_large(self):
        if False:
            while True:
                i = 10
        offset = '2141564789454123457895412237483648'
        resp = self.app.get('/v1/apikeys?offset=%s&limit=1' % offset, expect_errors=True)
        self.assertEqual(resp.status_int, 400)
        self.assertEqual(resp.json['faultstring'], 'Offset "%s" specified is more than 32 bit int' % offset)

    def test_get_one_by_id(self):
        if False:
            for i in range(10):
                print('nop')
        resp = self.app.get('/v1/apikeys/%s' % self.apikey1.id)
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.json['id'], str(self.apikey1.id), 'Incorrect api key retrieved.')
        self.assertEqual(resp.json['key_hash'], MASKED_ATTRIBUTE_VALUE, 'Key should be masked.')

    def test_get_one_by_key(self):
        if False:
            for i in range(10):
                print('nop')
        resp = self.app.get('/v1/apikeys/%s' % KEY1_KEY)
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.json['id'], str(self.apikey1.id), 'Incorrect api key retrieved.')
        self.assertEqual(resp.json['key_hash'], MASKED_ATTRIBUTE_VALUE, 'Key should be masked.')
        resp = self.app.get('/v1/apikeys/%s' % KEY2_KEY)
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.json['id'], str(self.apikey2.id), 'Incorrect api key retrieved.')
        self.assertEqual(resp.json['key_hash'], MASKED_ATTRIBUTE_VALUE, 'Key should be masked.')
        resp = self.app.get('/v1/apikeys/%s' % KEY3_KEY)
        self.assertEqual(resp.status_int, 200)
        self.assertEqual(resp.json['id'], str(self.apikey3.id), 'Incorrect api key retrieved.')
        self.assertEqual(resp.json['key_hash'], MASKED_ATTRIBUTE_VALUE, 'Key should be masked.')

    def test_get_show_secrets(self):
        if False:
            i = 10
            return i + 15
        resp = self.app.get('/v1/apikeys?show_secrets=True', expect_errors=True)
        self.assertEqual(resp.status_int, 200)
        for key in resp.json:
            self.assertNotEqual(key['key_hash'], MASKED_ATTRIBUTE_VALUE)
            self.assertNotEqual(key['uid'], MASKED_ATTRIBUTE_VALUE)

    def test_post_delete_key(self):
        if False:
            for i in range(10):
                print('nop')
        api_key = {'user': 'herge'}
        resp1 = self.app.post_json('/v1/apikeys', api_key)
        self.assertEqual(resp1.status_int, 201)
        self.assertTrue(resp1.json['key'], 'Key should be non-None.')
        self.assertNotEqual(resp1.json['key'], MASKED_ATTRIBUTE_VALUE, 'Key should not be masked.')
        resp2 = self.app.post_json('/v1/apikeys', api_key)
        self.assertEqual(resp2.status_int, 201)
        self.assertTrue(resp2.json['key'], 'Key should be non-None.')
        self.assertNotEqual(resp2.json['key'], MASKED_ATTRIBUTE_VALUE, 'Key should not be masked.')
        self.assertNotEqual(resp1.json['key'], resp2.json['key'], 'Should be different')
        resp = self.app.delete('/v1/apikeys/%s' % resp1.json['id'])
        self.assertEqual(resp.status_int, 204)
        resp = self.app.delete('/v1/apikeys/%s' % resp2.json['key'])
        self.assertEqual(resp.status_int, 204)
        resp3 = self.app.post_json('/v1/apikeys', {})
        self.assertEqual(resp3.status_int, 201)
        self.assertTrue(resp3.json['key'], 'Key should be non-None.')
        self.assertNotEqual(resp3.json['key'], MASKED_ATTRIBUTE_VALUE, 'Key should not be masked.')
        self.assertTrue(resp3.json['user'], cfg.CONF.system_user.user)

    def test_post_delete_same_key_hash(self):
        if False:
            i = 10
            return i + 15
        api_key = {'id': '5c5dbb576cb8de06a2d79a4d', 'user': 'herge', 'key_hash': 'ABCDE'}
        resp1 = self.app.post_json('/v1/apikeys', api_key)
        self.assertEqual(resp1.status_int, 201)
        self.assertEqual(resp1.json['key'], None, 'Key should be None.')
        api_key_db = ApiKey.get_by_id(resp1.json['id'])
        self.assertEqual(resp1.json['id'], api_key['id'], 'PK ID of created API should match.')
        self.assertEqual(api_key_db.key_hash, api_key['key_hash'], 'Key_hash should match.')
        self.assertEqual(api_key_db.user, api_key['user'], 'User should match.')
        resp = self.app.delete('/v1/apikeys/%s' % resp1.json['id'])
        self.assertEqual(resp.status_int, 204)

    def test_put_api_key(self):
        if False:
            print('Hello World!')
        resp = self.app.get('/v1/apikeys/%s' % self.apikey1.id)
        self.assertEqual(resp.status_int, 200)
        update_input = resp.json
        update_input['enabled'] = not update_input['enabled']
        put_resp = self.app.put_json('/v1/apikeys/%s' % self.apikey1.id, update_input, expect_errors=True)
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['enabled'], not resp.json['enabled'])
        update_input = put_resp.json
        update_input['enabled'] = not update_input['enabled']
        put_resp = self.app.put_json('/v1/apikeys/%s' % self.apikey1.id, update_input, expect_errors=True)
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['enabled'], resp.json['enabled'])

    def test_put_api_key_fail(self):
        if False:
            i = 10
            return i + 15
        resp = self.app.get('/v1/apikeys/%s' % self.apikey1.id)
        self.assertEqual(resp.status_int, 200)
        update_input = resp.json
        update_input['key_hash'] = '1'
        put_resp = self.app.put_json('/v1/apikeys/%s' % self.apikey1.id, update_input, expect_errors=True)
        self.assertEqual(put_resp.status_int, 400)
        self.assertTrue(put_resp.json['faultstring'])

    def test_post_no_user_fail(self):
        if False:
            while True:
                i = 10
        self.app.post_json('/v1/apikeys', {}, expect_errors=True)