import copy
import mock
import uuid
from oslo_config import cfg
from st2tests.api import FunctionalTest
from st2common.constants.keyvalue import ALL_SCOPE, FULL_SYSTEM_SCOPE, FULL_USER_SCOPE
from st2common.persistence.auth import User
from st2common.models.db.auth import UserDB
from st2common.rbac.backends.noop import NoOpRBACUtils
from six.moves import http_client
__all__ = ['KeyValuePairControllerTestCase', 'KeyValuePairControllerBaseTestCase', 'KeyValuePairControllerRBACTestCase']
KVP = {'name': 'keystone_endpoint', 'value': 'http://127.0.0.1:5000/v3'}
KVP_2 = {'name': 'keystone_version', 'value': 'v3'}
KVP_2_USER = {'name': 'keystone_version', 'value': 'user_v3', 'scope': 'st2kv.user'}
KVP_2_USER_LEGACY = {'name': 'keystone_version', 'value': 'user_v3', 'scope': 'user'}
KVP_3_USER = {'name': 'keystone_endpoint', 'value': 'http://127.0.1.1:5000/v3', 'scope': 'st2kv.user'}
KVP_4_USER = {'name': 'customer_ssn', 'value': '123-456-7890', 'secret': True, 'scope': 'st2kv.user'}
KVP_WITH_TTL = {'name': 'keystone_endpoint', 'value': 'http://127.0.0.1:5000/v3', 'ttl': 10}
SECRET_KVP = {'name': 'secret_key1', 'value': 'secret_value1', 'secret': True}
ENCRYPTED_KVP = {'name': 'secret_key1', 'value': '3030303030298D848B45A24EDCD1A82FAB4E831E3FCE6E60956817A48A180E4C040801EB30170DACF79498F30520236A629912C3584847098D', 'encrypted': True}
ENCRYPTED_KVP_SECRET_FALSE = {'name': 'secret_key2', 'value': '3030303030298D848B45A24EDCD1A82FAB4E831E3FCE6E60956817A48A180E4C040801EB30170DACF79498F30520236A629912C3584847098D', 'secret': True, 'encrypted': True}

class KeyValuePairControllerBaseTestCase(FunctionalTest):

    @staticmethod
    def _get_kvp_id(resp):
        if False:
            return 10
        return resp.json['name']

    def _do_get_one(self, kvp_id, expect_errors=False):
        if False:
            while True:
                i = 10
        return self.app.get('/v1/keys/%s' % kvp_id, expect_errors=expect_errors)

    def _do_put(self, kvp_id, kvp, expect_errors=False):
        if False:
            return 10
        return self.app.put_json('/v1/keys/%s' % kvp_id, kvp, expect_errors=expect_errors)

    def _do_delete(self, kvp_id, expect_errors=False):
        if False:
            i = 10
            return i + 15
        return self.app.delete('/v1/keys/%s' % kvp_id, expect_errors=expect_errors)

class KeyValuePairControllerTestCase(KeyValuePairControllerBaseTestCase):

    def test_get_all(self):
        if False:
            i = 10
            return i + 15
        resp = self.app.get('/v1/keys')
        self.assertEqual(resp.status_int, 200)

    def test_get_one(self):
        if False:
            for i in range(10):
                print('nop')
        put_resp = self._do_put('key1', KVP)
        kvp_id = self._get_kvp_id(put_resp)
        get_resp = self._do_get_one(kvp_id)
        self.assertEqual(get_resp.status_int, 200)
        self.assertEqual(self._get_kvp_id(get_resp), kvp_id)
        self._do_delete(kvp_id)

    def test_get_all_all_scope(self):
        if False:
            i = 10
            return i + 15
        user_db_1 = UserDB(name='user1')
        user_db_2 = UserDB(name='user2')
        user_db_3 = UserDB(name='user3')
        put_resp = self._do_put('system1', {'name': 'system1', 'value': 'val1', 'scope': 'st2kv.system'})
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['name'], 'system1')
        self.assertEqual(put_resp.json['scope'], 'st2kv.system')
        put_resp = self._do_put('system2', {'name': 'system2', 'value': 'val2', 'scope': 'st2kv.system'})
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['name'], 'system2')
        self.assertEqual(put_resp.json['scope'], 'st2kv.system')
        self.use_user(user_db_1)
        put_resp = self._do_put('user1', {'name': 'user1', 'value': 'user1', 'scope': 'st2kv.user'})
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['name'], 'user1')
        self.assertEqual(put_resp.json['scope'], 'st2kv.user')
        self.assertEqual(put_resp.json['value'], 'user1')
        put_resp = self._do_put('userkey', {'name': 'userkey', 'value': 'user1', 'scope': 'st2kv.user'})
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['name'], 'userkey')
        self.assertEqual(put_resp.json['scope'], 'st2kv.user')
        self.assertEqual(put_resp.json['value'], 'user1')
        self.use_user(user_db_2)
        put_resp = self._do_put('user2', {'name': 'user2', 'value': 'user2', 'scope': 'st2kv.user'})
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['name'], 'user2')
        self.assertEqual(put_resp.json['scope'], 'st2kv.user')
        self.assertEqual(put_resp.json['value'], 'user2')
        put_resp = self._do_put('userkey', {'name': 'userkey', 'value': 'user2', 'scope': 'st2kv.user'})
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['name'], 'userkey')
        self.assertEqual(put_resp.json['scope'], 'st2kv.user')
        self.assertEqual(put_resp.json['value'], 'user2')
        self.use_user(user_db_3)
        put_resp = self._do_put('user3', {'name': 'user3', 'value': 'user3', 'scope': 'st2kv.user'})
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['name'], 'user3')
        self.assertEqual(put_resp.json['scope'], 'st2kv.user')
        self.assertEqual(put_resp.json['value'], 'user3')
        put_resp = self._do_put('userkey', {'name': 'userkey', 'value': 'user3', 'scope': 'st2kv.user'})
        self.assertEqual(put_resp.status_int, 200)
        self.assertEqual(put_resp.json['name'], 'userkey')
        self.assertEqual(put_resp.json['scope'], 'st2kv.user')
        self.assertEqual(put_resp.json['value'], 'user3')
        self.use_user(user_db_1)
        resp = self.app.get('/v1/keys?scope=all')
        self.assertEqual(len(resp.json), 2 + 2)
        self.assertEqual(resp.json[0]['name'], 'system1')
        self.assertEqual(resp.json[0]['scope'], 'st2kv.system')
        self.assertEqual(resp.json[1]['name'], 'system2')
        self.assertEqual(resp.json[1]['scope'], 'st2kv.system')
        self.assertEqual(resp.json[2]['name'], 'user1')
        self.assertEqual(resp.json[2]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[2]['user'], 'user1')
        self.assertEqual(resp.json[3]['name'], 'userkey')
        self.assertEqual(resp.json[3]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[3]['user'], 'user1')
        resp = self.app.get('/v1/keys?scope=all&prefix=user2:')
        self.assertEqual(resp.json, [])
        resp = self.app.get('/v1/keys?scope=all&prefix=user')
        self.assertEqual(len(resp.json), 2)
        self.assertEqual(resp.json[0]['name'], 'user1')
        self.assertEqual(resp.json[0]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[0]['user'], 'user1')
        self.assertEqual(resp.json[1]['name'], 'userkey')
        self.assertEqual(resp.json[1]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[1]['user'], 'user1')
        self.use_user(user_db_2)
        resp = self.app.get('/v1/keys?scope=all')
        self.assertEqual(len(resp.json), 2 + 2)
        self.assertEqual(resp.json[0]['name'], 'system1')
        self.assertEqual(resp.json[0]['scope'], 'st2kv.system')
        self.assertEqual(resp.json[1]['name'], 'system2')
        self.assertEqual(resp.json[1]['scope'], 'st2kv.system')
        self.assertEqual(resp.json[2]['name'], 'user2')
        self.assertEqual(resp.json[2]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[2]['user'], 'user2')
        self.assertEqual(resp.json[3]['name'], 'userkey')
        self.assertEqual(resp.json[3]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[3]['user'], 'user2')
        resp = self.app.get('/v1/keys?scope=all&prefix=user1:')
        self.assertEqual(resp.json, [])
        resp = self.app.get('/v1/keys?scope=all&prefix=user')
        self.assertEqual(len(resp.json), 2)
        self.assertEqual(resp.json[0]['name'], 'user2')
        self.assertEqual(resp.json[0]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[0]['user'], 'user2')
        self.assertEqual(resp.json[1]['name'], 'userkey')
        self.assertEqual(resp.json[1]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[1]['user'], 'user2')
        resp = self.app.get('/v1/keys?scope=user&user=user1', expect_errors=True)
        expected_error = '"user" attribute can only be provided by admins when RBAC is enabled'
        self.assertEqual(resp.status_int, http_client.FORBIDDEN)
        self.assertEqual(resp.json['faultstring'], expected_error)
        self.use_user(user_db_3)
        resp = self.app.get('/v1/keys?scope=all')
        self.assertEqual(len(resp.json), 2 + 2)
        self.assertEqual(resp.json[0]['name'], 'system1')
        self.assertEqual(resp.json[0]['scope'], 'st2kv.system')
        self.assertEqual(resp.json[1]['name'], 'system2')
        self.assertEqual(resp.json[1]['scope'], 'st2kv.system')
        self.assertEqual(resp.json[2]['name'], 'user3')
        self.assertEqual(resp.json[2]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[2]['user'], 'user3')
        self.assertEqual(resp.json[3]['name'], 'userkey')
        self.assertEqual(resp.json[3]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[3]['user'], 'user3')
        resp = self.app.get('/v1/keys?scope=all&prefix=user1:')
        self.assertEqual(resp.json, [])
        resp = self.app.get('/v1/keys?scope=all&prefix=user')
        self.assertEqual(len(resp.json), 2)
        self.assertEqual(resp.json[0]['name'], 'user3')
        self.assertEqual(resp.json[0]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[0]['user'], 'user3')
        self.assertEqual(resp.json[1]['name'], 'userkey')
        self.assertEqual(resp.json[1]['scope'], 'st2kv.user')
        self.assertEqual(resp.json[1]['user'], 'user3')
        self._do_delete('system1')
        self._do_delete('system2')
        self.use_user(user_db_1)
        self._do_delete('user1?scope=user')
        self._do_delete('userkey?scope=user')
        self.use_user(user_db_2)
        self._do_delete('user2?scope=user')
        self._do_delete('userkey?scope=user')
        self.use_user(user_db_3)
        self._do_delete('user3?scope=user')
        self._do_delete('userkey?scope=user')

    def test_get_all_user_query_param_can_only_be_used_with_rbac(self):
        if False:
            while True:
                i = 10
        resp = self.app.get('/v1/keys?user=foousera', expect_errors=True)
        expected_error = '"user" attribute can only be provided by admins when RBAC is enabled'
        self.assertEqual(resp.status_int, http_client.FORBIDDEN)
        self.assertEqual(resp.json['faultstring'], expected_error)

    def test_get_one_user_query_param_can_only_be_used_with_rbac(self):
        if False:
            for i in range(10):
                print('nop')
        resp = self.app.get('/v1/keys/keystone_endpoint?user=foousera', expect_errors=True)
        expected_error = '"user" attribute can only be provided by admins when RBAC is enabled'
        self.assertEqual(resp.status_int, http_client.FORBIDDEN)
        self.assertEqual(resp.json['faultstring'], expected_error)

    def test_get_all_prefix_filtering(self):
        if False:
            i = 10
            return i + 15
        put_resp1 = self._do_put(KVP['name'], KVP)
        put_resp2 = self._do_put(KVP_2['name'], KVP_2)
        self.assertEqual(put_resp1.status_int, 200)
        self.assertEqual(put_resp2.status_int, 200)
        resp = self.app.get('/v1/keys?prefix=something')
        self.assertEqual(resp.json, [])
        resp = self.app.get('/v1/keys?prefix=keystone')
        self.assertEqual(len(resp.json), 2)
        resp = self.app.get('/v1/keys?prefix=keystone_endpoint')
        self.assertEqual(len(resp.json), 1)
        self._do_delete(self._get_kvp_id(put_resp1))
        self._do_delete(self._get_kvp_id(put_resp2))

    def test_get_one_fail(self):
        if False:
            while True:
                i = 10
        resp = self.app.get('/v1/keys/1', expect_errors=True)
        self.assertEqual(resp.status_int, 404)

    def test_put(self):
        if False:
            for i in range(10):
                print('nop')
        put_resp = self._do_put('key1', KVP)
        update_input = put_resp.json
        update_input['value'] = 'http://127.0.0.1:35357/v3'
        put_resp = self._do_put(self._get_kvp_id(put_resp), update_input)
        self.assertEqual(put_resp.status_int, 200)
        self._do_delete(self._get_kvp_id(put_resp))

    def test_put_with_scope(self):
        if False:
            while True:
                i = 10
        self.app.put_json('/v1/keys/%s' % 'keystone_endpoint', KVP, expect_errors=False)
        self.app.put_json('/v1/keys/%s?scope=st2kv.system' % 'keystone_version', KVP_2, expect_errors=False)
        get_resp_1 = self.app.get('/v1/keys/keystone_endpoint')
        self.assertTrue(get_resp_1.status_int, 200)
        self.assertEqual(self._get_kvp_id(get_resp_1), 'keystone_endpoint')
        get_resp_2 = self.app.get('/v1/keys/keystone_version?scope=st2kv.system')
        self.assertTrue(get_resp_2.status_int, 200)
        self.assertEqual(self._get_kvp_id(get_resp_2), 'keystone_version')
        get_resp_3 = self.app.get('/v1/keys/keystone_version')
        self.assertTrue(get_resp_3.status_int, 200)
        self.assertEqual(self._get_kvp_id(get_resp_3), 'keystone_version')
        self.app.delete('/v1/keys/keystone_endpoint?scope=st2kv.system')
        self.app.delete('/v1/keys/keystone_version?scope=st2kv.system')

    def test_put_user_scope_and_system_scope_dont_overlap(self):
        if False:
            print('Hello World!')
        self.app.put_json('/v1/keys/%s?scope=st2kv.system' % 'keystone_version', KVP_2, expect_errors=False)
        self.app.put_json('/v1/keys/%s?scope=st2kv.user' % 'keystone_version', KVP_2_USER, expect_errors=False)
        get_resp = self.app.get('/v1/keys/keystone_version?scope=st2kv.system')
        self.assertEqual(get_resp.json['value'], KVP_2['value'])
        get_resp = self.app.get('/v1/keys/keystone_version?scope=st2kv.user')
        self.assertEqual(get_resp.json['value'], KVP_2_USER['value'])
        self.app.delete('/v1/keys/keystone_version?scope=st2kv.system')
        self.app.delete('/v1/keys/keystone_version?scope=st2kv.user')

    def test_put_invalid_scope(self):
        if False:
            while True:
                i = 10
        put_resp = self.app.put_json('/v1/keys/keystone_version?scope=st2', KVP_2, expect_errors=True)
        self.assertTrue(put_resp.status_int, 400)

    def test_get_all_with_scope(self):
        if False:
            while True:
                i = 10
        self.app.put_json('/v1/keys/%s?scope=st2kv.system' % 'keystone_version', KVP_2, expect_errors=False)
        self.app.put_json('/v1/keys/%s?scope=st2kv.user' % 'keystone_version', KVP_2_USER, expect_errors=False)
        self.app.put_json('/v1/keys/%s?scope=system' % 'keystone_version', KVP_2, expect_errors=False)
        self.app.put_json('/v1/keys/%s?scope=user' % 'keystone_version', KVP_2_USER_LEGACY, expect_errors=False)
        get_resp_all = self.app.get('/v1/keys?scope=all')
        self.assertTrue(len(get_resp_all.json), 2)
        get_resp_sys = self.app.get('/v1/keys?scope=st2kv.system')
        self.assertTrue(len(get_resp_sys.json), 1)
        self.assertEqual(get_resp_sys.json[0]['value'], KVP_2['value'])
        get_resp_sys = self.app.get('/v1/keys?scope=system')
        self.assertTrue(len(get_resp_sys.json), 1)
        self.assertEqual(get_resp_sys.json[0]['value'], KVP_2['value'])
        get_resp_sys = self.app.get('/v1/keys?scope=st2kv.user')
        self.assertTrue(len(get_resp_sys.json), 1)
        self.assertEqual(get_resp_sys.json[0]['value'], KVP_2_USER['value'])
        get_resp_sys = self.app.get('/v1/keys?scope=user')
        self.assertTrue(len(get_resp_sys.json), 1)
        self.assertEqual(get_resp_sys.json[0]['value'], KVP_2_USER['value'])
        self.app.delete('/v1/keys/keystone_version?scope=st2kv.system')
        self.app.delete('/v1/keys/keystone_version?scope=st2kv.user')

    def test_get_all_with_scope_and_prefix_filtering(self):
        if False:
            i = 10
            return i + 15
        self.app.put_json('/v1/keys/%s?scope=st2kv.user' % 'keystone_version', KVP_2_USER, expect_errors=False)
        self.app.put_json('/v1/keys/%s?scope=st2kv.user' % 'keystone_endpoint', KVP_3_USER, expect_errors=False)
        self.app.put_json('/v1/keys/%s?scope=st2kv.user' % 'customer_ssn', KVP_4_USER, expect_errors=False)
        get_prefix = self.app.get('/v1/keys?scope=st2kv.user&prefix=keystone')
        self.assertEqual(len(get_prefix.json), 2)
        self.app.delete('/v1/keys/keystone_version?scope=st2kv.user')
        self.app.delete('/v1/keys/keystone_endpoint?scope=st2kv.user')
        self.app.delete('/v1/keys/customer_ssn?scope=st2kv.user')

    def test_put_with_ttl(self):
        if False:
            print('Hello World!')
        put_resp = self._do_put('key_with_ttl', KVP_WITH_TTL)
        self.assertEqual(put_resp.status_int, 200)
        get_resp = self.app.get('/v1/keys')
        self.assertTrue(get_resp.json[0]['expire_timestamp'])
        self._do_delete(self._get_kvp_id(put_resp))

    def test_put_secret(self):
        if False:
            i = 10
            return i + 15
        put_resp = self._do_put('secret_key1', SECRET_KVP)
        kvp_id = self._get_kvp_id(put_resp)
        get_resp = self._do_get_one(kvp_id)
        self.assertTrue(get_resp.json['encrypted'])
        crypto_val = get_resp.json['value']
        self.assertNotEqual(SECRET_KVP['value'], crypto_val)
        self._do_delete(self._get_kvp_id(put_resp))

    def test_get_one_secret_no_decrypt(self):
        if False:
            return 10
        put_resp = self._do_put('secret_key1', SECRET_KVP)
        kvp_id = self._get_kvp_id(put_resp)
        get_resp = self.app.get('/v1/keys/secret_key1')
        self.assertEqual(get_resp.status_int, 200)
        self.assertEqual(self._get_kvp_id(get_resp), kvp_id)
        self.assertTrue(get_resp.json['secret'])
        self.assertTrue(get_resp.json['encrypted'])
        self._do_delete(kvp_id)

    def test_get_one_secret_decrypt(self):
        if False:
            while True:
                i = 10
        put_resp = self._do_put('secret_key1', SECRET_KVP)
        kvp_id = self._get_kvp_id(put_resp)
        get_resp = self.app.get('/v1/keys/secret_key1?decrypt=true')
        self.assertEqual(get_resp.status_int, 200)
        self.assertEqual(self._get_kvp_id(get_resp), kvp_id)
        self.assertTrue(get_resp.json['secret'])
        self.assertFalse(get_resp.json['encrypted'])
        self.assertEqual(get_resp.json['value'], SECRET_KVP['value'])
        self._do_delete(kvp_id)

    def test_get_all_decrypt(self):
        if False:
            while True:
                i = 10
        put_resp = self._do_put('secret_key1', SECRET_KVP)
        kvp_id_1 = self._get_kvp_id(put_resp)
        put_resp = self._do_put('key1', KVP)
        kvp_id_2 = self._get_kvp_id(put_resp)
        kvps = {'key1': KVP, 'secret_key1': SECRET_KVP}
        stored_kvps = self.app.get('/v1/keys?decrypt=true').json
        self.assertTrue(len(stored_kvps), 2)
        for stored_kvp in stored_kvps:
            self.assertFalse(stored_kvp['encrypted'])
            exp_kvp = kvps.get(stored_kvp['name'])
            self.assertIsNotNone(exp_kvp)
            self.assertEqual(exp_kvp['value'], stored_kvp['value'])
        self._do_delete(kvp_id_1)
        self._do_delete(kvp_id_2)

    def test_put_encrypted_value(self):
        if False:
            print('Hello World!')
        put_resp = self._do_put('secret_key1', ENCRYPTED_KVP)
        kvp_id = self._get_kvp_id(put_resp)
        self.assertEqual(put_resp.status_code, 200)
        self.assertEqual(put_resp.json['name'], 'secret_key1')
        self.assertEqual(put_resp.json['scope'], 'st2kv.system')
        self.assertEqual(put_resp.json['encrypted'], True)
        self.assertEqual(put_resp.json['secret'], True)
        self.assertEqual(put_resp.json['value'], ENCRYPTED_KVP['value'])
        self.assertTrue(put_resp.json['value'] != 'S3cret!Value')
        self.assertTrue(len(put_resp.json['value']) > len('S3cret!Value') * 2)
        get_resp = self._do_get_one(kvp_id + '?decrypt=True')
        self.assertEqual(put_resp.json['name'], 'secret_key1')
        self.assertEqual(put_resp.json['scope'], 'st2kv.system')
        self.assertEqual(put_resp.json['encrypted'], True)
        self.assertEqual(put_resp.json['secret'], True)
        self.assertEqual(put_resp.json['value'], ENCRYPTED_KVP['value'])
        get_resp = self._do_get_one(kvp_id + '?decrypt=True')
        self.assertFalse(get_resp.json['encrypted'])
        self.assertEqual(get_resp.json['value'], 'S3cret!Value')
        self._do_delete(self._get_kvp_id(put_resp))
        put_resp = self._do_put('secret_key2', ENCRYPTED_KVP_SECRET_FALSE)
        kvp_id = self._get_kvp_id(put_resp)
        self.assertEqual(put_resp.status_code, 200)
        self.assertEqual(put_resp.json['name'], 'secret_key2')
        self.assertEqual(put_resp.json['scope'], 'st2kv.system')
        self.assertEqual(put_resp.json['encrypted'], True)
        self.assertEqual(put_resp.json['secret'], True)
        self.assertEqual(put_resp.json['value'], ENCRYPTED_KVP['value'])
        self.assertTrue(put_resp.json['value'] != 'S3cret!Value')
        self.assertTrue(len(put_resp.json['value']) > len('S3cret!Value') * 2)
        get_resp = self._do_get_one(kvp_id + '?decrypt=True')
        self.assertEqual(put_resp.json['name'], 'secret_key2')
        self.assertEqual(put_resp.json['scope'], 'st2kv.system')
        self.assertEqual(put_resp.json['encrypted'], True)
        self.assertEqual(put_resp.json['secret'], True)
        self.assertEqual(put_resp.json['value'], ENCRYPTED_KVP['value'])
        get_resp = self._do_get_one(kvp_id + '?decrypt=True')
        self.assertFalse(get_resp.json['encrypted'])
        self.assertEqual(get_resp.json['value'], 'S3cret!Value')
        self._do_delete(self._get_kvp_id(put_resp))

    def test_put_encrypted_value_integrity_check_failed(self):
        if False:
            while True:
                i = 10
        data = copy.deepcopy(ENCRYPTED_KVP)
        data['value'] = 'corrupted'
        put_resp = self._do_put('secret_key1', data, expect_errors=True)
        self.assertEqual(put_resp.status_code, 400)
        expected_error = 'Failed to verify the integrity of the provided value for key "secret_key1".'
        self.assertIn(expected_error, put_resp.json['faultstring'])
        data = copy.deepcopy(ENCRYPTED_KVP)
        data['value'] = str(data['value'][:-2])
        put_resp = self._do_put('secret_key1', data, expect_errors=True)
        self.assertEqual(put_resp.status_code, 400)
        expected_error = 'Failed to verify the integrity of the provided value for key "secret_key1".'
        self.assertIn(expected_error, put_resp.json['faultstring'])

    def test_put_delete(self):
        if False:
            return 10
        put_resp = self._do_put('key1', KVP)
        self.assertEqual(put_resp.status_int, 200)
        self._do_delete(self._get_kvp_id(put_resp))

    def test_delete(self):
        if False:
            i = 10
            return i + 15
        put_resp = self._do_put('key1', KVP)
        del_resp = self._do_delete(self._get_kvp_id(put_resp))
        self.assertEqual(del_resp.status_int, 204)

    def test_delete_fail(self):
        if False:
            return 10
        resp = self._do_delete('inexistentkey', expect_errors=True)
        self.assertEqual(resp.status_int, 404)

    @staticmethod
    def _get_kvp_id(resp):
        if False:
            return 10
        return resp.json['name']

    def _do_get_one(self, kvp_id, expect_errors=False):
        if False:
            while True:
                i = 10
        return self.app.get('/v1/keys/%s' % kvp_id, expect_errors=expect_errors)

    def _do_put(self, kvp_id, kvp, expect_errors=False):
        if False:
            for i in range(10):
                print('nop')
        return self.app.put_json('/v1/keys/%s' % kvp_id, kvp, expect_errors=expect_errors)

    def _do_delete(self, kvp_id, expect_errors=False):
        if False:
            i = 10
            return i + 15
        return self.app.delete('/v1/keys/%s' % kvp_id, expect_errors=expect_errors)

class KeyValuePairControllerRBACTestCase(KeyValuePairControllerBaseTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super(KeyValuePairControllerRBACTestCase, cls).setUpClass()
        cfg.CONF.set_override(group='rbac', name='enable', override=True)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cfg.CONF.set_override(group='rbac', name='enable', override=False)
        super(KeyValuePairControllerRBACTestCase, cls).tearDownClass()

    @mock.patch.object(NoOpRBACUtils, 'user_has_system_role', mock.MagicMock(return_value=True))
    @mock.patch('st2api.controllers.v1.keyvalue.get_all_system_kvp_names_for_user')
    def test_get_all_all_scope_admin(self, mock_get_system_kvp_names):
        if False:
            return 10
        user_1_db = UserDB(name='user-' + uuid.uuid4().hex)
        user_1_db = User.add_or_update(user_1_db)
        user_2_db = UserDB(name='user-' + uuid.uuid4().hex)
        user_2_db = User.add_or_update(user_2_db)
        for i in range(1, 3):
            (k, v) = ('s11' + str(i), 'v11' + str(i))
            put_resp = self._do_put(k, {'name': k, 'value': v, 'scope': FULL_SYSTEM_SCOPE})
            self.assertEqual(put_resp.status_int, 200)
        self.use_user(user_1_db)
        (k, v) = ('a111', 'v12345')
        put_resp = self._do_put(k, {'name': k, 'value': v, 'scope': FULL_USER_SCOPE})
        self.assertEqual(put_resp.status_int, 200)
        self.use_user(user_2_db)
        (k, v) = ('u111', 'v23456')
        put_resp = self._do_put(k, {'name': k, 'value': v, 'scope': FULL_USER_SCOPE})
        self.assertEqual(put_resp.status_int, 200)
        self.use_user(user_1_db)
        resp = self.app.get('/v1/keys', {'scope': ALL_SCOPE})
        self.assertEqual(len(resp.json), 3)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['a111', 's111', 's112'])
        self.assertFalse(mock_get_system_kvp_names.called)
        resp = self.app.get('/v1/keys', {'scope': FULL_SYSTEM_SCOPE})
        self.assertEqual(len(resp.json), 2)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['s111', 's112'])
        self.assertFalse(mock_get_system_kvp_names.called)
        resp = self.app.get('/v1/keys', {'scope': FULL_USER_SCOPE})
        self.assertEqual(len(resp.json), 1)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['a111'])
        self.assertFalse(mock_get_system_kvp_names.called)
        resp = self.app.get('/v1/keys', {'user': user_2_db.name})
        self.assertEqual(len(resp.json), 1)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['u111'])
        self.assertFalse(mock_get_system_kvp_names.called)

    @mock.patch.object(NoOpRBACUtils, 'user_has_system_role', mock.MagicMock(return_value=False))
    @mock.patch('st2api.controllers.v1.keyvalue.get_all_system_kvp_names_for_user')
    def test_get_all_all_scope_nonadmin(self, mock_get_system_kvp_names):
        if False:
            while True:
                i = 10
        user_1_db = UserDB(name='user-' + uuid.uuid4().hex)
        user_1_db = User.add_or_update(user_1_db)
        user_2_db = UserDB(name='user-' + uuid.uuid4().hex)
        user_2_db = User.add_or_update(user_2_db)
        for i in range(1, 7):
            (k, v) = ('s12' + str(i), 'v12' + str(i))
            put_resp = self._do_put(k, {'name': k, 'value': v, 'scope': FULL_SYSTEM_SCOPE})
            self.assertEqual(put_resp.status_int, 200)
        self.use_user(user_1_db)
        (k, v) = ('u121', 'v12345')
        put_resp = self._do_put(k, {'name': k, 'value': v, 'scope': FULL_USER_SCOPE})
        self.assertEqual(put_resp.status_int, 200)
        self.use_user(user_2_db)
        (k, v) = ('u122', 'v23456')
        put_resp = self._do_put(k, {'name': k, 'value': v, 'scope': FULL_USER_SCOPE})
        self.assertEqual(put_resp.status_int, 200)
        self.use_user(user_1_db)
        mock_get_system_kvp_names.return_value = ['s121', 's122']
        resp = self.app.get('/v1/keys', {'scope': ALL_SCOPE})
        self.assertEqual(len(resp.json), 3)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['s121', 's122', 'u121'])
        self.assertTrue(mock_get_system_kvp_names.called)
        mock_get_system_kvp_names.reset_mock()
        resp = self.app.get('/v1/keys', {'scope': FULL_SYSTEM_SCOPE})
        self.assertEqual(len(resp.json), 2)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['s121', 's122'])
        self.assertTrue(mock_get_system_kvp_names.called)
        mock_get_system_kvp_names.reset_mock()
        resp = self.app.get('/v1/keys', {'scope': FULL_USER_SCOPE})
        self.assertEqual(len(resp.json), 1)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['u121'])
        self.assertFalse(mock_get_system_kvp_names.called)
        mock_get_system_kvp_names.reset_mock()
        self.use_user(user_2_db)
        mock_get_system_kvp_names.return_value = ['s123', 's124']
        resp = self.app.get('/v1/keys', {'scope': ALL_SCOPE})
        self.assertEqual(len(resp.json), 3)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['s123', 's124', 'u122'])
        self.assertTrue(mock_get_system_kvp_names.called)
        mock_get_system_kvp_names.reset_mock()
        resp = self.app.get('/v1/keys', {'scope': FULL_SYSTEM_SCOPE})
        self.assertEqual(len(resp.json), 2)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['s123', 's124'])
        self.assertTrue(mock_get_system_kvp_names.called)
        mock_get_system_kvp_names.reset_mock()
        resp = self.app.get('/v1/keys', {'scope': FULL_USER_SCOPE})
        self.assertEqual(len(resp.json), 1)
        self.assertListEqual(sorted([i['name'] for i in resp.json]), ['u122'])
        self.assertFalse(mock_get_system_kvp_names.called)
        mock_get_system_kvp_names.reset_mock()