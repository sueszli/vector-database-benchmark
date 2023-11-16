import mock
import unittest2
from oslo_config import cfg
from st2common.util import keyvalue as kv_utl
from st2common.constants.keyvalue import FULL_SYSTEM_SCOPE, FULL_USER_SCOPE, USER_SCOPE, ALL_SCOPE, DATASTORE_PARENT_SCOPE, DATASTORE_SCOPE_SEPARATOR
from st2common.constants.types import ResourceType
from st2common.exceptions.rbac import AccessDeniedError
from st2common.exceptions.rbac import ResourceAccessDeniedError
from st2common.models.db import auth as auth_db
from st2common.models.db.keyvalue import KeyValuePairDB
from st2common.rbac.backends.noop import NoOpRBACUtils
from st2common.rbac.types import PermissionType
from st2tests import config
USER = 'stanley'
RESOURCE_UUID = '%s:%s:%s' % (ResourceType.KEY_VALUE_PAIR, FULL_USER_SCOPE, 'stanley:foobar')

class TestKeyValueUtil(unittest2.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(TestKeyValueUtil, cls).setUpClass()
        config.parse_args()
        cfg.CONF.set_override(name='backend', override='noop', group='rbac')

    def test_validate_scope(self):
        if False:
            return 10
        scope = FULL_USER_SCOPE
        kv_utl._validate_scope(scope)
        scope = FULL_SYSTEM_SCOPE
        kv_utl._validate_scope(scope)
        scope = USER_SCOPE
        kv_utl._validate_scope(scope)

    def test_validate_scope_with_invalid_scope(self):
        if False:
            while True:
                i = 10
        scope = 'INVALID_SCOPE'
        self.assertRaises(ValueError, kv_utl._validate_scope, scope)

    def test_validate_decrypt_query_parameter(self):
        if False:
            return 10
        test_params = [[False, USER_SCOPE, False, {}], [True, USER_SCOPE, False, {}], [True, FULL_SYSTEM_SCOPE, True, {}]]
        for params in test_params:
            kv_utl._validate_decrypt_query_parameter(*params)

    def test_validate_decrypt_query_parameter_access_denied(self):
        if False:
            i = 10
            return i + 15
        test_params = [[True, FULL_SYSTEM_SCOPE, False, {}]]
        for params in test_params:
            assert_params = [AccessDeniedError, kv_utl._validate_decrypt_query_parameter]
            assert_params.extend(params)
            self.assertRaises(*assert_params)

    def test_get_datastore_full_scope(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(kv_utl.get_datastore_full_scope(USER_SCOPE), DATASTORE_SCOPE_SEPARATOR.join([DATASTORE_PARENT_SCOPE, USER_SCOPE]))

    def test_get_datastore_full_scope_all_scope(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(kv_utl.get_datastore_full_scope(ALL_SCOPE), ALL_SCOPE)

    def test_get_datastore_full_scope_datastore_parent_scope(self):
        if False:
            while True:
                i = 10
        self.assertEqual(kv_utl.get_datastore_full_scope(DATASTORE_PARENT_SCOPE), DATASTORE_PARENT_SCOPE)

    def test_derive_scope_and_key(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'test'
        scope = USER_SCOPE
        result = kv_utl._derive_scope_and_key(key, scope)
        self.assertEqual((FULL_USER_SCOPE, 'user:%s' % key), result)

    def test_derive_scope_and_key_without_scope(self):
        if False:
            return 10
        key = 'test'
        scope = None
        result = kv_utl._derive_scope_and_key(key, scope)
        self.assertEqual((FULL_USER_SCOPE, 'None:%s' % key), result)

    def test_derive_scope_and_key_system_key(self):
        if False:
            return 10
        key = 'system.test'
        scope = None
        result = kv_utl._derive_scope_and_key(key, scope)
        self.assertEqual((FULL_SYSTEM_SCOPE, key.split('.')[1]), result)

    @mock.patch('st2common.util.keyvalue.KeyValuePair')
    @mock.patch('st2common.util.keyvalue.deserialize_key_value')
    def test_get_key(self, deseralize_key_value, KeyValuePair):
        if False:
            return 10
        (key, value) = ('Lindsay', 'Lohan')
        decrypt = False
        KeyValuePair.get_by_scope_and_name().value = value
        deseralize_key_value.return_value = value
        result = kv_utl.get_key(key=key, user_db=auth_db.UserDB(name=USER), decrypt=decrypt)
        self.assertEqual(result, value)
        KeyValuePair.get_by_scope_and_name.assert_called_with(FULL_USER_SCOPE, 'stanley:%s' % key)
        deseralize_key_value.assert_called_once_with(value, decrypt)

    def test_get_key_invalid_input(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, kv_utl.get_key, key=1)
        self.assertRaises(TypeError, kv_utl.get_key, key='test', decrypt='yep')

    @mock.patch('st2common.util.keyvalue.KeyValuePair')
    @mock.patch('st2common.util.keyvalue.deserialize_key_value')
    @mock.patch.object(NoOpRBACUtils, 'assert_user_has_resource_db_permission', mock.MagicMock(side_effect=ResourceAccessDeniedError(user_db=auth_db.UserDB(name=USER), resource_api_or_db=KeyValuePairDB(uid=RESOURCE_UUID), permission_type=PermissionType.KEY_VALUE_PAIR_VIEW)))
    def test_get_key_unauthorized(self, deseralize_key_value, KeyValuePair):
        if False:
            for i in range(10):
                print('nop')
        (key, value) = ('foobar', 'fubar')
        decrypt = False
        KeyValuePair.get_by_scope_and_name().value = value
        deseralize_key_value.return_value = value
        self.assertRaises(ResourceAccessDeniedError, kv_utl.get_key, key=key, user_db=auth_db.UserDB(name=USER), decrypt=decrypt)
        KeyValuePair.get_by_scope_and_name.assert_called_with(FULL_USER_SCOPE, 'stanley:%s' % key)