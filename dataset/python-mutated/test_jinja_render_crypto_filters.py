from __future__ import absolute_import
from oslo_config import cfg
from st2tests.base import CleanDbTestCase
from st2common.constants.keyvalue import FULL_SYSTEM_SCOPE, SYSTEM_SCOPE, DATASTORE_PARENT_SCOPE, FULL_USER_SCOPE, USER_SCOPE
from st2common.models.db.keyvalue import KeyValuePairDB
from st2common.persistence.keyvalue import KeyValuePair
from st2common.services.keyvalues import KeyValueLookup, UserKeyValueLookup
from st2common.util import jinja as jinja_utils
from st2common.util.crypto import read_crypto_key, symmetric_encrypt

class JinjaUtilsDecryptTestCase(CleanDbTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(JinjaUtilsDecryptTestCase, self).setUp()
        crypto_key_path = cfg.CONF.keyvalue.encryption_key_path
        crypto_key = read_crypto_key(key_path=crypto_key_path)
        self.secret = 'Build a wall'
        self.secret_value = symmetric_encrypt(encrypt_key=crypto_key, plaintext=self.secret)
        self.env = jinja_utils.get_jinja_environment()

    def test_filter_decrypt_kv(self):
        if False:
            for i in range(10):
                print('nop')
        KeyValuePair.add_or_update(KeyValuePairDB(name='k8', value=self.secret_value, scope=FULL_SYSTEM_SCOPE, secret=True))
        context = {}
        context.update({SYSTEM_SCOPE: KeyValueLookup(scope=SYSTEM_SCOPE)})
        context.update({DATASTORE_PARENT_SCOPE: {SYSTEM_SCOPE: KeyValueLookup(scope=FULL_SYSTEM_SCOPE)}})
        template = '{{st2kv.system.k8 | decrypt_kv}}'
        actual = self.env.from_string(template).render(context)
        self.assertEqual(actual, self.secret)

    def test_filter_decrypt_kv_datastore_value_doesnt_exist(self):
        if False:
            for i in range(10):
                print('nop')
        context = {}
        context.update({SYSTEM_SCOPE: KeyValueLookup(scope=SYSTEM_SCOPE)})
        context.update({DATASTORE_PARENT_SCOPE: {SYSTEM_SCOPE: KeyValueLookup(scope=FULL_SYSTEM_SCOPE)}})
        template = '{{st2kv.system.doesnt_exist | decrypt_kv}}'
        expected_msg = 'Referenced datastore item "st2kv.system.doesnt_exist" doesn\'t exist or it contains an empty string'
        self.assertRaisesRegexp(ValueError, expected_msg, self.env.from_string(template).render, context)

    def test_filter_decrypt_kv_with_user_scope_value(self):
        if False:
            for i in range(10):
                print('nop')
        KeyValuePair.add_or_update(KeyValuePairDB(name='stanley:k8', value=self.secret_value, scope=FULL_USER_SCOPE, secret=True))
        context = {}
        context.update({USER_SCOPE: UserKeyValueLookup(user='stanley', scope=USER_SCOPE)})
        context.update({DATASTORE_PARENT_SCOPE: {USER_SCOPE: UserKeyValueLookup(user='stanley', scope=FULL_USER_SCOPE)}})
        template = '{{st2kv.user.k8 | decrypt_kv}}'
        actual = self.env.from_string(template).render(context)
        self.assertEqual(actual, self.secret)

    def test_filter_decrypt_kv_with_user_scope_value_datastore_value_doesnt_exist(self):
        if False:
            while True:
                i = 10
        context = {}
        context.update({SYSTEM_SCOPE: KeyValueLookup(scope=SYSTEM_SCOPE)})
        context.update({DATASTORE_PARENT_SCOPE: {USER_SCOPE: UserKeyValueLookup(user='stanley', scope=FULL_USER_SCOPE)}})
        template = '{{st2kv.user.doesnt_exist | decrypt_kv}}'
        expected_msg = 'Referenced datastore item "st2kv.user.doesnt_exist" doesn\'t exist or it contains an empty string'
        self.assertRaisesRegexp(ValueError, expected_msg, self.env.from_string(template).render, context)