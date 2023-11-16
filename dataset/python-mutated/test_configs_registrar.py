from __future__ import absolute_import
import six
import mock
from st2common.content import utils as content_utils
from st2common.bootstrap.configsregistrar import ConfigsRegistrar
from st2common.persistence.pack import Pack
from st2common.persistence.pack import Config
from st2tests.api import SUPER_SECRET_PARAMETER
from st2tests.base import CleanDbTestCase
from st2tests.fixtures.packs.dummy_pack_1.fixture import PACK_NAME as DUMMY_PACK_1, PACK_PATH as PACK_1_PATH
from st2tests.fixtures.packs.dummy_pack_6.fixture import PACK_NAME as DUMMY_PACK_6, PACK_PATH as PACK_6_PATH
from st2tests.fixtures.packs.dummy_pack_11.fixture import PACK_NAME as DUMMY_PACK_11, PACK_PATH as PACK_11_PATH
from st2tests.fixtures.packs.dummy_pack_19.fixture import PACK_NAME as DUMMY_PACK_19, PACK_PATH as PACK_19_PATH
from st2tests.fixtures.packs.dummy_pack_22.fixture import PACK_NAME as DUMMY_PACK_22, PACK_PATH as PACK_22_PATH
__all__ = ['ConfigsRegistrarTestCase']

class ConfigsRegistrarTestCase(CleanDbTestCase):

    def test_register_configs_for_all_packs(self):
        if False:
            while True:
                i = 10
        pack_dbs = Pack.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(config_dbs), 0)
        registrar = ConfigsRegistrar(use_pack_cache=False)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_1: PACK_1_PATH}
        packs_base_paths = content_utils.get_packs_base_paths()
        registrar.register_from_packs(base_dirs=packs_base_paths)
        pack_dbs = Pack.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 1)
        self.assertEqual(len(config_dbs), 1)
        config_db = config_dbs[0]
        self.assertEqual(config_db.values['api_key'], '{{st2kv.user.api_key}}')
        self.assertEqual(config_db.values['api_secret'], SUPER_SECRET_PARAMETER)
        self.assertEqual(config_db.values['region'], 'us-west-1')

    def test_register_all_configs_invalid_config_no_config_schema(self):
        if False:
            while True:
                i = 10
        pack_dbs = Pack.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(config_dbs), 0)
        registrar = ConfigsRegistrar(use_pack_cache=False, validate_configs=False)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_6: PACK_6_PATH}
        packs_base_paths = content_utils.get_packs_base_paths()
        registrar.register_from_packs(base_dirs=packs_base_paths)
        pack_dbs = Pack.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 1)
        self.assertEqual(len(config_dbs), 1)

    def test_register_all_configs_with_config_schema_validation_validation_failure_1(self):
        if False:
            while True:
                i = 10
        pack_dbs = Pack.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(config_dbs), 0)
        registrar = ConfigsRegistrar(use_pack_cache=False, fail_on_failure=True, validate_configs=True)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_6: PACK_6_PATH}
        registrar._register_pack_db = mock.Mock()
        registrar._register_pack(pack_name='dummy_pack_5', pack_dir=PACK_6_PATH)
        packs_base_paths = content_utils.get_packs_base_paths()
        if six.PY3:
            expected_msg = 'Failed validating attribute "regions" in config for pack "dummy_pack_6" (.*?): 1000 is not of type \'array\''
        else:
            expected_msg = 'Failed validating attribute "regions" in config for pack "dummy_pack_6" (.*?): 1000 is not of type u\'array\''
        self.assertRaisesRegexp(ValueError, expected_msg, registrar.register_from_packs, base_dirs=packs_base_paths)

    def test_register_all_configs_with_config_schema_validation_validation_failure_2(self):
        if False:
            print('Hello World!')
        pack_dbs = Pack.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(config_dbs), 0)
        registrar = ConfigsRegistrar(use_pack_cache=False, fail_on_failure=True, validate_configs=True)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_19: PACK_19_PATH}
        registrar._register_pack_db = mock.Mock()
        registrar._register_pack(pack_name=DUMMY_PACK_19, pack_dir=PACK_19_PATH)
        packs_base_paths = content_utils.get_packs_base_paths()
        if six.PY3:
            expected_msg = 'Failed validating attribute "instances.0.alias" in config for pack "dummy_pack_19" (.*?): {\'not\': \'string\'} is not of type \'string\''
        else:
            expected_msg = 'Failed validating attribute "instances.0.alias" in config for pack "dummy_pack_19" (.*?): {\'not\': \'string\'} is not of type u\'string\''
        self.assertRaisesRegexp(ValueError, expected_msg, registrar.register_from_packs, base_dirs=packs_base_paths)

    def test_register_all_configs_with_config_schema_validation_validation_failure_3(self):
        if False:
            return 10
        pack_dbs = Pack.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(config_dbs), 0)
        registrar = ConfigsRegistrar(use_pack_cache=False, fail_on_failure=True, validate_configs=True)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_11: PACK_11_PATH}
        registrar._register_pack_db = mock.Mock()
        registrar._register_pack(pack_name=DUMMY_PACK_11, pack_dir=PACK_11_PATH)
        packs_base_paths = content_utils.get_packs_base_paths()
        expected_msg = 'Values specified as "secret: True" in config schema are automatically decrypted by default. Use of "decrypt_kv" jinja filter is not allowed for such values. Please check the specified values in the config or the default values in the schema.'
        self.assertRaisesRegexp(ValueError, expected_msg, registrar.register_from_packs, base_dirs=packs_base_paths)

    def test_register_all_configs_with_config_schema_validation_validation_failure_4(self):
        if False:
            for i in range(10):
                print('nop')
        pack_dbs = Pack.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(config_dbs), 0)
        registrar = ConfigsRegistrar(use_pack_cache=False, fail_on_failure=True, validate_configs=True)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_22: PACK_22_PATH}
        registrar._register_pack_db = mock.Mock()
        registrar._register_pack(pack_name=DUMMY_PACK_22, pack_dir=PACK_22_PATH)
        packs_base_paths = content_utils.get_packs_base_paths()
        expected_msg = 'Values specified as "secret: True" in config schema are automatically decrypted by default. Use of "decrypt_kv" jinja filter is not allowed for such values. Please check the specified values in the config or the default values in the schema.'
        self.assertRaisesRegexp(ValueError, expected_msg, registrar.register_from_packs, base_dirs=packs_base_paths)