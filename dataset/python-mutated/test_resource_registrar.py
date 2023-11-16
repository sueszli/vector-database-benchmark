from __future__ import absolute_import
import six
import mock
from jsonschema import ValidationError
from st2common.content import utils as content_utils
from st2common.bootstrap.base import ResourceRegistrar
from st2common.persistence.pack import Pack
from st2common.persistence.pack import ConfigSchema
from st2tests.base import CleanDbTestCase
from st2tests.fixtures.packs.dummy_pack_1.fixture import PACK_NAME as DUMMY_PACK_1, PACK_PATH as PACK_PATH_1
from st2tests.fixtures.packs.dummy_pack_6.fixture import PACK_NAME as DUMMY_PACK_6, PACK_PATH as PACK_PATH_6
from st2tests.fixtures.packs.dummy_pack_7.fixture import PACK_NAME as DUMMY_PACK_7_NAME, PACK_PATH as PACK_PATH_7
from st2tests.fixtures.packs.dummy_pack_8.fixture import PACK_PATH as PACK_PATH_8
from st2tests.fixtures.packs.dummy_pack_9.fixture import PACK_DIR_NAME as DUMMY_PACK_9, PACK_NAME as DUMMY_PACK_9_DEPS, PACK_PATH as PACK_PATH_9
from st2tests.fixtures.packs.dummy_pack_10.fixture import PACK_PATH as PACK_PATH_10
from st2tests.fixtures.packs.dummy_pack_13.fixture import PACK_PATH as PACK_PATH_13
from st2tests.fixtures.packs.dummy_pack_14.fixture import PACK_PATH as PACK_PATH_14
from st2tests.fixtures.packs.dummy_pack_20.fixture import PACK_NAME as DUMMY_PACK_20, PACK_PATH as PACK_PATH_20
from st2tests.fixtures.packs.dummy_pack_21.fixture import PACK_NAME as DUMMY_PACK_21, PACK_PATH as PACK_PATH_21
from st2tests.fixtures.packs_invalid.dummy_pack_17.fixture import PACK_NAME as DUMMY_PACK_17, PACK_PATH as PACK_PATH_17
from st2tests.fixtures.packs_invalid.dummy_pack_18.fixture import PACK_NAME as DUMMY_PACK_18, PACK_PATH as PACK_PATH_18
__all__ = ['ResourceRegistrarTestCase']

class ResourceRegistrarTestCase(CleanDbTestCase):

    def test_register_packs(self):
        if False:
            i = 10
            return i + 15
        pack_dbs = Pack.get_all()
        config_schema_dbs = ConfigSchema.get_all()
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(config_schema_dbs), 0)
        registrar = ResourceRegistrar(use_pack_cache=False)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_1: PACK_PATH_1}
        packs_base_paths = content_utils.get_packs_base_paths()
        registrar.register_packs(base_dirs=packs_base_paths)
        pack_dbs = Pack.get_all()
        config_schema_dbs = ConfigSchema.get_all()
        self.assertEqual(len(pack_dbs), 1)
        self.assertEqual(len(config_schema_dbs), 1)
        pack_db = pack_dbs[0]
        config_schema_db = config_schema_dbs[0]
        self.assertEqual(pack_db.name, DUMMY_PACK_1)
        self.assertEqual(len(pack_db.contributors), 2)
        self.assertEqual(pack_db.contributors[0], 'John Doe1 <john.doe1@gmail.com>')
        self.assertEqual(pack_db.contributors[1], 'John Doe2 <john.doe2@gmail.com>')
        self.assertIn('api_key', config_schema_db.attributes)
        self.assertIn('api_secret', config_schema_db.attributes)
        excluded_files = ['__init__.pyc', 'actions/dummy1.pyc', 'actions/dummy2.pyc']
        for excluded_file in excluded_files:
            self.assertNotIn(excluded_file, pack_db.files)

    def test_register_pack_arbitrary_properties_are_allowed(self):
        if False:
            i = 10
            return i + 15
        registrar = ResourceRegistrar(use_pack_cache=False)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_20: PACK_PATH_20}
        packs_base_paths = content_utils.get_packs_base_paths()
        registrar.register_packs(base_dirs=packs_base_paths)
        pack_db = Pack.get_by_name(DUMMY_PACK_20)
        self.assertEqual(pack_db.ref, 'dummy_pack_20_ref')
        self.assertEqual(len(pack_db.contributors), 0)

    def test_register_pack_pack_ref(self):
        if False:
            while True:
                i = 10
        pack_dbs = Pack.get_all()
        self.assertEqual(len(pack_dbs), 0)
        registrar = ResourceRegistrar(use_pack_cache=False)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_1: PACK_PATH_1, DUMMY_PACK_6: PACK_PATH_6}
        packs_base_paths = content_utils.get_packs_base_paths()
        registrar.register_packs(base_dirs=packs_base_paths)
        pack_db = Pack.get_by_name(DUMMY_PACK_6)
        self.assertEqual(pack_db.ref, 'dummy_pack_6_ref')
        self.assertEqual(len(pack_db.contributors), 0)
        pack_db = Pack.get_by_name(DUMMY_PACK_1)
        self.assertEqual(pack_db.ref, DUMMY_PACK_1)
        registrar._register_pack_db(pack_name=None, pack_dir=PACK_PATH_7)
        pack_db = Pack.get_by_name(DUMMY_PACK_7_NAME)
        self.assertEqual(pack_db.ref, DUMMY_PACK_7_NAME)
        expected_msg = 'contains invalid characters'
        self.assertRaisesRegexp(ValueError, expected_msg, registrar._register_pack_db, pack_name=None, pack_dir=PACK_PATH_8)

    def test_register_pack_invalid_ref_name_friendly_error_message(self):
        if False:
            while True:
                i = 10
        registrar = ResourceRegistrar(use_pack_cache=False)
        expected_msg = 'Pack ref / name can only contain valid word characters .*?, dashes are not allowed.'
        self.assertRaisesRegexp(ValidationError, expected_msg, registrar._register_pack_db, pack_name=None, pack_dir=PACK_PATH_13)
        try:
            registrar._register_pack_db(pack_name=None, pack_dir=PACK_PATH_13)
        except ValidationError as e:
            self.assertIn("'invalid-has-dash' does not match '^[a-z0-9_]+$'", six.text_type(e))
        else:
            self.fail('Exception not thrown')
        expected_msg = 'Pack name "dummy pack 14" contains invalid characters and "ref" attribute is not available. You either need to add'
        self.assertRaisesRegexp(ValueError, expected_msg, registrar._register_pack_db, pack_name=None, pack_dir=PACK_PATH_14)

    def test_register_pack_pack_stackstorm_version_and_future_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        pack_dbs = Pack.get_all()
        self.assertEqual(len(pack_dbs), 0)
        registrar = ResourceRegistrar(use_pack_cache=False)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_9: PACK_PATH_9}
        packs_base_paths = content_utils.get_packs_base_paths()
        registrar.register_packs(base_dirs=packs_base_paths)
        pack_db = Pack.get_by_name(DUMMY_PACK_9_DEPS)
        self.assertEqual(pack_db.dependencies, ['core=0.2.0'])
        self.assertEqual(pack_db.stackstorm_version, '>=1.6.0, <2.2.0')
        self.assertEqual(pack_db.system, {'centos': {'foo': '>= 1.0'}})
        self.assertEqual(pack_db.python_versions, ['2', '3'])
        self.assertTrue(not hasattr(pack_db, 'future'))
        self.assertTrue(not hasattr(pack_db, 'this'))
        expected_msg = "'wrongstackstormversion' does not match"
        self.assertRaisesRegexp(ValidationError, expected_msg, registrar._register_pack_db, pack_name=None, pack_dir=PACK_PATH_10)

    def test_register_pack_empty_and_invalid_config_schema(self):
        if False:
            print('Hello World!')
        registrar = ResourceRegistrar(use_pack_cache=False, fail_on_failure=True)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_17: PACK_PATH_17}
        packs_base_paths = content_utils.get_packs_base_paths()
        expected_msg = 'Config schema ".*?dummy_pack_17/config.schema.yaml" is empty and invalid.'
        self.assertRaisesRegexp(ValueError, expected_msg, registrar.register_packs, base_dirs=packs_base_paths)

    def test_register_pack_invalid_config_schema_invalid_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        registrar = ResourceRegistrar(use_pack_cache=False, fail_on_failure=True)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_18: PACK_PATH_18}
        packs_base_paths = content_utils.get_packs_base_paths()
        expected_msg = "Additional properties are not allowed \\(\\'invalid\\' was unexpected\\)"
        self.assertRaisesRegexp(ValueError, expected_msg, registrar.register_packs, base_dirs=packs_base_paths)

    def test_register_pack_invalid_python_versions_attribute(self):
        if False:
            while True:
                i = 10
        registrar = ResourceRegistrar(use_pack_cache=False, fail_on_failure=True)
        registrar._pack_loader.get_packs = mock.Mock()
        registrar._pack_loader.get_packs.return_value = {DUMMY_PACK_21: PACK_PATH_21}
        packs_base_paths = content_utils.get_packs_base_paths()
        expected_msg = "'4' is not one of \\['2', '3'\\]"
        self.assertRaisesRegexp(ValueError, expected_msg, registrar.register_packs, base_dirs=packs_base_paths)