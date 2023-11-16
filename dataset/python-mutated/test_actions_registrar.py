from __future__ import absolute_import
import six
import jsonschema
import mock
import yaml
import st2common.bootstrap.actionsregistrar as actions_registrar
from st2common.persistence.action import Action
import st2common.validators.api.action as action_validator
from st2common.models.db.runner import RunnerTypeDB
import st2tests.base as tests_base
from st2tests.fixtures.generic.fixture import PACK_NAME as GENERIC_PACK, PACK_PATH as GENERIC_PACK_PATH, PACK_BASE_PATH as PACKS_BASE_PATH
import st2tests.fixturesloader as fixtures_loader
MOCK_RUNNER_TYPE_DB = RunnerTypeDB(name='run-local', runner_module='st2.runners.local')

@mock.patch('st2common.content.utils.get_pack_base_path', mock.Mock(return_value=GENERIC_PACK_PATH))
class ActionsRegistrarTest(tests_base.DbTestCase):

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    @mock.patch.object(action_validator, 'get_runner_model', mock.MagicMock(return_value=MOCK_RUNNER_TYPE_DB))
    def test_register_all_actions(self):
        if False:
            return 10
        try:
            all_actions_in_db = Action.get_all()
            actions_registrar.register_actions(packs_base_paths=[PACKS_BASE_PATH])
        except Exception as e:
            print(six.text_type(e))
            self.fail('All actions must be registered without exceptions.')
        else:
            all_actions_in_db = Action.get_all()
            self.assertTrue(len(all_actions_in_db) > 0)
        expected_path = 'actions/action-with-no-parameters.yaml'
        self.assertEqual(all_actions_in_db[0].metadata_file, expected_path)

    def test_register_actions_from_bad_pack(self):
        if False:
            print('Hello World!')
        packs_base_path = tests_base.get_fixtures_path()
        try:
            actions_registrar.register_actions(packs_base_paths=[packs_base_path])
            self.fail('Should have thrown.')
        except:
            pass

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    @mock.patch.object(action_validator, 'get_runner_model', mock.MagicMock(return_value=MOCK_RUNNER_TYPE_DB))
    def test_pack_name_missing(self):
        if False:
            print('Hello World!')
        registrar = actions_registrar.ActionsRegistrar()
        loader = fixtures_loader.FixturesLoader()
        action_file = loader.get_fixture_file_path_abs(GENERIC_PACK, 'actions', 'action_3_pack_missing.yaml')
        registrar._register_action('dummy', action_file)
        action_name = None
        with open(action_file, 'r') as fd:
            content = yaml.safe_load(fd)
            action_name = str(content['name'])
            action_db = Action.get_by_name(action_name)
            expected_msg = 'Content pack must be set to dummy'
            self.assertEqual(action_db.pack, 'dummy', expected_msg)
            Action.delete(action_db)

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    @mock.patch.object(action_validator, 'get_runner_model', mock.MagicMock(return_value=MOCK_RUNNER_TYPE_DB))
    def test_register_action_with_no_params(self):
        if False:
            return 10
        registrar = actions_registrar.ActionsRegistrar()
        loader = fixtures_loader.FixturesLoader()
        action_file = loader.get_fixture_file_path_abs(GENERIC_PACK, 'actions', 'action-with-no-parameters.yaml')
        self.assertEqual(registrar._register_action('dummy', action_file), False)

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    @mock.patch.object(action_validator, 'get_runner_model', mock.MagicMock(return_value=MOCK_RUNNER_TYPE_DB))
    def test_register_action_invalid_parameter_type_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        registrar = actions_registrar.ActionsRegistrar()
        loader = fixtures_loader.FixturesLoader()
        action_file = loader.get_fixture_file_path_abs(GENERIC_PACK, 'actions', 'action_invalid_param_type.yaml')
        expected_msg = "'list' is not valid under any of the given schema"
        self.assertRaisesRegexp(jsonschema.ValidationError, expected_msg, registrar._register_action, 'dummy', action_file)

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    @mock.patch.object(action_validator, 'get_runner_model', mock.MagicMock(return_value=MOCK_RUNNER_TYPE_DB))
    def test_register_action_invalid_parameter_name(self):
        if False:
            print('Hello World!')
        registrar = actions_registrar.ActionsRegistrar()
        loader = fixtures_loader.FixturesLoader()
        action_file = loader.get_fixture_file_path_abs(GENERIC_PACK, 'actions', 'action_invalid_parameter_name.yaml')
        expected_msg = 'Parameter name "action-name" is invalid. Valid characters for parameter name are'
        self.assertRaisesRegexp(jsonschema.ValidationError, expected_msg, registrar._register_action, GENERIC_PACK, action_file)

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    @mock.patch.object(action_validator, 'get_runner_model', mock.MagicMock(return_value=MOCK_RUNNER_TYPE_DB))
    def test_invalid_params_schema(self):
        if False:
            for i in range(10):
                print('nop')
        registrar = actions_registrar.ActionsRegistrar()
        loader = fixtures_loader.FixturesLoader()
        action_file = loader.get_fixture_file_path_abs(GENERIC_PACK, 'actions', 'action-invalid-schema-params.yaml')
        try:
            registrar._register_action(GENERIC_PACK, action_file)
            self.fail('Invalid action schema. Should have failed.')
        except jsonschema.ValidationError:
            pass

    @mock.patch.object(action_validator, '_is_valid_pack', mock.MagicMock(return_value=True))
    @mock.patch.object(action_validator, 'get_runner_model', mock.MagicMock(return_value=MOCK_RUNNER_TYPE_DB))
    def test_action_update(self):
        if False:
            for i in range(10):
                print('nop')
        registrar = actions_registrar.ActionsRegistrar()
        loader = fixtures_loader.FixturesLoader()
        action_file = loader.get_fixture_file_path_abs(GENERIC_PACK, 'actions', 'action1.yaml')
        registrar._register_action('wolfpack', action_file)
        registrar._register_action('wolfpack', action_file)
        action_name = None
        with open(action_file, 'r') as fd:
            content = yaml.safe_load(fd)
            action_name = str(content['name'])
            action_db = Action.get_by_name(action_name)
            expected_msg = 'Content pack must be set to wolfpack'
            self.assertEqual(action_db.pack, 'wolfpack', expected_msg)
            Action.delete(action_db)