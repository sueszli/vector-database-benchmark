from __future__ import absolute_import
import copy
import os
import six
from st2common.content.loader import MetaLoader
from st2common.models.api.action import ActionAPI, LiveActionAPI, ActionExecutionStateAPI, RunnerTypeAPI, ActionAliasAPI
from st2common.models.api.auth import ApiKeyAPI, UserAPI
from st2common.models.api.execution import ActionExecutionAPI
from st2common.models.api.policy import PolicyTypeAPI, PolicyAPI
from st2common.models.api.rule import RuleAPI
from st2common.models.api.rule_enforcement import RuleEnforcementAPI
from st2common.models.api.sensor import SensorTypeAPI
from st2common.models.api.trace import TraceAPI
from st2common.models.api.trigger import TriggerAPI, TriggerTypeAPI, TriggerInstanceAPI
from st2common.models.db.action import ActionDB
from st2common.models.db.actionalias import ActionAliasDB
from st2common.models.db.auth import ApiKeyDB, UserDB
from st2common.models.db.liveaction import LiveActionDB
from st2common.models.db.executionstate import ActionExecutionStateDB
from st2common.models.db.runner import RunnerTypeDB
from st2common.models.db.execution import ActionExecutionDB
from st2common.models.db.policy import PolicyTypeDB, PolicyDB
from st2common.models.db.rule import RuleDB
from st2common.models.db.rule_enforcement import RuleEnforcementDB
from st2common.models.db.sensor import SensorTypeDB
from st2common.models.db.trace import TraceDB
from st2common.models.db.trigger import TriggerDB, TriggerTypeDB, TriggerInstanceDB
from st2common.persistence.action import Action
from st2common.persistence.actionalias import ActionAlias
from st2common.persistence.execution import ActionExecution
from st2common.persistence.executionstate import ActionExecutionState
from st2common.persistence.auth import ApiKey, User
from st2common.persistence.liveaction import LiveAction
from st2common.persistence.runner import RunnerType
from st2common.persistence.policy import PolicyType, Policy
from st2common.persistence.rule import Rule
from st2common.persistence.rule_enforcement import RuleEnforcement
from st2common.persistence.sensor import SensorType
from st2common.persistence.trace import Trace
from st2common.persistence.trigger import Trigger, TriggerType, TriggerInstance
ALLOWED_DB_FIXTURES = ['actions', 'actionstates', 'aliases', 'executions', 'liveactions', 'policies', 'policytypes', 'rules', 'runners', 'sensors', 'triggertypes', 'triggers', 'triggerinstances', 'traces', 'apikeys', 'users', 'enforcements']
ALLOWED_FIXTURES = copy.copy(ALLOWED_DB_FIXTURES)
ALLOWED_FIXTURES.extend(['actionchains', 'workflows'])
FIXTURE_DB_MODEL = {'actions': ActionDB, 'aliases': ActionAliasDB, 'actionstates': ActionExecutionStateDB, 'apikeys': ApiKeyDB, 'enforcements': RuleEnforcementDB, 'executions': ActionExecutionDB, 'liveactions': LiveActionDB, 'policies': PolicyDB, 'policytypes': PolicyTypeDB, 'rules': RuleDB, 'runners': RunnerTypeDB, 'sensors': SensorTypeDB, 'traces': TraceDB, 'triggertypes': TriggerTypeDB, 'triggers': TriggerDB, 'triggerinstances': TriggerInstanceDB, 'users': UserDB}
FIXTURE_API_MODEL = {'actions': ActionAPI, 'aliases': ActionAliasAPI, 'actionstates': ActionExecutionStateAPI, 'apikeys': ApiKeyAPI, 'enforcements': RuleEnforcementAPI, 'executions': ActionExecutionAPI, 'liveactions': LiveActionAPI, 'policies': PolicyAPI, 'policytypes': PolicyTypeAPI, 'rules': RuleAPI, 'runners': RunnerTypeAPI, 'sensors': SensorTypeAPI, 'traces': TraceAPI, 'triggertypes': TriggerTypeAPI, 'triggers': TriggerAPI, 'triggerinstances': TriggerInstanceAPI, 'users': UserAPI}
FIXTURE_PERSISTENCE_MODEL = {'actions': Action, 'aliases': ActionAlias, 'actionstates': ActionExecutionState, 'apikeys': ApiKey, 'enforcements': RuleEnforcement, 'executions': ActionExecution, 'liveactions': LiveAction, 'policies': Policy, 'policytypes': PolicyType, 'rules': Rule, 'runners': RunnerType, 'sensors': SensorType, 'traces': Trace, 'triggertypes': TriggerType, 'triggers': Trigger, 'triggerinstances': TriggerInstance, 'users': User}
GIT_SUBMODULES_NOT_CHECKED_OUT_ERROR = '\nGit submodule "%s" is not checked out. Make sure to run "git submodule update --init\n --recursive" in the repository root directory to check out all the\nsubmodules.\n'.replace('\n', '').strip()

def get_fixtures_base_path():
    if False:
        while True:
            i = 10
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'fixtures'))

def get_fixtures_packs_base_path():
    if False:
        return 10
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'fixtures/packs'))

def get_resources_base_path():
    if False:
        return 10
    return os.path.abspath(os.path.join(os.path.dirname(__file__), 'resources'))

def get_fixture_name_and_path(fixture_file):
    if False:
        i = 10
        return i + 15
    fixture_path = os.path.dirname(fixture_file)
    fixture_name = os.path.basename(fixture_path)
    return (fixture_name, fixture_path)

class FixturesLoader(object):

    def __init__(self):
        if False:
            return 10
        self.meta_loader = MetaLoader()

    def save_fixtures_to_db(self, fixtures_pack='generic', fixtures_dict=None, use_object_ids=False):
        if False:
            i = 10
            return i + 15
        "\n        Loads fixtures specified in fixtures_dict into the database\n        and returns DB models for the fixtures.\n\n        fixtures_dict should be of the form:\n        {\n            'actions': ['action-1.yaml', 'action-2.yaml'],\n            'rules': ['rule-1.yaml'],\n            'liveactions': ['execution-1.yaml']\n        }\n\n        :param fixtures_pack: Name of the pack to load fixtures from.\n        :type fixtures_pack: ``str``\n\n        :param fixtures_dict: Dictionary specifying the fixtures to load for each type.\n        :type fixtures_dict: ``dict``\n\n        :param use_object_ids: Use object id primary key from fixture file (if available) when\n                              storing objects in the database. By default id in\n                              file is discarded / not used and a new random one\n                              is generated.\n        :type use_object_ids: ``bool``\n\n        :rtype: ``dict``\n        "
        if fixtures_dict is None:
            fixtures_dict = {}
        fixtures_pack_path = self._validate_fixtures_pack(fixtures_pack)
        self._validate_fixture_dict(fixtures_dict, allowed=ALLOWED_DB_FIXTURES)
        db_models = {}
        for (fixture_type, fixtures) in six.iteritems(fixtures_dict):
            API_MODEL = FIXTURE_API_MODEL.get(fixture_type, None)
            PERSISTENCE_MODEL = FIXTURE_PERSISTENCE_MODEL.get(fixture_type, None)
            loaded_fixtures = {}
            for fixture in fixtures:
                if fixture in loaded_fixtures:
                    msg = 'Fixture "%s" is specified twice, probably a typo.' % fixture
                    raise ValueError(msg)
                fixture_dict = self.meta_loader.load(self._get_fixture_file_path_abs(fixtures_pack_path, fixture_type, fixture))
                api_model = API_MODEL(**fixture_dict)
                db_model = API_MODEL.to_model(api_model)
                if use_object_ids and 'id' in fixture_dict:
                    db_model.id = fixture_dict['id']
                db_model = PERSISTENCE_MODEL.add_or_update(db_model)
                loaded_fixtures[fixture] = db_model
            db_models[fixture_type] = loaded_fixtures
        return db_models

    def load_fixtures(self, fixtures_pack='generic', fixtures_dict=None):
        if False:
            i = 10
            return i + 15
        "\n        Loads fixtures specified in fixtures_dict. We\n        simply want to load the meta into dict objects.\n\n        fixtures_dict should be of the form:\n        {\n            'actionchains': ['actionchain1.yaml', 'actionchain2.yaml'],\n            'workflows': ['workflow.yaml']\n        }\n\n        :param fixtures_pack: Name of the pack to load fixtures from.\n        :type fixtures_pack: ``str``\n\n        :param fixtures_dict: Dictionary specifying the fixtures to load for each type.\n        :type fixtures_dict: ``dict``\n\n        :rtype: ``dict``\n        "
        if not fixtures_dict:
            return {}
        fixtures_pack_path = self._validate_fixtures_pack(fixtures_pack)
        self._validate_fixture_dict(fixtures_dict)
        all_fixtures = {}
        for (fixture_type, fixtures) in six.iteritems(fixtures_dict):
            loaded_fixtures = {}
            for fixture in fixtures:
                fixture_dict = self.meta_loader.load(self._get_fixture_file_path_abs(fixtures_pack_path, fixture_type, fixture))
                loaded_fixtures[fixture] = fixture_dict
            all_fixtures[fixture_type] = loaded_fixtures
        return all_fixtures

    def load_models(self, fixtures_pack='generic', fixtures_dict=None):
        if False:
            print('Hello World!')
        "\n        Loads fixtures specified in fixtures_dict as db models. This method must be\n        used for fixtures that have associated DB models. We simply want to load the\n        meta as DB models but don't want to save them to db.\n\n        fixtures_dict should be of the form:\n        {\n            'actions': ['action-1.yaml', 'action-2.yaml'],\n            'rules': ['rule-1.yaml'],\n            'liveactions': ['execution-1.yaml']\n        }\n\n        :param fixtures_pack: Name of the pack to load fixtures from.\n        :type fixtures_pack: ``str``\n\n        :param fixtures_dict: Dictionary specifying the fixtures to load for each type.\n        :type fixtures_dict: ``dict``\n\n        :rtype: ``dict``\n        "
        if not fixtures_dict:
            return {}
        fixtures_pack_path = self._validate_fixtures_pack(fixtures_pack)
        self._validate_fixture_dict(fixtures_dict, allowed=ALLOWED_DB_FIXTURES)
        all_fixtures = {}
        for (fixture_type, fixtures) in six.iteritems(fixtures_dict):
            API_MODEL = FIXTURE_API_MODEL.get(fixture_type, None)
            loaded_models = {}
            for fixture in fixtures:
                fixture_dict = self.meta_loader.load(self._get_fixture_file_path_abs(fixtures_pack_path, fixture_type, fixture))
                api_model = API_MODEL(**fixture_dict)
                db_model = API_MODEL.to_model(api_model)
                loaded_models[fixture] = db_model
            all_fixtures[fixture_type] = loaded_models
        return all_fixtures

    def delete_fixtures_from_db(self, fixtures_pack='generic', fixtures_dict=None, raise_on_fail=False):
        if False:
            while True:
                i = 10
        "\n        Deletes fixtures specified in fixtures_dict from the database.\n\n        fixtures_dict should be of the form:\n        {\n            'actions': ['action-1.yaml', 'action-2.yaml'],\n            'rules': ['rule-1.yaml'],\n            'liveactions': ['execution-1.yaml']\n        }\n\n        :param fixtures_pack: Name of the pack to delete fixtures from.\n        :type fixtures_pack: ``str``\n\n        :param fixtures_dict: Dictionary specifying the fixtures to delete for each type.\n        :type fixtures_dict: ``dict``\n\n        :param raise_on_fail: Optional If True, raises exception if delete fails on any fixture.\n        :type raise_on_fail: ``boolean``\n        "
        if not fixtures_dict:
            return
        fixtures_pack_path = self._validate_fixtures_pack(fixtures_pack)
        self._validate_fixture_dict(fixtures_dict)
        for (fixture_type, fixtures) in six.iteritems(fixtures_dict):
            API_MODEL = FIXTURE_API_MODEL.get(fixture_type, None)
            PERSISTENCE_MODEL = FIXTURE_PERSISTENCE_MODEL.get(fixture_type, None)
            for fixture in fixtures:
                fixture_dict = self.meta_loader.load(self._get_fixture_file_path_abs(fixtures_pack_path, fixture_type, fixture))
                api_model = API_MODEL(**fixture_dict)
                db_model = API_MODEL.to_model(api_model)
                try:
                    PERSISTENCE_MODEL.delete(db_model)
                except:
                    if raise_on_fail:
                        raise

    def delete_models_from_db(self, models_dict, raise_on_fail=False):
        if False:
            print('Hello World!')
        "\n        Deletes models specified in models_dict from the database.\n\n        models_dict should be of the form:\n        {\n            'actions': [ACTION1, ACTION2],\n            'rules': [RULE1],\n            'liveactions': [EXECUTION]\n        }\n\n        :param fixtures_dict: Dictionary specifying the fixtures to delete for each type.\n        :type fixtures_dict: ``dict``.\n\n        :param raise_on_fail: Optional If True, raises exception if delete fails on any model.\n        :type raise_on_fail: ``boolean``\n        "
        for (model_type, models) in six.iteritems(models_dict):
            PERSISTENCE_MODEL = FIXTURE_PERSISTENCE_MODEL.get(model_type, None)
            for model in models:
                try:
                    PERSISTENCE_MODEL.delete(model)
                except:
                    if raise_on_fail:
                        raise

    def _validate_fixtures_pack(self, fixtures_pack):
        if False:
            for i in range(10):
                print('nop')
        fixtures_pack_path = self._get_fixtures_pack_path(fixtures_pack)
        if not self._is_fixture_pack_exists(fixtures_pack_path):
            raise Exception('Fixtures pack not found ' + 'in fixtures path %s.' % get_fixtures_base_path())
        return fixtures_pack_path

    def _validate_fixture_dict(self, fixtures_dict, allowed=ALLOWED_FIXTURES):
        if False:
            return 10
        fixture_types = list(fixtures_dict.keys())
        for fixture_type in fixture_types:
            if fixture_type not in allowed:
                raise Exception('Disallowed fixture type: %s. Valid fixture types are: %s' % (fixture_type, ', '.join(allowed)))

    def _is_fixture_pack_exists(self, fixtures_pack_path):
        if False:
            while True:
                i = 10
        return os.path.exists(fixtures_pack_path)

    def _get_fixture_file_path_abs(self, fixtures_pack_path, fixtures_type, fixture_name):
        if False:
            return 10
        return os.path.join(fixtures_pack_path, fixtures_type, fixture_name)

    def _get_fixtures_pack_path(self, fixtures_pack_name):
        if False:
            return 10
        return os.path.join(get_fixtures_base_path(), fixtures_pack_name)

    def get_fixture_file_path_abs(self, fixtures_pack, fixtures_type, fixture_name):
        if False:
            while True:
                i = 10
        return os.path.join(get_fixtures_base_path(), fixtures_pack, fixtures_type, fixture_name)

def assert_submodules_are_checked_out():
    if False:
        return 10
    '\n    Function which verifies that user has ran "git submodule update --init --recursive" in the\n    root of the directory and that the "st2tests/st2tests/fixtures/packs/test" git repo submodule\n    used by the tests is checked out.\n    '
    from st2tests.fixtures.packs.test_content_version_fixture.fixture import PACK_PATH
    pack_path = os.path.abspath(PACK_PATH)
    submodule_git_dir_or_file_path = os.path.join(pack_path, '.git')
    if not os.path.exists(submodule_git_dir_or_file_path):
        raise ValueError(GIT_SUBMODULES_NOT_CHECKED_OUT_ERROR % pack_path)
    return True