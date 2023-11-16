from __future__ import absolute_import
from st2common.util.monkey_patch import monkey_patch
monkey_patch()
import mock
from st2common.runners import utils
from st2common.services import executions as exe_svc
from st2common.util import action_db as action_db_utils
from st2tests import base
from st2tests import fixturesloader
from st2tests.fixtures.generic.fixture import PACK_NAME as FIXTURES_PACK
from st2tests import config as tests_config
tests_config.parse_args()
TEST_FIXTURES = {'liveactions': ['liveaction1.yaml'], 'actions': ['local.yaml'], 'executions': ['execution1.yaml'], 'runners': ['run-local.yaml']}

class RunnersUtilityTests(base.CleanDbTestCase):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(RunnersUtilityTests, self).__init__(*args, **kwargs)
        self.models = None

    def setUp(self):
        if False:
            print('Hello World!')
        super(RunnersUtilityTests, self).setUp()
        loader = fixturesloader.FixturesLoader()
        self.models = loader.save_fixtures_to_db(fixtures_pack=FIXTURES_PACK, fixtures_dict=TEST_FIXTURES)
        self.liveaction_db = self.models['liveactions']['liveaction1.yaml']
        exe_svc.create_execution_object(self.liveaction_db)
        self.action_db = action_db_utils.get_action_by_ref(self.liveaction_db.action)

    @mock.patch.object(action_db_utils, 'get_action_by_ref', mock.MagicMock(return_value=None))
    def test_invoke_post_run_action_provided(self):
        if False:
            print('Hello World!')
        utils.invoke_post_run(self.liveaction_db, action_db=self.action_db)
        action_db_utils.get_action_by_ref.assert_not_called()

    def test_invoke_post_run_action_exists(self):
        if False:
            i = 10
            return i + 15
        utils.invoke_post_run(self.liveaction_db)

    @mock.patch.object(action_db_utils, 'get_action_by_ref', mock.MagicMock(return_value=None))
    @mock.patch.object(action_db_utils, 'get_runnertype_by_name', mock.MagicMock(return_value=None))
    def test_invoke_post_run_action_does_not_exist(self):
        if False:
            while True:
                i = 10
        utils.invoke_post_run(self.liveaction_db)
        action_db_utils.get_action_by_ref.assert_called_once()
        action_db_utils.get_runnertype_by_name.assert_not_called()