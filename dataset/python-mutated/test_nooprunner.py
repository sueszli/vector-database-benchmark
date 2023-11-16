from __future__ import absolute_import
import uuid
import mock
import st2tests.config as tests_config
tests_config.parse_args()
from unittest2 import TestCase
from st2common.constants import action as action_constants
from st2tests.fixtures.generic.fixture import PACK_NAME as GENERIC_PACK
from st2tests.fixturesloader import FixturesLoader
from noop_runner import noop_runner

class TestNoopRunner(TestCase):
    fixtures_loader = FixturesLoader()

    def test_noop_command_executes(self):
        if False:
            while True:
                i = 10
        models = TestNoopRunner.fixtures_loader.load_models(fixtures_pack=GENERIC_PACK, fixtures_dict={'actions': ['noop.yaml']})
        action_db = models['actions']['noop.yaml']
        runner = TestNoopRunner._get_runner(action_db)
        (status, result, _) = runner.run({})
        self.assertEqual(status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(result['failed'], False)
        self.assertEqual(result['succeeded'], True)
        self.assertEqual(result['return_code'], 0)

    @staticmethod
    def _get_runner(action_db):
        if False:
            print('Hello World!')
        runner = noop_runner.NoopRunner(uuid.uuid4().hex)
        runner.action = action_db
        runner.action_name = action_db.name
        runner.liveaction_id = uuid.uuid4().hex
        runner.entry_point = None
        runner.context = dict()
        runner.callback = dict()
        runner.libs_dir_path = None
        runner.auth_token = mock.Mock()
        runner.auth_token.token = 'mock-token'
        return runner