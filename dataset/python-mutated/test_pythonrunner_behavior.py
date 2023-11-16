"""
This tests whether an action which is python-script behaves as we expect.
"""
import os
import mock
import tempfile
from st2common.util.monkey_patch import use_select_poll_workaround
use_select_poll_workaround()
from oslo_config import cfg
from python_runner import python_runner
from st2common.util.virtualenvs import setup_pack_virtualenv
from st2tests import config
from st2tests.base import CleanFilesTestCase
from st2tests.base import CleanDbTestCase
from st2tests.fixtures.packs.test_library_dependencies.fixture import PACK_NAME as TEST_LIBRARY_DEPENDENCIES
from st2tests.fixturesloader import get_fixtures_base_path
__all__ = ['PythonRunnerBehaviorTestCase']
FIXTURES_BASE_PATH = get_fixtures_base_path()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WRAPPER_SCRIPT_PATH = os.path.join(BASE_DIR, '../../../python_runner/python_runner/python_action_wrapper.py')
WRAPPER_SCRIPT_PATH = os.path.abspath(WRAPPER_SCRIPT_PATH)

class PythonRunnerBehaviorTestCase(CleanFilesTestCase, CleanDbTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(PythonRunnerBehaviorTestCase, self).setUp()
        config.parse_args()
        dir_path = tempfile.mkdtemp()
        cfg.CONF.set_override(name='base_path', override=dir_path, group='system')
        self.base_path = dir_path
        self.virtualenvs_path = os.path.join(self.base_path, 'virtualenvs/')
        self.to_delete_directories.append(self.base_path)

    def test_priority_of_loading_library_after_setup_pack_virtualenv(self):
        if False:
            i = 10
            return i + 15
        "\n        This test checks priority of loading library, whether the library which is specified in\n        the 'requirements.txt' of pack is loaded when a same name module is also specified in the\n        'requirements.txt' of st2, at a subprocess in ActionRunner.\n\n        To test above, this uses 'get_library_path.py' action in 'test_library_dependencies' pack.\n        This action returns file-path of imported module which is specified by 'module' parameter.\n        "
        pack_name = TEST_LIBRARY_DEPENDENCIES
        setup_pack_virtualenv(pack_name=pack_name)
        self.assertTrue(os.path.exists(os.path.join(self.virtualenvs_path, pack_name)))
        (_, output, _) = self._run_action(pack_name, 'get_library_path.py', {'module': 'six'})
        self.assertEqual(output['result'].find(self.virtualenvs_path), 0)
        (_, output, _) = self._run_action(pack_name, 'get_library_path.py', {'module': 'mock'})
        self.assertEqual(output['result'].find(self.virtualenvs_path), -1)
        (_, output, _) = self._run_action(pack_name, 'get_library_path.py', {'module': 'six'}, {'_sandbox': False})
        self.assertEqual(output['result'].find(self.virtualenvs_path), -1)

    def _run_action(self, pack, action, params, runner_params={}):
        if False:
            while True:
                i = 10
        action_db = mock.Mock()
        action_db.pack = pack
        runner = python_runner.get_runner()
        runner.runner_parameters = {}
        runner.action = action_db
        runner._use_parent_args = False
        for (key, value) in runner_params.items():
            setattr(runner, key, value)
        runner.entry_point = os.path.join(FIXTURES_BASE_PATH, f'packs/{pack}/actions/{action}')
        runner.pre_run()
        return runner.run(params)