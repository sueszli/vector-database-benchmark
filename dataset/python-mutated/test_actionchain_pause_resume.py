from __future__ import absolute_import
import eventlet
import mock
import os
import tempfile
from st2tests import config as test_config
test_config.parse_args()
from st2common.bootstrap import actionsregistrar
from st2common.bootstrap import runnersregistrar
from st2common.constants import action as action_constants
from st2common.models.db.liveaction import LiveActionDB
from st2common.persistence.execution import ActionExecution
from st2common.persistence.liveaction import LiveAction
from st2common.services import action as action_service
from st2common.transport.liveaction import LiveActionPublisher
from st2common.transport.publishers import CUDPublisher
from st2common.util import action_db as action_utils
from st2common.util import date as date_utils
from st2tests import ExecutionDbTestCase
from st2tests.fixtures.packs.action_chain_tests.fixture import PACK_NAME as TEST_PACK, PACK_PATH as TEST_PACK_PATH
from st2tests.fixtures.packs.core.fixture import PACK_PATH as CORE_PACK_PATH
from st2tests.mocks.liveaction import MockLiveActionPublisherNonBlocking
from six.moves import range
TEST_FIXTURES = {'chains': ['test_pause_resume.yaml', 'test_pause_resume_context_result', 'test_pause_resume_with_published_vars.yaml', 'test_pause_resume_with_error.yaml', 'test_pause_resume_with_subworkflow.yaml', 'test_pause_resume_with_context_access.yaml', 'test_pause_resume_with_init_vars.yaml', 'test_pause_resume_with_no_more_task.yaml', 'test_pause_resume_last_task_failed_with_no_next_task.yaml'], 'actions': ['test_pause_resume.yaml', 'test_pause_resume_context_result', 'test_pause_resume_with_published_vars.yaml', 'test_pause_resume_with_error.yaml', 'test_pause_resume_with_subworkflow.yaml', 'test_pause_resume_with_context_access.yaml', 'test_pause_resume_with_init_vars.yaml', 'test_pause_resume_with_no_more_task.yaml', 'test_pause_resume_last_task_failed_with_no_next_task.yaml']}
PACKS = [TEST_PACK_PATH, CORE_PACK_PATH]
USERNAME = 'stanley'

@mock.patch.object(CUDPublisher, 'publish_update', mock.MagicMock(return_value=None))
@mock.patch.object(CUDPublisher, 'publish_create', mock.MagicMock(return_value=None))
@mock.patch.object(LiveActionPublisher, 'publish_state', mock.MagicMock(side_effect=MockLiveActionPublisherNonBlocking.publish_state))
class ActionChainRunnerPauseResumeTest(ExecutionDbTestCase):
    temp_file_path = None

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super(ActionChainRunnerPauseResumeTest, cls).setUpClass()
        runnersregistrar.register_runners()
        actions_registrar = actionsregistrar.ActionsRegistrar(use_pack_cache=False, fail_on_failure=True)
        for pack in PACKS:
            actions_registrar.register_from_pack(pack)

    def setUp(self):
        if False:
            while True:
                i = 10
        super(ActionChainRunnerPauseResumeTest, self).setUp()
        (_, self.temp_file_path) = tempfile.mkstemp()
        os.chmod(self.temp_file_path, 493)

    def tearDown(self):
        if False:
            return 10
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)
        super(ActionChainRunnerPauseResumeTest, self).tearDown()

    def _wait_for_status(self, liveaction, status, interval=0.1, retries=100):
        if False:
            return 10
        for i in range(0, retries):
            liveaction = LiveAction.get_by_id(str(liveaction.id))
            if liveaction.status != status:
                eventlet.sleep(interval)
                continue
            else:
                break
        return liveaction

    def _wait_for_children(self, execution, interval=0.1, retries=100):
        if False:
            print('Hello World!')
        for i in range(0, retries):
            execution = ActionExecution.get_by_id(str(execution.id))
            if len(getattr(execution, 'children', [])) <= 0:
                eventlet.sleep(interval)
                continue
        return execution

    def test_chain_pause_resume(self):
        if False:
            i = 10
            return i + 15
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 2)

    def test_chain_pause_resume_with_published_vars(self):
        if False:
            while True:
                i = 10
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_with_published_vars'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 2)
        self.assertIn('published', liveaction.result)
        self.assertDictEqual({'var1': 'foobar', 'var2': 'fubar'}, liveaction.result['published'])

    def test_chain_pause_resume_with_published_vars_display_false(self):
        if False:
            return 10
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_with_published_vars'
        params = {'tempfile': path, 'message': 'foobar', 'display_published': False}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 2)
        self.assertNotIn('published', liveaction.result)

    def test_chain_pause_resume_with_error(self):
        if False:
            i = 10
            return i + 15
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_with_error'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 2)
        self.assertTrue(liveaction.result['tasks'][0]['result']['failed'])
        self.assertEqual(1, liveaction.result['tasks'][0]['result']['return_code'])
        self.assertTrue(liveaction.result['tasks'][1]['result']['succeeded'])
        self.assertEqual(0, liveaction.result['tasks'][1]['result']['return_code'])

    def test_chain_pause_resume_cascade_to_subworkflow(self):
        if False:
            for i in range(10):
                print('nop')
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_with_subworkflow'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        execution = self._wait_for_children(execution)
        self.assertEqual(len(execution.children), 1)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_for_status(task1_live, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(task1_live.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        self.assertEqual(len(execution.children), 1)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_for_status(task1_live, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(task1_live)
        self.assertEqual(task1_live.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        self.assertEqual(len(execution.children), 1)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_for_status(task1_live, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(task1_live)
        self.assertEqual(task1_live.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 1)
        subworkflow = liveaction.result['tasks'][0]
        self.assertEqual(len(subworkflow['result']['tasks']), 1)
        self.assertEqual(subworkflow['state'], action_constants.LIVEACTION_STATUS_PAUSED)
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 2)
        subworkflow = liveaction.result['tasks'][0]
        self.assertEqual(len(subworkflow['result']['tasks']), 2)
        self.assertEqual(subworkflow['state'], action_constants.LIVEACTION_STATUS_SUCCEEDED)

    def test_chain_pause_resume_cascade_to_parent_workflow(self):
        if False:
            return 10
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_with_subworkflow'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        execution = self._wait_for_children(execution)
        self.assertEqual(len(execution.children), 1)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_for_status(task1_live, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(task1_live.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (task1_live, task1_exec) = action_service.request_pause(task1_live, USERNAME)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_for_status(task1_live, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(task1_live)
        self.assertEqual(task1_live.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_for_status(task1_live, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(task1_live)
        self.assertEqual(task1_live.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        self.assertEqual(len(execution.children), 1)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 1)
        subworkflow = liveaction.result['tasks'][0]
        self.assertEqual(len(subworkflow['result']['tasks']), 1)
        self.assertEqual(subworkflow['state'], action_constants.LIVEACTION_STATUS_PAUSED)
        (task1_live, task1_exec) = action_service.request_resume(task1_live, USERNAME)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_for_status(task1_live, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(task1_live.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 1)
        subworkflow = liveaction.result['tasks'][0]
        self.assertEqual(len(subworkflow['result']['tasks']), 1)
        self.assertEqual(subworkflow['state'], action_constants.LIVEACTION_STATUS_PAUSED)
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 2)
        subworkflow = liveaction.result['tasks'][0]
        self.assertEqual(len(subworkflow['result']['tasks']), 2)
        self.assertEqual(subworkflow['state'], action_constants.LIVEACTION_STATUS_SUCCEEDED)

    def test_chain_pause_resume_with_context_access(self):
        if False:
            print('Hello World!')
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_with_context_access'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 3)
        self.assertEqual(liveaction.result['tasks'][2]['result']['stdout'], 'foobar')

    def test_chain_pause_resume_with_init_vars(self):
        if False:
            i = 10
            return i + 15
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_with_init_vars'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 2)
        self.assertEqual(liveaction.result['tasks'][1]['result']['stdout'], 'FOOBAR')

    def test_chain_pause_resume_with_no_more_task(self):
        if False:
            print('Hello World!')
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_with_no_more_task'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 1)

    def test_chain_pause_resume_last_task_failed_with_no_next_task(self):
        if False:
            print('Hello World!')
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_pause_resume_last_task_failed_with_no_next_task'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_pause(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSING)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSING, extra_info)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_FAILED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_FAILED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 1)
        self.assertEqual(liveaction.result['tasks'][0]['state'], action_constants.LIVEACTION_STATUS_FAILED)

    def test_chain_pause_resume_status_change(self):
        if False:
            for i in range(10):
                print('nop')
        action = TEST_PACK + '.' + 'test_pause_resume_context_result'
        liveaction = LiveActionDB(action=action)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_PAUSED)
        extra_info = str(liveaction)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_PAUSED, extra_info)
        MockLiveActionPublisherNonBlocking.wait_all()
        last_task_liveaction_id = liveaction.result['tasks'][-1]['liveaction_id']
        action_utils.update_liveaction_status(status=action_constants.LIVEACTION_STATUS_SUCCEEDED, end_timestamp=date_utils.get_datetime_utc_now(), result={'foo': 'bar'}, liveaction_id=last_task_liveaction_id)
        (liveaction, execution) = action_service.request_resume(liveaction, USERNAME)
        liveaction = self._wait_for_status(liveaction, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(liveaction.status, action_constants.LIVEACTION_STATUS_SUCCEEDED, str(liveaction))
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 2)
        self.assertEqual(liveaction.result['tasks'][0]['result']['foo'], 'bar')