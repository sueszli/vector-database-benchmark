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
from st2tests import ExecutionDbTestCase
from st2tests.fixtures.packs.action_chain_tests.fixture import PACK_NAME as TEST_PACK, PACK_PATH as TEST_PACK_PATH
from st2tests.fixtures.packs.core.fixture import PACK_PATH as CORE_PACK_PATH
from st2tests.mocks.liveaction import MockLiveActionPublisherNonBlocking
from six.moves import range
TEST_FIXTURES = {'chains': ['test_cancel.yaml', 'test_cancel_with_subworkflow.yaml'], 'actions': ['test_cancel.yaml', 'test_cancel_with_subworkflow.yaml']}
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
            i = 10
            return i + 15
        super(ActionChainRunnerPauseResumeTest, cls).setUpClass()
        runnersregistrar.register_runners()
        actions_registrar = actionsregistrar.ActionsRegistrar(use_pack_cache=False, fail_on_failure=True)
        for pack in PACKS:
            actions_registrar.register_from_pack(pack)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ActionChainRunnerPauseResumeTest, self).setUp()
        (_, self.temp_file_path) = tempfile.mkstemp()
        os.chmod(self.temp_file_path, 493)

    def tearDown(self):
        if False:
            while True:
                i = 10
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)
        super(ActionChainRunnerPauseResumeTest, self).tearDown()

    def _wait_for_children(self, execution, interval=0.1, retries=100):
        if False:
            i = 10
            return i + 15
        for i in range(0, retries):
            execution = ActionExecution.get_by_id(str(execution.id))
            if len(getattr(execution, 'children', [])) <= 0:
                eventlet.sleep(interval)
                continue
        return execution

    def test_chain_cancel(self):
        if False:
            i = 10
            return i + 15
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_cancel'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_cancellation(liveaction, USERNAME)
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_CANCELING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_CANCELED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 1)

    def test_chain_cancel_cascade_to_subworkflow(self):
        if False:
            i = 10
            return i + 15
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_cancel_with_subworkflow'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        execution = self._wait_for_children(execution)
        self.assertEqual(len(execution.children), 1)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_on_status(task1_live, action_constants.LIVEACTION_STATUS_RUNNING)
        (liveaction, execution) = action_service.request_cancellation(liveaction, USERNAME)
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_CANCELING)
        self.assertEqual(len(execution.children), 1)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_on_status(task1_live, action_constants.LIVEACTION_STATUS_CANCELING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_CANCELED)
        self.assertEqual(len(execution.children), 1)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_on_status(task1_live, action_constants.LIVEACTION_STATUS_CANCELED)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 1)
        subworkflow = liveaction.result['tasks'][0]
        self.assertEqual(len(subworkflow['result']['tasks']), 1)
        self.assertEqual(subworkflow['state'], action_constants.LIVEACTION_STATUS_CANCELED)

    def test_chain_cancel_cascade_to_parent_workflow(self):
        if False:
            for i in range(10):
                print('nop')
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        action = TEST_PACK + '.' + 'test_cancel_with_subworkflow'
        params = {'tempfile': path, 'message': 'foobar'}
        liveaction = LiveActionDB(action=action, parameters=params)
        (liveaction, execution) = action_service.request(liveaction)
        liveaction = LiveAction.get_by_id(str(liveaction.id))
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_RUNNING)
        execution = self._wait_for_children(execution)
        self.assertEqual(len(execution.children), 1)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_on_status(task1_live, action_constants.LIVEACTION_STATUS_RUNNING)
        (task1_live, task1_exec) = action_service.request_cancellation(task1_live, USERNAME)
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_on_status(task1_live, action_constants.LIVEACTION_STATUS_CANCELING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        task1_exec = ActionExecution.get_by_id(execution.children[0])
        task1_live = LiveAction.get_by_id(task1_exec.liveaction['id'])
        task1_live = self._wait_on_status(task1_live, action_constants.LIVEACTION_STATUS_CANCELED)
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_CANCELED)
        self.assertEqual(len(execution.children), 1)
        MockLiveActionPublisherNonBlocking.wait_all()
        self.assertIn('tasks', liveaction.result)
        self.assertEqual(len(liveaction.result['tasks']), 1)
        subworkflow = liveaction.result['tasks'][0]
        self.assertEqual(len(subworkflow['result']['tasks']), 1)
        self.assertEqual(subworkflow['state'], action_constants.LIVEACTION_STATUS_CANCELED)