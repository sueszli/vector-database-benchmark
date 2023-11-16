from __future__ import absolute_import
import mock
from mock import call
import st2tests.config as tests_config
tests_config.parse_args()
import st2common
from st2actions.scheduler import handler as scheduling_queue
from st2common.bootstrap.policiesregistrar import register_policy_types
from st2common.constants import action as action_constants
from st2common.models.db.action import LiveActionDB
from st2common.persistence.action import LiveAction
from st2common.persistence.execution_queue import ActionExecutionSchedulingQueue
from st2common.persistence.policy import Policy
from st2common.services import action as action_service
from st2common.services import coordination
from st2common.transport.liveaction import LiveActionPublisher
from st2common.transport.publishers import CUDPublisher
from st2common.bootstrap import runnersregistrar as runners_registrar
from st2tests import ExecutionDbTestCase, EventletTestCase
import st2tests.config as tests_config
from st2tests.fixtures.generic.fixture import PACK_NAME as PACK
from st2tests.fixturesloader import FixturesLoader
from st2tests.mocks.execution import MockExecutionPublisher
from st2tests.mocks.liveaction import MockLiveActionPublisherSchedulingQueueOnly
from st2tests.mocks.runners import runner
from six.moves import range
__all__ = ['ConcurrencyByAttributePolicyTestCase']
TEST_FIXTURES = {'actions': ['action1.yaml', 'action2.yaml'], 'policies': ['policy_3.yaml', 'policy_7.yaml']}
NON_EMPTY_RESULT = {'data': 'non-empty'}
MOCK_RUN_RETURN_VALUE = (action_constants.LIVEACTION_STATUS_RUNNING, NON_EMPTY_RESULT, None)
SCHEDULED_STATES = [action_constants.LIVEACTION_STATUS_SCHEDULED, action_constants.LIVEACTION_STATUS_RUNNING, action_constants.LIVEACTION_STATUS_SUCCEEDED]

@mock.patch('st2common.runners.base.get_runner', mock.Mock(return_value=runner.get_runner()))
@mock.patch('st2actions.container.base.get_runner', mock.Mock(return_value=runner.get_runner()))
@mock.patch.object(CUDPublisher, 'publish_update', mock.MagicMock(side_effect=MockExecutionPublisher.publish_update))
@mock.patch.object(CUDPublisher, 'publish_create', mock.MagicMock(return_value=None))
class ConcurrencyByAttributePolicyTestCase(EventletTestCase, ExecutionDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        EventletTestCase.setUpClass()
        ExecutionDbTestCase.setUpClass()
        tests_config.parse_args(coordinator_noop=True)
        coordination.COORDINATOR = None
        runners_registrar.register_runners()
        register_policy_types(st2common)
        loader = FixturesLoader()
        loader.save_fixtures_to_db(fixtures_pack=PACK, fixtures_dict=TEST_FIXTURES)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        coordination.coordinator_teardown(coordination.COORDINATOR)
        coordination.COORDINATOR = None
        super(ConcurrencyByAttributePolicyTestCase, cls).tearDownClass()

    @mock.patch('st2actions.container.base.get_runner', mock.Mock(return_value=runner.get_runner()))
    def tearDown(self):
        if False:
            return 10
        for liveaction in LiveAction.get_all():
            action_service.update_status(liveaction, action_constants.LIVEACTION_STATUS_CANCELED)

    @staticmethod
    def _process_scheduling_queue():
        if False:
            for i in range(10):
                print('nop')
        for queued_req in ActionExecutionSchedulingQueue.get_all():
            scheduling_queue.get_handler()._handle_execution(queued_req)

    @mock.patch.object(runner.MockActionRunner, 'run', mock.MagicMock(return_value=MOCK_RUN_RETURN_VALUE))
    @mock.patch.object(LiveActionPublisher, 'publish_state', mock.MagicMock(side_effect=MockLiveActionPublisherSchedulingQueueOnly.publish_state))
    def test_over_threshold_delay_executions(self):
        if False:
            return 10
        policy_db = Policy.get_by_ref('wolfpack.action-1.concurrency.attr')
        self.assertGreater(policy_db.parameters['threshold'], 0)
        self.assertIn('actionstr', policy_db.parameters['attributes'])
        for i in range(0, policy_db.parameters['threshold']):
            liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'})
            action_service.request(liveaction)
        self._process_scheduling_queue()
        scheduled = [item for item in LiveAction.get_all() if item.status in SCHEDULED_STATES]
        self.assertEqual(len(scheduled), policy_db.parameters['threshold'])
        expected_num_exec = len(scheduled)
        expected_num_pubs = expected_num_exec * 3
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'})
        (liveaction, _) = action_service.request(liveaction)
        expected_num_pubs += 1
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self._process_scheduling_queue()
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_DELAYED)
        expected_num_exec += 0
        expected_num_pubs += 0
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'bar'})
        (liveaction, _) = action_service.request(liveaction)
        self._process_scheduling_queue()
        liveaction = self._wait_on_statuses(liveaction, SCHEDULED_STATES)
        expected_num_exec += 1
        expected_num_pubs += 3
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        action_service.update_status(scheduled[0], action_constants.LIVEACTION_STATUS_SUCCEEDED, publish=True)
        expected_num_pubs += 1
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self._process_scheduling_queue()
        liveaction = self._wait_on_statuses(liveaction, SCHEDULED_STATES)
        expected_num_exec += 1
        expected_num_pubs += 2
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)

    @mock.patch.object(runner.MockActionRunner, 'run', mock.MagicMock(return_value=MOCK_RUN_RETURN_VALUE))
    @mock.patch.object(LiveActionPublisher, 'publish_state', mock.MagicMock(side_effect=MockLiveActionPublisherSchedulingQueueOnly.publish_state))
    def test_over_threshold_cancel_executions(self):
        if False:
            return 10
        policy_db = Policy.get_by_ref('wolfpack.action-2.concurrency.attr.cancel')
        self.assertEqual(policy_db.parameters['action'], 'cancel')
        self.assertGreater(policy_db.parameters['threshold'], 0)
        self.assertIn('actionstr', policy_db.parameters['attributes'])
        for i in range(0, policy_db.parameters['threshold']):
            liveaction = LiveActionDB(action='wolfpack.action-2', parameters={'actionstr': 'foo'})
            action_service.request(liveaction)
        self._process_scheduling_queue()
        scheduled = [item for item in LiveAction.get_all() if item.status in SCHEDULED_STATES]
        self.assertEqual(len(scheduled), policy_db.parameters['threshold'])
        expected_num_exec = len(scheduled)
        expected_num_pubs = expected_num_exec * 3
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        liveaction = LiveActionDB(action='wolfpack.action-2', parameters={'actionstr': 'foo'})
        (liveaction, _) = action_service.request(liveaction)
        expected_num_pubs += 1
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self._process_scheduling_queue()
        calls = [call(liveaction, action_constants.LIVEACTION_STATUS_CANCELING)]
        LiveActionPublisher.publish_state.assert_has_calls(calls)
        expected_num_pubs += 2
        expected_num_exec += 0
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        canceled = LiveAction.get_by_id(str(liveaction.id))
        self.assertEqual(canceled.status, action_constants.LIVEACTION_STATUS_CANCELED)

    @mock.patch.object(runner.MockActionRunner, 'run', mock.MagicMock(return_value=MOCK_RUN_RETURN_VALUE))
    @mock.patch.object(LiveActionPublisher, 'publish_state', mock.MagicMock(side_effect=MockLiveActionPublisherSchedulingQueueOnly.publish_state))
    def test_on_cancellation(self):
        if False:
            for i in range(10):
                print('nop')
        policy_db = Policy.get_by_ref('wolfpack.action-1.concurrency.attr')
        self.assertGreater(policy_db.parameters['threshold'], 0)
        self.assertIn('actionstr', policy_db.parameters['attributes'])
        for i in range(0, policy_db.parameters['threshold']):
            liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'})
            action_service.request(liveaction)
        self._process_scheduling_queue()
        scheduled = [item for item in LiveAction.get_all() if item.status in SCHEDULED_STATES]
        self.assertEqual(len(scheduled), policy_db.parameters['threshold'])
        expected_num_exec = len(scheduled)
        expected_num_pubs = expected_num_exec * 3
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'})
        (liveaction, _) = action_service.request(liveaction)
        expected_num_pubs += 1
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self._process_scheduling_queue()
        liveaction = self._wait_on_status(liveaction, action_constants.LIVEACTION_STATUS_DELAYED)
        delayed = liveaction
        expected_num_exec += 0
        expected_num_pubs += 0
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'bar'})
        (liveaction, _) = action_service.request(liveaction)
        self._process_scheduling_queue()
        liveaction = self._wait_on_statuses(liveaction, SCHEDULED_STATES)
        expected_num_exec += 1
        expected_num_pubs += 3
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        action_service.request_cancellation(scheduled[0], 'stanley')
        expected_num_pubs += 2
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self._process_scheduling_queue()
        expected_num_exec += 1
        expected_num_pubs += 2
        self.assertEqual(expected_num_pubs, LiveActionPublisher.publish_state.call_count)
        self.assertEqual(expected_num_exec, runner.MockActionRunner.run.call_count)
        liveaction = LiveAction.get_by_id(str(delayed.id))
        liveaction = self._wait_on_statuses(liveaction, SCHEDULED_STATES)