from __future__ import absolute_import
import mock
import st2actions
from st2common.constants.action import LIVEACTION_STATUS_REQUESTED
from st2common.constants.action import LIVEACTION_STATUS_SUCCEEDED
from st2common.constants.action import LIVEACTION_STATUS_TIMED_OUT
from st2common.constants.action import LIVEACTION_STATUS_SCHEDULED
from st2common.constants.action import LIVEACTION_STATUS_DELAYED
from st2common.constants.action import LIVEACTION_STATUS_CANCELING
from st2common.constants.action import LIVEACTION_STATUS_CANCELED
from st2common.bootstrap.policiesregistrar import register_policy_types
from st2common.bootstrap import runnersregistrar as runners_registrar
from st2common.models.db.action import LiveActionDB
from st2common.persistence.action import LiveAction, ActionExecution
from st2common.services import action as action_service
from st2common.services import trace as trace_service
from st2actions.policies.retry import ExecutionRetryPolicyApplicator
from st2tests.base import DbTestCase
from st2tests.base import CleanDbTestCase
from st2tests.fixtures.generic.fixture import PACK_NAME as PACK
from st2tests.fixturesloader import FixturesLoader
__all__ = ['RetryPolicyTestCase']
TEST_FIXTURES = {'actions': ['action1.yaml'], 'policies': ['policy_4.yaml']}

class RetryPolicyTestCase(CleanDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        DbTestCase.setUpClass()
        super(RetryPolicyTestCase, cls).setUpClass()

    def setUp(self):
        if False:
            print('Hello World!')
        super(RetryPolicyTestCase, self).setUp()
        runners_registrar.register_runners()
        register_policy_types(st2actions)
        loader = FixturesLoader()
        models = loader.save_fixtures_to_db(fixtures_pack=PACK, fixtures_dict=TEST_FIXTURES)
        policy_db = models['policies']['policy_4.yaml']
        retry_on = policy_db.parameters['retry_on']
        max_retry_count = policy_db.parameters['max_retry_count']
        self.policy = ExecutionRetryPolicyApplicator(policy_ref='test_policy', policy_type='action.retry', retry_on=retry_on, max_retry_count=max_retry_count, delay=0)

    def test_retry_on_timeout_no_retry_since_no_timeout_reached(self):
        if False:
            return 10
        self.assertSequenceEqual(LiveAction.get_all(), [])
        self.assertSequenceEqual(ActionExecution.get_all(), [])
        liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'})
        (live_action_db, execution_db) = action_service.request(liveaction)
        live_action_db.status = LIVEACTION_STATUS_SUCCEEDED
        execution_db.status = LIVEACTION_STATUS_SUCCEEDED
        LiveAction.add_or_update(live_action_db)
        ActionExecution.add_or_update(execution_db)
        self.policy.apply_after(target=live_action_db)
        live_action_dbs = LiveAction.get_all()
        action_execution_dbs = ActionExecution.get_all()
        self.assertEqual(len(live_action_dbs), 1)
        self.assertEqual(len(action_execution_dbs), 1)
        self.assertEqual(action_execution_dbs[0].status, LIVEACTION_STATUS_SUCCEEDED)

    def test_retry_on_timeout_first_retry_is_successful(self):
        if False:
            while True:
                i = 10
        self.assertSequenceEqual(LiveAction.get_all(), [])
        self.assertSequenceEqual(ActionExecution.get_all(), [])
        liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'})
        (live_action_db, execution_db) = action_service.request(liveaction)
        live_action_db.status = LIVEACTION_STATUS_TIMED_OUT
        execution_db.status = LIVEACTION_STATUS_TIMED_OUT
        LiveAction.add_or_update(live_action_db)
        ActionExecution.add_or_update(execution_db)
        self.policy.apply_after(target=live_action_db)
        live_action_dbs = LiveAction.get_all()
        action_execution_dbs = ActionExecution.get_all()
        self.assertEqual(len(live_action_dbs), 2)
        self.assertEqual(len(action_execution_dbs), 2)
        self.assertEqual(action_execution_dbs[0].status, LIVEACTION_STATUS_TIMED_OUT)
        self.assertEqual(action_execution_dbs[1].status, LIVEACTION_STATUS_REQUESTED)
        original_liveaction_id = action_execution_dbs[0].liveaction['id']
        context = action_execution_dbs[1].context
        self.assertIn('policies', context)
        self.assertEqual(context['policies']['retry']['retry_count'], 1)
        self.assertEqual(context['policies']['retry']['applied_policy'], 'test_policy')
        self.assertEqual(context['policies']['retry']['retried_liveaction_id'], original_liveaction_id)
        live_action_db = live_action_dbs[1]
        live_action_db.status = LIVEACTION_STATUS_SUCCEEDED
        LiveAction.add_or_update(live_action_db)
        self.policy.apply_after(target=live_action_db)
        live_action_dbs = LiveAction.get_all()
        action_execution_dbs = ActionExecution.get_all()
        self.assertEqual(len(live_action_dbs), 2)
        self.assertEqual(len(action_execution_dbs), 2)
        self.assertEqual(live_action_dbs[0].status, LIVEACTION_STATUS_TIMED_OUT)
        self.assertEqual(live_action_dbs[1].status, LIVEACTION_STATUS_SUCCEEDED)

    def test_retry_on_timeout_policy_is_retried_twice(self):
        if False:
            print('Hello World!')
        self.assertSequenceEqual(LiveAction.get_all(), [])
        self.assertSequenceEqual(ActionExecution.get_all(), [])
        liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'})
        (live_action_db, execution_db) = action_service.request(liveaction)
        live_action_db.status = LIVEACTION_STATUS_TIMED_OUT
        execution_db.status = LIVEACTION_STATUS_TIMED_OUT
        LiveAction.add_or_update(live_action_db)
        ActionExecution.add_or_update(execution_db)
        self.policy.apply_after(target=live_action_db)
        live_action_dbs = LiveAction.get_all()
        action_execution_dbs = ActionExecution.get_all()
        self.assertEqual(len(live_action_dbs), 2)
        self.assertEqual(len(action_execution_dbs), 2)
        self.assertEqual(action_execution_dbs[0].status, LIVEACTION_STATUS_TIMED_OUT)
        self.assertEqual(action_execution_dbs[1].status, LIVEACTION_STATUS_REQUESTED)
        original_liveaction_id = action_execution_dbs[0].liveaction['id']
        context = action_execution_dbs[1].context
        self.assertIn('policies', context)
        self.assertEqual(context['policies']['retry']['retry_count'], 1)
        self.assertEqual(context['policies']['retry']['applied_policy'], 'test_policy')
        self.assertEqual(context['policies']['retry']['retried_liveaction_id'], original_liveaction_id)
        live_action_db = live_action_dbs[1]
        live_action_db.status = LIVEACTION_STATUS_TIMED_OUT
        LiveAction.add_or_update(live_action_db)
        execution_db = action_execution_dbs[1]
        execution_db.status = LIVEACTION_STATUS_TIMED_OUT
        ActionExecution.add_or_update(execution_db)
        self.policy.apply_after(target=live_action_db)
        live_action_dbs = LiveAction.get_all()
        action_execution_dbs = ActionExecution.get_all()
        self.assertEqual(len(live_action_dbs), 3)
        self.assertEqual(len(action_execution_dbs), 3)
        self.assertEqual(action_execution_dbs[0].status, LIVEACTION_STATUS_TIMED_OUT)
        self.assertEqual(action_execution_dbs[1].status, LIVEACTION_STATUS_TIMED_OUT)
        self.assertEqual(action_execution_dbs[2].status, LIVEACTION_STATUS_REQUESTED)
        original_liveaction_id = action_execution_dbs[1].liveaction['id']
        context = action_execution_dbs[2].context
        self.assertIn('policies', context)
        self.assertEqual(context['policies']['retry']['retry_count'], 2)
        self.assertEqual(context['policies']['retry']['applied_policy'], 'test_policy')
        self.assertEqual(context['policies']['retry']['retried_liveaction_id'], original_liveaction_id)

    def test_retry_on_timeout_max_retries_reached(self):
        if False:
            return 10
        self.assertSequenceEqual(LiveAction.get_all(), [])
        self.assertSequenceEqual(ActionExecution.get_all(), [])
        liveaction = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'})
        (live_action_db, execution_db) = action_service.request(liveaction)
        live_action_db.status = LIVEACTION_STATUS_TIMED_OUT
        live_action_db.context['policies'] = {}
        live_action_db.context['policies']['retry'] = {'retry_count': 2}
        execution_db.status = LIVEACTION_STATUS_TIMED_OUT
        LiveAction.add_or_update(live_action_db)
        ActionExecution.add_or_update(execution_db)
        self.policy.apply_after(target=live_action_db)
        live_action_dbs = LiveAction.get_all()
        action_execution_dbs = ActionExecution.get_all()
        self.assertEqual(len(live_action_dbs), 1)
        self.assertEqual(len(action_execution_dbs), 1)
        self.assertEqual(action_execution_dbs[0].status, LIVEACTION_STATUS_TIMED_OUT)

    @mock.patch.object(trace_service, 'get_trace_db_by_live_action', mock.MagicMock(return_value=(None, None)))
    def test_no_retry_on_workflow_task(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertSequenceEqual(LiveAction.get_all(), [])
        self.assertSequenceEqual(ActionExecution.get_all(), [])
        live_action_db = LiveActionDB(action='wolfpack.action-1', parameters={'actionstr': 'foo'}, context={'parent': {'execution_id': 'abcde'}})
        (live_action_db, execution_db) = action_service.request(live_action_db)
        live_action_db = LiveAction.get_by_id(str(live_action_db.id))
        self.assertEqual(live_action_db.status, LIVEACTION_STATUS_REQUESTED)
        live_action_db.status = LIVEACTION_STATUS_TIMED_OUT
        live_action_db.context['policies'] = {}
        execution_db.status = LIVEACTION_STATUS_TIMED_OUT
        LiveAction.add_or_update(live_action_db)
        ActionExecution.add_or_update(execution_db)
        self.policy.apply_after(target=live_action_db)
        live_action_dbs = LiveAction.get_all()
        action_execution_dbs = ActionExecution.get_all()
        self.assertEqual(len(live_action_dbs), 1)
        self.assertEqual(len(action_execution_dbs), 1)
        self.assertEqual(action_execution_dbs[0].status, LIVEACTION_STATUS_TIMED_OUT)

    def test_no_retry_on_non_applicable_statuses(self):
        if False:
            return 10
        self.assertSequenceEqual(LiveAction.get_all(), [])
        self.assertSequenceEqual(ActionExecution.get_all(), [])
        non_retry_statuses = [LIVEACTION_STATUS_REQUESTED, LIVEACTION_STATUS_SCHEDULED, LIVEACTION_STATUS_DELAYED, LIVEACTION_STATUS_CANCELING, LIVEACTION_STATUS_CANCELED]
        action_ref = 'wolfpack.action-1'
        for status in non_retry_statuses:
            liveaction = LiveActionDB(action=action_ref, parameters={'actionstr': 'foo'})
            (live_action_db, execution_db) = action_service.request(liveaction)
            live_action_db.status = status
            execution_db.status = status
            LiveAction.add_or_update(live_action_db)
            ActionExecution.add_or_update(execution_db)
            self.policy.apply_after(target=live_action_db)
        live_action_dbs = LiveAction.get_all()
        action_execution_dbs = ActionExecution.get_all()
        self.assertEqual(len(live_action_dbs), len(non_retry_statuses))
        self.assertEqual(len(action_execution_dbs), len(non_retry_statuses))