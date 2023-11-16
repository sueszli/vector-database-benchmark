from __future__ import absolute_import
import eventlet
import mock
import st2tests
from orquesta import statuses as wf_statuses
from oslo_config import cfg
from tooz import coordination
import st2tests.config as tests_config
tests_config.parse_args()
from st2actions.workflows import workflows
from st2common.bootstrap import actionsregistrar
from st2common.bootstrap import runnersregistrar
from st2common.constants import action as action_constants
from st2common.models.db import liveaction as lv_db_models
from st2common.persistence import execution as ex_db_access
from st2common.persistence import liveaction as lv_db_access
from st2common.persistence import workflow as wf_db_access
from st2common.services import action as action_service
from st2common.services import coordination as coordination_service
from st2common.transport import liveaction as lv_ac_xport
from st2common.transport import workflow as wf_ex_xport
from st2common.transport import publishers
from st2reactor.garbage_collector import base as garbage_collector
from st2tests.fixtures.packs.core.fixture import PACK_PATH as CORE_PACK_PATH
from st2tests.fixtures.packs.orquesta_tests.fixture import PACK_PATH as TEST_PACK_PATH
from st2tests.mocks import liveaction as mock_lv_ac_xport
from st2tests.mocks import workflow as mock_wf_ex_xport
PACKS = [TEST_PACK_PATH, CORE_PACK_PATH]

@mock.patch.object(publishers.CUDPublisher, 'publish_update', mock.MagicMock(return_value=None))
@mock.patch.object(lv_ac_xport.LiveActionPublisher, 'publish_create', mock.MagicMock(side_effect=mock_lv_ac_xport.MockLiveActionPublisher.publish_create))
@mock.patch.object(lv_ac_xport.LiveActionPublisher, 'publish_state', mock.MagicMock(side_effect=mock_lv_ac_xport.MockLiveActionPublisher.publish_state))
@mock.patch.object(wf_ex_xport.WorkflowExecutionPublisher, 'publish_create', mock.MagicMock(side_effect=mock_wf_ex_xport.MockWorkflowExecutionPublisher.publish_create))
@mock.patch.object(wf_ex_xport.WorkflowExecutionPublisher, 'publish_state', mock.MagicMock(side_effect=mock_wf_ex_xport.MockWorkflowExecutionPublisher.publish_state))
class WorkflowExecutionHandlerTest(st2tests.WorkflowTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(WorkflowExecutionHandlerTest, cls).setUpClass()
        runnersregistrar.register_runners()
        actions_registrar = actionsregistrar.ActionsRegistrar(use_pack_cache=False, fail_on_failure=True)
        for pack in PACKS:
            actions_registrar.register_from_pack(pack)

    def test_process(self):
        if False:
            print('Hello World!')
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task1'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        workflows.get_engine().process(t1_ac_ex_db)
        t1_ex_db = wf_db_access.TaskExecution.get_by_id(t1_ex_db.id)
        self.assertEqual(t1_ex_db.status, wf_statuses.SUCCEEDED)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task2'}
        t2_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_ex_db.id))[0]
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(t2_ex_db.id)
        self.assertEqual(t2_ex_db.status, wf_statuses.SUCCEEDED)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task3'}
        t3_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t3_ex_db.id))[0]
        workflows.get_engine().process(t3_ac_ex_db)
        t3_ex_db = wf_db_access.TaskExecution.get_by_id(t3_ex_db.id)
        self.assertEqual(t3_ex_db.status, wf_statuses.SUCCEEDED)
        expected_output = {'msg': 'Stanley, All your base are belong to us!'}
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.SUCCEEDED)
        self.assertDictEqual(wf_ex_db.output, expected_output)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)

    @mock.patch.object(coordination_service.NoOpDriver, 'get_lock')
    def test_process_error_handling(self, mock_get_lock):
        if False:
            i = 10
            return i + 15
        expected_errors = [{'message': 'Execution failed. See result for details.', 'type': 'error', 'task_id': 'task1'}, {'type': 'error', 'message': 'ToozConnectionError: foobar', 'task_id': 'task1', 'route': 0}]
        mock_get_lock.side_effect = coordination_service.NoOpLock(name='noop')
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task1'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        mock_get_lock.side_effect = [coordination.ToozConnectionError('foobar'), coordination.ToozConnectionError('foobar'), coordination.ToozConnectionError('foobar'), coordination.ToozConnectionError('foobar'), coordination.ToozConnectionError('foobar'), coordination_service.NoOpLock(name='noop'), coordination_service.NoOpLock(name='noop')]
        workflows.get_engine().process(t1_ac_ex_db)
        t1_ex_db = wf_db_access.TaskExecution.get_by_id(str(t1_ex_db.id))
        self.assertEqual(t1_ex_db.status, wf_statuses.FAILED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.FAILED)
        self.assertListEqual(wf_ex_db.errors, expected_errors)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_FAILED)

    @mock.patch.object(coordination_service.NoOpDriver, 'get_lock')
    @mock.patch.object(workflows.WorkflowExecutionHandler, 'fail_workflow_execution', mock.MagicMock(side_effect=Exception('Unexpected error.')))
    def test_process_error_handling_has_error(self, mock_get_lock):
        if False:
            i = 10
            return i + 15
        mock_get_lock.side_effect = coordination_service.NoOpLock(name='noop')
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task1'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        mock_get_lock.side_effect = [coordination.ToozConnectionError('foobar'), coordination.ToozConnectionError('foobar'), coordination.ToozConnectionError('foobar'), coordination.ToozConnectionError('foobar'), coordination.ToozConnectionError('foobar')]
        self.assertRaisesRegexp(Exception, 'Unexpected error.', workflows.get_engine().process, t1_ac_ex_db)
        self.assertTrue(workflows.WorkflowExecutionHandler.fail_workflow_execution.called)
        mock_get_lock.side_effect = coordination_service.NoOpLock(name='noop')
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        eventlet.sleep(cfg.CONF.workflow_engine.gc_max_idle_sec)
        gc = garbage_collector.GarbageCollectorService()
        gc._purge_orphaned_workflow_executions()
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_CANCELED)

    @mock.patch.object(coordination_service.NoOpDriver, 'get_members', mock.MagicMock(return_value=coordination_service.NoOpAsyncResult('')))
    def test_workflow_engine_shutdown(self):
        if False:
            print('Hello World!')
        cfg.CONF.set_override(name='service_registry', override=True, group='coordination')
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        workflow_engine = workflows.get_engine()
        eventlet.spawn(workflow_engine.shutdown)
        eventlet.sleep(5)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_PAUSING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task1'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        workflows.get_engine().process(t1_ac_ex_db)
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        self.assertEqual(t1_ac_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_PAUSED)
        workflow_engine = workflows.get_engine()
        workflow_engine._delay = 0
        workflow_engine.start(False)
        eventlet.sleep(workflow_engine._delay + 5)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertTrue(lv_ac_db.status in [action_constants.LIVEACTION_STATUS_RESUMING, action_constants.LIVEACTION_STATUS_RUNNING, action_constants.LIVEACTION_STATUS_SUCCEEDED])

    @mock.patch.object(coordination_service.NoOpDriver, 'get_members', mock.MagicMock(return_value=coordination_service.NoOpAsyncResult('member-1')))
    def test_workflow_engine_shutdown_with_multiple_members(self):
        if False:
            for i in range(10):
                print('nop')
        cfg.CONF.set_override(name='service_registry', override=True, group='coordination')
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        workflow_engine = workflows.get_engine()
        eventlet.spawn(workflow_engine.shutdown)
        eventlet.sleep(5)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task1'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        workflows.get_engine().process(t1_ac_ex_db)
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        self.assertEqual(t1_ac_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)

    def test_workflow_engine_shutdown_with_service_registry_disabled(self):
        if False:
            for i in range(10):
                print('nop')
        cfg.CONF.set_override(name='service_registry', override=False, group='coordination')
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        workflow_engine = workflows.get_engine()
        eventlet.spawn(workflow_engine.shutdown)
        eventlet.sleep(5)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)

    @mock.patch.object(coordination_service.NoOpDriver, 'get_lock', mock.MagicMock(return_value=coordination_service.NoOpLock(name='noop')))
    def test_workflow_engine_shutdown_first_then_start(self):
        if False:
            print('Hello World!')
        cfg.CONF.set_override(name='service_registry', override=True, group='coordination')
        cfg.CONF.set_override(name='exit_still_active_check', override=0, group='workflow_engine')
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        workflow_engine = workflows.get_engine()
        workflow_engine._delay = 5
        eventlet.spawn(workflow_engine.shutdown)
        eventlet.spawn_after(1, workflow_engine.start, True)
        eventlet.sleep(2)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_PAUSING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task1'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        workflows.get_engine().process(t1_ac_ex_db)
        eventlet.sleep(workflow_engine._delay + 5)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertTrue(lv_ac_db.status in [action_constants.LIVEACTION_STATUS_RESUMING, action_constants.LIVEACTION_STATUS_RUNNING, action_constants.LIVEACTION_STATUS_SUCCEEDED])

    @mock.patch.object(coordination_service.NoOpDriver, 'get_lock', mock.MagicMock(return_value=coordination_service.NoOpLock(name='noop')))
    def test_workflow_engine_start_first_then_shutdown(self):
        if False:
            return 10
        cfg.CONF.set_override(name='service_registry', override=True, group='coordination')
        cfg.CONF.set_override(name='exit_still_active_check', override=0, group='workflow_engine')
        wf_meta = self.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        workflow_engine = workflows.get_engine()
        workflow_engine._delay = 0
        eventlet.spawn(workflow_engine.start, True)
        eventlet.spawn_after(1, workflow_engine.shutdown)
        coordination_service.NoOpDriver.get_members = mock.MagicMock(return_value=coordination_service.NoOpAsyncResult('member-1'))
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        eventlet.sleep(workflow_engine._delay + 5)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)