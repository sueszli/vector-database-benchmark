from __future__ import absolute_import
import mock
import st2tests
from orquesta import statuses as wf_statuses
from oslo_config import cfg
import st2tests.config as tests_config
tests_config.parse_args()
from tests.unit import base
from st2actions.workflows import workflows
from st2common.bootstrap import actionsregistrar
from st2common.bootstrap import runnersregistrar
from st2common.constants import action as ac_const
from st2common.models.db import liveaction as lv_db_models
from st2common.persistence import execution as ex_db_access
from st2common.persistence import liveaction as lv_db_access
from st2common.persistence import workflow as wf_db_access
from st2common.runners import base as runners
from st2common.services import action as ac_svc
from st2common.services import workflows as wf_svc
from st2common.transport import liveaction as lv_ac_xport
from st2common.transport import workflow as wf_ex_xport
from st2common.transport import publishers
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
class OrquestaRunnerPauseResumeTest(st2tests.ExecutionDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super(OrquestaRunnerPauseResumeTest, cls).setUpClass()
        runnersregistrar.register_runners()
        actions_registrar = actionsregistrar.ActionsRegistrar(use_pack_cache=False, fail_on_failure=True)
        for pack in PACKS:
            actions_registrar.register_from_pack(pack)

    @classmethod
    def get_runner_class(cls, runner_name):
        if False:
            for i in range(10):
                print('nop')
        return runners.get_runner(runner_name, runner_name).__class__

    @mock.patch.object(ac_svc, 'is_children_active', mock.MagicMock(return_value=False))
    def test_pause(self):
        if False:
            return 10
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        (lv_ac_db, ac_ex_db) = ac_svc.request_pause(lv_ac_db, cfg.CONF.system_user.user)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        self.assertEqual(lv_ac_db.context['paused_by'], cfg.CONF.system_user.user)

    @mock.patch.object(ac_svc, 'is_children_active', mock.MagicMock(return_value=True))
    def test_pause_with_active_children(self):
        if False:
            i = 10
            return i + 15
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        (lv_ac_db, ac_ex_db) = ac_svc.request_pause(lv_ac_db, cfg.CONF.system_user.user)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)

    def test_pause_subworkflow_not_cascade_up_to_workflow(self):
        if False:
            while True:
                i = 10
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'subworkflow.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        wf_ex_dbs = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))
        self.assertEqual(len(wf_ex_dbs), 1)
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_dbs[0].id))
        self.assertEqual(len(tk_ex_dbs), 1)
        tk_ac_ex_dbs = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[0].id))
        self.assertEqual(len(tk_ac_ex_dbs), 1)
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(tk_ac_ex_dbs[0].liveaction['id'])
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        (tk_lv_ac_db, tk_ac_ex_db) = ac_svc.request_pause(tk_lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)

    def test_pause_workflow_cascade_down_to_subworkflow(self):
        if False:
            while True:
                i = 10
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'subworkflow.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        wf_ex_dbs = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))
        self.assertEqual(len(wf_ex_dbs), 1)
        wf_ex_db = wf_ex_dbs[0]
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 1)
        tk_ex_db = tk_ex_dbs[0]
        tk_ac_ex_dbs = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_db.id))
        self.assertEqual(len(tk_ac_ex_dbs), 1)
        tk_ac_ex_db = tk_ac_ex_dbs[0]
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(tk_ac_ex_db.liveaction['id'])
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        sub_wf_ex_dbs = wf_db_access.WorkflowExecution.query(action_execution=str(tk_ac_ex_db.id))
        self.assertEqual(len(sub_wf_ex_dbs), 1)
        sub_wf_ex_db = sub_wf_ex_dbs[0]
        sub_tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(sub_wf_ex_db.id))
        self.assertEqual(len(sub_tk_ex_dbs), 1)
        sub_tk_ex_db = sub_tk_ex_dbs[0]
        sub_tk_ac_ex_dbs = ex_db_access.ActionExecution.query(task_execution=str(sub_tk_ex_db.id))
        self.assertEqual(len(sub_tk_ac_ex_dbs), 1)
        (lv_ac_db, ac_ex_db) = ac_svc.request_pause(lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(tk_lv_ac_db.id))
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        sub_tk_ac_ex_db = sub_tk_ac_ex_dbs[0]
        self.assertEqual(sub_tk_ac_ex_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(sub_tk_ac_ex_db)
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(tk_lv_ac_db.id))
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        tk_ac_ex_db = ex_db_access.ActionExecution.get_by_id(str(tk_ac_ex_db.id))
        self.assertEqual(tk_ac_ex_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(tk_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)

    def test_pause_subworkflow_while_another_subworkflow_running(self):
        if False:
            return 10
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'subworkflows.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 2)
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[0].id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        t1_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t1_ac_ex_db.id))[0]
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t1_wf_ex_db.status, wf_statuses.RUNNING)
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[1].id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        t2_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t2_ac_ex_db.id))[0]
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.RUNNING)
        (t1_lv_ac_db, t1_ac_ex_db) = ac_svc.request_pause(t1_lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t1_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[0]
        t1_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t1_ex_db.id))[0]
        workflows.get_engine().process(t1_t1_ac_ex_db)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        t1_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t1_ac_ex_db.id)
        self.assertEqual(t1_ac_ex_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(t1_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[0]
        t2_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t1_ex_db.id))[0]
        workflows.get_engine().process(t2_t1_ac_ex_db)
        t2_t2_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[1]
        t2_t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t2_ex_db.id))[0]
        workflows.get_engine().process(t2_t2_ac_ex_db)
        t2_t3_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[2]
        t2_t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t3_ex_db.id))[0]
        workflows.get_engine().process(t2_t3_ac_ex_db)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_lv_ac_db.id))
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t2_ac_ex_db.id)
        workflows.get_engine().process(t2_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)

    def test_pause_subworkflow_while_another_subworkflow_completed(self):
        if False:
            i = 10
            return i + 15
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'subworkflows.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 2)
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[0].id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        t1_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t1_ac_ex_db.id))[0]
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t1_wf_ex_db.status, wf_statuses.RUNNING)
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[1].id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        t2_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t2_ac_ex_db.id))[0]
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.RUNNING)
        (t1_lv_ac_db, t1_ac_ex_db) = ac_svc.request_pause(t1_lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[0]
        t2_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t1_ex_db.id))[0]
        workflows.get_engine().process(t2_t1_ac_ex_db)
        t2_t2_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[1]
        t2_t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t2_ex_db.id))[0]
        workflows.get_engine().process(t2_t2_ac_ex_db)
        t2_t3_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[2]
        t2_t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t3_ex_db.id))[0]
        workflows.get_engine().process(t2_t3_ac_ex_db)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_lv_ac_db.id))
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t2_ac_ex_db.id)
        workflows.get_engine().process(t2_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        t1_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[0]
        t1_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t1_ex_db.id))[0]
        workflows.get_engine().process(t1_t1_ac_ex_db)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        t1_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t1_ac_ex_db.id)
        self.assertEqual(t1_ac_ex_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(t1_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)

    @mock.patch.object(ac_svc, 'is_children_active', mock.MagicMock(return_value=False))
    def test_resume(self):
        if False:
            for i in range(10):
                print('nop')
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'sequential.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        (lv_ac_db, ac_ex_db) = ac_svc.request_pause(lv_ac_db, cfg.CONF.system_user.user)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        wf_ex_dbs = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_dbs[0].id))
        self.assertEqual(len(tk_ex_dbs), 1)
        tk_ac_ex_dbs = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[0].id))
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(tk_ac_ex_dbs[0].liveaction['id'])
        self.assertEqual(tk_ac_ex_dbs[0].status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        wf_svc.handle_action_execution_completion(tk_ac_ex_dbs[0])
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED, lv_ac_db.result)
        wf_ex_dbs = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))
        self.assertEqual(wf_ex_dbs[0].status, wf_statuses.PAUSED)
        (lv_ac_db, ac_ex_db) = ac_svc.request_resume(lv_ac_db, cfg.CONF.system_user.user)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        wf_ex_dbs = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))
        self.assertEqual(wf_ex_dbs[0].status, wf_statuses.RUNNING)
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_dbs[0].id))
        self.assertEqual(len(tk_ex_dbs), 2)
        self.assertEqual(lv_ac_db.context['resumed_by'], cfg.CONF.system_user.user)

    def test_resume_cascade_to_subworkflow(self):
        if False:
            while True:
                i = 10
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'subworkflow.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        wf_ex_dbs = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))
        self.assertEqual(len(wf_ex_dbs), 1)
        wf_ex_db = wf_ex_dbs[0]
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 1)
        tk_ex_db = tk_ex_dbs[0]
        tk_ac_ex_dbs = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_db.id))
        self.assertEqual(len(tk_ac_ex_dbs), 1)
        tk_ac_ex_db = tk_ac_ex_dbs[0]
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(tk_ac_ex_db.liveaction['id'])
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        sub_wf_ex_dbs = wf_db_access.WorkflowExecution.query(action_execution=str(tk_ac_ex_db.id))
        self.assertEqual(len(sub_wf_ex_dbs), 1)
        sub_wf_ex_db = sub_wf_ex_dbs[0]
        sub_tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(sub_wf_ex_db.id))
        self.assertEqual(len(sub_tk_ex_dbs), 1)
        sub_tk_ex_db = sub_tk_ex_dbs[0]
        sub_tk_ac_ex_dbs = ex_db_access.ActionExecution.query(task_execution=str(sub_tk_ex_db.id))
        self.assertEqual(len(sub_tk_ac_ex_dbs), 1)
        (lv_ac_db, ac_ex_db) = ac_svc.request_pause(lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(tk_lv_ac_db.id))
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        sub_tk_ac_ex_db = sub_tk_ac_ex_dbs[0]
        self.assertEqual(sub_tk_ac_ex_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(sub_tk_ac_ex_db)
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(tk_lv_ac_db.id))
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        tk_ac_ex_db = ex_db_access.ActionExecution.get_by_id(str(tk_ac_ex_db.id))
        self.assertEqual(tk_ac_ex_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(tk_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        (lv_ac_db, ac_ex_db) = ac_svc.request_resume(lv_ac_db, cfg.CONF.system_user.user)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        tk_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(tk_lv_ac_db.id))
        self.assertEqual(tk_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)

    def test_resume_from_each_subworkflow_when_parent_is_paused(self):
        if False:
            i = 10
            return i + 15
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'subworkflows.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 2)
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[0].id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        t1_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t1_ac_ex_db.id))[0]
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t1_wf_ex_db.status, wf_statuses.RUNNING)
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[1].id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        t2_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t2_ac_ex_db.id))[0]
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.RUNNING)
        (t1_lv_ac_db, t1_ac_ex_db) = ac_svc.request_pause(t1_lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t1_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[0]
        t1_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t1_ex_db.id))[0]
        workflows.get_engine().process(t1_t1_ac_ex_db)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        t1_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t1_ac_ex_db.id)
        self.assertEqual(t1_ac_ex_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(t1_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        (t2_lv_ac_db, t2_ac_ex_db) = ac_svc.request_pause(t2_lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[0]
        t2_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t1_ex_db.id))[0]
        workflows.get_engine().process(t2_t1_ac_ex_db)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_lv_ac_db.id))
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t2_ac_ex_db.id)
        self.assertEqual(t2_ac_ex_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(t2_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        (t1_lv_ac_db, t1_ac_ex_db) = ac_svc.request_resume(t1_lv_ac_db, cfg.CONF.system_user.user)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t1_t2_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[1]
        t1_t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t2_ex_db.id))[0]
        workflows.get_engine().process(t1_t2_ac_ex_db)
        t1_t3_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[2]
        t1_t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t3_ex_db.id))[0]
        workflows.get_engine().process(t1_t3_ac_ex_db)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        t1_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t1_ac_ex_db.id)
        workflows.get_engine().process(t1_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)

    def test_resume_from_subworkflow_when_parent_is_paused(self):
        if False:
            for i in range(10):
                print('nop')
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'subworkflows.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 2)
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[0].id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        t1_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t1_ac_ex_db.id))[0]
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t1_wf_ex_db.status, wf_statuses.RUNNING)
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[1].id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        t2_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t2_ac_ex_db.id))[0]
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.RUNNING)
        (t1_lv_ac_db, t1_ac_ex_db) = ac_svc.request_pause(t1_lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t1_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[0]
        t1_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t1_ex_db.id))[0]
        workflows.get_engine().process(t1_t1_ac_ex_db)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        t1_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t1_ac_ex_db.id)
        self.assertEqual(t1_ac_ex_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(t1_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[0]
        t2_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t1_ex_db.id))[0]
        workflows.get_engine().process(t2_t1_ac_ex_db)
        t2_t2_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[1]
        t2_t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t2_ex_db.id))[0]
        workflows.get_engine().process(t2_t2_ac_ex_db)
        t2_t3_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[2]
        t2_t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t3_ex_db.id))[0]
        workflows.get_engine().process(t2_t3_ac_ex_db)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_lv_ac_db.id))
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t2_ac_ex_db.id)
        workflows.get_engine().process(t2_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        (t1_lv_ac_db, t1_ac_ex_db) = ac_svc.request_resume(t1_lv_ac_db, cfg.CONF.system_user.user)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t1_t2_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[1]
        t1_t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t2_ex_db.id))[0]
        workflows.get_engine().process(t1_t2_ac_ex_db)
        t1_t3_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[2]
        t1_t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t3_ex_db.id))[0]
        workflows.get_engine().process(t1_t3_ac_ex_db)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        t1_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t1_ac_ex_db.id)
        workflows.get_engine().process(t1_ac_ex_db)
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 3)
        t3_ex_db_qry = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task3'}
        t3_ex_db = wf_db_access.TaskExecution.query(**t3_ex_db_qry)[0]
        t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t3_ex_db.id))[0]
        t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(t3_ac_ex_db.liveaction['id'])
        self.assertEqual(t3_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        wf_svc.handle_action_execution_completion(t3_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_resume_from_subworkflow_when_parent_is_running(self):
        if False:
            i = 10
            return i + 15
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'subworkflows.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = ac_svc.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING, lv_ac_db.result)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 2)
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[0].id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        t1_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t1_ac_ex_db.id))[0]
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t1_wf_ex_db.status, wf_statuses.RUNNING)
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(tk_ex_dbs[1].id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        t2_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t2_ac_ex_db.id))[0]
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.RUNNING)
        (t1_lv_ac_db, t1_ac_ex_db) = ac_svc.request_pause(t1_lv_ac_db, cfg.CONF.system_user.user)
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t1_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[0]
        t1_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t1_ex_db.id))[0]
        workflows.get_engine().process(t1_t1_ac_ex_db)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        t1_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t1_ac_ex_db.id)
        self.assertEqual(t1_ac_ex_db.status, ac_const.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(t1_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        (t1_lv_ac_db, t1_ac_ex_db) = ac_svc.request_resume(t1_lv_ac_db, cfg.CONF.system_user.user)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_RUNNING)
        t1_t2_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[1]
        t1_t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t2_ex_db.id))[0]
        workflows.get_engine().process(t1_t2_ac_ex_db)
        t1_t3_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t1_wf_ex_db.id))[2]
        t1_t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_t3_ex_db.id))[0]
        workflows.get_engine().process(t1_t3_ac_ex_db)
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t1_lv_ac_db.id))
        self.assertEqual(t1_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        t1_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t1_ac_ex_db.id)
        workflows.get_engine().process(t1_ac_ex_db)
        t2_t1_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[0]
        t2_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t1_ex_db.id))[0]
        workflows.get_engine().process(t2_t1_ac_ex_db)
        t2_t2_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[1]
        t2_t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t2_ex_db.id))[0]
        workflows.get_engine().process(t2_t2_ac_ex_db)
        t2_t3_ex_db = wf_db_access.TaskExecution.query(workflow_execution=str(t2_wf_ex_db.id))[2]
        t2_t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t3_ex_db.id))[0]
        workflows.get_engine().process(t2_t3_ac_ex_db)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_lv_ac_db.id))
        self.assertEqual(t2_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(t2_ac_ex_db.id)
        workflows.get_engine().process(t2_ac_ex_db)
        tk_ex_dbs = wf_db_access.TaskExecution.query(workflow_execution=str(wf_ex_db.id))
        self.assertEqual(len(tk_ex_dbs), 3)
        t3_ex_db_qry = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'task3'}
        t3_ex_db = wf_db_access.TaskExecution.query(**t3_ex_db_qry)[0]
        t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t3_ex_db.id))[0]
        t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(t3_ac_ex_db.liveaction['id'])
        self.assertEqual(t3_lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        wf_svc.handle_action_execution_completion(t3_ac_ex_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)