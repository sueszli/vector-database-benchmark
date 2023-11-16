from __future__ import absolute_import
import mock
from orquesta import statuses as wf_statuses
import st2tests
import st2tests.config as tests_config
tests_config.parse_args()
from tests.unit import base
from st2actions.workflows import workflows
from st2common.bootstrap import actionsregistrar
from st2common.bootstrap import runnersregistrar
from st2common.constants import action as action_constants
from st2common.models.api import inquiry as inqy_api_models
from st2common.models.db import liveaction as lv_db_models
from st2common.persistence import execution as ex_db_access
from st2common.persistence import liveaction as lv_db_access
from st2common.persistence import workflow as wf_db_access
from st2common.services import action as action_service
from st2common.services import inquiry as inquiry_service
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
class OrquestaRunnerTest(st2tests.ExecutionDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(OrquestaRunnerTest, cls).setUpClass()
        runnersregistrar.register_runners()
        actions_registrar = actionsregistrar.ActionsRegistrar(use_pack_cache=False, fail_on_failure=True)
        for pack in PACKS:
            actions_registrar.register_from_pack(pack)

    def test_inquiry(self):
        if False:
            while True:
                i = 10
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'ask-approval.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'start'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        self.assertEqual(t1_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t1_ac_ex_db)
        t1_ex_db = wf_db_access.TaskExecution.get_by_id(t1_ex_db.id)
        self.assertEqual(t1_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'get_approval'}
        t2_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_ex_db.id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_PENDING)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(t2_ex_db.id)
        self.assertEqual(t2_ex_db.status, wf_statuses.PENDING)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.PAUSED)
        inquiry_api = inqy_api_models.InquiryAPI.from_model(t2_ac_ex_db)
        inquiry_response = {'approved': True}
        inquiry_service.respond(inquiry_api, inquiry_response)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_lv_ac_db.id))
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(str(t2_ac_ex_db.id))
        self.assertEqual(t2_ac_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(str(t2_ex_db.id))
        self.assertEqual(t2_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'finish'}
        t3_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t3_ex_db.id))[0]
        t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(t3_ac_ex_db.liveaction['id'])
        self.assertEqual(t3_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t3_ac_ex_db)
        t3_ex_db = wf_db_access.TaskExecution.get_by_id(t3_ex_db.id)
        self.assertEqual(t3_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.SUCCEEDED)

    def test_consecutive_inquiries(self):
        if False:
            print('Hello World!')
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'ask-consecutive-approvals.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'start'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        self.assertEqual(t1_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t1_ac_ex_db)
        t1_ex_db = wf_db_access.TaskExecution.get_by_id(t1_ex_db.id)
        self.assertEqual(t1_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'get_approval'}
        t2_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_ex_db.id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_PENDING)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(t2_ex_db.id)
        self.assertEqual(t2_ex_db.status, wf_statuses.PENDING)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.PAUSED)
        inquiry_api = inqy_api_models.InquiryAPI.from_model(t2_ac_ex_db)
        inquiry_response = {'approved': True}
        inquiry_service.respond(inquiry_api, inquiry_response)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_lv_ac_db.id))
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(str(t2_ac_ex_db.id))
        self.assertEqual(t2_ac_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(str(t2_ex_db.id))
        self.assertEqual(t2_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'get_confirmation'}
        t3_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t3_ex_db.id))[0]
        t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(t3_ac_ex_db.liveaction['id'])
        self.assertEqual(t3_lv_ac_db.status, action_constants.LIVEACTION_STATUS_PENDING)
        workflows.get_engine().process(t3_ac_ex_db)
        t3_ex_db = wf_db_access.TaskExecution.get_by_id(t3_ex_db.id)
        self.assertEqual(t3_ex_db.status, wf_statuses.PENDING)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.PAUSED)
        inquiry_api = inqy_api_models.InquiryAPI.from_model(t3_ac_ex_db)
        inquiry_response = {'approved': True}
        inquiry_service.respond(inquiry_api, inquiry_response)
        t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t3_lv_ac_db.id))
        self.assertEqual(t3_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        t3_ac_ex_db = ex_db_access.ActionExecution.get_by_id(str(t3_ac_ex_db.id))
        self.assertEqual(t3_ac_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t3_ac_ex_db)
        t3_ex_db = wf_db_access.TaskExecution.get_by_id(str(t3_ex_db.id))
        self.assertEqual(t3_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'finish'}
        t4_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t4_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t4_ex_db.id))[0]
        t4_lv_ac_db = lv_db_access.LiveAction.get_by_id(t4_ac_ex_db.liveaction['id'])
        self.assertEqual(t4_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t4_ac_ex_db)
        t4_ex_db = wf_db_access.TaskExecution.get_by_id(t4_ex_db.id)
        self.assertEqual(t4_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.SUCCEEDED)

    def test_parallel_inquiries(self):
        if False:
            while True:
                i = 10
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'ask-parallel-approvals.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'start'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        self.assertEqual(t1_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t1_ac_ex_db)
        t1_ex_db = wf_db_access.TaskExecution.get_by_id(t1_ex_db.id)
        self.assertEqual(t1_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'ask_jack'}
        t2_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_ex_db.id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_PENDING)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(t2_ex_db.id)
        self.assertEqual(t2_ex_db.status, wf_statuses.PENDING)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.PAUSING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'ask_jill'}
        t3_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t3_ex_db.id))[0]
        t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(t3_ac_ex_db.liveaction['id'])
        self.assertEqual(t3_lv_ac_db.status, action_constants.LIVEACTION_STATUS_PENDING)
        workflows.get_engine().process(t3_ac_ex_db)
        t3_ex_db = wf_db_access.TaskExecution.get_by_id(t3_ex_db.id)
        self.assertEqual(t3_ex_db.status, wf_statuses.PENDING)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.PAUSED)
        inquiry_api = inqy_api_models.InquiryAPI.from_model(t2_ac_ex_db)
        inquiry_response = {'approved': True}
        inquiry_service.respond(inquiry_api, inquiry_response)
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_lv_ac_db.id))
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(str(t2_ac_ex_db.id))
        self.assertEqual(t2_ac_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(str(t2_ex_db.id))
        self.assertEqual(t2_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.PAUSED)
        inquiry_api = inqy_api_models.InquiryAPI.from_model(t3_ac_ex_db)
        inquiry_response = {'approved': True}
        inquiry_service.respond(inquiry_api, inquiry_response)
        t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t3_lv_ac_db.id))
        self.assertEqual(t3_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        t3_ac_ex_db = ex_db_access.ActionExecution.get_by_id(str(t3_ac_ex_db.id))
        self.assertEqual(t3_ac_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t3_ac_ex_db)
        t3_ex_db = wf_db_access.TaskExecution.get_by_id(str(t3_ex_db.id))
        self.assertEqual(t3_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'finish'}
        t4_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t4_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t4_ex_db.id))[0]
        t4_lv_ac_db = lv_db_access.LiveAction.get_by_id(t4_ac_ex_db.liveaction['id'])
        self.assertEqual(t4_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t4_ac_ex_db)
        t4_ex_db = wf_db_access.TaskExecution.get_by_id(t4_ex_db.id)
        self.assertEqual(t4_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.SUCCEEDED)

    def test_nested_inquiry(self):
        if False:
            return 10
        wf_meta = base.get_wf_fixture_meta_data(TEST_PACK_PATH, 'ask-nested-approval.yaml')
        lv_ac_db = lv_db_models.LiveActionDB(action=wf_meta['name'])
        (lv_ac_db, ac_ex_db) = action_service.request(lv_ac_db)
        lv_ac_db = lv_db_access.LiveAction.get_by_id(str(lv_ac_db.id))
        self.assertEqual(lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(ac_ex_db.id))[0]
        self.assertEqual(wf_ex_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'start'}
        t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t1_ex_db.id))[0]
        t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t1_ac_ex_db.liveaction['id'])
        self.assertEqual(t1_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t1_ac_ex_db)
        t1_ex_db = wf_db_access.TaskExecution.get_by_id(t1_ex_db.id)
        self.assertEqual(t1_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'get_approval'}
        t2_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_ex_db.id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_RUNNING)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(t2_ex_db.id)
        self.assertEqual(t2_ex_db.status, wf_statuses.RUNNING)
        t2_wf_ex_db = wf_db_access.WorkflowExecution.query(action_execution=str(t2_ac_ex_db.id))[0]
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(t2_wf_ex_db.id), 'task_id': 'start'}
        t2_t1_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t2_t1_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t1_ex_db.id))[0]
        t2_t1_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_t1_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_t1_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t2_t1_ac_ex_db)
        t2_t1_ex_db = wf_db_access.TaskExecution.get_by_id(t2_t1_ex_db.id)
        self.assertEqual(t2_t1_ex_db.status, wf_statuses.SUCCEEDED)
        t2_wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(str(t2_wf_ex_db.id))
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(t2_wf_ex_db.id), 'task_id': 'get_approval'}
        t2_t2_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t2_t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t2_ex_db.id))[0]
        t2_t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_PENDING)
        workflows.get_engine().process(t2_t2_ac_ex_db)
        t2_t2_ex_db = wf_db_access.TaskExecution.get_by_id(t2_t2_ex_db.id)
        self.assertEqual(t2_t2_ex_db.status, wf_statuses.PENDING)
        t2_wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(str(t2_wf_ex_db.id))
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.PAUSED)
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_ex_db.id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_PAUSED)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(t2_ex_db.id)
        self.assertEqual(t2_ex_db.status, wf_statuses.PAUSED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.PAUSED)
        inquiry_api = inqy_api_models.InquiryAPI.from_model(t2_t2_ac_ex_db)
        inquiry_response = {'approved': True}
        inquiry_service.respond(inquiry_api, inquiry_response)
        t2_t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(str(t2_t2_lv_ac_db.id))
        self.assertEqual(t2_t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        t2_t2_ac_ex_db = ex_db_access.ActionExecution.get_by_id(str(t2_t2_ac_ex_db.id))
        self.assertEqual(t2_t2_ac_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t2_t2_ac_ex_db)
        t2_t2_ex_db = wf_db_access.TaskExecution.get_by_id(str(t2_t2_ex_db.id))
        self.assertEqual(t2_t2_ex_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(t2_wf_ex_db.id), 'task_id': 'finish'}
        t2_t3_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t2_t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_t3_ex_db.id))[0]
        t2_t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_t3_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_t3_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t2_t3_ac_ex_db)
        t2_t3_ex_db = wf_db_access.TaskExecution.get_by_id(t2_t3_ex_db.id)
        self.assertEqual(t2_t3_ex_db.status, wf_statuses.SUCCEEDED)
        t2_wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(str(t2_wf_ex_db.id))
        self.assertEqual(t2_wf_ex_db.status, wf_statuses.SUCCEEDED)
        t2_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t2_ex_db.id))[0]
        t2_lv_ac_db = lv_db_access.LiveAction.get_by_id(t2_ac_ex_db.liveaction['id'])
        self.assertEqual(t2_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t2_ac_ex_db)
        t2_ex_db = wf_db_access.TaskExecution.get_by_id(t2_ex_db.id)
        self.assertEqual(t2_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.RUNNING)
        query_filters = {'workflow_execution': str(wf_ex_db.id), 'task_id': 'finish'}
        t3_ex_db = wf_db_access.TaskExecution.query(**query_filters)[0]
        t3_ac_ex_db = ex_db_access.ActionExecution.query(task_execution=str(t3_ex_db.id))[0]
        t3_lv_ac_db = lv_db_access.LiveAction.get_by_id(t3_ac_ex_db.liveaction['id'])
        self.assertEqual(t3_lv_ac_db.status, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        workflows.get_engine().process(t3_ac_ex_db)
        t3_ex_db = wf_db_access.TaskExecution.get_by_id(t3_ex_db.id)
        self.assertEqual(t3_ex_db.status, wf_statuses.SUCCEEDED)
        wf_ex_db = wf_db_access.WorkflowExecution.get_by_id(wf_ex_db.id)
        self.assertEqual(wf_ex_db.status, wf_statuses.SUCCEEDED)