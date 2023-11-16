from __future__ import absolute_import
import sys
import traceback
import uuid
import six
from oslo_config import cfg
from orquesta import exceptions as wf_exc
from orquesta import statuses as wf_statuses
from st2common.constants import action as ac_const
from st2common import log as logging
from st2common.exceptions import workflow as wf_svc_exc
from st2common.models.api import notification as notify_api_models
from st2common.persistence import execution as ex_db_access
from st2common.persistence import liveaction as lv_db_access
from st2common.runners import base as runners
from st2common.services import action as ac_svc
from st2common.services import workflows as wf_svc
from st2common.util import api as api_util
from st2common.util import deep_copy
__all__ = ['OrquestaRunner', 'get_runner', 'get_metadata']
LOG = logging.getLogger(__name__)

class OrquestaRunner(runners.AsyncActionRunner):

    @staticmethod
    def get_workflow_definition(entry_point):
        if False:
            i = 10
            return i + 15
        with open(entry_point, 'r') as def_file:
            return def_file.read()

    def _get_notify_config(self):
        if False:
            while True:
                i = 10
        return notify_api_models.NotificationsHelper.from_model(notify_model=self.liveaction.notify) if self.liveaction.notify else None

    def _construct_context(self, wf_ex):
        if False:
            i = 10
            return i + 15
        ctx = deep_copy.fast_deepcopy_dict(self.context)
        ctx['workflow_execution'] = str(wf_ex.id)
        return ctx

    def _construct_st2_context(self):
        if False:
            i = 10
            return i + 15
        st2_ctx = {'st2': {'action_execution_id': str(self.execution.id), 'api_url': api_util.get_full_public_api_url(), 'user': self.execution.context.get('user', cfg.CONF.system_user.user), 'pack': self.execution.context.get('pack', None), 'action': self.execution.action.get('ref', None), 'runner': self.execution.action.get('runner_type', None)}}
        if self.execution.context.get('api_user'):
            st2_ctx['st2']['api_user'] = self.execution.context.get('api_user')
        if self.execution.context.get('source_channel'):
            st2_ctx['st2']['source_channel'] = self.execution.context.get('source_channel')
        if self.execution.context:
            st2_ctx['parent'] = self.execution.context
        return st2_ctx

    def _handle_workflow_return_value(self, wf_ex_db):
        if False:
            return 10
        if wf_ex_db.status in wf_statuses.COMPLETED_STATUSES:
            status = wf_ex_db.status
            result = {'output': wf_ex_db.output or None}
            if wf_ex_db.status in wf_statuses.ABENDED_STATUSES:
                result['errors'] = wf_ex_db.errors
            for wf_ex_error in wf_ex_db.errors:
                msg = 'Workflow execution completed with errors.'
                wf_svc.update_progress(wf_ex_db, '%s %s' % (msg, str(wf_ex_error)), log=False)
                LOG.error('[%s] %s', str(self.execution.id), msg, extra=wf_ex_error)
            return (status, result, self.context)
        status = ac_const.LIVEACTION_STATUS_RUNNING
        partial_results = {}
        ctx = self._construct_context(wf_ex_db)
        return (status, partial_results, ctx)

    def run(self, action_parameters):
        if False:
            for i in range(10):
                print('nop')
        rerun_options = self.context.get('re-run', {})
        rerun_task_options = rerun_options.get('tasks', [])
        if self.rerun_ex_ref and rerun_task_options:
            return self.rerun_workflow(self.rerun_ex_ref, options=rerun_options)
        return self.start_workflow(action_parameters)

    def start_workflow(self, action_parameters):
        if False:
            print('Hello World!')
        wf_def = self.get_workflow_definition(self.entry_point)
        try:
            st2_ctx = self._construct_st2_context()
            notify_cfg = self._get_notify_config()
            wf_ex_db = wf_svc.request(wf_def, self.execution, st2_ctx, notify_cfg=notify_cfg)
        except wf_exc.WorkflowInspectionError as e:
            status = ac_const.LIVEACTION_STATUS_FAILED
            result = {'errors': e.args[1], 'output': None}
            return (status, result, self.context)
        except Exception as e:
            status = ac_const.LIVEACTION_STATUS_FAILED
            result = {'errors': [{'message': six.text_type(e)}], 'output': None}
            return (status, result, self.context)
        return self._handle_workflow_return_value(wf_ex_db)

    def rerun_workflow(self, ac_ex_ref, options=None):
        if False:
            i = 10
            return i + 15
        try:
            wf_ex_id = ac_ex_ref.context.get('workflow_execution')
            st2_ctx = self._construct_st2_context()
            st2_ctx['workflow_execution_id'] = wf_ex_id
            wf_ex_db = wf_svc.request_rerun(self.execution, st2_ctx, options=options)
        except Exception as e:
            status = ac_const.LIVEACTION_STATUS_FAILED
            result = {'errors': [{'message': six.text_type(e)}], 'output': None}
            return (status, result, self.context)
        return self._handle_workflow_return_value(wf_ex_db)

    @staticmethod
    def task_pauseable(ac_ex):
        if False:
            while True:
                i = 10
        wf_ex_pauseable = ac_ex.runner['name'] in ac_const.WORKFLOW_RUNNER_TYPES and ac_ex.status == ac_const.LIVEACTION_STATUS_RUNNING
        return wf_ex_pauseable

    def pause(self):
        if False:
            i = 10
            return i + 15
        wf_ex_db = wf_svc.request_pause(self.execution)
        for child_ex_id in self.execution.children:
            child_ex = ex_db_access.ActionExecution.get(id=child_ex_id)
            if self.task_pauseable(child_ex):
                ac_svc.request_pause(lv_db_access.LiveAction.get(id=child_ex.liveaction['id']), self.context.get('user', None))
        if wf_ex_db.status == wf_statuses.PAUSING or ac_svc.is_children_active(self.liveaction.id):
            status = ac_const.LIVEACTION_STATUS_PAUSING
        else:
            status = ac_const.LIVEACTION_STATUS_PAUSED
        return (status, self.liveaction.result, self.liveaction.context)

    @staticmethod
    def task_resumeable(ac_ex):
        if False:
            return 10
        wf_ex_resumeable = ac_ex.runner['name'] in ac_const.WORKFLOW_RUNNER_TYPES and ac_ex.status == ac_const.LIVEACTION_STATUS_PAUSED
        return wf_ex_resumeable

    def resume(self):
        if False:
            for i in range(10):
                print('nop')
        wf_ex_db = wf_svc.request_resume(self.execution)
        for child_ex_id in self.execution.children:
            child_ex = ex_db_access.ActionExecution.get(id=child_ex_id)
            if self.task_resumeable(child_ex):
                ac_svc.request_resume(lv_db_access.LiveAction.get(id=child_ex.liveaction['id']), self.context.get('user', None))
        return (wf_ex_db.status if wf_ex_db else ac_const.LIVEACTION_STATUS_RUNNING, self.liveaction.result, self.liveaction.context)

    @staticmethod
    def task_cancelable(ac_ex):
        if False:
            for i in range(10):
                print('nop')
        wf_ex_cancelable = ac_ex.runner['name'] in ac_const.WORKFLOW_RUNNER_TYPES and ac_ex.status in ac_const.LIVEACTION_CANCELABLE_STATES
        ac_ex_cancelable = ac_ex.runner['name'] not in ac_const.WORKFLOW_RUNNER_TYPES and ac_ex.status in ac_const.LIVEACTION_DELAYED_STATES
        return wf_ex_cancelable or ac_ex_cancelable

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        result = None
        wf_ex_db = None
        try:
            wf_ex_db = wf_svc.request_cancellation(self.execution)
        except (wf_svc_exc.WorkflowExecutionNotFoundException, wf_svc_exc.WorkflowExecutionIsCompletedException):
            pass
        except Exception:
            (_, ex, tb) = sys.exc_info()
            msg = 'Error encountered when canceling workflow execution.'
            LOG.exception('[%s] %s', str(self.execution.id), msg)
            msg = 'Error encountered when canceling workflow execution. %s'
            wf_svc.update_progress(wf_ex_db, msg % str(ex), log=False)
            result = {'error': msg % str(ex), 'traceback': ''.join(traceback.format_tb(tb, 20))}
        for child_ex_id in self.execution.children:
            child_ex = ex_db_access.ActionExecution.get(id=child_ex_id)
            if self.task_cancelable(child_ex):
                ac_svc.request_cancellation(lv_db_access.LiveAction.get(id=child_ex.liveaction['id']), self.context.get('user', None))
        status = ac_const.LIVEACTION_STATUS_CANCELING if ac_svc.is_children_active(self.liveaction.id) else ac_const.LIVEACTION_STATUS_CANCELED
        return (status, result if result else self.liveaction.result, self.liveaction.context)

def get_runner():
    if False:
        for i in range(10):
            print('nop')
    return OrquestaRunner(str(uuid.uuid4()))

def get_metadata():
    if False:
        return 10
    return runners.get_metadata('orquesta_runner')[0]