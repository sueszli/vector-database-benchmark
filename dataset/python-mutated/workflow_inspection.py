from __future__ import absolute_import
import uuid
from orquesta.specs import loader as specs_loader
from oslo_config import cfg
from st2common import log as logging
from st2common import router
from st2common.services import workflows as workflow_service
from st2common.util import api as api_utils
LOG = logging.getLogger(__name__)

class WorkflowInspectionController(object):

    def mock_st2_ctx(self):
        if False:
            i = 10
            return i + 15
        st2_ctx = {'st2': {'api_url': api_utils.get_full_public_api_url(), 'action_execution_id': uuid.uuid4().hex, 'user': cfg.CONF.system_user.user}}
        return st2_ctx

    def post(self, wf_def):
        if False:
            while True:
                i = 10
        spec_module = specs_loader.get_spec_module('native')
        wf_spec = spec_module.instantiate(wf_def)
        st2_ctx = self.mock_st2_ctx()
        errors = workflow_service.inspect(wf_spec, st2_ctx, raise_exception=False)
        return router.Response(json=errors)
workflow_inspection_controller = WorkflowInspectionController()