from st2api.controllers.resource import ResourceController
from st2common.models.api.trace import TraceAPI
from st2common.persistence.trace import Trace
from st2common.rbac.types import PermissionType
__all__ = ['TracesController']

class TracesController(ResourceController):
    model = TraceAPI
    access = Trace
    supported_filters = {'trace_tag': 'trace_tag', 'execution': 'action_executions.object_id', 'rule': 'rules.object_id', 'trigger_instance': 'trigger_instances.object_id'}
    query_options = {'sort': ['-start_timestamp', 'trace_tag']}

    def get_all(self, exclude_attributes=None, include_attributes=None, sort=None, offset=0, limit=None, requester_user=None, **raw_filters):
        if False:
            i = 10
            return i + 15
        query_options = None
        if 'sort_desc' in raw_filters and raw_filters['sort_desc'] == 'True':
            query_options = {'sort': ['-start_timestamp', 'trace_tag']}
        elif 'sort_asc' in raw_filters and raw_filters['sort_asc'] == 'True':
            query_options = {'sort': ['+start_timestamp', 'trace_tag']}
        return self._get_all(exclude_fields=exclude_attributes, include_fields=include_attributes, sort=sort, offset=offset, limit=limit, query_options=query_options, raw_filters=raw_filters, requester_user=requester_user)

    def get_one(self, id, requester_user):
        if False:
            i = 10
            return i + 15
        return self._get_one_by_id(id, requester_user=requester_user, permission_type=PermissionType.TRACE_VIEW)
traces_controller = TracesController()