import six
from st2common import log as logging
from st2common.models.api.rule_enforcement import RuleEnforcementAPI
from st2common.persistence.rule_enforcement import RuleEnforcement
from st2common.util import isotime
from st2common.rbac.types import PermissionType
from st2api.controllers.resource import ResourceController
__all__ = ['RuleEnforcementController', 'SUPPORTED_FILTERS', 'QUERY_OPTIONS', 'FILTER_TRANSFORM_FUNCTIONS']
http_client = six.moves.http_client
LOG = logging.getLogger(__name__)
SUPPORTED_FILTERS = {'rule_ref': 'rule.ref', 'rule_id': 'rule.id', 'execution': 'execution_id', 'trigger_instance': 'trigger_instance_id', 'enforced_at': 'enforced_at', 'enforced_at_gt': 'enforced_at.gt', 'enforced_at_lt': 'enforced_at.lt'}
QUERY_OPTIONS = {'sort': ['-enforced_at', 'rule.ref']}
FILTER_TRANSFORM_FUNCTIONS = {'enforced_at': lambda value: isotime.parse(value=value), 'enforced_at_gt': lambda value: isotime.parse(value=value), 'enforced_at_lt': lambda value: isotime.parse(value=value)}

class RuleEnforcementController(ResourceController):
    model = RuleEnforcementAPI
    access = RuleEnforcement
    query_options = QUERY_OPTIONS
    supported_filters = SUPPORTED_FILTERS
    filter_transform_functions = FILTER_TRANSFORM_FUNCTIONS

    def get_all(self, exclude_attributes=None, include_attributes=None, sort=None, offset=0, limit=None, requester_user=None, **raw_filters):
        if False:
            for i in range(10):
                print('nop')
        return super(RuleEnforcementController, self)._get_all(exclude_fields=exclude_attributes, include_fields=include_attributes, sort=sort, offset=offset, limit=limit, raw_filters=raw_filters, requester_user=requester_user)

    def get_one(self, id, requester_user):
        if False:
            print('Hello World!')
        return super(RuleEnforcementController, self)._get_one_by_id(id, requester_user=requester_user, permission_type=PermissionType.RULE_ENFORCEMENT_VIEW)
rule_enforcements_controller = RuleEnforcementController()