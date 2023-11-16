import six
from st2common import log as logging
from st2common.persistence.execution import ActionExecution
LOG = logging.getLogger(__name__)
SUPPORTED_FILTERS = {'action': 'action.ref', 'status': 'status', 'liveaction': 'liveaction.id', 'parent': 'parent', 'rule': 'rule.name', 'runner': 'runner.name', 'timestamp': 'start_timestamp', 'trigger': 'trigger.name', 'trigger_type': 'trigger_type.name', 'trigger_instance': 'trigger_instance.id', 'user': 'context.user'}
FILTERS_WITH_VALID_NULL_VALUES = ['parent', 'rule', 'trigger', 'trigger_type', 'trigger_instance']
IGNORE_FILTERS = ['parent', 'timestamp', 'liveaction', 'trigger_instance']

class FiltersController(object):

    def get_all(self, types=None):
        if False:
            while True:
                i = 10
        '\n        List all distinct filters.\n\n        Handles requests:\n            GET /executions/views/filters[?types=action,rule]\n\n        :param types: Comma delimited string of filter types to output.\n        :type types: ``str``\n        '
        filters = {}
        for (name, field) in six.iteritems(SUPPORTED_FILTERS):
            if name not in IGNORE_FILTERS and (not types or name in types):
                if name not in FILTERS_WITH_VALID_NULL_VALUES:
                    query = {field.replace('.', '__'): {'$ne': None}}
                else:
                    query = {}
                filters[name] = ActionExecution.distinct(field=field, **query)
        return filters

class ExecutionViewsController(object):
    filters = FiltersController()
filters_controller = FiltersController()