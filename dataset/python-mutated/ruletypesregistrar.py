from __future__ import absolute_import
import copy
from st2common import log as logging
from st2common.constants.rules import RULE_TYPE_STANDARD, RULE_TYPE_BACKSTOP
from st2common.models.api.rule import RuleTypeAPI
from st2common.persistence.rule import RuleType
from st2common.exceptions.db import StackStormDBObjectNotFoundError
__all__ = ['register_rule_types', 'RULE_TYPES']
LOG = logging.getLogger(__name__)
RULE_TYPES = [{'name': RULE_TYPE_STANDARD, 'description': 'standard rule that is always applicable.', 'enabled': True, 'parameters': {}}, {'name': RULE_TYPE_BACKSTOP, 'description': 'Rule that applies when no other rule has matched for a specific Trigger.', 'enabled': True, 'parameters': {}}]

def register_rule_types():
    if False:
        print('Hello World!')
    LOG.debug('Start : register default RuleTypes.')
    registered_count = 0
    for rule_type in RULE_TYPES:
        rule_type = copy.deepcopy(rule_type)
        try:
            rule_type_db = RuleType.get_by_name(rule_type['name'])
            update = True
        except StackStormDBObjectNotFoundError:
            rule_type_db = None
            update = False
        rule_type_api = RuleTypeAPI(**rule_type)
        rule_type_api.validate()
        rule_type_model = RuleTypeAPI.to_model(rule_type_api)
        if rule_type_db:
            rule_type_model.id = rule_type_db.id
        try:
            rule_type_db = RuleType.add_or_update(rule_type_model)
            extra = {'rule_type_db': rule_type_db}
            if update:
                LOG.audit('RuleType updated. RuleType %s', rule_type_db, extra=extra)
            else:
                LOG.audit('RuleType created. RuleType %s', rule_type_db, extra=extra)
        except Exception:
            LOG.exception('Unable to register RuleType %s.', rule_type['name'])
        else:
            registered_count += 1
    LOG.debug('End : register default RuleTypes.')
    return registered_count