from __future__ import absolute_import
import six
import uuid
from oslo_config import cfg
from apscheduler.triggers.cron import CronTrigger
from st2common import log as logging
from st2common.exceptions.apivalidation import ValueValidationException
from st2common.constants.triggers import SYSTEM_TRIGGER_TYPES
from st2common.constants.triggers import CRON_TIMER_TRIGGER_REF
from st2common.util import schema as util_schema
import st2common.operators as criteria_operators
from st2common.services import triggers
__all__ = ['validate_criteria', 'validate_trigger_parameters', 'validate_trigger_payload']
LOG = logging.getLogger(__name__)
allowed_operators = criteria_operators.get_allowed_operators()

def validate_criteria(criteria):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(criteria, dict):
        raise ValueValidationException('Criteria should be a dict.')
    for (key, value) in six.iteritems(criteria):
        operator = value.get('type', None)
        if operator is None:
            raise ValueValidationException('Operator not specified for field: ' + key)
        if operator not in allowed_operators:
            raise ValueValidationException('For field: ' + key + ', operator ' + operator + ' not in list of allowed operators: ' + str(list(allowed_operators.keys())))
        pattern = value.get('pattern', None)
        if pattern is None:
            raise ValueValidationException('For field: ' + key + ', no pattern specified ' + 'for operator ' + operator)

def validate_trigger_parameters(trigger_type_ref, parameters):
    if False:
        return 10
    '\n    This function validates parameters for system and user-defined triggers.\n\n    :param trigger_type_ref: Reference of a trigger type.\n    :type trigger_type_ref: ``str``\n\n    :param parameters: Trigger parameters.\n    :type parameters: ``dict``\n\n    :return: Cleaned parameters on success, None if validation is not performed.\n    '
    if not trigger_type_ref:
        return None
    is_system_trigger = trigger_type_ref in SYSTEM_TRIGGER_TYPES
    if is_system_trigger:
        parameters_schema = SYSTEM_TRIGGER_TYPES[trigger_type_ref]['parameters_schema']
    else:
        trigger_type_db = triggers.get_trigger_type_db(trigger_type_ref)
        if not trigger_type_db:
            return None
        parameters_schema = getattr(trigger_type_db, 'parameters_schema', {})
        if not parameters_schema:
            return None
    if not is_system_trigger and (not cfg.CONF.system.validate_trigger_parameters):
        LOG.debug('Got non-system trigger "%s", but trigger parameter validation for non-systemtriggers is disabled, skipping validation.' % trigger_type_ref)
        return None
    cleaned = util_schema.validate(instance=parameters, schema=parameters_schema, cls=util_schema.CustomValidator, use_default=True, allow_default_none=True)
    if trigger_type_ref == CRON_TIMER_TRIGGER_REF:
        CronTrigger(**parameters)
    return cleaned

def validate_trigger_payload(trigger_type_ref, payload, throw_on_inexistent_trigger=False):
    if False:
        print('Hello World!')
    '\n    This function validates trigger payload parameters for system and user-defined triggers.\n\n    :param trigger_type_ref: Reference of a trigger type / trigger / trigger dictionary object.\n    :type trigger_type_ref: ``str``\n\n    :param payload: Trigger payload.\n    :type payload: ``dict``\n\n    :return: Cleaned payload on success, None if validation is not performed.\n    '
    if not trigger_type_ref:
        return None
    if isinstance(trigger_type_ref, dict):
        if trigger_type_ref.get('type', None):
            trigger_type_ref = trigger_type_ref['type']
        else:
            trigger_db = triggers.get_trigger_db_by_ref_or_dict(trigger_type_ref)
            if not trigger_db:
                return None
            trigger_type_ref = trigger_db.type
    is_system_trigger = trigger_type_ref in SYSTEM_TRIGGER_TYPES
    if is_system_trigger:
        payload_schema = SYSTEM_TRIGGER_TYPES[trigger_type_ref]['payload_schema']
    else:
        try:
            trigger_uuid = uuid.UUID(trigger_type_ref.split('.')[-1])
        except ValueError:
            is_trigger_db = False
        else:
            is_trigger_db = trigger_uuid.version == 4
        if is_trigger_db:
            trigger_db = triggers.get_trigger_db_by_ref(trigger_type_ref)
            if trigger_db:
                trigger_type_ref = trigger_db.type
        trigger_type_db = triggers.get_trigger_type_db(trigger_type_ref)
        if not trigger_type_db:
            if throw_on_inexistent_trigger:
                msg = 'Trigger type with reference "%s" doesn\'t exist in the database' % trigger_type_ref
                raise ValueError(msg)
            return None
        payload_schema = getattr(trigger_type_db, 'payload_schema', {})
        if not payload_schema:
            return None
    if not is_system_trigger and (not cfg.CONF.system.validate_trigger_payload):
        LOG.debug('Got non-system trigger "%s", but trigger payload validation for non-systemtriggers is disabled, skipping validation.' % trigger_type_ref)
        return None
    cleaned = util_schema.validate(instance=payload, schema=payload_schema, cls=util_schema.CustomValidator, use_default=True, allow_default_none=True)
    return cleaned