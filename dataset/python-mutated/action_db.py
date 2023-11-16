from __future__ import absolute_import
from typing import Optional
from typing import List
from collections import OrderedDict
from oslo_config import cfg
import six
from mongoengine import ValidationError
from st2common import log as logging
from st2common.constants.action import LIVEACTION_STATUSES, LIVEACTION_STATUS_CANCELED, LIVEACTION_STATUS_SUCCEEDED
from st2common.exceptions.db import StackStormDBObjectNotFoundError
from st2common.persistence.action import Action
from st2common.persistence.liveaction import LiveAction
from st2common.persistence.runner import RunnerType
from st2common.metrics.base import get_driver
from st2common.util import output_schema
from st2common.util.jsonify import json_encode
LOG = logging.getLogger(__name__)
__all__ = ['get_action_parameters_specs', 'get_runnertype_by_id', 'get_runnertype_by_name', 'get_action_by_id', 'get_action_by_ref', 'get_liveaction_by_id', 'update_liveaction_status', 'serialize_positional_argument', 'get_args']

def get_action_parameters_specs(action_ref):
    if False:
        i = 10
        return i + 15
    '\n    Retrieve parameters specifications schema for the provided action reference.\n\n    Note: This function returns a union of action and action runner parameters.\n\n    :param action_ref: Action reference.\n    :type action_ref: ``str``\n\n    :rtype: ``dict``\n    '
    action_db = get_action_by_ref(ref=action_ref)
    parameters = {}
    if not action_db:
        return parameters
    runner_type_name = action_db.runner_type['name']
    runner_type_db = get_runnertype_by_name(runnertype_name=runner_type_name)
    parameters.update(runner_type_db['runner_parameters'])
    parameters.update(action_db.parameters)
    return parameters

def get_runnertype_by_id(runnertype_id):
    if False:
        i = 10
        return i + 15
    '\n    Get RunnerType by id.\n\n    On error, raise StackStormDBObjectNotFoundError\n    '
    try:
        runnertype = RunnerType.get_by_id(runnertype_id)
    except (ValueError, ValidationError) as e:
        LOG.warning('Database lookup for runnertype with id="%s" resulted in exception: %s', runnertype_id, e)
        raise StackStormDBObjectNotFoundError('Unable to find runnertype with id="%s"' % runnertype_id)
    return runnertype

def get_runnertype_by_name(runnertype_name):
    if False:
        print('Hello World!')
    '\n    Get an runnertype by name.\n    On error, raise ST2ObjectNotFoundError.\n    '
    try:
        runnertypes = RunnerType.query(name=runnertype_name)
    except (ValueError, ValidationError) as e:
        LOG.error('Database lookup for name="%s" resulted in exception: %s', runnertype_name, e)
        raise StackStormDBObjectNotFoundError('Unable to find runnertype with name="%s"' % runnertype_name)
    if not runnertypes:
        raise StackStormDBObjectNotFoundError('Unable to find RunnerType with name="%s"' % runnertype_name)
    if len(runnertypes) > 1:
        LOG.warning('More than one RunnerType returned from DB lookup by name. Result list is: %s', runnertypes)
    return runnertypes[0]

def get_action_by_id(action_id):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get Action by id.\n\n    On error, raise StackStormDBObjectNotFoundError\n    '
    action = None
    try:
        action = Action.get_by_id(action_id)
    except (ValueError, ValidationError) as e:
        LOG.warning('Database lookup for action with id="%s" resulted in exception: %s', action_id, e)
        raise StackStormDBObjectNotFoundError('Unable to find action with id="%s"' % action_id)
    return action

def get_action_by_ref(ref, only_fields: Optional[List[str]]=None):
    if False:
        while True:
            i = 10
    '\n    Returns the action object from db given a string ref.\n\n    :param ref: Reference to the trigger type db object.\n    :type ref: ``str``\n\n    :param: only_field: Optional lists if fields to retrieve. If not specified, it defaults to all\n                        fields.\n\n    :rtype action: ``object``\n    '
    try:
        return Action.get_by_ref(ref, only_fields=only_fields)
    except ValueError as e:
        LOG.debug('Database lookup for ref="%s" resulted ' + 'in exception : %s.', ref, e, exc_info=True)
        return None

def get_liveaction_by_id(liveaction_id):
    if False:
        print('Hello World!')
    '\n    Get LiveAction by id.\n\n    On error, raise ST2DBObjectNotFoundError.\n    '
    liveaction = None
    try:
        liveaction = LiveAction.get_by_id(liveaction_id)
    except (ValidationError, ValueError) as e:
        LOG.error('Database lookup for LiveAction with id="%s" resulted in exception: %s', liveaction_id, e)
        raise StackStormDBObjectNotFoundError('Unable to find LiveAction with id="%s"' % liveaction_id)
    return liveaction

def update_liveaction_status(status=None, result=None, context=None, end_timestamp=None, liveaction_id=None, runner_info=None, liveaction_db=None, publish=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update the status of the specified LiveAction to the value provided in\n    new_status.\n\n    The LiveAction may be specified using either liveaction_id, or as an\n    liveaction_db instance.\n    '
    if liveaction_id is None and liveaction_db is None:
        raise ValueError('Must specify an liveaction_id or an liveaction_db when calling update_LiveAction_status')
    if liveaction_db is None:
        liveaction_db = get_liveaction_by_id(liveaction_id)
    if status not in LIVEACTION_STATUSES:
        raise ValueError('Attempting to set status for LiveAction "%s" to unknown status string. Unknown status is "%s"' % (liveaction_db, status))
    if result and cfg.CONF.system.validate_output_schema and (status == LIVEACTION_STATUS_SUCCEEDED):
        action_db = get_action_by_ref(liveaction_db.action)
        runner_db = get_runnertype_by_name(action_db.runner_type['name'])
        (result, status) = output_schema.validate_output(runner_db.output_schema, action_db.output_schema, result, status, runner_db.output_key)
    if liveaction_db.status:
        get_driver().dec_counter('action.executions.%s' % liveaction_db.status)
    if status:
        get_driver().inc_counter('action.executions.%s' % status)
    extra = {'liveaction_db': liveaction_db}
    LOG.debug('Updating ActionExection: "%s" with status="%s"', liveaction_db.id, status, extra=extra)
    if liveaction_db.status == LIVEACTION_STATUS_CANCELED and status != LIVEACTION_STATUS_CANCELED:
        LOG.info('Unable to update ActionExecution "%s" with status="%s". ActionExecution is already canceled.', liveaction_db.id, status, extra=extra)
        return liveaction_db
    old_status = liveaction_db.status
    liveaction_db.status = status
    if result:
        liveaction_db.result = result
    if context:
        liveaction_db.context.update(context)
    if end_timestamp:
        liveaction_db.end_timestamp = end_timestamp
    if runner_info:
        liveaction_db.runner_info = runner_info
    liveaction_db = LiveAction.add_or_update(liveaction_db)
    LOG.debug('Updated status for LiveAction object.', extra=extra)
    if publish and status != old_status:
        LiveAction.publish_status(liveaction_db)
        LOG.debug('Published status for LiveAction object.', extra=extra)
    return liveaction_db

def serialize_positional_argument(argument_type, argument_value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Serialize the provided positional argument.\n\n    Note: Serialization is NOT performed recursively since it doesn't make much\n    sense for shell script actions (only the outter / top level value is\n    serialized).\n    "
    if argument_type in ['string', 'number', 'float']:
        if argument_value is None:
            argument_value = six.text_type('')
            return argument_value
        if isinstance(argument_value, (int, float)):
            argument_value = str(argument_value)
        if not isinstance(argument_value, six.text_type):
            argument_value = argument_value.decode('utf-8')
    elif argument_type == 'boolean':
        if argument_value is not None:
            argument_value = '1' if bool(argument_value) else '0'
        else:
            argument_value = ''
    elif argument_type in ['array', 'list']:
        argument_value = ','.join(map(str, argument_value)) if argument_value else ''
    elif argument_type == 'object':
        argument_value = json_encode(argument_value) if argument_value else ''
    elif argument_type == 'null':
        argument_value = ''
    else:
        argument_value = six.text_type(argument_value) if argument_value else ''
    return argument_value

def get_args(action_parameters, action_db):
    if False:
        for i in range(10):
            print('nop')
    '\n\n    Get and serialize positional and named arguments.\n\n    :return: (positional_args, named_args)\n    :rtype: (``str``, ``dict``)\n    '
    position_args_dict = _get_position_arg_dict(action_parameters, action_db)
    action_db_parameters = action_db.parameters or {}
    positional_args = []
    positional_args_keys = set()
    for (_, arg) in six.iteritems(position_args_dict):
        arg_type = action_db_parameters.get(arg, {}).get('type', None)
        arg_value = action_parameters.get(arg, None)
        arg_value = serialize_positional_argument(argument_type=arg_type, argument_value=arg_value)
        positional_args.append(arg_value)
        positional_args_keys.add(arg)
    named_args = {}
    for param in action_parameters:
        if param not in positional_args_keys:
            named_args[param] = action_parameters.get(param)
    return (positional_args, named_args)

def _get_position_arg_dict(action_parameters, action_db):
    if False:
        return 10
    action_db_params = action_db.parameters
    args_dict = {}
    for param in action_db_params:
        param_meta = action_db_params.get(param, None)
        if param_meta is not None:
            pos = param_meta.get('position')
            if pos is not None:
                args_dict[pos] = param
    args_dict = OrderedDict(sorted(args_dict.items()))
    return args_dict