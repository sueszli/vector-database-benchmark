"""
Management of Zabbix Action object over Zabbix API.

.. versionadded:: 2017.7.0

:codeauthor: Jakub Sliva <jakub.sliva@ultimum.io>
"""
import json
import logging
from salt.exceptions import SaltException
log = logging.getLogger(__name__)
__deprecated__ = (3009, 'zabbix', 'https://github.com/salt-extensions/saltext-zabbix')

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only make these states available if Zabbix module and run_query function is available\n    and all 3rd party modules imported.\n    '
    if 'zabbix.run_query' in __salt__:
        return True
    return (False, 'Import zabbix or other needed modules failed.')

def present(name, params, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Creates Zabbix Action object or if differs update it according defined parameters\n\n    :param name: Zabbix Action name\n    :param params: Definition of the Zabbix Action\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module\'s docstring)\n\n    If there is a need to get a value from current zabbix online (e.g. id of a hostgroup you want to put a discovered\n    system into), put a dictionary with two keys "query_object" and "query_name" instead of the value.\n    In this example we want to get object id of hostgroup named "Virtual machines" and "Databases".\n\n    .. code-block:: yaml\n\n        zabbix-action-present:\n            zabbix_action.present:\n                - name: VMs\n                - params:\n                    eventsource: 2\n                    status: 0\n                    filter:\n                        evaltype: 2\n                        conditions:\n                            - conditiontype: 24\n                              operator: 2\n                              value: \'virtual\'\n                            - conditiontype: 24\n                              operator: 2\n                              value: \'kvm\'\n                    operations:\n                        - operationtype: 2\n                        - operationtype: 4\n                          opgroup:\n                              - groupid:\n                                  query_object: hostgroup\n                                  query_name: Virtual machines\n                              - groupid:\n                                  query_object: hostgroup\n                                  query_name: Databases\n    '
    zabbix_id_mapper = __salt__['zabbix.get_zabbix_id_mapper']()
    dry_run = __opts__['test']
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    params['name'] = name
    params['operations'] = params['operations'] if 'operations' in params else []
    if 'filter' in params:
        params['filter']['conditions'] = params['filter']['conditions'] if 'conditions' in params['filter'] else []
    input_params = __salt__['zabbix.substitute_params'](params, **kwargs)
    log.info('Zabbix Action: input params: %s', str(json.dumps(input_params, indent=4)))
    search = {'output': 'extend', 'selectOperations': 'extend', 'selectFilter': 'extend', 'filter': {'name': name}}
    action_get = __salt__['zabbix.run_query']('action.get', search, **kwargs)
    log.info('Zabbix Action: action.get result: %s', str(json.dumps(action_get, indent=4)))
    existing_obj = __salt__['zabbix.substitute_params'](action_get[0], **kwargs) if action_get and len(action_get) == 1 else False
    if existing_obj:
        diff_params = __salt__['zabbix.compare_params'](input_params, existing_obj)
        log.info('Zabbix Action: input params: {%s', str(json.dumps(input_params, indent=4)))
        log.info('Zabbix Action: Object comparison result. Differences: %s', str(diff_params))
        if diff_params:
            diff_params[zabbix_id_mapper['action']] = existing_obj[zabbix_id_mapper['action']]
            log.info('Zabbix Action: update params: %s', str(json.dumps(diff_params, indent=4)))
            if dry_run:
                ret['result'] = True
                ret['comment'] = f'Zabbix Action "{name}" would be fixed.'
                ret['changes'] = {name: {'old': 'Zabbix Action "{}" differs in following parameters: {}'.format(name, diff_params), 'new': 'Zabbix Action "{}" would correspond to definition.'.format(name)}}
            else:
                action_update = __salt__['zabbix.run_query']('action.update', diff_params, **kwargs)
                log.info('Zabbix Action: action.update result: %s', str(action_update))
                if action_update:
                    ret['result'] = True
                    ret['comment'] = f'Zabbix Action "{name}" updated.'
                    ret['changes'] = {name: {'old': 'Zabbix Action "{}" differed in following parameters: {}'.format(name, diff_params), 'new': f'Zabbix Action "{name}" fixed.'}}
        else:
            ret['result'] = True
            ret['comment'] = 'Zabbix Action "{}" already exists and corresponds to a definition.'.format(name)
    elif dry_run:
        ret['result'] = True
        ret['comment'] = f'Zabbix Action "{name}" would be created.'
        ret['changes'] = {name: {'old': f'Zabbix Action "{name}" does not exist.', 'new': 'Zabbix Action "{}" would be created according definition.'.format(name)}}
    else:
        action_create = __salt__['zabbix.run_query']('action.create', input_params, **kwargs)
        log.info('Zabbix Action: action.create result: %s', str(action_create))
        if action_create:
            ret['result'] = True
            ret['comment'] = f'Zabbix Action "{name}" created.'
            ret['changes'] = {name: {'old': f'Zabbix Action "{name}" did not exist.', 'new': 'Zabbix Action "{}" created according definition.'.format(name)}}
    return ret

def absent(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Makes the Zabbix Action to be absent (either does not exist or delete it).\n\n    :param name: Zabbix Action name\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    .. code-block:: yaml\n\n        zabbix-action-absent:\n            zabbix_action.absent:\n                - name: Action name\n    "
    dry_run = __opts__['test']
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    try:
        object_id = __salt__['zabbix.get_object_id_by_params']('action', {'filter': {'name': name}}, **kwargs)
    except SaltException:
        object_id = False
    if not object_id:
        ret['result'] = True
        ret['comment'] = f'Zabbix Action "{name}" does not exist.'
    elif dry_run:
        ret['result'] = True
        ret['comment'] = f'Zabbix Action "{name}" would be deleted.'
        ret['changes'] = {name: {'old': f'Zabbix Action "{name}" exists.', 'new': f'Zabbix Action "{name}" would be deleted.'}}
    else:
        action_delete = __salt__['zabbix.run_query']('action.delete', [object_id], **kwargs)
        if action_delete:
            ret['result'] = True
            ret['comment'] = f'Zabbix Action "{name}" deleted.'
            ret['changes'] = {name: {'old': f'Zabbix Action "{name}" existed.', 'new': f'Zabbix Action "{name}" deleted.'}}
    return ret