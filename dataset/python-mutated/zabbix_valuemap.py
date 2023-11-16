"""
Management of Zabbix Valuemap object over Zabbix API.

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
        for i in range(10):
            print('nop')
    '\n    Only make these states available if Zabbix module and run_query function is available\n    and all 3rd party modules imported.\n    '
    if 'zabbix.run_query' in __salt__:
        return True
    return (False, 'Import zabbix or other needed modules failed.')

def present(name, params, **kwargs):
    if False:
        return 10
    "\n    Creates Zabbix Value map object or if differs update it according defined parameters\n\n    :param name: Zabbix Value map name\n    :param params: Definition of the Zabbix Value map\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    .. code-block:: yaml\n\n        zabbix-valuemap-present:\n            zabbix_valuemap.present:\n                - name: Number mapping\n                - params:\n                    mappings:\n                        - value: 1\n                          newvalue: one\n                        - value: 2\n                          newvalue: two\n    "
    zabbix_id_mapper = __salt__['zabbix.get_zabbix_id_mapper']()
    dry_run = __opts__['test']
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    params['name'] = name
    input_params = __salt__['zabbix.substitute_params'](params, **kwargs)
    log.info('Zabbix Value map: input params: %s', str(json.dumps(input_params, indent=4)))
    search = {'output': 'extend', 'selectMappings': 'extend', 'filter': {'name': name}}
    valuemap_get = __salt__['zabbix.run_query']('valuemap.get', search, **kwargs)
    log.info('Zabbix Value map: valuemap.get result: %s', str(json.dumps(valuemap_get, indent=4)))
    existing_obj = __salt__['zabbix.substitute_params'](valuemap_get[0], **kwargs) if valuemap_get and len(valuemap_get) == 1 else False
    if existing_obj:
        diff_params = __salt__['zabbix.compare_params'](input_params, existing_obj)
        log.info('Zabbix Value map: input params: {%s', str(json.dumps(input_params, indent=4)))
        log.info('Zabbix Value map: Object comparison result. Differences: %s', str(diff_params))
        if diff_params:
            diff_params[zabbix_id_mapper['valuemap']] = existing_obj[zabbix_id_mapper['valuemap']]
            log.info('Zabbix Value map: update params: %s', str(json.dumps(diff_params, indent=4)))
            if dry_run:
                ret['result'] = True
                ret['comment'] = f'Zabbix Value map "{name}" would be fixed.'
                ret['changes'] = {name: {'old': 'Zabbix Value map "{}" differs in following parameters: {}'.format(name, diff_params), 'new': 'Zabbix Value map "{}" would correspond to definition.'.format(name)}}
            else:
                valuemap_update = __salt__['zabbix.run_query']('valuemap.update', diff_params, **kwargs)
                log.info('Zabbix Value map: valuemap.update result: %s', str(valuemap_update))
                if valuemap_update:
                    ret['result'] = True
                    ret['comment'] = f'Zabbix Value map "{name}" updated.'
                    ret['changes'] = {name: {'old': 'Zabbix Value map "{}" differed in following parameters: {}'.format(name, diff_params), 'new': f'Zabbix Value map "{name}" fixed.'}}
        else:
            ret['result'] = True
            ret['comment'] = 'Zabbix Value map "{}" already exists and corresponds to a definition.'.format(name)
    elif dry_run:
        ret['result'] = True
        ret['comment'] = f'Zabbix Value map "{name}" would be created.'
        ret['changes'] = {name: {'old': f'Zabbix Value map "{name}" does not exist.', 'new': 'Zabbix Value map "{}" would be created according definition.'.format(name)}}
    else:
        valuemap_create = __salt__['zabbix.run_query']('valuemap.create', input_params, **kwargs)
        log.info('Zabbix Value map: valuemap.create result: %s', str(valuemap_create))
        if valuemap_create:
            ret['result'] = True
            ret['comment'] = f'Zabbix Value map "{name}" created.'
            ret['changes'] = {name: {'old': f'Zabbix Value map "{name}" did not exist.', 'new': 'Zabbix Value map "{}" created according definition.'.format(name)}}
    return ret

def absent(name, **kwargs):
    if False:
        return 10
    "\n    Makes the Zabbix Value map to be absent (either does not exist or delete it).\n\n    :param name: Zabbix Value map name\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    .. code-block:: yaml\n\n        zabbix-valuemap-absent:\n            zabbix_valuemap.absent:\n                - name: Value map name\n    "
    dry_run = __opts__['test']
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    try:
        object_id = __salt__['zabbix.get_object_id_by_params']('valuemap', {'filter': {'name': name}}, **kwargs)
    except SaltException:
        object_id = False
    if not object_id:
        ret['result'] = True
        ret['comment'] = f'Zabbix Value map "{name}" does not exist.'
    elif dry_run:
        ret['result'] = True
        ret['comment'] = f'Zabbix Value map "{name}" would be deleted.'
        ret['changes'] = {name: {'old': f'Zabbix Value map "{name}" exists.', 'new': f'Zabbix Value map "{name}" would be deleted.'}}
    else:
        valuemap_delete = __salt__['zabbix.run_query']('valuemap.delete', [object_id], **kwargs)
        if valuemap_delete:
            ret['result'] = True
            ret['comment'] = f'Zabbix Value map "{name}" deleted.'
            ret['changes'] = {name: {'old': f'Zabbix Value map "{name}" existed.', 'new': f'Zabbix Value map "{name}" deleted.'}}
    return ret