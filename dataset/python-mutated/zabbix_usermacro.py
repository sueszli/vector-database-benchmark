"""
Management of Zabbix usermacros.
:codeauthor: Raymond Kuiper <qix@the-wired.net>

"""
__deprecated__ = (3009, 'zabbix', 'https://github.com/salt-extensions/saltext-zabbix')

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only make these states available if Zabbix module is available.\n    '
    if 'zabbix.usermacro_create' in __salt__:
        return True
    return (False, 'zabbix module could not be loaded')

def present(name, value, hostid=None, **kwargs):
    if False:
        return 10
    "\n    Creates a new usermacro.\n\n    :param name: name of the usermacro\n    :param value: value of the usermacro\n    :param hostid: id's of the hosts to apply the usermacro on, if missing a global usermacro is assumed.\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    .. code-block:: yaml\n\n        override host usermacro:\n            zabbix_usermacro.present:\n                - name: '{$SNMP_COMMUNITY}''\n                - value: 'public'\n                - hostid: 21\n\n    "
    connection_args = {}
    if '_connection_user' in kwargs:
        connection_args['_connection_user'] = kwargs['_connection_user']
    if '_connection_password' in kwargs:
        connection_args['_connection_password'] = kwargs['_connection_password']
    if '_connection_url' in kwargs:
        connection_args['_connection_url'] = kwargs['_connection_url']
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    if hostid:
        comment_usermacro_created = 'Usermacro {} created on hostid {}.'.format(name, hostid)
        comment_usermacro_updated = 'Usermacro {} updated on hostid {}.'.format(name, hostid)
        comment_usermacro_notcreated = f'Unable to create usermacro: {name} on hostid {hostid}. '
        comment_usermacro_exists = 'Usermacro {} already exists on hostid {}.'.format(name, hostid)
        changes_usermacro_created = {name: {'old': f'Usermacro {name} does not exist on hostid {hostid}.', 'new': f'Usermacro {name} created on hostid {hostid}.'}}
    else:
        comment_usermacro_created = f'Usermacro {name} created.'
        comment_usermacro_updated = f'Usermacro {name} updated.'
        comment_usermacro_notcreated = f'Unable to create usermacro: {name}. '
        comment_usermacro_exists = f'Usermacro {name} already exists.'
        changes_usermacro_created = {name: {'old': f'Usermacro {name} does not exist.', 'new': f'Usermacro {name} created.'}}
    if 'exec_params' in kwargs:
        if isinstance(kwargs['exec_params'], list):
            kwargs['exec_params'] = '\n'.join(kwargs['exec_params']) + '\n'
        else:
            kwargs['exec_params'] = str(kwargs['exec_params']) + '\n'
    if hostid:
        usermacro_exists = __salt__['zabbix.usermacro_get'](name, hostids=hostid, **connection_args)
    else:
        usermacro_exists = __salt__['zabbix.usermacro_get'](name, globalmacro=True, **connection_args)
    if usermacro_exists:
        usermacroobj = usermacro_exists[0]
        if hostid:
            usermacroid = int(usermacroobj['hostmacroid'])
        else:
            usermacroid = int(usermacroobj['globalmacroid'])
        update_value = False
        if str(value) != usermacroobj['value']:
            update_value = True
    if __opts__['test']:
        if usermacro_exists:
            if update_value:
                ret['result'] = None
                ret['comment'] = comment_usermacro_updated
            else:
                ret['result'] = True
                ret['comment'] = comment_usermacro_exists
        else:
            ret['result'] = None
            ret['comment'] = comment_usermacro_created
        return ret
    error = []
    if usermacro_exists:
        if update_value:
            ret['result'] = True
            ret['comment'] = comment_usermacro_updated
            if hostid:
                updated_value = __salt__['zabbix.usermacro_update'](usermacroid, value=value, **connection_args)
            else:
                updated_value = __salt__['zabbix.usermacro_updateglobal'](usermacroid, value=value, **connection_args)
            if not isinstance(updated_value, int):
                if 'error' in updated_value:
                    error.append(updated_value['error'])
                else:
                    ret['changes']['value'] = value
        else:
            ret['result'] = True
            ret['comment'] = comment_usermacro_exists
    else:
        if hostid:
            usermacro_create = __salt__['zabbix.usermacro_create'](name, value, hostid, **connection_args)
        else:
            usermacro_create = __salt__['zabbix.usermacro_createglobal'](name, value, **connection_args)
        if 'error' not in usermacro_create:
            ret['result'] = True
            ret['comment'] = comment_usermacro_created
            ret['changes'] = changes_usermacro_created
        else:
            ret['result'] = False
            ret['comment'] = comment_usermacro_notcreated + str(usermacro_create['error'])
    if error:
        ret['changes'] = {}
        ret['result'] = False
        ret['comment'] = str(error)
    return ret

def absent(name, hostid=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Ensures that the mediatype does not exist, eventually deletes the mediatype.\n\n    :param name: name of the usermacro\n    :param hostid: id's of the hosts to apply the usermacro on, if missing a global usermacro is assumed.\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    .. code-block:: yaml\n\n        delete_usermacro:\n            zabbix_usermacro.absent:\n                - name: '{$SNMP_COMMUNITY}'\n\n    "
    connection_args = {}
    if '_connection_user' in kwargs:
        connection_args['_connection_user'] = kwargs['_connection_user']
    if '_connection_password' in kwargs:
        connection_args['_connection_password'] = kwargs['_connection_password']
    if '_connection_url' in kwargs:
        connection_args['_connection_url'] = kwargs['_connection_url']
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    if hostid:
        comment_usermacro_deleted = 'Usermacro {} deleted from hostid {}.'.format(name, hostid)
        comment_usermacro_notdeleted = f'Unable to delete usermacro: {name} from hostid {hostid}.'
        comment_usermacro_notexists = f'Usermacro {name} does not exist on hostid {hostid}.'
        changes_usermacro_deleted = {name: {'old': f'Usermacro {name} exists on hostid {hostid}.', 'new': f'Usermacro {name} deleted from {hostid}.'}}
    else:
        comment_usermacro_deleted = f'Usermacro {name} deleted.'
        comment_usermacro_notdeleted = f'Unable to delete usermacro: {name}.'
        comment_usermacro_notexists = f'Usermacro {name} does not exist.'
        changes_usermacro_deleted = {name: {'old': f'Usermacro {name} exists.', 'new': f'Usermacro {name} deleted.'}}
    if hostid:
        usermacro_exists = __salt__['zabbix.usermacro_get'](name, hostids=hostid, **connection_args)
    else:
        usermacro_exists = __salt__['zabbix.usermacro_get'](name, globalmacro=True, **connection_args)
    if __opts__['test']:
        if not usermacro_exists:
            ret['result'] = True
            ret['comment'] = comment_usermacro_notexists
        else:
            ret['result'] = None
            ret['comment'] = comment_usermacro_deleted
        return ret
    if not usermacro_exists:
        ret['result'] = True
        ret['comment'] = comment_usermacro_notexists
    else:
        try:
            if hostid:
                usermacroid = usermacro_exists[0]['hostmacroid']
                usermacro_delete = __salt__['zabbix.usermacro_delete'](usermacroid, **connection_args)
            else:
                usermacroid = usermacro_exists[0]['globalmacroid']
                usermacro_delete = __salt__['zabbix.usermacro_deleteglobal'](usermacroid, **connection_args)
        except KeyError:
            usermacro_delete = False
        if usermacro_delete and 'error' not in usermacro_delete:
            ret['result'] = True
            ret['comment'] = comment_usermacro_deleted
            ret['changes'] = changes_usermacro_deleted
        else:
            ret['result'] = False
            ret['comment'] = comment_usermacro_notdeleted + str(usermacro_delete['error'])
    return ret