"""
Management of InfluxDB users
============================

(compatible with InfluxDB version 0.9+)
"""

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if the influxdb module is available\n    '
    if 'influxdb.db_exists' in __salt__:
        return 'influxdb_user'
    return (False, 'influxdb module could not be loaded')

def present(name, passwd, admin=False, grants=None, **client_args):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that given user is present.\n\n    name\n        Name of the user to manage\n\n    passwd\n        Password of the user\n\n    admin : False\n        Whether the user should have cluster administration\n        privileges or not.\n\n    grants\n        Optional - Dict of database:privilege items associated with\n        the user. Example:\n\n        grants:\n          foo_db: read\n          bar_db: all\n\n    **Example:**\n\n    .. code-block:: yaml\n\n        example user present in influxdb:\n          influxdb_user.present:\n            - name: example\n            - passwd: somepassword\n            - admin: False\n            - grants:\n                foo_db: read\n                bar_db: all\n    '
    create = False
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'User {} is present and up to date'.format(name)}
    if not __salt__['influxdb.user_exists'](name, **client_args):
        create = True
        if __opts__['test']:
            ret['comment'] = 'User {} will be created'.format(name)
            ret['result'] = None
            return ret
        elif not __salt__['influxdb.create_user'](name, passwd, admin=admin, **client_args):
            ret['comment'] = 'Failed to create user {}'.format(name)
            ret['result'] = False
            return ret
    else:
        user = __salt__['influxdb.user_info'](name, **client_args)
        if user['admin'] != admin:
            if not __opts__['test']:
                if admin:
                    __salt__['influxdb.grant_admin_privileges'](name, **client_args)
                else:
                    __salt__['influxdb.revoke_admin_privileges'](name, **client_args)
                if admin != __salt__['influxdb.user_info'](name, **client_args)['admin']:
                    ret['comment'] = 'Failed to set admin privilege to user {}'.format(name)
                    ret['result'] = False
                    return ret
            ret['changes']['Admin privileges'] = admin
    if grants:
        db_privileges = __salt__['influxdb.list_privileges'](name, **client_args)
        for (database, privilege) in grants.items():
            privilege = privilege.lower()
            if privilege != db_privileges.get(database, privilege):
                if not __opts__['test']:
                    __salt__['influxdb.revoke_privilege'](database, 'all', name, **client_args)
                del db_privileges[database]
            if database not in db_privileges:
                ret['changes']['Grant on database {} to user {}'.format(database, name)] = privilege
                if not __opts__['test']:
                    __salt__['influxdb.grant_privilege'](database, privilege, name, **client_args)
    if ret['changes']:
        if create:
            ret['comment'] = 'Created user {}'.format(name)
            ret['changes'][name] = 'User created'
        elif __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'User {} will be updated with the following changes:'.format(name)
            for (k, v) in ret['changes'].items():
                ret['comment'] += '\n{} => {}'.format(k, v)
            ret['changes'] = {}
        else:
            ret['comment'] = 'Updated user {}'.format(name)
    return ret

def absent(name, **client_args):
    if False:
        return 10
    '\n    Ensure that given user is absent.\n\n    name\n        The name of the user to manage\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'User {} is not present'.format(name)}
    if __salt__['influxdb.user_exists'](name, **client_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'User {} will be removed'.format(name)
            return ret
        elif __salt__['influxdb.remove_user'](name, **client_args):
            ret['comment'] = 'Removed user {}'.format(name)
            ret['changes'][name] = 'removed'
            return ret
        else:
            ret['comment'] = 'Failed to remove user {}'.format(name)
            ret['result'] = False
            return ret
    return ret