"""
Management of InfluxDB 0.8 users
================================

(compatible with InfluxDB version 0.5-0.8)

.. versionadded:: 2014.7.0

"""

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if the influxdb08 module is available\n    '
    if 'influxdb08.db_exists' in __salt__:
        return 'influxdb08_user'
    return (False, 'influxdb08 module could not be loaded')

def present(name, passwd, database=None, user=None, password=None, host=None, port=None):
    if False:
        while True:
            i = 10
    '\n    Ensure that the cluster admin or database user is present.\n\n    name\n        The name of the user to manage\n\n    passwd\n        The password of the user\n\n    database\n        The database to create the user in\n\n    user\n        The user to connect as (must be able to create the user)\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if database and (not __salt__['influxdb08.db_exists'](database, user, password, host, port)):
        ret['result'] = False
        ret['comment'] = 'Database {} does not exist'.format(database)
        return ret
    if not __salt__['influxdb08.user_exists'](name, database, user, password, host, port):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'User {} is not present and needs to be created'.format(name)
            return ret
        if __salt__['influxdb08.user_create'](name, passwd, database, user, password, host, port):
            ret['comment'] = 'User {} has been created'.format(name)
            ret['changes'][name] = 'Present'
            return ret
        else:
            ret['comment'] = 'Failed to create user {}'.format(name)
            ret['result'] = False
            return ret
    ret['comment'] = 'User {} is already present'.format(name)
    return ret

def absent(name, database=None, user=None, password=None, host=None, port=None):
    if False:
        return 10
    '\n    Ensure that the named cluster admin or database user is absent.\n\n    name\n        The name of the user to remove\n\n    database\n        The database to remove the user from\n\n    user\n        The user to connect as (must be able to remove the user)\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if __salt__['influxdb08.user_exists'](name, database, user, password, host, port):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'User {} is present and needs to be removed'.format(name)
            return ret
        if __salt__['influxdb08.user_remove'](name, database, user, password, host, port):
            ret['comment'] = 'User {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['comment'] = 'Failed to remove user {}'.format(name)
            ret['result'] = False
            return ret
    ret['comment'] = 'User {} is not present, so it cannot be removed'.format(name)
    return ret