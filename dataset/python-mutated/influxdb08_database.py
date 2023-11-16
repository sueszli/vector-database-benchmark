"""
Management of Influxdb 0.8 databases
====================================

(compatible with InfluxDB version 0.5-0.8)

.. versionadded:: 2014.7.0

"""

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if the influxdb08 module is available\n    '
    if 'influxdb08.db_exists' in __salt__:
        return 'influxdb08_database'
    return (False, 'influxdb08 module could not be loaded')

def present(name, user=None, password=None, host=None, port=None):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that the named database is present\n\n    name\n        The name of the database to create\n\n    user\n        The user to connect as (must be able to remove the database)\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if not __salt__['influxdb08.db_exists'](name, user, password, host, port):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Database {} is absent and needs to be created'.format(name)
            return ret
        if __salt__['influxdb08.db_create'](name, user, password, host, port):
            ret['comment'] = 'Database {} has been created'.format(name)
            ret['changes'][name] = 'Present'
            return ret
        else:
            ret['comment'] = 'Failed to create database {}'.format(name)
            ret['result'] = False
            return ret
    ret['comment'] = 'Database {} is already present, so cannot be created'.format(name)
    return ret

def absent(name, user=None, password=None, host=None, port=None):
    if False:
        return 10
    '\n    Ensure that the named database is absent\n\n    name\n        The name of the database to remove\n\n    user\n        The user to connect as (must be able to remove the database)\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if __salt__['influxdb08.db_exists'](name, user, password, host, port):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Database {} is present and needs to be removed'.format(name)
            return ret
        if __salt__['influxdb08.db_remove'](name, user, password, host, port):
            ret['comment'] = 'Database {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['comment'] = 'Failed to remove database {}'.format(name)
            ret['result'] = False
            return ret
    ret['comment'] = 'Database {} is not present, so it cannot be removed'.format(name)
    return ret