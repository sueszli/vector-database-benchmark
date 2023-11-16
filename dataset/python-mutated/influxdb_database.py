"""
Management of Influxdb databases
================================

(compatible with InfluxDB version 0.9+)
"""

def __virtual__():
    if False:
        return 10
    '\n    Only load if the influxdb module is available\n    '
    if 'influxdb.db_exists' in __salt__:
        return 'influxdb_database'
    return (False, 'influxdb module could not be loaded')

def present(name, **client_args):
    if False:
        print('Hello World!')
    '\n    Ensure that given database is present.\n\n    name\n        Name of the database to create.\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'Database {} is already present'.format(name)}
    if not __salt__['influxdb.db_exists'](name, **client_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Database {} is absent and will be created'.format(name)
            return ret
        if __salt__['influxdb.create_db'](name, **client_args):
            ret['comment'] = 'Database {} has been created'.format(name)
            ret['changes'][name] = 'Present'
            return ret
        else:
            ret['comment'] = 'Failed to create database {}'.format(name)
            ret['result'] = False
            return ret
    return ret

def absent(name, **client_args):
    if False:
        print('Hello World!')
    '\n    Ensure that given database is absent.\n\n    name\n        Name of the database to remove.\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'Database {} is not present'.format(name)}
    if __salt__['influxdb.db_exists'](name, **client_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Database {} is present and needs to be removed'.format(name)
            return ret
        if __salt__['influxdb.drop_db'](name, **client_args):
            ret['comment'] = 'Database {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['comment'] = 'Failed to remove database {}'.format(name)
            ret['result'] = False
            return ret
    return ret