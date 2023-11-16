"""
Management of Influxdb continuous queries
=========================================

.. versionadded:: 2017.7.0

(compatible with InfluxDB version 0.9+)
"""

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if the influxdb module is available\n    '
    if 'influxdb.db_exists' in __salt__:
        return 'influxdb_continuous_query'
    return (False, 'influxdb module could not be loaded')

def present(name, database, query, resample_time=None, coverage_period=None, **client_args):
    if False:
        return 10
    '\n    Ensure that given continuous query is present.\n\n    name\n        Name of the continuous query to create.\n\n    database\n        Database to create continuous query on.\n\n    query\n        The query content\n\n    resample_time : None\n        Duration between continuous query resampling.\n\n    coverage_period : None\n        Duration specifying time period per sample.\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'continuous query {} is already present'.format(name)}
    if not __salt__['influxdb.continuous_query_exists'](name=name, database=database, **client_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = ' {} is absent and will be created'.format(name)
            return ret
        if __salt__['influxdb.create_continuous_query'](database, name, query, resample_time, coverage_period, **client_args):
            ret['comment'] = 'continuous query {} has been created'.format(name)
            ret['changes'][name] = 'Present'
            return ret
        else:
            ret['comment'] = 'Failed to create continuous query {}'.format(name)
            ret['result'] = False
            return ret
    return ret

def absent(name, database, **client_args):
    if False:
        while True:
            i = 10
    '\n    Ensure that given continuous query is absent.\n\n    name\n        Name of the continuous query to remove.\n\n    database\n        Name of the database that the continuous query was defined on.\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'continuous query {} is not present'.format(name)}
    if __salt__['influxdb.continuous_query_exists'](database, name, **client_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'continuous query {} is present and needs to be removed'.format(name)
            return ret
        if __salt__['influxdb.drop_continuous_query'](database, name, **client_args):
            ret['comment'] = 'continuous query {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['comment'] = 'Failed to remove continuous query {}'.format(name)
            ret['result'] = False
            return ret
    return ret