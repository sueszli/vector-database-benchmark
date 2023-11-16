"""
Management of Microsoft SQLServer Users
=======================================

The mssql_user module is used to create
and manage SQL Server Users

.. code-block:: yaml

    frank:
      mssql_user.present:
        - database: yolo
"""
import collections

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if the mssql module is present\n    '
    if 'mssql.version' in __salt__:
        return True
    return (False, 'mssql module could not be loaded')

def _normalize_options(options):
    if False:
        for i in range(10):
            print('nop')
    if type(options) in [dict, collections.OrderedDict]:
        return ['{}={}'.format(k, v) for (k, v) in options.items()]
    if type(options) is list and (not options or type(options[0]) is str):
        return options
    if type(options) is not list or type(options[0]) not in [dict, collections.OrderedDict]:
        return []
    return [o for d in options for o in _normalize_options(d)]

def present(name, login=None, domain=None, database=None, roles=None, options=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Checks existence of the named user.\n    If not present, creates the user with the specified roles and options.\n\n    name\n        The name of the user to manage\n    login\n        If not specified, will be created WITHOUT LOGIN\n    domain\n        Creates a Windows authentication user.\n        Needs to be NetBIOS domain or hostname\n    database\n        The database of the user (not the login)\n    roles\n        Add this user to all the roles in the list\n    options\n        Can be a list of strings, a dictionary, or a list of dictionaries\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if domain and (not login):
        ret['result'] = False
        ret['comment'] = 'domain cannot be set without login'
        return ret
    if __salt__['mssql.user_exists'](name, domain=domain, database=database, **kwargs):
        ret['comment'] = 'User {} is already present (Not going to try to set its roles or options)'.format(name)
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = 'User {} is set to be added'.format(name)
        return ret
    user_created = __salt__['mssql.user_create'](name, login=login, domain=domain, database=database, roles=roles, options=_normalize_options(options), **kwargs)
    if user_created is not True:
        ret['result'] = False
        ret['comment'] += 'User {} failed to be added: {}'.format(name, user_created)
        return ret
    ret['comment'] += 'User {} has been added'.format(name)
    ret['changes'][name] = 'Present'
    return ret

def absent(name, **kwargs):
    if False:
        return 10
    '\n    Ensure that the named user is absent\n\n    name\n        The username of the user to remove\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if not __salt__['mssql.user_exists'](name):
        ret['comment'] = 'User {} is not present'.format(name)
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = 'User {} is set to be removed'.format(name)
        return ret
    if __salt__['mssql.user_remove'](name, **kwargs):
        ret['comment'] = 'User {} has been removed'.format(name)
        ret['changes'][name] = 'Absent'
        return ret
    ret['result'] = False
    ret['comment'] = 'User {} failed to be removed'.format(name)
    return ret