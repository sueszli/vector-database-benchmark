"""
Management of Microsoft SQLServer Databases
===========================================

The mssql_role module is used to create
and manage SQL Server Roles

.. code-block:: yaml

    yolo:
      mssql_role.present
"""

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if the mssql module is present\n    '
    if 'mssql.version' in __salt__:
        return True
    return (False, 'mssql module could not be loaded')

def present(name, owner=None, grants=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that the named database is present with the specified options\n\n    name\n        The name of the database to manage\n    owner\n        Adds owner using AUTHORIZATION option\n    Grants\n        Can only be a list of strings\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if __salt__['mssql.role_exists'](name, **kwargs):
        ret['comment'] = 'Role {} is already present (Not going to try to set its grants)'.format(name)
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = 'Role {} is set to be added'.format(name)
        return ret
    role_created = __salt__['mssql.role_create'](name, owner=owner, grants=grants, **kwargs)
    if role_created is not True:
        ret['result'] = False
        ret['comment'] += 'Role {} failed to be created: {}'.format(name, role_created)
        return ret
    ret['comment'] += 'Role {} has been added'.format(name)
    ret['changes'][name] = 'Present'
    return ret

def absent(name, **kwargs):
    if False:
        print('Hello World!')
    '\n    Ensure that the named database is absent\n\n    name\n        The name of the database to remove\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if not __salt__['mssql.role_exists'](name):
        ret['comment'] = 'Role {} is not present'.format(name)
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = 'Role {} is set to be removed'.format(name)
        return ret
    if __salt__['mssql.role_remove'](name, **kwargs):
        ret['comment'] = 'Role {} has been removed'.format(name)
        ret['changes'][name] = 'Absent'
        return ret
    ret['result'] = False
    ret['comment'] = 'Role {} failed to be removed'.format(name)
    return ret