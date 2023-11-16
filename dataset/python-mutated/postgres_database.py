"""
Management of PostgreSQL databases
==================================

The postgres_database module is used to create and manage Postgres databases.
Databases can be set as either absent or present

.. code-block:: yaml

    frank:
      postgres_database.present
"""

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if the postgres module is present\n    '
    if 'postgres.user_exists' not in __salt__:
        return (False, 'Unable to load postgres module.  Make sure `postgres.bins_dir` is set.')
    return True

def present(name, tablespace=None, encoding=None, lc_collate=None, lc_ctype=None, owner=None, owner_recurse=False, template=None, user=None, maintenance_db=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that the named database is present with the specified properties.\n    For more information about all of these options see man createdb(1)\n\n    name\n        The name of the database to manage\n\n    tablespace\n        Default tablespace for the database\n\n    encoding\n        The character encoding scheme to be used in this database\n\n    lc_collate\n        The LC_COLLATE setting to be used in this database\n\n    lc_ctype\n        The LC_CTYPE setting to be used in this database\n\n    owner\n        The username of the database owner\n\n    owner_recurse\n        Recurse owner change to all relations in the database\n\n    template\n        The template database from which to build this database\n\n    user\n        System user all operations should be performed on behalf of\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n\n        .. versionadded:: 0.17.0\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'Database {} is already present'.format(name)}
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    dbs = __salt__['postgres.db_list'](**db_args)
    db_params = dbs.get(name, {})
    if name in dbs and all((db_params.get('Tablespace') == tablespace if tablespace else True, db_params.get('Encoding').lower() == encoding.lower() if encoding else True, db_params.get('Collate') == lc_collate if lc_collate else True, db_params.get('Ctype') == lc_ctype if lc_ctype else True, db_params.get('Owner') == owner if owner else True)):
        return ret
    elif name in dbs and any((db_params.get('Encoding').lower() != encoding.lower() if encoding else False, db_params.get('Collate') != lc_collate if lc_collate else False, db_params.get('Ctype') != lc_ctype if lc_ctype else False)):
        ret['comment'] = "Database {} has wrong parameters which couldn't be changed on fly.".format(name)
        ret['result'] = False
        return ret
    if __opts__['test']:
        ret['result'] = None
        if name not in dbs:
            ret['comment'] = 'Database {} is set to be created'.format(name)
        else:
            ret['comment'] = 'Database {} exists, but parameters need to be changed'.format(name)
        return ret
    if name not in dbs and __salt__['postgres.db_create'](name, tablespace=tablespace, encoding=encoding, lc_collate=lc_collate, lc_ctype=lc_ctype, owner=owner, template=template, **db_args):
        ret['comment'] = 'The database {} has been created'.format(name)
        ret['changes'][name] = 'Present'
    elif name in dbs and __salt__['postgres.db_alter'](name, tablespace=tablespace, owner=owner, owner_recurse=owner_recurse, **db_args):
        ret['comment'] = 'Parameters for database {} have been changed'.format(name)
        ret['changes'][name] = 'Parameters changed'
    elif name in dbs:
        ret['comment'] = 'Failed to change parameters for database {}'.format(name)
        ret['result'] = False
    else:
        ret['comment'] = 'Failed to create database {}'.format(name)
        ret['result'] = False
    return ret

def absent(name, user=None, maintenance_db=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        print('Hello World!')
    '\n    Ensure that the named database is absent\n\n    name\n        The name of the database to remove\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n\n    user\n        System user all operations should be performed on behalf of\n\n        .. versionadded:: 0.17.0\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    if __salt__['postgres.db_exists'](name, **db_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Database {} is set to be removed'.format(name)
            return ret
        if __salt__['postgres.db_remove'](name, **db_args):
            ret['comment'] = 'Database {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
    ret['comment'] = 'Database {} is not present, so it cannot be removed'.format(name)
    return ret