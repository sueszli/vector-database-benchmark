"""
Management of PostgreSQL languages
==================================

The postgres_language module is used to create and manage Postgres languages.
Languages can be set as either absent or present

.. versionadded:: 2016.3.0

.. code-block:: yaml

    plpgsql:
      postgres_language.present:
        - maintenance_db: testdb

.. code-block:: yaml

    plpgsql:
      postgres_language.absent:
        - maintenance_db: testdb

"""

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if the postgres module is present\n    '
    if 'postgres.language_create' not in __salt__:
        return (False, 'Unable to load postgres module.  Make sure `postgres.bins_dir` is set.')
    return True

def present(name, maintenance_db, user=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that a named language is present in the specified\n    database.\n\n    name\n        The name of the language to install\n\n    maintenance_db\n        The name of the database in which the language is to be installed\n\n    user\n        System user all operations should be performed on behalf of\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'Language {} is already installed'.format(name)}
    dbargs = {'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    languages = __salt__['postgres.language_list'](maintenance_db, **dbargs)
    if name not in languages:
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Language {} is set to be installed'.format(name)
            return ret
        if __salt__['postgres.language_create'](name, maintenance_db, **dbargs):
            ret['comment'] = 'Language {} has been installed'.format(name)
            ret['changes'][name] = 'Present'
        else:
            ret['comment'] = 'Failed to install language {}'.format(name)
            ret['result'] = False
    return ret

def absent(name, maintenance_db, user=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        print('Hello World!')
    '\n    Ensure that a named language is absent in the specified\n    database.\n\n    name\n        The name of the language to remove\n\n    maintenance_db\n        The name of the database in which the language is to be installed\n\n    user\n        System user all operations should be performed on behalf of\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    dbargs = {'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    if __salt__['postgres.language_exists'](name, maintenance_db, **dbargs):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Language {} is set to be removed'.format(name)
            return ret
        if __salt__['postgres.language_remove'](name, **dbargs):
            ret['comment'] = 'Language {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['comment'] = 'Failed to remove language {}'.format(name)
            ret['result'] = False
    ret['comment'] = 'Language {} is not present so it cannot be removed'.format(name)
    return ret