"""
Management of PostgreSQL schemas
================================

The postgres_schemas module is used to create and manage Postgres schemas.

.. code-block:: yaml

    public:
      postgres_schema.present 'dbname' 'name'
"""
import logging
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if the postgres module is present\n    '
    if 'postgres.schema_exists' not in __salt__:
        return (False, 'Unable to load postgres module.  Make sure `postgres.bins_dir` is set.')
    return True

def present(dbname, name, owner=None, user=None, db_user=None, db_password=None, db_host=None, db_port=None):
    if False:
        while True:
            i = 10
    "\n    Ensure that the named schema is present in the database.\n\n    dbname\n        The database's name will work on\n\n    name\n        The name of the schema to manage\n\n    user\n        system user all operations should be performed on behalf of\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    "
    ret = {'dbname': dbname, 'name': name, 'changes': {}, 'result': True, 'comment': 'Schema {} is already present in database {}'.format(name, dbname)}
    db_args = {'db_user': db_user, 'db_password': db_password, 'db_host': db_host, 'db_port': db_port, 'user': user}
    schema_attr = __salt__['postgres.schema_get'](dbname, name, **db_args)
    cret = None
    if schema_attr is None:
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Schema {} is set to be created in database {}.'.format(name, dbname)
            return ret
        cret = __salt__['postgres.schema_create'](dbname, name, owner=owner, **db_args)
    else:
        msg = 'Schema {0} already exists in database {1}'
        cret = None
    if cret:
        msg = 'Schema {0} has been created in database {1}'
        ret['result'] = True
        ret['changes'][name] = 'Present'
    elif cret is not None:
        msg = 'Failed to create schema {0} in database {1}'
        ret['result'] = False
    else:
        msg = 'Schema {0} already exists in database {1}'
        ret['result'] = True
    ret['comment'] = msg.format(name, dbname)
    return ret

def absent(dbname, name, user=None, db_user=None, db_password=None, db_host=None, db_port=None):
    if False:
        i = 10
        return i + 15
    "\n    Ensure that the named schema is absent.\n\n    dbname\n        The database's name will work on\n\n    name\n        The name of the schema to remove\n\n    user\n        system user all operations should be performed on behalf of\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    "
    ret = {'name': name, 'dbname': dbname, 'changes': {}, 'result': True, 'comment': ''}
    db_args = {'db_user': db_user, 'db_password': db_password, 'db_host': db_host, 'db_port': db_port, 'user': user}
    if __salt__['postgres.schema_exists'](dbname, name, **db_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Schema {} is set to be removed from database {}'.format(name, dbname)
            return ret
        elif __salt__['postgres.schema_remove'](dbname, name, **db_args):
            ret['comment'] = 'Schema {} has been removed from database {}'.format(name, dbname)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['result'] = False
            ret['comment'] = 'Schema {} failed to be removed'.format(name)
            return ret
    else:
        ret['comment'] = 'Schema {} is not present in database {}, so it cannot be removed'.format(name, dbname)
    return ret