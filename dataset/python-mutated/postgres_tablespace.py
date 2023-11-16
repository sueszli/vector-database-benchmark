"""
Management of PostgreSQL tablespace
===================================

A module used to create and manage PostgreSQL tablespaces.

.. code-block:: yaml

    ssd-tablespace:
      postgres_tablespace.present:
        - name: indexes
        - directory: /mnt/ssd-data

.. versionadded:: 2015.8.0

"""
import salt.utils.dictupdate as dictupdate

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if the postgres module is present and is new enough (has ts funcs)\n    '
    if 'postgres.tablespace_exists' not in __salt__:
        return (False, 'Unable to load postgres module.  Make sure `postgres.bins_dir` is set.')
    return True

def present(name, directory, options=None, owner=None, user=None, maintenance_db=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        while True:
            i = 10
    '\n    Ensure that the named tablespace is present with the specified properties.\n    For more information about all of these options run ``man 7\n    create_tablespace``.\n\n    name\n        The name of the tablespace to create/manage.\n\n    directory\n        The directory where the tablespace will be located, must already exist\n\n    options\n        A dictionary of options to specify for the tablespace.\n        Currently, the only tablespace options supported are ``seq_page_cost``\n        and ``random_page_cost``. Default values are shown in the example below:\n\n        .. code-block:: yaml\n\n            my_space:\n              postgres_tablespace.present:\n                - directory: /srv/my_tablespace\n                - options:\n                    seq_page_cost: 1.0\n                    random_page_cost: 4.0\n\n    owner\n        The database user that will be the owner of the tablespace.\n        Defaults to the user executing the command (i.e. the `user` option)\n\n    user\n        System user all operations should be performed on behalf of\n\n    maintenance_db\n        Database to act on\n\n    db_user\n        Database username if different from config or default\n\n    db_password\n        User password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'Tablespace {} is already present'.format(name)}
    dbargs = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    tblspaces = __salt__['postgres.tablespace_list'](**dbargs)
    if name not in tblspaces:
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Tablespace {} is set to be created'.format(name)
            return ret
        if __salt__['postgres.tablespace_create'](name, directory, options, owner, **dbargs):
            ret['comment'] = 'The tablespace {} has been created'.format(name)
            ret['changes'][name] = 'Present'
            return ret
    if tblspaces[name]['Location'] != directory and (not __opts__['test']):
        ret['comment'] = 'Tablespace {} is not at the right location. This is\n            unfixable without dropping and recreating the tablespace.'.format(name)
        ret['result'] = False
        return ret
    if owner and (not tblspaces[name]['Owner'] == owner):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Tablespace {} owner to be altered'.format(name)
        if __salt__['postgres.tablespace_alter'](name, new_owner=owner) and (not __opts__['test']):
            ret['comment'] = 'Tablespace {} owner changed'.format(name)
            ret['changes'][name] = {'owner': owner}
            ret['result'] = True
    if options:
        for (k, v) in options.items():
            if '{}={}'.format(k, v) not in tblspaces[name]['Opts']:
                if __opts__['test']:
                    ret['result'] = None
                    ret['comment'] = 'Tablespace {} options to be\n                        altered'.format(name)
                    break
                if __salt__['postgres.tablespace_alter'](name, set_option={k: v}):
                    ret['comment'] = 'Tablespace {} opts changed'.format(name)
                    dictupdate.update(ret['changes'], {name: {'options': {k: v}}})
                    ret['result'] = True
    return ret

def absent(name, user=None, maintenance_db=None, db_user=None, db_password=None, db_host=None, db_port=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that the named tablespace is absent.\n\n    name\n        The name of the tablespace to remove\n\n    user\n        System user all operations should be performed on behalf of\n\n    maintenance_db\n        Database to act on\n\n    db_user\n        Database username if different from config or default\n\n    db_password\n        User password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    if __salt__['postgres.tablespace_exists'](name, **db_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Tablespace {} is set to be removed'.format(name)
            return ret
        if __salt__['postgres.tablespace_remove'](name, **db_args):
            ret['comment'] = 'Tablespace {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
    ret['comment'] = 'Tablespace {} is not present, so it cannot be removed'.format(name)
    return ret