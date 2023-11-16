"""
Management of PostgreSQL groups (roles)
=======================================

The postgres_group module is used to create and manage Postgres groups.

.. code-block:: yaml

    frank:
      postgres_group.present
"""
import logging
from salt.modules import postgres
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if the postgres module is present\n    '
    if 'postgres.group_create' not in __salt__:
        return (False, 'Unable to load postgres module.  Make sure `postgres.bins_dir` is set.')
    return True

def present(name, createdb=None, createroles=None, encrypted=None, superuser=None, inherit=None, login=None, replication=None, password=None, refresh_password=None, groups=None, user=None, maintenance_db=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        while True:
            i = 10
    '\n    Ensure that the named group is present with the specified privileges\n    Please note that the user/group notion in postgresql is just abstract, we\n    have roles, where users can be seen as roles with the ``LOGIN`` privilege\n    and groups the others.\n\n    name\n        The name of the group to manage\n\n    createdb\n        Is the group allowed to create databases?\n\n    createroles\n        Is the group allowed to create other roles/users\n\n    encrypted\n        How the password should be stored.\n\n        If encrypted is ``None``, ``True``, or ``md5``, it will use\n        PostgreSQL\'s MD5 algorithm.\n\n        If encrypted is ``False``, it will be stored in plaintext.\n\n        If encrypted is ``scram-sha-256``, it will use the algorithm described\n        in RFC 7677.\n\n        .. versionchanged:: 3003\n\n            Prior versions only supported ``True`` and ``False``\n\n    login\n        Should the group have login perm\n\n    inherit\n        Should the group inherit permissions\n\n    superuser\n        Should the new group be a "superuser"\n\n    replication\n        Should the new group be allowed to initiate streaming replication\n\n    password\n        The group\'s password.\n        It can be either a plain string or a pre-hashed password::\n\n            \'md5{MD5OF({password}{role}}\'\n            \'SCRAM-SHA-256${iterations}:{salt}${stored_key}:{server_key}\'\n\n        If encrypted is not ``False``, then the password will be converted\n        to the appropriate format above, if not already. As a consequence,\n        passwords that start with "md5" or "SCRAM-SHA-256" cannot be used.\n\n    refresh_password\n        Password refresh flag\n\n        Boolean attribute to specify whether to password comparison check\n        should be performed.\n\n        If refresh_password is ``True``, the password will be automatically\n        updated without extra password change check.\n\n        This behaviour makes it possible to execute in environments without\n        superuser access available, e.g. Amazon RDS for PostgreSQL\n\n    groups\n        A string of comma separated groups the group should be in\n\n    user\n        System user all operations should be performed on behalf of\n\n        .. versionadded:: 0.17.0\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'Group {} is already present'.format(name)}
    if encrypted is None:
        encrypted = postgres._DEFAULT_PASSWORDS_ENCRYPTION
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    mode = 'create'
    group_attr = __salt__['postgres.role_get'](name, return_password=not refresh_password, **db_args)
    if group_attr is not None:
        mode = 'update'
    if password is not None:
        if mode == 'update' and (not refresh_password) and postgres._verify_password(name, password, group_attr['password'], encrypted):
            password = None
        else:
            password = postgres._maybe_encrypt_password(name, password, encrypted=encrypted)
    update = {}
    if mode == 'update':
        role_groups = group_attr.get('groups', [])
        if createdb is not None and group_attr['can create databases'] != createdb:
            update['createdb'] = createdb
        if inherit is not None and group_attr['inherits privileges'] != inherit:
            update['inherit'] = inherit
        if login is not None and group_attr['can login'] != login:
            update['login'] = login
        if createroles is not None and group_attr['can create roles'] != createroles:
            update['createroles'] = createroles
        if replication is not None and group_attr['replication'] != replication:
            update['replication'] = replication
        if superuser is not None and group_attr['superuser'] != superuser:
            update['superuser'] = superuser
        if password is not None:
            update['password'] = True
        if groups is not None:
            lgroups = groups
            if isinstance(groups, str):
                lgroups = lgroups.split(',')
            if isinstance(lgroups, list):
                missing_groups = [a for a in lgroups if a not in role_groups]
                if missing_groups:
                    update['groups'] = missing_groups
    if mode == 'create' or (mode == 'update' and update):
        if __opts__['test']:
            if update:
                ret['changes'][name] = update
            ret['result'] = None
            ret['comment'] = 'Group {} is set to be {}d'.format(name, mode)
            return ret
        cret = __salt__['postgres.group_{}'.format(mode)](groupname=name, createdb=createdb, createroles=createroles, encrypted=encrypted, login=login, inherit=inherit, superuser=superuser, replication=replication, rolepassword=password, groups=groups, **db_args)
    else:
        cret = None
    if cret:
        ret['comment'] = 'The group {} has been {}d'.format(name, mode)
        if update:
            ret['changes'][name] = update
        else:
            ret['changes'][name] = 'Present'
    elif cret is not None:
        ret['comment'] = 'Failed to {} group {}'.format(mode, name)
        ret['result'] = False
    else:
        ret['result'] = True
    return ret

def absent(name, user=None, maintenance_db=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        return 10
    '\n    Ensure that the named group is absent\n\n    name\n        The groupname of the group to remove\n\n    user\n        System user all operations should be performed on behalf of\n\n        .. versionadded:: 0.17.0\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    if __salt__['postgres.user_exists'](name, **db_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Group {} is set to be removed'.format(name)
            return ret
        if __salt__['postgres.group_remove'](name, **db_args):
            ret['comment'] = 'Group {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['result'] = False
            ret['comment'] = 'Group {} failed to be removed'.format(name)
            return ret
    else:
        ret['comment'] = 'Group {} is not present, so it cannot be removed'.format(name)
    return ret