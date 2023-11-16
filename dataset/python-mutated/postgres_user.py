"""
Management of PostgreSQL users (roles)
======================================

The postgres_users module is used to create and manage Postgres users.

.. code-block:: yaml

    frank:
      postgres_user.present
"""
import datetime
import logging
from salt.modules import postgres
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if the postgres module is present\n    '
    if 'postgres.user_exists' not in __salt__:
        return (False, 'Unable to load postgres module.  Make sure `postgres.bins_dir` is set.')
    return True

def present(name, createdb=None, createroles=None, encrypted=None, superuser=None, replication=None, inherit=None, login=None, password=None, default_password=None, refresh_password=None, valid_until=None, groups=None, user=None, maintenance_db=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        print('Hello World!')
    '\n    Ensure that the named user is present with the specified privileges\n    Please note that the user/group notion in postgresql is just abstract, we\n    have roles, where users can be seen as roles with the LOGIN privilege\n    and groups the others.\n\n    name\n        The name of the system user to manage.\n\n    createdb\n        Is the user allowed to create databases?\n\n    createroles\n        Is the user allowed to create other users?\n\n    encrypted\n        How the password should be stored.\n\n        If encrypted is ``None``, ``True``, or ``md5``, it will use\n        PostgreSQL\'s MD5 algorithm.\n\n        If encrypted is ``False``, it will be stored in plaintext.\n\n        If encrypted is ``scram-sha-256``, it will use the algorithm described\n        in RFC 7677.\n\n        .. versionchanged:: 3003\n\n            Prior versions only supported ``True`` and ``False``\n\n    login\n        Should the group have login perm\n\n    inherit\n        Should the group inherit permissions\n\n    superuser\n        Should the new user be a "superuser"\n\n    replication\n        Should the new user be allowed to initiate streaming replication\n\n    password\n        The user\'s password.\n        It can be either a plain string or a pre-hashed password::\n\n            \'md5{MD5OF({password}{role}}\'\n            \'SCRAM-SHA-256${iterations}:{salt}${stored_key}:{server_key}\'\n\n        If encrypted is not ``False``, then the password will be converted\n        to the appropriate format above, if not already. As a consequence,\n        passwords that start with "md5" or "SCRAM-SHA-256" cannot be used.\n\n    default_password\n        The password used only when creating the user, unless password is set.\n\n        .. versionadded:: 2016.3.0\n\n    refresh_password\n        Password refresh flag\n\n        Boolean attribute to specify whether to password comparison check\n        should be performed.\n\n        If refresh_password is ``True``, the password will be automatically\n        updated without extra password change check.\n\n        This behaviour makes it possible to execute in environments without\n        superuser access available, e.g. Amazon RDS for PostgreSQL\n\n    valid_until\n        A date and time after which the role\'s password is no longer valid.\n\n    groups\n        A string of comma separated groups the user should be in\n\n    user\n        System user all operations should be performed on behalf of\n\n        .. versionadded:: 0.17.0\n\n    db_user\n        Postgres database username, if different from config or default.\n\n    db_password\n        Postgres user\'s password, if any password, for a specified db_user.\n\n    db_host\n        Postgres database host, if different from config or default.\n\n    db_port\n        Postgres database port, if different from config or default.\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'User {} is already present'.format(name)}
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    if encrypted is None:
        encrypted = postgres._DEFAULT_PASSWORDS_ENCRYPTION
    mode = 'create'
    user_attr = __salt__['postgres.role_get'](name, return_password=not refresh_password, **db_args)
    if user_attr is not None:
        mode = 'update'
    if mode == 'create' and password is None:
        password = default_password
    if password is not None:
        if mode == 'update' and (not refresh_password) and postgres._verify_password(name, password, user_attr['password'], encrypted):
            password = None
        else:
            password = postgres._maybe_encrypt_password(name, password, encrypted=encrypted)
    update = {}
    if mode == 'update':
        user_groups = user_attr.get('groups', [])
        if createdb is not None and user_attr['can create databases'] != createdb:
            update['createdb'] = createdb
        if inherit is not None and user_attr['inherits privileges'] != inherit:
            update['inherit'] = inherit
        if login is not None and user_attr['can login'] != login:
            update['login'] = login
        if createroles is not None and user_attr['can create roles'] != createroles:
            update['createroles'] = createroles
        if replication is not None and user_attr['replication'] != replication:
            update['replication'] = replication
        if superuser is not None and user_attr['superuser'] != superuser:
            update['superuser'] = superuser
        if password is not None:
            update['password'] = True
        if valid_until is not None:
            valid_until_dt = __salt__['postgres.psql_query']("SELECT '{}'::timestamp(0) as dt;".format(valid_until.replace("'", "''")), **db_args)[0]['dt']
            try:
                valid_until_dt = datetime.datetime.strptime(valid_until_dt, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                valid_until_dt = None
            if valid_until_dt != user_attr['expiry time']:
                update['valid_until'] = valid_until
        if groups is not None:
            lgroups = groups
            if isinstance(groups, str):
                lgroups = lgroups.split(',')
            if isinstance(lgroups, list):
                missing_groups = [a for a in lgroups if a not in user_groups]
                if missing_groups:
                    update['groups'] = missing_groups
    if mode == 'create' or (mode == 'update' and update):
        if __opts__['test']:
            if update:
                ret['changes'][name] = update
            ret['result'] = None
            ret['comment'] = 'User {} is set to be {}d'.format(name, mode)
            return ret
        cret = __salt__['postgres.user_{}'.format(mode)](username=name, createdb=createdb, createroles=createroles, encrypted=encrypted, superuser=superuser, login=login, inherit=inherit, replication=replication, rolepassword=password, valid_until=valid_until, groups=groups, **db_args)
    else:
        cret = None
    if cret:
        ret['comment'] = 'The user {} has been {}d'.format(name, mode)
        if update:
            ret['changes'][name] = update
        else:
            ret['changes'][name] = 'Present'
    elif cret is not None:
        ret['comment'] = 'Failed to {} user {}'.format(mode, name)
        ret['result'] = False
    else:
        ret['result'] = True
    return ret

def absent(name, user=None, maintenance_db=None, db_password=None, db_host=None, db_port=None, db_user=None):
    if False:
        print('Hello World!')
    '\n    Ensure that the named user is absent\n\n    name\n        The username of the user to remove\n\n    user\n        System user all operations should be performed on behalf of\n\n        .. versionadded:: 0.17.0\n\n    db_user\n        database username if different from config or default\n\n    db_password\n        user password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    if __salt__['postgres.user_exists'](name, **db_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'User {} is set to be removed'.format(name)
            return ret
        if __salt__['postgres.user_remove'](name, **db_args):
            ret['comment'] = 'User {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['result'] = False
            ret['comment'] = 'User {} failed to be removed'.format(name)
            return ret
    else:
        ret['comment'] = 'User {} is not present, so it cannot be removed'.format(name)
    return ret