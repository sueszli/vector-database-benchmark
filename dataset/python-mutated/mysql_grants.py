"""
Management of MySQL grants (user permissions)
=============================================

:depends:   - MySQLdb Python module
:configuration: See :py:mod:`salt.modules.mysql` for setup instructions.

The mysql_grants module is used to grant and revoke MySQL permissions.

The ``name`` you pass in purely symbolic and does not have anything to do
with the grant itself.

The ``database`` parameter needs to specify a 'priv_level' in the same
specification as defined in the MySQL documentation:

* \\*
* \\*.\\*
* db_name.\\*
* db_name.tbl_name
* etc...

This state is not able to set password for the permission from the
specified host. See :py:mod:`salt.states.mysql_user` for further
instructions.

.. code-block:: yaml

   frank_exampledb:
      mysql_grants.present:
       - grant: select,insert,update
       - database: exampledb.*
       - user: frank
       - host: localhost

   frank_otherdb:
     mysql_grants.present:
       - grant: all privileges
       - database: otherdb.*
       - user: frank

   restricted_singletable:
     mysql_grants.present:
       - grant: select
       - database: somedb.sometable
       - user: joe
"""
import sys

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if the mysql module is available\n    '
    if 'mysql.grant_exists' in __salt__:
        return True
    return (False, 'mysql module could not be loaded')

def _get_mysql_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Look in module context for a MySQL error. Eventually we should make a less\n    ugly way of doing this.\n    '
    return sys.modules[__salt__['test.ping'].__module__].__context__.pop('mysql.error', None)

def present(name, grant=None, database=None, user=None, host='localhost', grant_option=False, escape=True, revoke_first=False, ssl_option=False, **connection_args):
    if False:
        print('Hello World!')
    "\n    Ensure that the grant is present with the specified properties\n\n    name\n        The name (key) of the grant to add\n\n    grant\n        The grant priv_type (i.e. select,insert,update OR all privileges)\n\n    database\n        The database priv_level (i.e. db.tbl OR db.*)\n\n    user\n        The user to apply the grant to\n\n    host\n        The network/host that the grant should apply to\n\n    grant_option\n        Adds the WITH GRANT OPTION to the defined grant. Default is ``False``\n\n    escape\n        Defines if the database value gets escaped or not. Default is ``True``\n\n    revoke_first\n        By default, MySQL will not do anything if you issue a command to grant\n        privileges that are more restrictive than what's already in place. This\n        effectively means that you cannot downgrade permissions without first\n        revoking permissions applied to a db.table/user pair first.\n\n        To have Salt forcibly revoke perms before applying a new grant, enable\n        the 'revoke_first options.\n\n        WARNING: This will *remove* permissions for a database before attempting\n        to apply new permissions. There is no guarantee that new permissions\n        will be applied correctly which can leave your database security in an\n        unknown and potentially dangerous state.\n        Use with caution!\n\n        Default is ``False``\n\n    ssl_option\n        Adds the specified ssl options for the connecting user as requirements for\n        this grant. Value is a list of single-element dicts corresponding to the\n        list of ssl options to use.\n\n        Possible key/value pairings for the dicts in the value:\n\n        .. code-block:: text\n\n            - SSL: True\n            - X509: True\n            - SUBJECT: <subject>\n            - ISSUER: <issuer>\n            - CIPHER: <cipher>\n\n        The non-boolean ssl options take a string as their values, which should\n        be an appropriate value as specified by the MySQL documentation for these\n        options.\n\n        Default is ``False`` (no ssl options will be used)\n    "
    comment = 'Grant {0} on {1} to {2}@{3} is already present'
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': comment.format(grant, database, user, host)}
    if __salt__['mysql.grant_exists'](grant, database, user, host, grant_option, escape, **connection_args):
        return ret
    else:
        err = _get_mysql_error()
        if err is not None:
            ret['comment'] = err
            ret['result'] = False
            return ret
    if revoke_first and (not __opts__['test']):
        user_grants = __salt__['mysql.user_grants'](user, host, **connection_args)
        if not user_grants:
            user_grants = []
        for user_grant in user_grants:
            token_grants = __salt__['mysql.tokenize_grant'](user_grant)
            db_part = database.rpartition('.')
            my_db = db_part[0]
            my_table = db_part[2]
            my_db = __salt__['mysql.quote_identifier'](my_db, my_table == '*')
            my_table = __salt__['mysql.quote_identifier'](my_table)
            if token_grants['database'] == my_db:
                grant_to_revoke = ','.join(token_grants['grant']).rstrip(',')
                __salt__['mysql.grant_revoke'](grant=grant_to_revoke, database=database, user=user, host=host, grant_option=grant_option, escape=escape, **connection_args)
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = 'MySQL grant {} is set to be created'.format(name)
        return ret
    if __salt__['mysql.grant_add'](grant, database, user, host, grant_option, escape, ssl_option, **connection_args):
        ret['comment'] = 'Grant {0} on {1} to {2}@{3} has been added'
        ret['comment'] = ret['comment'].format(grant, database, user, host)
        ret['changes'][name] = 'Present'
    else:
        ret['comment'] = 'Failed to execute: "GRANT {0} ON {1} TO {2}@{3}"'
        ret['comment'] = ret['comment'].format(grant, database, user, host)
        err = _get_mysql_error()
        if err is not None:
            ret['comment'] += ' ({})'.format(err)
        ret['result'] = False
    return ret

def absent(name, grant=None, database=None, user=None, host='localhost', grant_option=False, escape=True, **connection_args):
    if False:
        return 10
    '\n    Ensure that the grant is absent\n\n    name\n        The name (key) of the grant to add\n\n    grant\n        The grant priv_type (i.e. select,insert,update OR all privileges)\n\n    database\n        The database priv_level (i.e. db.tbl OR db.*)\n\n    user\n        The user to apply the grant to\n\n    host\n        The network/host that the grant should apply to\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if __salt__['mysql.grant_exists'](grant, database, user, host, grant_option, escape, **connection_args):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'MySQL grant {} is set to be revoked'.format(name)
            return ret
        if __salt__['mysql.grant_revoke'](grant, database, user, host, grant_option, **connection_args):
            ret['comment'] = 'Grant {} on {} for {}@{} has been revoked'.format(grant, database, user, host)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            err = _get_mysql_error()
            if err is not None:
                ret['comment'] = 'Unable to revoke grant {} on {} for {}@{} ({})'.format(grant, database, user, host, err)
                ret['result'] = False
                return ret
    else:
        err = _get_mysql_error()
        if err is not None:
            ret['comment'] = 'Unable to determine if grant {} on {} for {}@{} exists ({})'.format(grant, database, user, host, err)
            ret['result'] = False
            return ret
    ret['comment'] = 'Grant {} on {} to {}@{} is not present, so it cannot be revoked'.format(grant, database, user, host)
    return ret