"""
Management of PostgreSQL extensions
===================================

A module used to install and manage PostgreSQL extensions.

.. code-block:: yaml

    adminpack:
      postgres_extension.present


.. versionadded:: 2014.7.0
"""
import logging
from salt.modules import postgres
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if the postgres module is present\n    '
    if 'postgres.create_extension' not in __salt__:
        return (False, 'Unable to load postgres module.  Make sure `postgres.bins_dir` is set.')
    return True

def present(name, if_not_exists=None, schema=None, ext_version=None, from_version=None, user=None, maintenance_db=None, db_user=None, db_password=None, db_host=None, db_port=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensure that the named extension is present.\n\n    .. note::\n\n        Before you can use the state to load an extension into a database, the\n        extension's supporting files must be already installed.\n\n    For more information about all of these options see ``CREATE EXTENSION`` SQL\n    command reference in the PostgreSQL documentation.\n\n    name\n        The name of the extension to be installed\n\n    if_not_exists\n        Add an ``IF NOT EXISTS`` parameter to the DDL statement\n\n    schema\n        Schema to install the extension into\n\n    ext_version\n        Version to install\n\n    from_version\n        Old extension version if already installed\n\n    user\n        System user all operations should be performed on behalf of\n\n    maintenance_db\n        Database to act on\n\n    db_user\n        Database username if different from config or default\n\n    db_password\n        User password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    "
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': 'Extension {} is already present'.format(name)}
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'user': db_user, 'password': db_password, 'host': db_host, 'port': db_port}
    mode = 'create'
    mtdata = __salt__['postgres.create_metadata'](name, schema=schema, ext_version=ext_version, **db_args)
    toinstall = postgres._EXTENSION_NOT_INSTALLED in mtdata
    if toinstall:
        mode = 'install'
    toupgrade = False
    if postgres._EXTENSION_INSTALLED in mtdata:
        for flag in [postgres._EXTENSION_TO_MOVE, postgres._EXTENSION_TO_UPGRADE]:
            if flag in mtdata:
                toupgrade = True
                mode = 'upgrade'
    cret = None
    if toinstall or toupgrade:
        if __opts__['test']:
            ret['result'] = None
            if mode:
                ret['comment'] = 'Extension {} is set to be {}ed'.format(name, mode).replace('eed', 'ed')
            return ret
        cret = __salt__['postgres.create_extension'](name=name, if_not_exists=if_not_exists, schema=schema, ext_version=ext_version, from_version=from_version, **db_args)
    if cret:
        if mode.endswith('e'):
            suffix = 'd'
        else:
            suffix = 'ed'
        ret['comment'] = 'The extension {} has been {}{}'.format(name, mode, suffix)
        ret['changes'][name] = '{}{}'.format(mode.capitalize(), suffix)
    elif cret is not None:
        ret['comment'] = 'Failed to {1} extension {0}'.format(name, mode)
        ret['result'] = False
    return ret

def absent(name, if_exists=None, restrict=None, cascade=None, user=None, maintenance_db=None, db_user=None, db_password=None, db_host=None, db_port=None):
    if False:
        while True:
            i = 10
    '\n    Ensure that the named extension is absent.\n\n    name\n        Extension name of the extension to remove\n\n    if_exists\n        Add if exist slug\n\n    restrict\n        Add restrict slug\n\n    cascade\n        Drop on cascade\n\n    user\n        System user all operations should be performed on behalf of\n\n    maintenance_db\n        Database to act on\n\n    db_user\n        Database username if different from config or default\n\n    db_password\n        User password if any password for a specified user\n\n    db_host\n        Database host if different from config or default\n\n    db_port\n        Database port if different from config or default\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    db_args = {'maintenance_db': maintenance_db, 'runas': user, 'host': db_host, 'user': db_user, 'port': db_port, 'password': db_password}
    exists = __salt__['postgres.is_installed_extension'](name, **db_args)
    if exists:
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Extension {} is set to be removed'.format(name)
            return ret
        if __salt__['postgres.drop_extension'](name, if_exists=if_exists, restrict=restrict, cascade=cascade, **db_args):
            ret['comment'] = 'Extension {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['result'] = False
            ret['comment'] = 'Extension {} failed to be removed'.format(name)
            return ret
    else:
        ret['comment'] = 'Extension {} is not present, so it cannot be removed'.format(name)
    return ret