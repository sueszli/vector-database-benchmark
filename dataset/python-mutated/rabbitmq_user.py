"""
Manage RabbitMQ Users
=====================

Example:

.. code-block:: yaml

    rabbit_user:
      rabbitmq_user.present:
        - password: password
        - force: True
        - tags:
          - monitoring
          - user
        - perms:
          - '/':
            - '.*'
            - '.*'
            - '.*'
        - runas: rabbitmq
"""
import logging
import salt.utils.path
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if RabbitMQ is installed.\n    '
    if salt.utils.path.which('rabbitmqctl'):
        return True
    return (False, 'Command not found: rabbitmqctl')

def _check_perms_changes(name, newperms, runas=None, existing=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check whether Rabbitmq user's permissions need to be changed.\n    "
    if not newperms:
        return False
    if existing is None:
        try:
            existing = __salt__['rabbitmq.list_user_permissions'](name, runas=runas)
        except CommandExecutionError as err:
            log.error('Error: %s', err)
            return False
    empty_perms = {'configure': '', 'write': '', 'read': ''}
    perm_need_change = False
    for vhost_perms in newperms:
        for (vhost, perms) in vhost_perms.items():
            if vhost in existing:
                new_perms = {'configure': perms[0], 'write': perms[1], 'read': perms[2]}
                existing_vhost = existing[vhost]
                if new_perms != existing_vhost:
                    if existing_vhost == empty_perms and perms == empty_perms:
                        continue
                    perm_need_change = True
            else:
                perm_need_change = True
    return perm_need_change

def _get_current_tags(name, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Whether Rabbitmq user's tags need to be changed\n    "
    try:
        return list(__salt__['rabbitmq.list_users'](runas=runas)[name])
    except CommandExecutionError as err:
        log.error('Error: %s', err)
        return []

def present(name, password=None, force=False, tags=None, perms=(), runas=None):
    if False:
        return 10
    "\n    Ensure the RabbitMQ user exists.\n\n    name\n        User name\n    password\n        The user's password\n    force\n        If force is ``True``, the password will be automatically updated without extra password change check.\n    tags\n        Optional list of tags for the user\n    perms\n        A list of dicts with vhost keys and 3-tuple values\n    runas\n        Name of the user to run the command\n    "
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    try:
        user = __salt__['rabbitmq.user_exists'](name, runas=runas)
    except CommandExecutionError as err:
        ret['comment'] = 'Error: {}'.format(err)
        return ret
    passwd_reqs_update = False
    if user and password is not None:
        try:
            if not __salt__['rabbitmq.check_password'](name, password, runas=runas):
                passwd_reqs_update = True
                log.debug('RabbitMQ user %s password update required', name)
        except CommandExecutionError as err:
            ret['comment'] = 'Error: {}'.format(err)
            return ret
    if user and (not any((force, perms, tags, passwd_reqs_update))):
        log.debug("RabbitMQ user '%s' exists, password is up to date and force is not set.", name)
        ret['comment'] = "User '{}' is already present.".format(name)
        ret['result'] = True
        return ret
    if not user:
        ret['changes'].update({'user': {'old': '', 'new': name}})
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = "User '{}' is set to be created.".format(name)
            return ret
        log.debug("RabbitMQ user '%s' doesn't exist - Creating.", name)
        try:
            __salt__['rabbitmq.add_user'](name, password, runas=runas)
        except CommandExecutionError as err:
            ret['comment'] = 'Error: {}'.format(err)
            return ret
    else:
        log.debug("RabbitMQ user '%s' exists", name)
        if force or passwd_reqs_update:
            if password is not None:
                if not __opts__['test']:
                    try:
                        __salt__['rabbitmq.change_password'](name, password, runas=runas)
                    except CommandExecutionError as err:
                        ret['comment'] = 'Error: {}'.format(err)
                        return ret
                ret['changes'].update({'password': {'old': '', 'new': 'Set password.'}})
            else:
                if not __opts__['test']:
                    log.debug('Password for %s is not set - Clearing password.', name)
                    try:
                        __salt__['rabbitmq.clear_password'](name, runas=runas)
                    except CommandExecutionError as err:
                        ret['comment'] = 'Error: {}'.format(err)
                        return ret
                ret['changes'].update({'password': {'old': 'Removed password.', 'new': ''}})
    if tags is not None:
        current_tags = _get_current_tags(name, runas=runas)
        if isinstance(tags, str):
            tags = tags.split()
        if set(tags) ^ set(current_tags):
            if not __opts__['test']:
                try:
                    __salt__['rabbitmq.set_user_tags'](name, tags, runas=runas)
                except CommandExecutionError as err:
                    ret['comment'] = 'Error: {}'.format(err)
                    return ret
            ret['changes'].update({'tags': {'old': current_tags, 'new': tags}})
    try:
        existing_perms = __salt__['rabbitmq.list_user_permissions'](name, runas=runas)
    except CommandExecutionError as err:
        ret['comment'] = 'Error: {}'.format(err)
        return ret
    if _check_perms_changes(name, perms, runas=runas, existing=existing_perms):
        for vhost_perm in perms:
            for (vhost, perm) in vhost_perm.items():
                if not __opts__['test']:
                    try:
                        __salt__['rabbitmq.set_permissions'](vhost, name, perm[0], perm[1], perm[2], runas=runas)
                    except CommandExecutionError as err:
                        ret['comment'] = 'Error: {}'.format(err)
                        return ret
                new_perms = {vhost: {'configure': perm[0], 'write': perm[1], 'read': perm[2]}}
                if vhost in existing_perms:
                    if existing_perms[vhost] != new_perms[vhost]:
                        if ret['changes'].get('perms') is None:
                            ret['changes'].update({'perms': {'old': {}, 'new': {}}})
                        ret['changes']['perms']['old'].update(existing_perms[vhost])
                        ret['changes']['perms']['new'].update(new_perms)
                else:
                    ret['changes'].update({'perms': {'new': {}}})
                    ret['changes']['perms']['new'].update(new_perms)
    ret['result'] = True
    if ret['changes'] == {}:
        ret['comment'] = "'{}' is already in the desired state.".format(name)
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = "Configuration for '{}' will change.".format(name)
        return ret
    ret['comment'] = "'{}' was configured.".format(name)
    return ret

def absent(name, runas=None):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the named user is absent\n\n    name\n        The name of the user to remove\n    runas\n        User to run the command\n    '
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    try:
        user_exists = __salt__['rabbitmq.user_exists'](name, runas=runas)
    except CommandExecutionError as err:
        ret['comment'] = 'Error: {}'.format(err)
        return ret
    if user_exists:
        if not __opts__['test']:
            try:
                __salt__['rabbitmq.delete_user'](name, runas=runas)
            except CommandExecutionError as err:
                ret['comment'] = 'Error: {}'.format(err)
                return ret
        ret['changes'].update({'name': {'old': name, 'new': ''}})
    else:
        ret['result'] = True
        ret['comment'] = "The user '{}' is not present.".format(name)
        return ret
    if __opts__['test'] and ret['changes']:
        ret['result'] = None
        ret['comment'] = "The user '{}' will be removed.".format(name)
        return ret
    ret['result'] = True
    ret['comment'] = "The user '{}' was removed.".format(name)
    return ret