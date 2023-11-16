"""
Manage LXD profiles.

.. versionadded:: 2019.2.0

.. note:

    - `pylxd`_ version 2 is required to let this work,
      currently only available via pip.

        To install on Ubuntu:

        $ apt-get install libssl-dev python-pip
        $ pip install -U pylxd

    - you need lxd installed on the minion
      for the init() and version() methods.

    - for the config_get() and config_get() methods
      you need to have lxd-client installed.

.. _pylxd: https://github.com/lxc/pylxd/blob/master/doc/source/installation.rst

:maintainer: René Jochum <rene@jochums.at>
:maturity: new
:depends: python-pylxd
:platform: Linux
"""
from salt.exceptions import CommandExecutionError, SaltInvocationError
__docformat__ = 'restructuredtext en'
__virtualname__ = 'lxd_profile'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if the lxd module is available in __salt__\n    '
    if 'lxd.version' in __salt__:
        return __virtualname__
    return (False, 'lxd module could not be loaded')

def present(name, description=None, config=None, devices=None, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    "\n    Creates or updates LXD profiles\n\n    name :\n        The name of the profile to create/update\n\n    description :\n        A description string\n\n    config :\n        A config dict or None (None = unset).\n\n        Can also be a list:\n            [{'key': 'boot.autostart', 'value': 1},\n             {'key': 'security.privileged', 'value': '1'}]\n\n    devices :\n        A device dict or None (None = unset).\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Zertifikate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    See the `lxd-docs`_ for the details about the config and devices dicts.\n    See the `requests-docs` for the SSL stuff.\n\n    .. _lxd-docs: https://github.com/lxc/lxd/blob/master/doc/rest-api.md#post-10\n    .. _requests-docs: http://docs.python-requests.org/en/master/user/advanced/#ssl-cert-verification  # noqa\n    "
    ret = {'name': name, 'description': description, 'config': config, 'devices': devices, 'remote_addr': remote_addr, 'cert': cert, 'key': key, 'verify_cert': verify_cert, 'changes': {}}
    profile = None
    try:
        profile = __salt__['lxd.profile_get'](name, remote_addr, cert, key, verify_cert, _raw=True)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        pass
    if description is None:
        description = ''
    if profile is None:
        if __opts__['test']:
            msg = 'Would create the profile "{}"'.format(name)
            ret['changes'] = {'created': msg}
            return _unchanged(ret, msg)
        try:
            __salt__['lxd.profile_create'](name, config, devices, description, remote_addr, cert, key, verify_cert)
        except CommandExecutionError as e:
            return _error(ret, str(e))
        msg = 'Profile "{}" has been created'.format(name)
        ret['changes'] = {'created': msg}
        return _success(ret, msg)
    (config, devices) = __salt__['lxd.normalize_input_values'](config, devices)
    if str(profile.description) != str(description):
        ret['changes']['description'] = 'Description changed, from "{}" to "{}".'.format(profile.description, description)
        profile.description = description
    changes = __salt__['lxd.sync_config_devices'](profile, config, devices, __opts__['test'])
    ret['changes'].update(changes)
    if not ret['changes']:
        return _success(ret, 'No changes')
    if __opts__['test']:
        return _unchanged(ret, 'Profile "{}" would get changed.'.format(name))
    try:
        __salt__['lxd.pylxd_save_object'](profile)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    return _success(ret, '{} changes'.format(len(ret['changes'].keys())))

def absent(name, remote_addr=None, cert=None, key=None, verify_cert=True):
    if False:
        while True:
            i = 10
    '\n    Ensure a LXD profile is not present, removing it if present.\n\n    name :\n        The name of the profile to remove.\n\n    remote_addr :\n        An URL to a remote Server, you also have to give cert and key if you\n        provide remote_addr!\n\n        Examples:\n            https://myserver.lan:8443\n            /var/lib/mysocket.sock\n\n    cert :\n        PEM Formatted SSL Zertifikate.\n\n        Examples:\n            ~/.config/lxc/client.crt\n\n    key :\n        PEM Formatted SSL Key.\n\n        Examples:\n            ~/.config/lxc/client.key\n\n    verify_cert : True\n        Wherever to verify the cert, this is by default True\n        but in the most cases you want to set it off as LXD\n        normally uses self-signed certificates.\n\n    See the `requests-docs` for the SSL stuff.\n\n    .. _requests-docs: http://docs.python-requests.org/en/master/user/advanced/#ssl-cert-verification  # noqa\n    '
    ret = {'name': name, 'remote_addr': remote_addr, 'cert': cert, 'key': key, 'verify_cert': verify_cert, 'changes': {}}
    if __opts__['test']:
        try:
            __salt__['lxd.profile_get'](name, remote_addr, cert, key, verify_cert)
        except CommandExecutionError as e:
            return _error(ret, str(e))
        except SaltInvocationError as e:
            return _success(ret, 'Profile "{}" not found.'.format(name))
        ret['changes'] = {'removed': 'Profile "{}" would get deleted.'.format(name)}
        return _success(ret, ret['changes']['removed'])
    try:
        __salt__['lxd.profile_delete'](name, remote_addr, cert, key, verify_cert)
    except CommandExecutionError as e:
        return _error(ret, str(e))
    except SaltInvocationError as e:
        return _success(ret, 'Profile "{}" not found.'.format(name))
    ret['changes'] = {'removed': 'Profile "{}" has been deleted.'.format(name)}
    return _success(ret, ret['changes']['removed'])

def _success(ret, success_msg):
    if False:
        i = 10
        return i + 15
    ret['result'] = True
    ret['comment'] = success_msg
    if 'changes' not in ret:
        ret['changes'] = {}
    return ret

def _unchanged(ret, msg):
    if False:
        i = 10
        return i + 15
    ret['result'] = None
    ret['comment'] = msg
    if 'changes' not in ret:
        ret['changes'] = {}
    return ret

def _error(ret, err_msg):
    if False:
        for i in range(10):
            print('nop')
    ret['result'] = False
    ret['comment'] = err_msg
    if 'changes' not in ret:
        ret['changes'] = {}
    return ret