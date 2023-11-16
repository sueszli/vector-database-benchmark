"""
Set defaults on Mac OS

"""
import logging
import salt.utils.platform
log = logging.getLogger(__name__)
__virtualname__ = 'macdefaults'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only work on Mac OS\n    '
    if salt.utils.platform.is_darwin():
        return __virtualname__
    return False

def write(domain, key, value, type='string', user=None):
    if False:
        print('Hello World!')
    "\n    Write a default to the system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macdefaults.write com.apple.CrashReporter DialogType Server\n\n        salt '*' macdefaults.write NSGlobalDomain ApplePersistence True type=bool\n\n    domain\n        The name of the domain to write to\n\n    key\n        The key of the given domain to write to\n\n    value\n        The value to write to the given key\n\n    type\n        The type of value to be written, valid types are string, data, int[eger],\n        float, bool[ean], date, array, array-add, dict, dict-add\n\n    user\n        The user to write the defaults to\n\n\n    "
    if type == 'bool' or type == 'boolean':
        if value is True:
            value = 'TRUE'
        elif value is False:
            value = 'FALSE'
    cmd = 'defaults write "{}" "{}" -{} "{}"'.format(domain, key, type, value)
    return __salt__['cmd.run_all'](cmd, runas=user)

def read(domain, key, user=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Read a default from the system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macdefaults.read com.apple.CrashReporter DialogType\n\n        salt '*' macdefaults.read NSGlobalDomain ApplePersistence\n\n    domain\n        The name of the domain to read from\n\n    key\n        The key of the given domain to read from\n\n    user\n        The user to read the defaults as\n\n    "
    cmd = 'defaults read "{}" "{}"'.format(domain, key)
    return __salt__['cmd.run'](cmd, runas=user)

def delete(domain, key, user=None):
    if False:
        while True:
            i = 10
    "\n    Delete a default from the system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' macdefaults.delete com.apple.CrashReporter DialogType\n\n        salt '*' macdefaults.delete NSGlobalDomain ApplePersistence\n\n    domain\n        The name of the domain to delete from\n\n    key\n        The key of the given domain to delete\n\n    user\n        The user to delete the defaults with\n\n    "
    cmd = 'defaults delete "{}" "{}"'.format(domain, key)
    return __salt__['cmd.run_all'](cmd, runas=user, output_loglevel='debug')