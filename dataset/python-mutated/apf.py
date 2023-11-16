"""
Support for Advanced Policy Firewall (APF)
==========================================
:maintainer: Mostafa Hussein <mostafa.hussein91@gmail.com>
:maturity: new
:depends: python-iptables
:platform: Linux
"""
import salt.utils.path
from salt.exceptions import CommandExecutionError
try:
    import iptc
    IPTC_IMPORTED = True
except ImportError:
    IPTC_IMPORTED = False

def __virtual__():
    if False:
        return 10
    '\n    Only load if apf exists on the system\n    '
    if salt.utils.path.which('apf') is None:
        return (False, 'The apf execution module cannot be loaded: apf unavailable.')
    elif not IPTC_IMPORTED:
        return (False, 'The apf execution module cannot be loaded: python-iptables is missing.')
    else:
        return True

def __apf_cmd(cmd):
    if False:
        return 10
    '\n    Return the apf location\n    '
    apf_cmd = '{} {}'.format(salt.utils.path.which('apf'), cmd)
    out = __salt__['cmd.run_all'](apf_cmd)
    if out['retcode'] != 0:
        if not out['stderr']:
            msg = out['stdout']
        else:
            msg = out['stderr']
        raise CommandExecutionError('apf failed: {}'.format(msg))
    return out['stdout']

def _status_apf():
    if False:
        return 10
    '\n    Return True if apf is running otherwise return False\n    '
    status = 0
    table = iptc.Table(iptc.Table.FILTER)
    for chain in table.chains:
        if 'sanity' in chain.name.lower():
            status = 1
    return True if status else False

def running():
    if False:
        for i in range(10):
            print('nop')
    "\n    Check apf status\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apf.running\n    "
    return True if _status_apf() else False

def disable():
    if False:
        while True:
            i = 10
    "\n    Stop (flush) all firewall rules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apf.disable\n    "
    if _status_apf():
        return __apf_cmd('-f')

def enable():
    if False:
        for i in range(10):
            print('nop')
    "\n    Load all firewall rules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apf.enable\n    "
    if not _status_apf():
        return __apf_cmd('-s')

def reload():
    if False:
        return 10
    "\n    Stop (flush) & reload firewall rules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apf.reload\n    "
    if not _status_apf():
        return __apf_cmd('-r')

def refresh():
    if False:
        for i in range(10):
            print('nop')
    "\n    Refresh & resolve dns names in trust rules\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apf.refresh\n    "
    return __apf_cmd('-e')

def allow(ip, port=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add host (IP/FQDN) to allow_hosts.rules and immediately load new rule into firewall\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apf.allow 127.0.0.1\n    "
    if port is None:
        return __apf_cmd('-a {}'.format(ip))

def deny(ip):
    if False:
        i = 10
        return i + 15
    "\n    Add host (IP/FQDN) to deny_hosts.rules and immediately load new rule into firewall\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apf.deny 1.2.3.4\n    "
    return __apf_cmd('-d {}'.format(ip))

def remove(ip):
    if False:
        print('Hello World!')
    "\n    Remove host from [glob]*_hosts.rules and immediately remove rule from firewall\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' apf.remove 1.2.3.4\n    "
    return __apf_cmd('-u {}'.format(ip))