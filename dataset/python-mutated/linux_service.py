"""
If Salt's OS detection does not identify a different virtual service module, the minion will fall back to using this basic module, which simply wraps sysvinit scripts.
"""
import fnmatch
import os
import re
__func_alias__ = {'reload_': 'reload'}
_GRAINMAP = {'Arch': '/etc/rc.d', 'Arch ARM': '/etc/rc.d'}

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only work on systems which exclusively use sysvinit\n    '
    disable = {'RedHat', 'CentOS', 'Amazon', 'ScientificLinux', 'CloudLinux', 'Fedora', 'Gentoo', 'Ubuntu', 'Debian', 'Devuan', 'ALT', 'OEL', 'Linaro', 'elementary OS', 'McAfee  OS Server', 'Raspbian', 'SUSE', 'Slackware'}
    if __grains__.get('os') in disable:
        return (False, 'Your OS is on the disabled list')
    if __grains__['kernel'] != 'Linux':
        return (False, 'Non Linux OSes are not supported')
    init_grain = __grains__.get('init')
    if init_grain not in (None, 'sysvinit', 'unknown'):
        return (False, 'Minion is running {}'.format(init_grain))
    elif __utils__['systemd.booted'](__context__):
        return (False, 'Minion is running systemd')
    return 'service'

def run(name, action):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run the specified service with an action.\n\n    .. versionadded:: 2015.8.1\n\n    name\n        Service name.\n\n    action\n        Action name (like start,  stop,  reload,  restart).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.run apache2 reload\n        salt '*' service.run postgresql initdb\n    "
    cmd = os.path.join(_GRAINMAP.get(__grains__.get('os'), '/etc/init.d'), name) + ' ' + action
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def start(name):
    if False:
        while True:
            i = 10
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    return run(name, 'start')

def stop(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    return run(name, 'stop')

def restart(name):
    if False:
        return 10
    "\n    Restart the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    return run(name, 'restart')

def status(name, sig=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the status for a service.\n    If the name contains globbing, a dict mapping service name to PID or empty\n    string is returned.\n\n    .. versionchanged:: 2018.3.0\n        The service name can now be a glob (e.g. ``salt*``)\n\n    Args:\n        name (str): The name of the service to check\n        sig (str): Signature to use to find the service via ps\n\n    Returns:\n        string: PID if running, empty otherwise\n        dict: Maps service name to PID if running, empty string otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name> [service signature]\n    "
    if sig:
        return __salt__['status.pid'](sig)
    contains_globbing = bool(re.search('\\*|\\?|\\[.+\\]', name))
    if contains_globbing:
        services = fnmatch.filter(get_all(), name)
    else:
        services = [name]
    results = {}
    for service in services:
        results[service] = __salt__['status.pid'](service)
    if contains_globbing:
        return results
    return results[name]

def reload_(name):
    if False:
        while True:
            i = 10
    "\n    Refreshes config files by calling service reload. Does not perform a full\n    restart.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    return run(name, 'reload')

def available(name):
    if False:
        while True:
            i = 10
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    return name in get_all()

def missing(name):
    if False:
        i = 10
        return i + 15
    "\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return name not in get_all()

def get_all():
    if False:
        print('Hello World!')
    "\n    Return a list of all available services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    if not os.path.isdir(_GRAINMAP.get(__grains__.get('os'), '/etc/init.d')):
        return []
    return sorted(os.listdir(_GRAINMAP.get(__grains__.get('os'), '/etc/init.d')))