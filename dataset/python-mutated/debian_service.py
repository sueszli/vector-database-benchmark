"""
Service support for Debian systems (uses update-rc.d and /sbin/service)

.. important::
    If you feel that Salt should be using this module to manage services on a
    minion, and it is using a different module (or gives an error similar to
    *'service.start' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import fnmatch
import glob
import logging
import os
import re
import shlex
import salt.utils.systemd
__func_alias__ = {'reload_': 'reload'}
__virtualname__ = 'service'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    "\n    Only work on Debian and when systemd isn't running\n    "
    if __grains__['os'] in ('Debian', 'Raspbian', 'Devuan', 'NILinuxRT') and (not salt.utils.systemd.booted(__context__)):
        return __virtualname__
    else:
        return (False, 'The debian_service module could not be loaded: unsupported OS family and/or systemd running.')

def _service_cmd(*args):
    if False:
        i = 10
        return i + 15
    return 'service {} {}'.format(args[0], ' '.join(args[1:]))

def _get_runlevel():
    if False:
        for i in range(10):
            print('nop')
    '\n    returns the current runlevel\n    '
    out = __salt__['cmd.run']('runlevel')
    if 'unknown' in out:
        return '2'
    else:
        return out.split()[1]

def get_enabled():
    if False:
        print('Hello World!')
    "\n    Return a list of service that are enabled on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    prefix = '/etc/rc[S{}].d/S'.format(_get_runlevel())
    ret = set()
    for line in [x.rsplit(os.sep, 1)[-1] for x in glob.glob('{}*'.format(prefix))]:
        ret.add(re.split('\\d+', line)[-1])
    return sorted(ret)

def get_disabled():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a set of services that are installed but disabled\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    return sorted(set(get_all()) - set(get_enabled()))

def available(name):
    if False:
        return 10
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    return name in get_all()

def missing(name):
    if False:
        return 10
    "\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return name not in get_all()

def get_all():
    if False:
        print('Hello World!')
    "\n    Return all available boot services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    ret = set()
    lines = glob.glob('/etc/init.d/*')
    for line in lines:
        service = line.split('/etc/init.d/')[1]
        if service != 'README':
            ret.add(service)
    return sorted(ret | set(get_enabled()))

def start(name):
    if False:
        return 10
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = _service_cmd(name, 'start')
    return not __salt__['cmd.retcode'](cmd)

def stop(name):
    if False:
        return 10
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = _service_cmd(name, 'stop')
    return not __salt__['cmd.retcode'](cmd)

def restart(name):
    if False:
        i = 10
        return i + 15
    "\n    Restart the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = _service_cmd(name, 'restart')
    return not __salt__['cmd.retcode'](cmd)

def reload_(name):
    if False:
        print('Hello World!')
    "\n    Reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = _service_cmd(name, 'reload')
    return not __salt__['cmd.retcode'](cmd)

def force_reload(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Force-reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.force_reload <service name>\n    "
    cmd = _service_cmd(name, 'force-reload')
    return not __salt__['cmd.retcode'](cmd)

def status(name, sig=None):
    if False:
        while True:
            i = 10
    "\n    Return the status for a service.\n    If the name contains globbing, a dict mapping service name to True/False\n    values is returned.\n\n    .. versionchanged:: 2018.3.0\n        The service name can now be a glob (e.g. ``salt*``)\n\n    Args:\n        name (str): The name of the service to check\n        sig (str): Signature to use to find the service via ps\n\n    Returns:\n        bool: True if running, False otherwise\n        dict: Maps service name to True if running, False otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name> [service signature]\n    "
    if sig:
        return bool(__salt__['status.pid'](sig))
    contains_globbing = bool(re.search('\\*|\\?|\\[.+\\]', name))
    if contains_globbing:
        services = fnmatch.filter(get_all(), name)
    else:
        services = [name]
    results = {}
    for service in services:
        cmd = _service_cmd(service, 'status')
        results[service] = not __salt__['cmd.retcode'](cmd, ignore_retcode=True)
    if contains_globbing:
        return results
    return results[name]

def enable(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Enable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name>\n    "
    cmd = 'insserv {0} && update-rc.d {0} enable'.format(shlex.quote(name))
    return not __salt__['cmd.retcode'](cmd, python_shell=True)

def disable(name, **kwargs):
    if False:
        return 10
    "\n    Disable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name>\n    "
    cmd = 'update-rc.d {} disable'.format(name)
    return not __salt__['cmd.retcode'](cmd)

def enabled(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return True if the named service is enabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    return name in get_enabled()

def disabled(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return True if the named service is disabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    return name in get_disabled()