"""
The service module for NetBSD

.. important::
    If you feel that Salt should be using this module to manage services on a
    minion, and it is using a different module (or gives an error similar to
    *'service.start' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import fnmatch
import glob
import os
import re
__func_alias__ = {'reload_': 'reload'}
__virtualname__ = 'service'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only work on NetBSD\n    '
    if __grains__['os'] == 'NetBSD' and os.path.exists('/etc/rc.subr'):
        return __virtualname__
    return (False, 'The netbsdservice execution module failed to load: only available on NetBSD.')

def start(name):
    if False:
        i = 10
        return i + 15
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = f'/etc/rc.d/{name} onestart'
    return not __salt__['cmd.retcode'](cmd)

def stop(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = f'/etc/rc.d/{name} onestop'
    return not __salt__['cmd.retcode'](cmd)

def restart(name):
    if False:
        i = 10
        return i + 15
    "\n    Restart the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = f'/etc/rc.d/{name} onerestart'
    return not __salt__['cmd.retcode'](cmd)

def reload_(name):
    if False:
        i = 10
        return i + 15
    "\n    Reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = f'/etc/rc.d/{name} onereload'
    return not __salt__['cmd.retcode'](cmd)

def force_reload(name):
    if False:
        while True:
            i = 10
    "\n    Force-reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.force_reload <service name>\n    "
    cmd = f'/etc/rc.d/{name} forcereload'
    return not __salt__['cmd.retcode'](cmd)

def status(name, sig=None):
    if False:
        for i in range(10):
            print('nop')
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
        cmd = f'/etc/rc.d/{service} onestatus'
        results[service] = not __salt__['cmd.retcode'](cmd, ignore_retcode=True)
    if contains_globbing:
        return results
    return results[name]

def _get_svc(rcd, service_status):
    if False:
        return 10
    '\n    Returns a unique service status\n    '
    ena = None
    lines = __salt__['cmd.run'](f'{rcd} rcvar').splitlines()
    for rcvar in lines:
        if rcvar.startswith('$') and f'={service_status}' in rcvar:
            ena = 'yes'
        elif rcvar.startswith('#'):
            svc = rcvar.split(' ', 1)[1]
        else:
            continue
    if ena and svc:
        return svc
    return None

def _get_svc_list(service_status):
    if False:
        return 10
    '\n    Returns all service statuses\n    '
    prefix = '/etc/rc.d/'
    ret = set()
    lines = glob.glob(f'{prefix}*')
    for line in lines:
        svc = _get_svc(line, service_status)
        if svc is not None:
            ret.add(svc)
    return sorted(ret)

def get_enabled():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of service that are enabled on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    return _get_svc_list('YES')

def get_disabled():
    if False:
        print('Hello World!')
    "\n    Return a set of services that are installed but disabled\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    return _get_svc_list('NO')

def available(name):
    if False:
        i = 10
        return i + 15
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    return name in get_all()

def missing(name):
    if False:
        return 10
    "\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return name not in get_all()

def get_all():
    if False:
        return 10
    "\n    Return all available boot services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    return _get_svc_list('')

def _rcconf_status(name, service_status):
    if False:
        for i in range(10):
            print('nop')
    '\n    Modifies /etc/rc.conf so a service is started or not at boot time and\n    can be started via /etc/rc.d/<service>\n    '
    rcconf = '/etc/rc.conf'
    rxname = f'^{name}=.*'
    newstatus = f'{name}={service_status}'
    ret = __salt__['cmd.retcode'](f"grep '{rxname}' {rcconf}")
    if ret == 0:
        __salt__['file.replace'](rcconf, rxname, newstatus)
    else:
        ret = __salt__['file.append'](rcconf, newstatus)
    return ret

def enable(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Enable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name>\n    "
    return _rcconf_status(name, 'YES')

def disable(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Disable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name>\n    "
    return _rcconf_status(name, 'NO')

def enabled(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return True if the named service is enabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    return _get_svc(f'/etc/rc.d/{name}', 'YES')

def disabled(name):
    if False:
        i = 10
        return i + 15
    "\n    Return True if the named service is enabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    return _get_svc(f'/etc/rc.d/{name}', 'NO')