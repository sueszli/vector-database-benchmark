"""
The service module for Slackware

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
prefix = '/etc/rc.d/rc'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on Slackware\n    '
    if __grains__['os'] == 'Slackware':
        return __virtualname__
    return (False, 'The slackware_service execution module failed to load: only available on Slackware.')

def start(name):
    if False:
        while True:
            i = 10
    "\n    Start the specified service\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = f'/bin/sh {prefix}.{name} start'
    return not __salt__['cmd.retcode'](cmd)

def stop(name):
    if False:
        i = 10
        return i + 15
    "\n    Stop the specified service\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = f'/bin/sh {prefix}.{name} stop'
    return not __salt__['cmd.retcode'](cmd)

def restart(name):
    if False:
        while True:
            i = 10
    "\n    Restart the named service\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = f'/bin/sh {prefix}.{name} restart'
    return not __salt__['cmd.retcode'](cmd)

def reload_(name):
    if False:
        return 10
    "\n    Reload the named service\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = f'/bin/sh {prefix}.{name} reload'
    return not __salt__['cmd.retcode'](cmd)

def force_reload(name):
    if False:
        return 10
    "\n    Force-reload the named service\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.force_reload <service name>\n    "
    cmd = f'/bin/sh {prefix}.{name} forcereload'
    return not __salt__['cmd.retcode'](cmd)

def status(name, sig=None):
    if False:
        return 10
    "\n    Return the status for a service.\n    If the name contains globbing, a dict mapping service name to True/False\n    values is returned.\n\n    .. versionadded:: 3002\n\n    Args:\n        name (str): The name of the service to check\n        sig (str): Signature to use to find the service via ps\n\n    Returns:\n        bool: True if running, False otherwise\n        dict: Maps service name to True if running, False otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name> [service signature]\n    "
    if sig:
        return bool(__salt__['status.pid'](sig))
    contains_globbing = bool(re.search('\\*|\\?|\\[.+\\]', name))
    if contains_globbing:
        services = fnmatch.filter(get_all(), name)
    else:
        services = [name]
    results = {}
    for service in services:
        cmd = f'/bin/sh {prefix}.{service} status'
        results[service] = not __salt__['cmd.retcode'](cmd, ignore_retcode=True)
    if contains_globbing:
        return results
    return results[name]

def _get_svc(rcd, service_status):
    if False:
        print('Hello World!')
    '\n    Returns a unique service status\n    '
    if os.path.exists(rcd):
        ena = os.access(rcd, os.X_OK)
        svc = rcd.split('.')[2]
        if service_status == '':
            return svc
        elif service_status == 'ON' and ena:
            return svc
        elif service_status == 'OFF' and (not ena):
            return svc
    return None

def _get_svc_list(service_status):
    if False:
        print('Hello World!')
    '\n    Returns all service statuses\n    '
    notservice = re.compile('{}.([A-Za-z0-9_-]+\\.conf|0|4|6|K|M|S|inet1|inet2|local|modules.*|wireless)$'.format(prefix))
    ret = set()
    lines = glob.glob(f'{prefix}.*')
    for line in lines:
        if not notservice.match(line):
            svc = _get_svc(line, service_status)
            if svc is not None:
                ret.add(svc)
    return sorted(ret)

def get_enabled():
    if False:
        print('Hello World!')
    "\n    Return a list of service that are enabled on boot\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    return _get_svc_list('ON')

def get_disabled():
    if False:
        return 10
    "\n    Return a set of services that are installed but disabled\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    return _get_svc_list('OFF')

def available(name):
    if False:
        while True:
            i = 10
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    return name in get_all()

def missing(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return name not in get_all()

def get_all():
    if False:
        while True:
            i = 10
    "\n    Return all available boot services\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    return _get_svc_list('')

def _rcd_mode(name, ena):
    if False:
        print('Hello World!')
    '\n    Enable/Disable a service\n    '
    rcd = prefix + '.' + name
    if os.path.exists(rcd):
        perms = os.stat(rcd).st_mode
        if ena == 'ON':
            perms |= 73
            os.chmod(rcd, perms)
        elif ena == 'OFF':
            perms &= 262070
            os.chmod(rcd, perms)
        return True
    return False

def enable(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Enable the named service to start at boot\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name>\n    "
    return _rcd_mode(name, 'ON')

def disable(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Disable the named service to start at boot\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name>\n    "
    return _rcd_mode(name, 'OFF')

def enabled(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return True if the named service is enabled, false otherwise\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    ret = True
    if _get_svc(f'{prefix}.{name}', 'ON') is None:
        ret = False
    return ret

def disabled(name):
    if False:
        while True:
            i = 10
    "\n    Return True if the named service is enabled, false otherwise\n\n    .. versionadded:: 3002\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    ret = True
    if _get_svc(f'{prefix}.{name}', 'OFF') is None:
        ret = False
    return ret