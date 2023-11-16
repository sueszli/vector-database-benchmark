"""
Service support for Solaris 10 and 11, should work with other systems
that use SMF also. (e.g. SmartOS)

.. important::
    If you feel that Salt should be using this module to manage services on a
    minion, and it is using a different module (or gives an error similar to
    *'service.start' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import fnmatch
import re
__func_alias__ = {'reload_': 'reload'}
__virtualname__ = 'service'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only work on systems which default to SMF\n    '
    if 'Solaris' in __grains__['os_family']:
        if __grains__['kernelrelease'] == '5.9':
            return (False, 'The smf execution module failed to load: SMF not available on Solaris 9.')
        return __virtualname__
    return (False, 'The smf execution module failed to load: only available on Solaris.')

def _get_enabled_disabled(enabled_prop='true'):
    if False:
        for i in range(10):
            print('nop')
    '\n    DRY: Get all service FMRIs and their enabled property\n    '
    ret = set()
    cmd = '/usr/bin/svcprop -c -p general/enabled "*"'
    lines = __salt__['cmd.run_stdout'](cmd, python_shell=False).splitlines()
    for line in lines:
        comps = line.split()
        if not comps:
            continue
        if comps[2] == enabled_prop:
            ret.add(comps[0].split('/:properties')[0])
    return sorted(ret)

def get_running():
    if False:
        return 10
    "\n    Return the running services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_running\n    "
    ret = set()
    cmd = '/usr/bin/svcs -H -o FMRI,STATE -s FMRI'
    lines = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    for line in lines:
        comps = line.split()
        if not comps:
            continue
        if 'online' in line:
            ret.add(comps[0])
    return sorted(ret)

def get_stopped():
    if False:
        return 10
    "\n    Return the stopped services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_stopped\n    "
    ret = set()
    cmd = '/usr/bin/svcs -aH -o FMRI,STATE -s FMRI'
    lines = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    for line in lines:
        comps = line.split()
        if not comps:
            continue
        if 'online' not in line and 'legacy_run' not in line:
            ret.add(comps[0])
    return sorted(ret)

def available(name):
    if False:
        print('Hello World!')
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    We look up the name with the svcs command to get back the FMRI\n    This allows users to use simpler service names\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available net-snmp\n    "
    cmd = '/usr/bin/svcs -H -o FMRI {}'.format(name)
    name = __salt__['cmd.run'](cmd, python_shell=False)
    return name in get_all()

def missing(name):
    if False:
        i = 10
        return i + 15
    "\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing net-snmp\n    "
    cmd = '/usr/bin/svcs -H -o FMRI {}'.format(name)
    name = __salt__['cmd.run'](cmd, python_shell=False)
    return name not in get_all()

def get_all():
    if False:
        return 10
    "\n    Return all installed services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    ret = set()
    cmd = '/usr/bin/svcs -aH -o FMRI,STATE -s FMRI'
    lines = __salt__['cmd.run'](cmd).splitlines()
    for line in lines:
        comps = line.split()
        if not comps:
            continue
        ret.add(comps[0])
    return sorted(ret)

def start(name):
    if False:
        while True:
            i = 10
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = '/usr/sbin/svcadm enable -s -t {}'.format(name)
    retcode = __salt__['cmd.retcode'](cmd, python_shell=False)
    if not retcode:
        return True
    if retcode == 3:
        clear_cmd = '/usr/sbin/svcadm clear {}'.format(name)
        __salt__['cmd.retcode'](clear_cmd, python_shell=False)
        return not __salt__['cmd.retcode'](cmd, python_shell=False)
    return False

def stop(name):
    if False:
        i = 10
        return i + 15
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = '/usr/sbin/svcadm disable -s -t {}'.format(name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def restart(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Restart the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = '/usr/sbin/svcadm restart {}'.format(name)
    if not __salt__['cmd.retcode'](cmd, python_shell=False):
        return start(name)
    return False

def reload_(name):
    if False:
        return 10
    "\n    Reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = '/usr/sbin/svcadm refresh {}'.format(name)
    if not __salt__['cmd.retcode'](cmd, python_shell=False):
        return start(name)
    return False

def status(name, sig=None):
    if False:
        while True:
            i = 10
    "\n    Return the status for a service.\n    If the name contains globbing, a dict mapping service name to True/False\n    values is returned.\n\n    .. versionchanged:: 2018.3.0\n        The service name can now be a glob (e.g. ``salt*``)\n\n    Args:\n        name (str): The name of the service to check\n        sig (str): Not implemented\n\n    Returns:\n        bool: True if running, False otherwise\n        dict: Maps service name to True if running, False otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name>\n    "
    contains_globbing = bool(re.search('\\*|\\?|\\[.+\\]', name))
    if contains_globbing:
        services = fnmatch.filter(get_all(), name)
    else:
        services = [name]
    results = {}
    for service in services:
        cmd = '/usr/bin/svcs -H -o STATE {}'.format(service)
        line = __salt__['cmd.run'](cmd, python_shell=False)
        results[service] = line == 'online'
    if contains_globbing:
        return results
    return results[name]

def enable(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Enable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name>\n    "
    cmd = '/usr/sbin/svcadm enable {}'.format(name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def disable(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Disable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name>\n    "
    cmd = '/usr/sbin/svcadm disable {}'.format(name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def enabled(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Check to see if the named service is enabled to start on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    fmri_cmd = '/usr/bin/svcs -H -o FMRI {}'.format(name)
    fmri = __salt__['cmd.run'](fmri_cmd, python_shell=False)
    cmd = '/usr/sbin/svccfg -s {} listprop general/enabled'.format(fmri)
    comps = __salt__['cmd.run'](cmd, python_shell=False).split()
    if comps[2] == 'true':
        return True
    else:
        return False

def disabled(name):
    if False:
        i = 10
        return i + 15
    "\n    Check to see if the named service is disabled to start on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    return not enabled(name)

def get_enabled():
    if False:
        print('Hello World!')
    "\n    Return the enabled services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    return _get_enabled_disabled('true')

def get_disabled():
    if False:
        while True:
            i = 10
    "\n    Return the disabled services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    return _get_enabled_disabled('false')