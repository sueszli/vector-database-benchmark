"""
Service support for RHEL-based systems, including support for both upstart and sysvinit

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
import stat
import salt.utils.path
log = logging.getLogger(__name__)
__func_alias__ = {'reload_': 'reload'}
__virtualname__ = 'service'
HAS_UPSTART = False
if salt.utils.path.which('initctl'):
    try:
        from salt.modules.upstart_service import _upstart_disable, _upstart_enable, _upstart_is_enabled
    except Exception as exc:
        log.error('Unable to import helper functions from salt.modules.upstart: %s', exc)
    else:
        HAS_UPSTART = True

def __virtual__():
    if False:
        while True:
            i = 10
    "\n    Only work on select distros which still use Red Hat's /usr/bin/service for\n    management of either sysvinit or a hybrid sysvinit/upstart init system.\n    "
    if __utils__['systemd.booted'](__context__):
        return (False, 'The rh_service execution module failed to load: this system was booted with systemd.')
    enable = {'XenServer', 'XCP-ng', 'RedHat', 'CentOS', 'ScientificLinux', 'CloudLinux', 'Amazon', 'Fedora', 'ALT', 'OEL', 'SUSE  Enterprise Server', 'SUSE', 'McAfee  OS Server', 'VirtuozzoLinux'}
    if __grains__['os'] in enable:
        if __grains__['os'] == 'SUSE':
            if str(__grains__['osrelease']).startswith('11'):
                return __virtualname__
            else:
                return (False, 'Cannot load rh_service module on SUSE > 11')
        osrelease_major = __grains__.get('osrelease_info', [0])[0]
        if __grains__['os'] in ('XenServer', 'XCP-ng'):
            if osrelease_major >= 7:
                return (False, "XenServer and XCP-ng >= 7 use systemd, will not load rh_service.py as virtual 'service'")
            return __virtualname__
        if __grains__['os'] == 'Fedora':
            if osrelease_major >= 15:
                return (False, "Fedora >= 15 uses systemd, will not load rh_service.py as virtual 'service'")
        if __grains__['os'] in ('RedHat', 'CentOS', 'ScientificLinux', 'OEL', 'CloudLinux'):
            if osrelease_major >= 7:
                return (False, "RedHat-based distros >= version 7 use systemd, will not load rh_service.py as virtual 'service'")
        return __virtualname__
    return (False, f'Cannot load rh_service module: OS not in {enable}')

def _runlevel():
    if False:
        i = 10
        return i + 15
    '\n    Return the current runlevel\n    '
    out = __salt__['cmd.run']('/sbin/runlevel')
    if 'unknown' in out:
        return '3'
    else:
        return out.split()[1]

def _chkconfig_add(name):
    if False:
        i = 10
        return i + 15
    "\n    Run 'chkconfig --add' for a service whose script is installed in\n    /etc/init.d.  The service is initially configured to be disabled at all\n    run-levels.\n    "
    cmd = f'/sbin/chkconfig --add {name}'
    if __salt__['cmd.retcode'](cmd, python_shell=False) == 0:
        log.info('Added initscript "%s" to chkconfig', name)
        return True
    else:
        log.error('Unable to add initscript "%s" to chkconfig', name)
        return False

def _service_is_upstart(name):
    if False:
        print('Hello World!')
    '\n    Return True if the service is an upstart service, otherwise return False.\n    '
    return HAS_UPSTART and os.path.exists(f'/etc/init/{name}.conf')

def _service_is_sysv(name):
    if False:
        return 10
    '\n    Return True if the service is a System V service (includes those managed by\n    chkconfig); otherwise return False.\n    '
    try:
        return bool(os.stat(os.path.join('/etc/init.d', name)).st_mode & stat.S_IXUSR)
    except OSError:
        return False

def _service_is_chkconfig(name):
    if False:
        print('Hello World!')
    '\n    Return True if the service is managed by chkconfig.\n    '
    cmdline = f'/sbin/chkconfig --list {name}'
    return __salt__['cmd.retcode'](cmdline, python_shell=False, ignore_retcode=True) == 0

def _sysv_is_enabled(name, runlevel=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return True if the sysv (or chkconfig) service is enabled for the specified\n    runlevel; otherwise return False.  If `runlevel` is None, then use the\n    current runlevel.\n    '
    result = _chkconfig_is_enabled(name, runlevel)
    if result:
        return True
    if runlevel is None:
        runlevel = _runlevel()
    return len(glob.glob(f'/etc/rc.d/rc{runlevel}.d/S??{name}')) > 0

def _chkconfig_is_enabled(name, runlevel=None):
    if False:
        return 10
    '\n    Return ``True`` if the service is enabled according to chkconfig; otherwise\n    return ``False``.  If ``runlevel`` is ``None``, then use the current\n    runlevel.\n    '
    cmdline = f'/sbin/chkconfig --list {name}'
    result = __salt__['cmd.run_all'](cmdline, python_shell=False)
    if runlevel is None:
        runlevel = _runlevel()
    if result['retcode'] == 0:
        for row in result['stdout'].splitlines():
            if f'{runlevel}:on' in row:
                if row.split()[0] == name:
                    return True
            elif row.split() == [name, 'on']:
                return True
    return False

def _sysv_enable(name):
    if False:
        print('Hello World!')
    '\n    Enable the named sysv service to start at boot.  The service will be enabled\n    using chkconfig with default run-levels if the service is chkconfig\n    compatible.  If chkconfig is not available, then this will fail.\n    '
    if not _service_is_chkconfig(name) and (not _chkconfig_add(name)):
        return False
    cmd = f'/sbin/chkconfig {name} on'
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def _sysv_disable(name):
    if False:
        return 10
    '\n    Disable the named sysv service from starting at boot.  The service will be\n    disabled using chkconfig with default run-levels if the service is chkconfig\n    compatible; otherwise, the service will be disabled for the current\n    run-level only.\n    '
    if not _service_is_chkconfig(name) and (not _chkconfig_add(name)):
        return False
    cmd = f'/sbin/chkconfig {name} off'
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def _sysv_delete(name):
    if False:
        print('Hello World!')
    '\n    Delete the named sysv service from the system. The service will be\n    deleted using chkconfig.\n    '
    if not _service_is_chkconfig(name):
        return False
    cmd = f'/sbin/chkconfig --del {name}'
    return not __salt__['cmd.retcode'](cmd)

def _upstart_delete(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Delete an upstart service. This will only rename the .conf file\n    '
    if HAS_UPSTART:
        if os.path.exists(f'/etc/init/{name}.conf'):
            os.rename(f'/etc/init/{name}.conf', f'/etc/init/{name}.conf.removed')
    return True

def _upstart_services():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return list of upstart services.\n    '
    if HAS_UPSTART:
        return [os.path.basename(name)[:-5] for name in glob.glob('/etc/init/*.conf')]
    else:
        return []

def _sysv_services():
    if False:
        i = 10
        return i + 15
    '\n    Return list of sysv services.\n    '
    _services = []
    output = __salt__['cmd.run'](['chkconfig', '--list'], python_shell=False)
    for line in output.splitlines():
        comps = line.split()
        try:
            if comps[1].startswith('0:'):
                _services.append(comps[0])
        except IndexError:
            continue
    return [x for x in _services if _service_is_sysv(x)]

def get_enabled(limit=''):
    if False:
        return 10
    "\n    Return the enabled services. Use the ``limit`` param to restrict results\n    to services of that type.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n        salt '*' service.get_enabled limit=upstart\n        salt '*' service.get_enabled limit=sysvinit\n    "
    limit = limit.lower()
    if limit == 'upstart':
        return sorted((name for name in _upstart_services() if _upstart_is_enabled(name)))
    elif limit == 'sysvinit':
        runlevel = _runlevel()
        return sorted((name for name in _sysv_services() if _sysv_is_enabled(name, runlevel)))
    else:
        runlevel = _runlevel()
        return sorted([name for name in _upstart_services() if _upstart_is_enabled(name)] + [name for name in _sysv_services() if _sysv_is_enabled(name, runlevel)])

def get_disabled(limit=''):
    if False:
        i = 10
        return i + 15
    "\n    Return the disabled services. Use the ``limit`` param to restrict results\n    to services of that type.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n        salt '*' service.get_disabled limit=upstart\n        salt '*' service.get_disabled limit=sysvinit\n    "
    limit = limit.lower()
    if limit == 'upstart':
        return sorted((name for name in _upstart_services() if not _upstart_is_enabled(name)))
    elif limit == 'sysvinit':
        runlevel = _runlevel()
        return sorted((name for name in _sysv_services() if not _sysv_is_enabled(name, runlevel)))
    else:
        runlevel = _runlevel()
        return sorted([name for name in _upstart_services() if not _upstart_is_enabled(name)] + [name for name in _sysv_services() if not _sysv_is_enabled(name, runlevel)])

def get_all(limit=''):
    if False:
        print('Hello World!')
    "\n    Return all installed services. Use the ``limit`` param to restrict results\n    to services of that type.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n        salt '*' service.get_all limit=upstart\n        salt '*' service.get_all limit=sysvinit\n    "
    limit = limit.lower()
    if limit == 'upstart':
        return sorted(_upstart_services())
    elif limit == 'sysvinit':
        return sorted(_sysv_services())
    else:
        return sorted(_sysv_services() + _upstart_services())

def available(name, limit=''):
    if False:
        while True:
            i = 10
    "\n    Return True if the named service is available.  Use the ``limit`` param to\n    restrict results to services of that type.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n        salt '*' service.available sshd limit=upstart\n        salt '*' service.available sshd limit=sysvinit\n    "
    if limit == 'upstart':
        return _service_is_upstart(name)
    elif limit == 'sysvinit':
        return _service_is_sysv(name)
    else:
        return _service_is_upstart(name) or _service_is_sysv(name) or _service_is_chkconfig(name)

def missing(name, limit=''):
    if False:
        while True:
            i = 10
    "\n    The inverse of service.available.\n    Return True if the named service is not available.  Use the ``limit`` param to\n    restrict results to services of that type.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n        salt '*' service.missing sshd limit=upstart\n        salt '*' service.missing sshd limit=sysvinit\n    "
    if limit == 'upstart':
        return not _service_is_upstart(name)
    elif limit == 'sysvinit':
        return not _service_is_sysv(name)
    elif _service_is_upstart(name) or _service_is_sysv(name):
        return False
    else:
        return True

def start(name):
    if False:
        return 10
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    if _service_is_upstart(name):
        cmd = f'start {name}'
    else:
        cmd = f'/sbin/service {name} start'
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def stop(name):
    if False:
        print('Hello World!')
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    if _service_is_upstart(name):
        cmd = f'stop {name}'
    else:
        cmd = f'/sbin/service {name} stop'
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def restart(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Restart the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    if _service_is_upstart(name):
        cmd = f'restart {name}'
    else:
        cmd = f'/sbin/service {name} restart'
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def reload_(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    if _service_is_upstart(name):
        cmd = f'reload {name}'
    else:
        cmd = f'/sbin/service {name} reload'
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def status(name, sig=None):
    if False:
        print('Hello World!')
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
        if _service_is_upstart(service):
            cmd = f'status {service}'
            results[service] = 'start/running' in __salt__['cmd.run'](cmd, python_shell=False)
        else:
            cmd = f'/sbin/service {service} status'
            results[service] = __salt__['cmd.retcode'](cmd, python_shell=False, ignore_retcode=True) == 0
    if contains_globbing:
        return results
    return results[name]

def delete(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Delete the named service\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.delete <service name>\n    "
    if _service_is_upstart(name):
        return _upstart_delete(name)
    else:
        return _sysv_delete(name)

def enable(name, **kwargs):
    if False:
        return 10
    "\n    Enable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name>\n    "
    if _service_is_upstart(name):
        return _upstart_enable(name)
    else:
        return _sysv_enable(name)

def disable(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Disable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name>\n    "
    if _service_is_upstart(name):
        return _upstart_disable(name)
    else:
        return _sysv_disable(name)

def enabled(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Check to see if the named service is enabled to start on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    if _service_is_upstart(name):
        return _upstart_is_enabled(name)
    else:
        return _sysv_is_enabled(name)

def disabled(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check to see if the named service is disabled to start on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    if _service_is_upstart(name):
        return not _upstart_is_enabled(name)
    else:
        return not _sysv_is_enabled(name)