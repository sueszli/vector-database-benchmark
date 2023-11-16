"""
Top level package command wrapper, used to translate the os detected by grains
to the correct service manager

.. important::
    If you feel that Salt should be using this module to manage services on a
    minion, and it is using a different module (or gives an error similar to
    *'service.start' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import fnmatch
import logging
import re
import salt.utils.odict as odict
import salt.utils.systemd
log = logging.getLogger(__name__)
__virtualname__ = 'service'
__func_alias__ = {'reload_': 'reload'}

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only work on systems which default to OpenRC\n    '
    if __grains__['os'] == 'Gentoo' and (not salt.utils.systemd.booted(__context__)):
        return __virtualname__
    if __grains__['os'] == 'Alpine':
        return __virtualname__
    return (False, 'The gentoo_service execution module cannot be loaded: only available on Gentoo/Open-RC systems.')

def _ret_code(cmd, ignore_retcode=False):
    if False:
        while True:
            i = 10
    log.debug('executing [%s]', cmd)
    sts = __salt__['cmd.retcode'](cmd, python_shell=False, ignore_retcode=ignore_retcode)
    return sts

def _list_services():
    if False:
        while True:
            i = 10
    return __salt__['cmd.run']('rc-update -v show').splitlines()

def _get_service_list(include_enabled=True, include_disabled=False):
    if False:
        for i in range(10):
            print('nop')
    enabled_services = dict()
    disabled_services = set()
    lines = _list_services()
    for line in lines:
        if '|' not in line:
            continue
        service = [l.strip() for l in line.split('|')]
        if service[1]:
            if include_enabled:
                enabled_services.update({service[0]: sorted(service[1].split())})
            continue
        if include_disabled:
            disabled_services.update({service[0]: []})
    return (enabled_services, disabled_services)

def _enable_delta(name, requested_runlevels):
    if False:
        while True:
            i = 10
    all_enabled = get_enabled()
    current_levels = set(all_enabled[name] if name in all_enabled else [])
    enabled_levels = requested_runlevels - current_levels
    disabled_levels = current_levels - requested_runlevels
    return (enabled_levels, disabled_levels)

def _disable_delta(name, requested_runlevels):
    if False:
        while True:
            i = 10
    all_enabled = get_enabled()
    current_levels = set(all_enabled[name] if name in all_enabled else [])
    return current_levels & requested_runlevels

def _service_cmd(*args):
    if False:
        return 10
    return '/etc/init.d/{} {}'.format(args[0], ' '.join(args[1:]))

def _enable_disable_cmd(name, command, runlevels=()):
    if False:
        print('Hello World!')
    return 'rc-update {} {} {}'.format(command, name, ' '.join(sorted(runlevels))).strip()

def get_enabled():
    if False:
        return 10
    "\n    Return a list of service that are enabled on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    (enabled_services, disabled_services) = _get_service_list()
    return odict.OrderedDict(enabled_services)

def get_disabled():
    if False:
        while True:
            i = 10
    "\n    Return a set of services that are installed but disabled\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    (enabled_services, disabled_services) = _get_service_list(include_enabled=False, include_disabled=True)
    return sorted(disabled_services)

def available(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    (enabled_services, disabled_services) = _get_service_list(include_enabled=True, include_disabled=True)
    return name in enabled_services or name in disabled_services

def missing(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return not available(name)

def get_all():
    if False:
        i = 10
        return i + 15
    "\n    Return all available boot services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    (enabled_services, disabled_services) = _get_service_list(include_enabled=True, include_disabled=True)
    enabled_services.update({s: [] for s in disabled_services})
    return odict.OrderedDict(enabled_services)

def start(name):
    if False:
        print('Hello World!')
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = _service_cmd(name, 'start')
    return not _ret_code(cmd)

def stop(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = _service_cmd(name, 'stop')
    return not _ret_code(cmd)

def restart(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Restart the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = _service_cmd(name, 'restart')
    return not _ret_code(cmd)

def reload_(name):
    if False:
        return 10
    "\n    Reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = _service_cmd(name, 'reload')
    return not _ret_code(cmd)

def zap(name):
    if False:
        return 10
    "\n    Resets service state\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.zap <service name>\n    "
    cmd = _service_cmd(name, 'zap')
    return not _ret_code(cmd)

def status(name, sig=None):
    if False:
        i = 10
        return i + 15
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
        results[service] = not _ret_code(cmd, ignore_retcode=True)
    if contains_globbing:
        return results
    return results[name]

def enable(name, **kwargs):
    if False:
        return 10
    "\n    Enable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name> <runlevels=single-runlevel>\n        salt '*' service.enable <service name> <runlevels=[runlevel1,runlevel2]>\n    "
    if 'runlevels' in kwargs:
        requested_levels = set(kwargs['runlevels'] if isinstance(kwargs['runlevels'], list) else [kwargs['runlevels']])
        (enabled_levels, disabled_levels) = _enable_delta(name, requested_levels)
        commands = []
        if disabled_levels:
            commands.append(_enable_disable_cmd(name, 'delete', disabled_levels))
        if enabled_levels:
            commands.append(_enable_disable_cmd(name, 'add', enabled_levels))
        if not commands:
            return True
    else:
        commands = [_enable_disable_cmd(name, 'add')]
    for cmd in commands:
        if _ret_code(cmd):
            return False
    return True

def disable(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Disable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name> <runlevels=single-runlevel>\n        salt '*' service.disable <service name> <runlevels=[runlevel1,runlevel2]>\n    "
    levels = []
    if 'runlevels' in kwargs:
        requested_levels = set(kwargs['runlevels'] if isinstance(kwargs['runlevels'], list) else [kwargs['runlevels']])
        levels = _disable_delta(name, requested_levels)
        if not levels:
            return True
    cmd = _enable_disable_cmd(name, 'delete', levels)
    return not _ret_code(cmd)

def enabled(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Return True if the named service is enabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name> <runlevels=single-runlevel>\n        salt '*' service.enabled <service name> <runlevels=[runlevel1,runlevel2]>\n    "
    enabled_services = get_enabled()
    if name not in enabled_services:
        return False
    if 'runlevels' not in kwargs:
        return True
    requested_levels = set(kwargs['runlevels'] if isinstance(kwargs['runlevels'], list) else [kwargs['runlevels']])
    return len(requested_levels - set(enabled_services[name])) == 0

def disabled(name):
    if False:
        while True:
            i = 10
    "\n    Return True if the named service is enabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name> <runlevels=[runlevel]>\n    "
    return name in get_disabled()