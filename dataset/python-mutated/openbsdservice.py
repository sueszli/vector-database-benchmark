"""
The service module for OpenBSD

.. important::
    If you feel that Salt should be using this module to manage services on a
    minion, and it is using a different module (or gives an error similar to
    *'service.start' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import fnmatch
import logging
import os
import re
import salt.utils.data
import salt.utils.files
log = logging.getLogger(__name__)
__virtualname__ = 'service'
__func_alias__ = {'reload_': 'reload'}

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only work on OpenBSD\n    '
    if __grains__['os'] == 'OpenBSD' and os.path.exists('/etc/rc.d/rc.subr'):
        krel = list(list(map(int, __grains__['kernelrelease'].split('.'))))
        if krel[0] > 5 or (krel[0] == 5 and krel[1] > 0):
            if not os.path.exists('/usr/sbin/rcctl'):
                return __virtualname__
    return (False, 'The openbsdservice execution module cannot be loaded: only available on OpenBSD systems.')

def start(name):
    if False:
        while True:
            i = 10
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = '/etc/rc.d/{} -f start'.format(name)
    return not __salt__['cmd.retcode'](cmd)

def stop(name):
    if False:
        while True:
            i = 10
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = '/etc/rc.d/{} -f stop'.format(name)
    return not __salt__['cmd.retcode'](cmd)

def restart(name):
    if False:
        print('Hello World!')
    "\n    Restart the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = '/etc/rc.d/{} -f restart'.format(name)
    return not __salt__['cmd.retcode'](cmd)

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
        cmd = '/etc/rc.d/{} -f check'.format(service)
        results[service] = not __salt__['cmd.retcode'](cmd, ignore_retcode=True)
    if contains_globbing:
        return results
    return results[name]

def reload_(name):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2014.7.0\n\n    Reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = '/etc/rc.d/{} -f reload'.format(name)
    return not __salt__['cmd.retcode'](cmd)
service_flags_regex = re.compile('^\\s*(\\w[\\d\\w]*)_flags=(?:(NO)|.*)$')
pkg_scripts_regex = re.compile("^\\s*pkg_scripts=\\'(.*)\\'$")
start_daemon_call_regex = re.compile('(\\s*start_daemon(?!\\(\\)))')
start_daemon_parameter_regex = re.compile('(?:\\s+(\\w[\\w\\d]*))')

def _get_rc():
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a dict where the key is the daemon's name and\n    the value a boolean indicating its status (True: enabled or False: disabled).\n    Check the daemons started by the system in /etc/rc and\n    configured in /etc/rc.conf and /etc/rc.conf.local.\n    Also add to the dict all the localy enabled daemons via $pkg_scripts.\n    "
    daemons_flags = {}
    try:
        with salt.utils.files.fopen('/etc/rc', 'r') as handle:
            lines = salt.utils.data.decode(handle.readlines())
    except OSError:
        log.error('Unable to read /etc/rc')
    else:
        for line in lines:
            match = start_daemon_call_regex.match(line)
            if match:
                line = line[len(match.group(1)):]
                for daemon in start_daemon_parameter_regex.findall(line):
                    daemons_flags[daemon] = True
    variables = __salt__['cmd.run']('(. /etc/rc.conf && set)', clean_env=True, output_loglevel='quiet', python_shell=True).split('\n')
    for var in variables:
        match = service_flags_regex.match(var)
        if match:
            if match.group(2) == 'NO':
                daemons_flags[match.group(1)] = False
        else:
            match = pkg_scripts_regex.match(var)
            if match:
                for daemon in match.group(1).split():
                    daemons_flags[daemon] = True
    return daemons_flags

def available(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2014.7.0\n\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    path = '/etc/rc.d/{}'.format(name)
    return os.path.isfile(path) and os.access(path, os.X_OK)

def missing(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2014.7.0\n\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return not available(name)

def get_all():
    if False:
        return 10
    "\n    .. versionadded:: 2014.7.0\n\n    Return all available boot services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    services = []
    if not os.path.isdir('/etc/rc.d'):
        return services
    for service in os.listdir('/etc/rc.d'):
        if available(service):
            services.append(service)
    return sorted(services)

def get_enabled():
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2014.7.0\n\n    Return a list of service that are enabled on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    services = []
    for (daemon, is_enabled) in _get_rc().items():
        if is_enabled:
            services.append(daemon)
    return sorted(set(get_all()) & set(services))

def enabled(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2014.7.0\n\n    Return True if the named service is enabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    return name in get_enabled()

def get_disabled():
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.7.0\n\n    Return a set of services that are installed but disabled\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    services = []
    for (daemon, is_enabled) in _get_rc().items():
        if not is_enabled:
            services.append(daemon)
    return sorted(set(get_all()) & set(services))

def disabled(name):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.7.0\n\n    Return True if the named service is disabled, false otherwise\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    return name in get_disabled()