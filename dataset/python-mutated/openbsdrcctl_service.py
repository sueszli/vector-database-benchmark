"""
The rcctl service module for OpenBSD
"""
import os
import salt.utils.decorators as decorators
import salt.utils.path
from salt.exceptions import CommandNotFoundError
__func_alias__ = {'reload_': 'reload'}
__virtualname__ = 'service'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    rcctl(8) is only available on OpenBSD.\n    '
    if __grains__['os'] == 'OpenBSD' and os.path.exists('/usr/sbin/rcctl'):
        return __virtualname__
    return (False, 'The openbsdpkg execution module cannot be loaded: only available on OpenBSD systems.')

@decorators.memoize
def _cmd():
    if False:
        while True:
            i = 10
    '\n    Return the full path to the rcctl(8) command.\n    '
    rcctl = salt.utils.path.which('rcctl')
    if not rcctl:
        raise CommandNotFoundError
    return rcctl

def _get_flags(**kwargs):
    if False:
        print('Hello World!')
    '\n    Return the configured service flags.\n    '
    flags = kwargs.get('flags', __salt__['config.option']('service.flags', default=''))
    return flags

def available(name):
    if False:
        while True:
            i = 10
    "\n    Return True if the named service is available.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    cmd = '{} get {}'.format(_cmd(), name)
    if __salt__['cmd.retcode'](cmd, ignore_retcode=True) == 2:
        return False
    return True

def missing(name):
    if False:
        return 10
    "\n    The inverse of service.available.\n    Return True if the named service is not available.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return not available(name)

def get_all():
    if False:
        return 10
    "\n    Return all installed services.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    ret = []
    service = _cmd()
    for svc in __salt__['cmd.run']('{} ls all'.format(service)).splitlines():
        ret.append(svc)
    return sorted(ret)

def get_disabled():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return what services are available but not enabled to start at boot.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    ret = []
    service = _cmd()
    for svc in __salt__['cmd.run']('{} ls off'.format(service)).splitlines():
        ret.append(svc)
    return sorted(ret)

def get_enabled():
    if False:
        return 10
    "\n    Return what services are set to run on boot.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    ret = []
    service = _cmd()
    for svc in __salt__['cmd.run']('{} ls on'.format(service)).splitlines():
        ret.append(svc)
    return sorted(ret)

def start(name):
    if False:
        return 10
    "\n    Start the named service.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = '{} -f start {}'.format(_cmd(), name)
    return not __salt__['cmd.retcode'](cmd)

def stop(name):
    if False:
        return 10
    "\n    Stop the named service.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = '{} stop {}'.format(_cmd(), name)
    return not __salt__['cmd.retcode'](cmd)

def restart(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Restart the named service.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = '{} -f restart {}'.format(_cmd(), name)
    return not __salt__['cmd.retcode'](cmd)

def reload_(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reload the named service.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = '{} reload {}'.format(_cmd(), name)
    return not __salt__['cmd.retcode'](cmd)

def status(name, sig=None):
    if False:
        return 10
    "\n    Return the status for a service, returns a bool whether the service is\n    running.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.status <service name>\n    "
    if sig:
        return bool(__salt__['status.pid'](sig))
    cmd = '{} check {}'.format(_cmd(), name)
    return not __salt__['cmd.retcode'](cmd, ignore_retcode=True)

def enable(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Enable the named service to start at boot.\n\n    flags : None\n        Set optional flags to run the service with.\n\n    service.flags can be used to change the default flags.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name>\n        salt '*' service.enable <service name> flags=<flags>\n    "
    stat_cmd = '{} set {} status on'.format(_cmd(), name)
    stat_retcode = __salt__['cmd.retcode'](stat_cmd)
    flag_retcode = None
    if os.path.exists('/etc/rc.d/{}'.format(name)):
        flags = _get_flags(**kwargs)
        flag_cmd = '{} set {} flags {}'.format(_cmd(), name, flags)
        flag_retcode = __salt__['cmd.retcode'](flag_cmd)
    return not any([stat_retcode, flag_retcode])

def disable(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Disable the named service to not start at boot.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name>\n    "
    cmd = '{} set {} status off'.format(_cmd(), name)
    return not __salt__['cmd.retcode'](cmd)

def disabled(name):
    if False:
        i = 10
        return i + 15
    "\n    Return True if the named service is disabled at boot, False otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    cmd = '{} get {} status'.format(_cmd(), name)
    return not __salt__['cmd.retcode'](cmd, ignore_retcode=True) == 0

def enabled(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return True if the named service is enabled at boot and the provided\n    flags match the configured ones (if any). Return False otherwise.\n\n    name\n        Service name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n        salt '*' service.enabled <service name> flags=<flags>\n    "
    cmd = '{} get {} status'.format(_cmd(), name)
    if not __salt__['cmd.retcode'](cmd, ignore_retcode=True):
        flags = _get_flags(**kwargs)
        cur_flags = __salt__['cmd.run_stdout']('{} get {} flags'.format(_cmd(), name))
        if format(flags) == format(cur_flags):
            return True
        if not flags:
            def_flags = __salt__['cmd.run_stdout']('{} getdef {} flags'.format(_cmd(), name))
            if format(cur_flags) == format(def_flags):
                return True
    return False