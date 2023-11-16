"""
Module for the management of upstart systems. The Upstart system only supports
service starting, stopping and restarting.

.. important::
    If you feel that Salt should be using this module to manage services on a
    minion, and it is using a different module (or gives an error similar to
    *'service.start' is not available*), see :ref:`here
    <module-provider-override>`.

Currently (as of Ubuntu 12.04) there is no tool available to disable
Upstart services (like update-rc.d). This[1] is the recommended way to
disable an Upstart service. So we assume that all Upstart services
that have not been disabled in this manner are enabled.

But this is broken because we do not check to see that the dependent
services are enabled. Otherwise we would have to do something like
parse the output of "initctl show-config" to determine if all service
dependencies are enabled to start on boot. For example, see the "start
on" condition for the lightdm service below[2]. And this would be too
hard. So we wait until the upstart developers have solved this
problem. :) This is to say that an Upstart service that is enabled may
not really be enabled.

Also, when an Upstart service is enabled, should the dependent
services be enabled too? Probably not. But there should be a notice
about this, at least.

[1] http://upstart.ubuntu.com/cookbook/#disabling-a-job-from-automatically-starting

[2] example upstart configuration file::

    lightdm
    emits login-session-start
    emits desktop-session-start
    emits desktop-shutdown
    start on ((((filesystem and runlevel [!06]) and started dbus) and (drm-device-added card0 PRIMARY_DEVICE_FOR_DISPLAY=1 or stopped udev-fallback-graphics)) or runlevel PREVLEVEL=S)
    stop on runlevel [016]

.. warning::
    This module should not be used on Red Hat systems. For these,
    the :mod:`rh_service <salt.modules.rh_service>` module should be
    used, as it supports the hybrid upstart/sysvinit system used in
    RHEL/CentOS 6.
"""
import fnmatch
import glob
import os
import re
import salt.modules.cmdmod
import salt.utils.files
import salt.utils.path
import salt.utils.systemd
__func_alias__ = {'reload_': 'reload'}
__virtualname__ = 'service'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on Ubuntu\n    '
    if salt.utils.systemd.booted(__context__):
        return (False, 'The upstart execution module failed to load: this system was booted with systemd.')
    elif __grains__['os'] in ('Ubuntu', 'Linaro', 'elementary OS', 'Mint'):
        return __virtualname__
    elif __grains__['os'] in ('Debian', 'Raspbian'):
        debian_initctl = '/sbin/initctl'
        if os.path.isfile(debian_initctl):
            initctl_version = salt.modules.cmdmod._run_quiet(debian_initctl + ' version')
            if 'upstart' in initctl_version:
                return __virtualname__
    return (False, 'The upstart execution module failed to load:  the system must be Ubuntu-based, or Debian-based with upstart support.')

def _find_utmp():
    if False:
        i = 10
        return i + 15
    "\n    Figure out which utmp file to use when determining runlevel.\n    Sometimes /var/run/utmp doesn't exist, /run/utmp is the new hotness.\n    "
    result = {}
    for utmp in ('/var/run/utmp', '/run/utmp'):
        try:
            result[os.stat(utmp).st_mtime] = utmp
        except Exception:
            pass
    if result:
        return result[sorted(result).pop()]
    else:
        return False

def _default_runlevel():
    if False:
        while True:
            i = 10
    '\n    Try to figure out the default runlevel.  It is kept in\n    /etc/init/rc-sysinit.conf, but can be overridden with entries\n    in /etc/inittab, or via the kernel command-line at boot\n    '
    try:
        with salt.utils.files.fopen('/etc/init/rc-sysinit.conf') as fp_:
            for line in fp_:
                line = salt.utils.stringutils.to_unicode(line)
                if line.startswith('env DEFAULT_RUNLEVEL'):
                    runlevel = line.split('=')[-1].strip()
    except Exception:
        return '2'
    try:
        with salt.utils.files.fopen('/etc/inittab') as fp_:
            for line in fp_:
                line = salt.utils.stringutils.to_unicode(line)
                if not line.startswith('#') and 'initdefault' in line:
                    runlevel = line.split(':')[1]
    except Exception:
        pass
    try:
        valid_strings = {'0', '1', '2', '3', '4', '5', '6', 's', 'S', '-s', 'single'}
        with salt.utils.files.fopen('/proc/cmdline') as fp_:
            for line in fp_:
                line = salt.utils.stringutils.to_unicode(line)
                for arg in line.strip().split():
                    if arg in valid_strings:
                        runlevel = arg
                        break
    except Exception:
        pass
    return runlevel

def _runlevel():
    if False:
        i = 10
        return i + 15
    '\n    Return the current runlevel\n    '
    if 'upstart._runlevel' in __context__:
        return __context__['upstart._runlevel']
    ret = _default_runlevel()
    utmp = _find_utmp()
    if utmp:
        out = __salt__['cmd.run'](['runlevel', '{}'.format(utmp)], python_shell=False)
        try:
            ret = out.split()[1]
        except IndexError:
            pass
    __context__['upstart._runlevel'] = ret
    return ret

def _is_symlink(name):
    if False:
        for i in range(10):
            print('nop')
    return os.path.abspath(name) != os.path.realpath(name)

def _service_is_upstart(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    From "Writing Jobs" at\n    http://upstart.ubuntu.com/getting-started.html:\n\n    Jobs are defined in files placed in /etc/init, the name of the job\n    is the filename under this directory without the .conf extension.\n    '
    return os.access('/etc/init/{}.conf'.format(name), os.R_OK)

def _upstart_is_disabled(name):
    if False:
        print('Hello World!')
    '\n    An Upstart service is assumed disabled if a manual stanza is\n    placed in /etc/init/[name].override.\n    NOTE: An Upstart service can also be disabled by placing "manual"\n    in /etc/init/[name].conf.\n    '
    files = ['/etc/init/{}.conf'.format(name), '/etc/init/{}.override'.format(name)]
    for file_name in filter(os.path.isfile, files):
        with salt.utils.files.fopen(file_name) as fp_:
            if re.search('^\\s*manual', salt.utils.stringutils.to_unicode(fp_.read()), re.MULTILINE):
                return True
    return False

def _upstart_is_enabled(name):
    if False:
        print('Hello World!')
    '\n    Assume that if an Upstart service is not disabled then it must be\n    enabled.\n    '
    return not _upstart_is_disabled(name)

def _service_is_sysv(name):
    if False:
        i = 10
        return i + 15
    "\n    A System-V style service will have a control script in\n    /etc/init.d. We make sure to skip over symbolic links that point\n    to Upstart's /lib/init/upstart-job, and anything that isn't an\n    executable, like README or skeleton.\n    "
    script = '/etc/init.d/{}'.format(name)
    return not _service_is_upstart(name) and os.access(script, os.X_OK)

def _sysv_is_disabled(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    A System-V style service is assumed disabled if there is no\n    start-up link (starts with "S") to its script in /etc/init.d in\n    the current runlevel.\n    '
    return not bool(glob.glob('/etc/rc{}.d/S*{}'.format(_runlevel(), name)))

def _sysv_is_enabled(name):
    if False:
        print('Hello World!')
    '\n    Assume that if a System-V style service is not disabled then it\n    must be enabled.\n    '
    return not _sysv_is_disabled(name)

def _iter_service_names():
    if False:
        print('Hello World!')
    '\n    Detect all of the service names available to upstart via init configuration\n    files and via classic sysv init scripts\n    '
    found = set()
    for line in glob.glob('/etc/init.d/*'):
        name = os.path.basename(line)
        found.add(name)
        yield name
    init_root = '/etc/init/'
    for (root, dirnames, filenames) in salt.utils.path.os_walk(init_root):
        relpath = os.path.relpath(root, init_root)
        for filename in fnmatch.filter(filenames, '*.conf'):
            if relpath == '.':
                name = filename[:-5]
            else:
                name = os.path.join(relpath, filename[:-5])
            if name in found:
                continue
            yield name

def get_enabled():
    if False:
        i = 10
        return i + 15
    "\n    Return the enabled services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_enabled\n    "
    ret = set()
    for name in _iter_service_names():
        if _service_is_upstart(name):
            if _upstart_is_enabled(name):
                ret.add(name)
        elif _service_is_sysv(name):
            if _sysv_is_enabled(name):
                ret.add(name)
    return sorted(ret)

def get_disabled():
    if False:
        return 10
    "\n    Return the disabled services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_disabled\n    "
    ret = set()
    for name in _iter_service_names():
        if _service_is_upstart(name):
            if _upstart_is_disabled(name):
                ret.add(name)
        elif _service_is_sysv(name):
            if _sysv_is_disabled(name):
                ret.add(name)
    return sorted(ret)

def available(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns ``True`` if the specified service is available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.available sshd\n    "
    return name in get_all()

def missing(name):
    if False:
        print('Hello World!')
    "\n    The inverse of service.available.\n    Returns ``True`` if the specified service is not available, otherwise returns\n    ``False``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.missing sshd\n    "
    return name not in get_all()

def get_all():
    if False:
        while True:
            i = 10
    "\n    Return all installed services\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.get_all\n    "
    return sorted(get_enabled() + get_disabled())

def start(name):
    if False:
        print('Hello World!')
    "\n    Start the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.start <service name>\n    "
    cmd = ['service', name, 'start']
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def stop(name):
    if False:
        return 10
    "\n    Stop the specified service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.stop <service name>\n    "
    cmd = ['service', name, 'stop']
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def restart(name):
    if False:
        return 10
    "\n    Restart the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.restart <service name>\n    "
    cmd = ['service', name, 'restart']
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def full_restart(name):
    if False:
        return 10
    "\n    Do a full restart (stop/start) of the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.full_restart <service name>\n    "
    cmd = ['service', name, '--full-restart']
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def reload_(name):
    if False:
        return 10
    "\n    Reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.reload <service name>\n    "
    cmd = ['service', name, 'reload']
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def force_reload(name):
    if False:
        return 10
    "\n    Force-reload the named service\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.force_reload <service name>\n    "
    cmd = ['service', name, 'force-reload']
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

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
        cmd = ['service', service, 'status']
        if _service_is_upstart(service):
            results[service] = 'start/running' in __salt__['cmd.run'](cmd, python_shell=False, ignore_retcode=True)
        else:
            results[service] = not bool(__salt__['cmd.retcode'](cmd, python_shell=False, ignore_retcode=True, quite=True))
    if contains_globbing:
        return results
    return results[name]

def _get_service_exec():
    if False:
        for i in range(10):
            print('nop')
    '\n    Debian uses update-rc.d to manage System-V style services.\n    http://www.debian.org/doc/debian-policy/ch-opersys.html#s9.3.3\n    '
    executable = 'update-rc.d'
    salt.utils.path.check_or_die(executable)
    return executable

def _upstart_disable(name):
    if False:
        i = 10
        return i + 15
    '\n    Disable an Upstart service.\n    '
    if _upstart_is_disabled(name):
        return _upstart_is_disabled(name)
    override = '/etc/init/{}.override'.format(name)
    with salt.utils.files.fopen(override, 'a') as ofile:
        ofile.write(salt.utils.stringutils.to_str('manual\n'))
    return _upstart_is_disabled(name)

def _upstart_enable(name):
    if False:
        print('Hello World!')
    '\n    Enable an Upstart service.\n    '
    if _upstart_is_enabled(name):
        return _upstart_is_enabled(name)
    override = '/etc/init/{}.override'.format(name)
    files = ['/etc/init/{}.conf'.format(name), override]
    for file_name in filter(os.path.isfile, files):
        with salt.utils.files.fopen(file_name, 'r+') as fp_:
            new_text = re.sub('^\\s*manual\\n?', '', salt.utils.stringutils.to_unicode(fp_.read()), 0, re.MULTILINE)
            fp_.seek(0)
            fp_.write(salt.utils.stringutils.to_str(new_text))
            fp_.truncate()
    if os.access(override, os.R_OK) and os.path.getsize(override) == 0:
        os.unlink(override)
    return _upstart_is_enabled(name)

def enable(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Enable the named service to start at boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enable <service name>\n    "
    if _service_is_upstart(name):
        return _upstart_enable(name)
    executable = _get_service_exec()
    cmd = '{} -f {} defaults'.format(executable, name)
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def disable(name, **kwargs):
    if False:
        return 10
    "\n    Disable the named service from starting on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disable <service name>\n    "
    if _service_is_upstart(name):
        return _upstart_disable(name)
    executable = _get_service_exec()
    cmd = [executable, '-f', name, 'remove']
    return not __salt__['cmd.retcode'](cmd, python_shell=False)

def enabled(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Check to see if the named service is enabled to start on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.enabled <service name>\n    "
    if _service_is_upstart(name):
        return _upstart_is_enabled(name)
    elif _service_is_sysv(name):
        return _sysv_is_enabled(name)
    return None

def disabled(name):
    if False:
        i = 10
        return i + 15
    "\n    Check to see if the named service is disabled to start on boot\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' service.disabled <service name>\n    "
    if _service_is_upstart(name):
        return _upstart_is_disabled(name)
    elif _service_is_sysv(name):
        return _sysv_is_disabled(name)
    return None