"""
Manage Linux kernel packages on YUM-based systems
"""
import functools
import logging
import salt.modules.yumpkg
import salt.utils.data
import salt.utils.functools
import salt.utils.systemd
from salt.exceptions import CommandExecutionError
from salt.utils.versions import LooseVersion
log = logging.getLogger(__name__)
__virtualname__ = 'kernelpkg'
_yum = salt.utils.functools.namespaced_function(salt.modules.yumpkg._yum, globals())

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Load this module on RedHat-based systems only\n    '
    if __grains__.get('os_family', '') == 'RedHat':
        return __virtualname__
    elif __grains__.get('os', '').lower() in ('amazon', 'xcp', 'xenserver', 'virtuozzolinux'):
        return __virtualname__
    return (False, 'Module kernelpkg_linux_yum: no YUM based system detected')

def active():
    if False:
        return 10
    "\n    Return the version of the running kernel.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.active\n    "
    if 'pkg.normalize_name' in __salt__:
        return __salt__['pkg.normalize_name'](__grains__['kernelrelease'])
    return __grains__['kernelrelease']

def list_installed():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of all installed kernels.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.list_installed\n    "
    result = __salt__['pkg.version'](_package_name(), versions_as_list=True)
    if result is None:
        return []
    return sorted(result, key=functools.cmp_to_key(_cmp_version))

def latest_available():
    if False:
        while True:
            i = 10
    "\n    Return the version of the latest kernel from the package repositories.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.latest_available\n    "
    result = __salt__['pkg.latest_version'](_package_name())
    if result == '':
        result = latest_installed()
    return result

def latest_installed():
    if False:
        return 10
    "\n    Return the version of the latest installed kernel.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.latest_installed\n\n    .. note::\n        This function may not return the same value as\n        :py:func:`~salt.modules.kernelpkg_linux_yum.active` if a new kernel\n        has been installed and the system has not yet been rebooted.\n        The :py:func:`~salt.modules.kernelpkg_linux_yum.needs_reboot` function\n        exists to detect this condition.\n    "
    pkgs = list_installed()
    if pkgs:
        return pkgs[-1]
    return None

def needs_reboot():
    if False:
        for i in range(10):
            print('nop')
    "\n    Detect if a new kernel version has been installed but is not running.\n    Returns True if a new kernel is installed, False otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.needs_reboot\n    "
    return LooseVersion(active()) < LooseVersion(latest_installed())

def upgrade(reboot=False, at_time=None):
    if False:
        while True:
            i = 10
    "\n    Upgrade the kernel and optionally reboot the system.\n\n    reboot : False\n        Request a reboot if a new kernel is available.\n\n    at_time : immediate\n        Schedule the reboot at some point in the future. This argument\n        is ignored if ``reboot=False``. See\n        :py:func:`~salt.modules.system.reboot` for more details\n        on this argument.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.upgrade\n        salt '*' kernelpkg.upgrade reboot=True at_time=1\n\n    .. note::\n        An immediate reboot often shuts down the system before the minion has a\n        chance to return, resulting in errors. A minimal delay (1 minute) is\n        useful to ensure the result is delivered to the master.\n    "
    result = __salt__['pkg.upgrade'](name=_package_name())
    _needs_reboot = needs_reboot()
    ret = {'upgrades': result, 'active': active(), 'latest_installed': latest_installed(), 'reboot_requested': reboot, 'reboot_required': _needs_reboot}
    if reboot and _needs_reboot:
        log.warning('Rebooting system due to kernel upgrade')
        __salt__['system.reboot'](at_time=at_time)
    return ret

def upgrade_available():
    if False:
        for i in range(10):
            print('nop')
    "\n    Detect if a new kernel version is available in the repositories.\n    Returns True if a new kernel is available, False otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.upgrade_available\n    "
    return LooseVersion(latest_available()) > LooseVersion(latest_installed())

def remove(release):
    if False:
        i = 10
        return i + 15
    "\n    Remove a specific version of the kernel.\n\n    release\n        The release number of an installed kernel. This must be the entire release\n        number as returned by :py:func:`~salt.modules.kernelpkg_linux_yum.list_installed`,\n        not the package name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.remove 3.10.0-327.el7\n    "
    if release not in list_installed():
        raise CommandExecutionError("Kernel release '{}' is not installed".format(release))
    if release == active():
        raise CommandExecutionError('Active kernel cannot be removed')
    target = '{}-{}'.format(_package_name(), release)
    log.info('Removing kernel package %s', target)
    old = __salt__['pkg.list_pkgs']()
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.extend([_yum(), '-y', 'remove', target])
    out = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    __context__.pop('pkg.list_pkgs', None)
    new = __salt__['pkg.list_pkgs']()
    ret = salt.utils.data.compare_dicts(old, new)
    if out['retcode'] != 0:
        raise CommandExecutionError('Error occurred removing package(s)', info={'errors': [out['stderr']], 'changes': ret})
    return {'removed': [target]}

def cleanup(keep_latest=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove all unused kernel packages from the system.\n\n    keep_latest : True\n        In the event that the active kernel is not the latest one installed, setting this to True\n        will retain the latest kernel package, in addition to the active one. If False, all kernel\n        packages other than the active one will be removed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.cleanup\n    "
    removed = []
    for kernel in list_installed():
        if kernel == active():
            continue
        if keep_latest and kernel == latest_installed():
            continue
        removed.extend(remove(kernel)['removed'])
    return {'removed': removed}

def _package_name():
    if False:
        while True:
            i = 10
    '\n    Return static string for the package name\n    '
    return 'kernel'

def _cmp_version(item1, item2):
    if False:
        i = 10
        return i + 15
    '\n    Compare function for package version sorting\n    '
    vers1 = LooseVersion(item1)
    vers2 = LooseVersion(item2)
    if vers1 < vers2:
        return -1
    if vers1 > vers2:
        return 1
    return 0