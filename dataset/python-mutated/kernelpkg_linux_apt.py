"""
Manage Linux kernel packages on APT-based systems
"""
import functools
import logging
import re
from salt.exceptions import CommandExecutionError
from salt.utils.versions import LooseVersion
log = logging.getLogger(__name__)
__virtualname__ = 'kernelpkg'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Load this module on Debian-based systems only\n    '
    if __grains__.get('os_family', '') in ('Kali', 'Debian'):
        return __virtualname__
    elif __grains__.get('os_family', '') == 'Cumulus':
        return __virtualname__
    return (False, 'Module kernelpkg_linux_apt: no APT based system detected')

def active():
    if False:
        i = 10
        return i + 15
    "\n    Return the version of the running kernel.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.active\n    "
    if 'pkg.normalize_name' in __salt__:
        return __salt__['pkg.normalize_name'](__grains__['kernelrelease'])
    return __grains__['kernelrelease']

def list_installed():
    if False:
        print('Hello World!')
    "\n    Return a list of all installed kernels.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.list_installed\n    "
    pkg_re = re.compile('^{}-[\\d.-]+-{}$'.format(_package_prefix(), _kernel_type()))
    pkgs = __salt__['pkg.list_pkgs'](versions_as_list=True)
    if pkgs is None:
        pkgs = []
    result = list(filter(pkg_re.match, pkgs))
    if result is None:
        return []
    prefix_len = len(_package_prefix()) + 1
    return sorted((pkg[prefix_len:] for pkg in result), key=functools.cmp_to_key(_cmp_version))

def latest_available():
    if False:
        return 10
    "\n    Return the version of the latest kernel from the package repositories.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.latest_available\n    "
    result = __salt__['pkg.latest_version']('{}-{}'.format(_package_prefix(), _kernel_type()))
    if result == '':
        return latest_installed()
    version = re.match('^(\\d+\\.\\d+\\.\\d+)\\.(\\d+)', result)
    return '{}-{}-{}'.format(version.group(1), version.group(2), _kernel_type())

def latest_installed():
    if False:
        while True:
            i = 10
    "\n    Return the version of the latest installed kernel.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.latest_installed\n\n    .. note::\n        This function may not return the same value as\n        :py:func:`~salt.modules.kernelpkg_linux_apt.active` if a new kernel\n        has been installed and the system has not yet been rebooted.\n        The :py:func:`~salt.modules.kernelpkg_linux_apt.needs_reboot` function\n        exists to detect this condition.\n    "
    pkgs = list_installed()
    if pkgs:
        return pkgs[-1]
    return None

def needs_reboot():
    if False:
        return 10
    "\n    Detect if a new kernel version has been installed but is not running.\n    Returns True if a new kernel is installed, False otherwise.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.needs_reboot\n    "
    return LooseVersion(active()) < LooseVersion(latest_installed())

def upgrade(reboot=False, at_time=None):
    if False:
        while True:
            i = 10
    "\n    Upgrade the kernel and optionally reboot the system.\n\n    reboot : False\n        Request a reboot if a new kernel is available.\n\n    at_time : immediate\n        Schedule the reboot at some point in the future. This argument\n        is ignored if ``reboot=False``. See\n        :py:func:`~salt.modules.system.reboot` for more details\n        on this argument.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.upgrade\n        salt '*' kernelpkg.upgrade reboot=True at_time=1\n\n    .. note::\n        An immediate reboot often shuts down the system before the minion has a\n        chance to return, resulting in errors. A minimal delay (1 minute) is\n        useful to ensure the result is delivered to the master.\n    "
    result = __salt__['pkg.install'](name='{}-{}'.format(_package_prefix(), latest_available()))
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
    "\n    Remove a specific version of the kernel.\n\n    release\n        The release number of an installed kernel. This must be the entire release\n        number as returned by :py:func:`~salt.modules.kernelpkg_linux_apt.list_installed`,\n        not the package name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.remove 4.4.0-70-generic\n    "
    if release not in list_installed():
        raise CommandExecutionError("Kernel release '{}' is not installed".format(release))
    if release == active():
        raise CommandExecutionError('Active kernel cannot be removed')
    target = '{}-{}'.format(_package_prefix(), release)
    log.info('Removing kernel package %s', target)
    __salt__['pkg.purge'](target)
    return {'removed': [target]}

def cleanup(keep_latest=True):
    if False:
        return 10
    "\n    Remove all unused kernel packages from the system.\n\n    keep_latest : True\n        In the event that the active kernel is not the latest one installed, setting this to True\n        will retain the latest kernel package, in addition to the active one. If False, all kernel\n        packages other than the active one will be removed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' kernelpkg.cleanup\n    "
    removed = []
    for kernel in list_installed():
        if kernel == active():
            continue
        if keep_latest and kernel == latest_installed():
            continue
        removed.extend(remove(kernel)['removed'])
    return {'removed': removed}

def _package_prefix():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return static string for the package prefix\n    '
    return 'linux-image'

def _kernel_type():
    if False:
        i = 10
        return i + 15
    '\n    Parse the kernel name and return its type\n    '
    return re.match('^[\\d.-]+-(.+)$', active()).group(1)

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