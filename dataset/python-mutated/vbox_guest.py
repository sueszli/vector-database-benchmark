"""
VirtualBox Guest Additions installer
"""
import contextlib
import functools
import glob
import logging
import os
import re
import tempfile
log = logging.getLogger(__name__)
__virtualname__ = 'vbox_guest'
_additions_dir_prefix = 'VBoxGuestAdditions'
_shared_folders_group = 'vboxsf'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Set the vbox_guest module if the OS Linux\n    '
    if __grains__.get('kernel', '') not in ('Linux',):
        return (False, 'The vbox_guest execution module failed to load: only available on Linux systems.')
    return __virtualname__

def additions_mount():
    if False:
        while True:
            i = 10
    "\n    Mount VirtualBox Guest Additions CD to the temp directory.\n\n    To connect VirtualBox Guest Additions via VirtualBox graphical interface\n    press 'Host+D' ('Host' is usually 'Right Ctrl').\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vbox_guest.additions_mount\n\n    :return: True or OSError exception\n    "
    mount_point = tempfile.mkdtemp()
    ret = __salt__['mount.mount'](mount_point, '/dev/cdrom')
    if ret is True:
        return mount_point
    else:
        raise OSError(ret)

def additions_umount(mount_point):
    if False:
        print('Hello World!')
    "\n    Unmount VirtualBox Guest Additions CD from the temp directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vbox_guest.additions_umount\n\n    :param mount_point: directory VirtualBox Guest Additions is mounted to\n    :return: True or an string with error\n    "
    ret = __salt__['mount.umount'](mount_point)
    if ret:
        os.rmdir(mount_point)
    return ret

@contextlib.contextmanager
def _additions_mounted():
    if False:
        while True:
            i = 10
    mount_point = additions_mount()
    yield mount_point
    additions_umount(mount_point)

def _return_mount_error(f):
    if False:
        i = 10
        return i + 15

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            return f(*args, **kwargs)
        except OSError as e:
            return str(e)
    return wrapper

def _additions_install_program_path(mount_point):
    if False:
        while True:
            i = 10
    return os.path.join(mount_point, {'Linux': 'VBoxLinuxAdditions.run', 'Solaris': 'VBoxSolarisAdditions.pkg', 'Windows': 'VBoxWindowsAdditions.exe'}[__grains__.get('kernel', '')])

def _additions_install_opensuse(**kwargs):
    if False:
        i = 10
        return i + 15
    kernel_type = re.sub('^(\\d|\\.|-)*', '', __grains__.get('kernelrelease', ''))
    kernel_devel = 'kernel-{}-devel'.format(kernel_type)
    return __states__['pkg.installed'](None, pkgs=['make', 'gcc', kernel_devel])

def _additions_install_ubuntu(**kwargs):
    if False:
        while True:
            i = 10
    return __states__['pkg.installed'](None, pkgs=['dkms'])

def _additions_install_fedora(**kwargs):
    if False:
        while True:
            i = 10
    return __states__['pkg.installed'](None, pkgs=['dkms', 'gcc'])

def _additions_install_linux(mount_point, **kwargs):
    if False:
        print('Hello World!')
    reboot = kwargs.pop('reboot', False)
    restart_x11 = kwargs.pop('restart_x11', False)
    upgrade_os = kwargs.pop('upgrade_os', False)
    if upgrade_os:
        __salt__['pkg.upgrade']()
    guest_os = __grains__.get('os', '')
    if guest_os == 'openSUSE':
        _additions_install_opensuse(**kwargs)
    elif guest_os == 'ubuntu':
        _additions_install_ubuntu(**kwargs)
    elif guest_os == 'fedora':
        _additions_install_fedora(**kwargs)
    else:
        log.warning('%s is not fully supported yet.', guest_os)
    installer_path = _additions_install_program_path(mount_point)
    installer_ret = __salt__['cmd.run_all'](installer_path)
    if installer_ret['retcode'] in (0, 1):
        if reboot:
            __salt__['system.reboot']()
        elif restart_x11:
            raise NotImplementedError('Restarting x11 is not supported yet.')
        else:
            pass
        return additions_version()
    elif installer_ret['retcode'] in (127, '127'):
        return "'{}' not found on CD. Make sure that VirtualBox Guest Additions CD is attached to the CD IDE Controller.".format(os.path.basename(installer_path))
    else:
        return installer_ret['stderr']

@_return_mount_error
def additions_install(**kwargs):
    if False:
        print('Hello World!')
    "\n    Install VirtualBox Guest Additions. Uses the CD, connected by VirtualBox.\n\n    To connect VirtualBox Guest Additions via VirtualBox graphical interface\n    press 'Host+D' ('Host' is usually 'Right Ctrl').\n\n    See https://www.virtualbox.org/manual/ch04.html#idp52733088 for more details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vbox_guest.additions_install\n        salt '*' vbox_guest.additions_install reboot=True\n        salt '*' vbox_guest.additions_install upgrade_os=True\n\n    :param reboot: reboot computer to complete installation\n    :type reboot: bool\n    :param upgrade_os: upgrade OS (to ensure the latests version of kernel and developer tools are installed)\n    :type upgrade_os: bool\n    :return: version of VirtualBox Guest Additions or string with error\n    "
    with _additions_mounted() as mount_point:
        kernel = __grains__.get('kernel', '')
        if kernel == 'Linux':
            return _additions_install_linux(mount_point, **kwargs)

def _additions_dir():
    if False:
        i = 10
        return i + 15
    root = '/opt'
    dirs = glob.glob(os.path.join(root, _additions_dir_prefix) + '*')
    if dirs:
        return dirs[0]
    else:
        raise OSError('No VirtualBox Guest Additions dirs found!')

def _additions_remove_linux_run(cmd):
    if False:
        i = 10
        return i + 15
    uninstaller_ret = __salt__['cmd.run_all'](cmd)
    return uninstaller_ret['retcode'] in (0,)

def _additions_remove_linux(**kwargs):
    if False:
        i = 10
        return i + 15
    try:
        return _additions_remove_linux_run(os.path.join(_additions_dir(), 'uninstall.sh'))
    except OSError:
        return False

def _additions_remove_linux_use_cd(mount_point, **kwargs):
    if False:
        return 10
    force = kwargs.pop('force', False)
    args = ''
    if force:
        args += '--force'
    return _additions_remove_linux_run('{program} uninstall {args}'.format(program=_additions_install_program_path(mount_point), args=args))

@_return_mount_error
def _additions_remove_use_cd(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove VirtualBox Guest Additions.\n\n    It uses the CD, connected by VirtualBox.\n    '
    with _additions_mounted() as mount_point:
        kernel = __grains__.get('kernel', '')
        if kernel == 'Linux':
            return _additions_remove_linux_use_cd(mount_point, **kwargs)

def additions_remove(**kwargs):
    if False:
        return 10
    "\n    Remove VirtualBox Guest Additions.\n\n    Firstly it tries to uninstall itself by executing\n    '/opt/VBoxGuestAdditions-VERSION/uninstall.run uninstall'.\n    It uses the CD, connected by VirtualBox if it failes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vbox_guest.additions_remove\n        salt '*' vbox_guest.additions_remove force=True\n\n    :param force: force VirtualBox Guest Additions removing\n    :type force: bool\n    :return: True if VirtualBox Guest Additions were removed successfully else False\n\n    "
    kernel = __grains__.get('kernel', '')
    if kernel == 'Linux':
        ret = _additions_remove_linux()
    if not ret:
        ret = _additions_remove_use_cd(**kwargs)
    return ret

def additions_version():
    if False:
        return 10
    "\n    Check VirtualBox Guest Additions version.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vbox_guest.additions_version\n\n    :return: version of VirtualBox Guest Additions or False if they are not installed\n    "
    try:
        d = _additions_dir()
    except OSError:
        return False
    if d and len(os.listdir(d)) > 0:
        return re.sub('^{}-'.format(_additions_dir_prefix), '', os.path.basename(d))
    return False

def grant_access_to_shared_folders_to(name, users=None):
    if False:
        print('Hello World!')
    "\n    Grant access to auto-mounted shared folders to the users.\n\n    User is specified by its name. To grant access for several users use argument `users`.\n    Access will be denied to the users not listed in `users` argument.\n\n    See https://www.virtualbox.org/manual/ch04.html#sf_mount_auto for more details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vbox_guest.grant_access_to_shared_folders_to fred\n        salt '*' vbox_guest.grant_access_to_shared_folders_to users ['fred', 'roman']\n\n    :param name: name of the user to grant access to auto-mounted shared folders to\n    :type name: str\n    :param users: list of names of users to grant access to auto-mounted shared folders to (if specified, `name` will not be taken into account)\n    :type users: list of str\n    :return: list of users who have access to auto-mounted shared folders\n    "
    if users is None:
        users = [name]
    if __salt__['group.members'](_shared_folders_group, ','.join(users)):
        return users
    elif not __salt__['group.info'](_shared_folders_group):
        if not additions_version:
            return 'VirtualBox Guest Additions are not installed. Ιnstall them firstly. You can do it with the help of command vbox_guest.additions_install.'
        else:
            return "VirtualBox Guest Additions seems to be installed, but group '{}' not found. Check your installation and fix it. You can uninstall VirtualBox Guest Additions with the help of command :py:func:`vbox_guest.additions_remove <salt.modules.vbox_guest.additions_remove> (it has `force` argument to fix complex situations; use it with care) and then install it again. You can do it with the help of :py:func:`vbox_guest.additions_install <salt.modules.vbox_guest.additions_install>`.".format(_shared_folders_group)
    else:
        return "Cannot replace members of the '{}' group.".format(_shared_folders_group)

def list_shared_folders_users():
    if False:
        i = 10
        return i + 15
    "\n    List users who have access to auto-mounted shared folders.\n\n    See https://www.virtualbox.org/manual/ch04.html#sf_mount_auto for more details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' vbox_guest.list_shared_folders_users\n\n    :return: list of users who have access to auto-mounted shared folders\n    "
    try:
        return __salt__['group.info'](_shared_folders_group)['members']
    except KeyError:
        return []