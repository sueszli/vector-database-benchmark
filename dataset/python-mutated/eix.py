"""
Support for Eix
"""
import salt.utils.path

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only works on Gentoo systems with eix installed\n    '
    if __grains__['os'] == 'Gentoo' and salt.utils.path.which('eix'):
        return 'eix'
    return (False, 'The eix execution module cannot be loaded: either the system is not Gentoo or the eix binary is not in the path.')

def sync():
    if False:
        i = 10
        return i + 15
    "\n    Sync portage/overlay trees and update the eix database\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' eix.sync\n    "
    cmd = 'eix-sync -q -C "--ask" -C "n"'
    if 'makeconf.features_contains' in __salt__ and __salt__['makeconf.features_contains']('webrsync-gpg'):
        if salt.utils.path.which('emerge-delta-webrsync'):
            cmd += ' -W'
        else:
            cmd += ' -w'
        return __salt__['cmd.retcode'](cmd) == 0
    else:
        if __salt__['cmd.retcode'](cmd) == 0:
            return True
        if salt.utils.path.which('emerge-delta-webrsync'):
            cmd += ' -W'
        else:
            cmd += ' -w'
        return __salt__['cmd.retcode'](cmd) == 0

def update():
    if False:
        print('Hello World!')
    "\n    Update the eix database\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' eix.update\n    "
    cmd = 'eix-update --quiet'
    return __salt__['cmd.retcode'](cmd) == 0