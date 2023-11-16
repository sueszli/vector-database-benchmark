"""
Support for Layman
"""
import salt.exceptions
import salt.utils.path

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only work on Gentoo systems with layman installed\n    '
    if __grains__['os'] == 'Gentoo' and salt.utils.path.which('layman'):
        return 'layman'
    return (False, 'layman execution module cannot be loaded: only available on Gentoo with layman installed.')

def _get_makeconf():
    if False:
        return 10
    '\n    Find the correct make.conf. Gentoo recently moved the make.conf\n    but still supports the old location, using the old location first\n    '
    old_conf = '/etc/make.conf'
    new_conf = '/etc/portage/make.conf'
    if __salt__['file.file_exists'](old_conf):
        return old_conf
    elif __salt__['file.file_exists'](new_conf):
        return new_conf

def add(overlay):
    if False:
        return 10
    "\n    Add the given overlay from the cached remote list to your locally\n    installed overlays. Specify 'ALL' to add all overlays from the\n    remote list.\n\n    Return a list of the new overlay(s) added:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' layman.add <overlay name>\n    "
    ret = list()
    old_overlays = list_local()
    cmd = 'layman --quietness=0 --add {}'.format(overlay)
    add_attempt = __salt__['cmd.run_all'](cmd, python_shell=False, stdin='y')
    if add_attempt['retcode'] != 0:
        raise salt.exceptions.CommandExecutionError(add_attempt['stdout'])
    new_overlays = list_local()
    if not old_overlays and new_overlays:
        srcline = 'source /var/lib/layman/make.conf'
        makeconf = _get_makeconf()
        if not __salt__['file.contains'](makeconf, 'layman'):
            __salt__['file.append'](makeconf, srcline)
    ret = [overlay for overlay in new_overlays if overlay not in old_overlays]
    return ret

def delete(overlay):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove the given overlay from the your locally installed overlays.\n    Specify 'ALL' to remove all overlays.\n\n    Return a list of the overlays(s) that were removed:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' layman.delete <overlay name>\n    "
    ret = list()
    old_overlays = list_local()
    cmd = 'layman --quietness=0 --delete {}'.format(overlay)
    delete_attempt = __salt__['cmd.run_all'](cmd, python_shell=False)
    if delete_attempt['retcode'] != 0:
        raise salt.exceptions.CommandExecutionError(delete_attempt['stdout'])
    new_overlays = list_local()
    if not new_overlays:
        srcline = 'source /var/lib/layman/make.conf'
        makeconf = _get_makeconf()
        if __salt__['file.contains'](makeconf, 'layman'):
            __salt__['file.sed'](makeconf, srcline, '')
    ret = [overlay for overlay in old_overlays if overlay not in new_overlays]
    return ret

def sync(overlay='ALL'):
    if False:
        while True:
            i = 10
    "\n    Update the specified overlay. Use 'ALL' to synchronize all overlays.\n    This is the default if no overlay is specified.\n\n    overlay\n        Name of the overlay to sync. (Defaults to 'ALL')\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' layman.sync\n    "
    cmd = 'layman --quietness=0 --sync {}'.format(overlay)
    return __salt__['cmd.retcode'](cmd, python_shell=False) == 0

def list_local():
    if False:
        for i in range(10):
            print('nop')
    "\n    List the locally installed overlays.\n\n    Return a list of installed overlays:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' layman.list_local\n    "
    cmd = 'layman --quietness=1 --list-local --nocolor'
    out = __salt__['cmd.run'](cmd, python_shell=False).split('\n')
    ret = [line.split()[1] for line in out if len(line.split()) > 2]
    return ret

def list_all():
    if False:
        while True:
            i = 10
    "\n    List all overlays, including remote ones.\n\n    Return a list of available overlays:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' layman.list_all\n    "
    cmd = 'layman --quietness=1 --list --nocolor'
    out = __salt__['cmd.run'](cmd, python_shell=False).split('\n')
    ret = [line.split()[1] for line in out if len(line.split()) > 2]
    return ret