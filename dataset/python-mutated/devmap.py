"""
Device-Mapper module
"""
import os.path

def multipath_list():
    if False:
        i = 10
        return i + 15
    "\n    Device-Mapper Multipath list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' devmap.multipath_list\n    "
    cmd = 'multipath -l'
    return __salt__['cmd.run'](cmd).splitlines()

def multipath_flush(device):
    if False:
        print('Hello World!')
    "\n    Device-Mapper Multipath flush\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' devmap.multipath_flush mpath1\n    "
    if not os.path.exists(device):
        return '{} does not exist'.format(device)
    cmd = 'multipath -f {}'.format(device)
    return __salt__['cmd.run'](cmd).splitlines()