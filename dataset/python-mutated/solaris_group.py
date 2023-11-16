"""
Manage groups on Solaris

.. important::
    If you feel that Salt should be using this module to manage groups on a
    minion, and it is using a different module (or gives an error similar to
    *'group.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import logging
import salt.utils.data
log = logging.getLogger(__name__)
try:
    import grp
except ImportError:
    pass
__virtualname__ = 'group'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set the group module if the kernel is SunOS\n    '
    if __grains__['kernel'] == 'SunOS':
        return __virtualname__
    return (False, 'The solaris_group execution module failed to load: only available on Solaris systems.')

def add(name, gid=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Add the specified group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.add foo 3456\n    "
    if salt.utils.data.is_true(kwargs.pop('system', False)):
        log.warning("solaris_group module does not support the 'system' argument")
    if kwargs:
        log.warning('Invalid kwargs passed to group.add')
    cmd = 'groupadd '
    if gid:
        cmd += '-g {} '.format(gid)
    cmd += name
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    return not ret['retcode']

def delete(name):
    if False:
        while True:
            i = 10
    "\n    Remove the named group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.delete foo\n    "
    ret = __salt__['cmd.run_all']('groupdel {}'.format(name), python_shell=False)
    return not ret['retcode']

def info(name):
    if False:
        print('Hello World!')
    "\n    Return information about a group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.info foo\n    "
    try:
        grinfo = grp.getgrnam(name)
    except KeyError:
        return {}
    else:
        return {'name': grinfo.gr_name, 'passwd': grinfo.gr_passwd, 'gid': grinfo.gr_gid, 'members': grinfo.gr_mem}

def getent(refresh=False):
    if False:
        return 10
    "\n    Return info on all groups\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.getent\n    "
    if 'group.getent' in __context__ and (not refresh):
        return __context__['group.getent']
    ret = []
    for grinfo in grp.getgrall():
        ret.append(info(grinfo.gr_name))
    __context__['group.getent'] = ret
    return ret

def chgid(name, gid):
    if False:
        for i in range(10):
            print('nop')
    "\n    Change the gid for a named group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.chgid foo 4376\n    "
    pre_gid = __salt__['file.group_to_gid'](name)
    if gid == pre_gid:
        return True
    cmd = 'groupmod -g {} {}'.format(gid, name)
    __salt__['cmd.run'](cmd, python_shell=False)
    post_gid = __salt__['file.group_to_gid'](name)
    if post_gid != pre_gid:
        return post_gid == gid
    return False