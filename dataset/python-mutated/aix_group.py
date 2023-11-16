"""
Manage groups on Solaris

.. important::
    If you feel that Salt should be using this module to manage groups on a
    minion, and it is using a different module (or gives an error similar to
    *'group.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import logging
try:
    import grp
except ImportError:
    pass
log = logging.getLogger(__name__)
__virtualname__ = 'group'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the group module if the kernel is AIX\n    '
    if __grains__['kernel'] == 'AIX':
        return __virtualname__
    return (False, 'The aix_group execution module failed to load: only available on AIX systems.')

def add(name, gid=None, system=False, root=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Add the specified group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.add foo 3456\n    "
    cmd = 'mkgroup '
    if system and root is not None:
        cmd += '-a '
    if gid:
        cmd += 'id={} '.format(gid)
    cmd += name
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    return not ret['retcode']

def delete(name):
    if False:
        while True:
            i = 10
    "\n    Remove the named group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.delete foo\n    "
    ret = __salt__['cmd.run_all']('rmgroup {}'.format(name), python_shell=False)
    return not ret['retcode']

def info(name):
    if False:
        return 10
    "\n    Return information about a group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.info foo\n    "
    try:
        grinfo = grp.getgrnam(name)
    except KeyError:
        return {}
    else:
        return {'name': grinfo.gr_name, 'passwd': grinfo.gr_passwd, 'gid': grinfo.gr_gid, 'members': grinfo.gr_mem}

def getent(refresh=False):
    if False:
        for i in range(10):
            print('nop')
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
        while True:
            i = 10
    "\n    Change the gid for a named group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.chgid foo 4376\n    "
    pre_gid = __salt__['file.group_to_gid'](name)
    if gid == pre_gid:
        return True
    cmd = 'chgroup id={} {}'.format(gid, name)
    __salt__['cmd.run'](cmd, python_shell=False)
    post_gid = __salt__['file.group_to_gid'](name)
    if post_gid != pre_gid:
        return post_gid == gid
    return False

def adduser(name, username, root=None):
    if False:
        return 10
    "\n    Add a user in the group.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' group.adduser foo bar\n\n    Verifies if a valid username 'bar' as a member of an existing group 'foo',\n    if not then adds it.\n    "
    cmd = 'chgrpmem -m + {} {}'.format(username, name)
    retcode = __salt__['cmd.retcode'](cmd, python_shell=False)
    return not retcode

def deluser(name, username, root=None):
    if False:
        print('Hello World!')
    "\n    Remove a user from the group.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' group.deluser foo bar\n\n    Removes a member user 'bar' from a group 'foo'. If group is not present\n    then returns True.\n    "
    grp_info = __salt__['group.info'](name)
    try:
        if username in grp_info['members']:
            cmd = 'chgrpmem -m - {} {}'.format(username, name)
            ret = __salt__['cmd.run'](cmd, python_shell=False)
            return not ret['retcode']
        else:
            return True
    except Exception:
        return True

def members(name, members_list, root=None):
    if False:
        return 10
    "\n    Replaces members of the group with a provided list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.members foo 'user1,user2,user3,...'\n\n    Replaces a membership list for a local group 'foo'.\n        foo:x:1234:user1,user2,user3,...\n    "
    cmd = 'chgrpmem -m = {} {}'.format(members_list, name)
    retcode = __salt__['cmd.retcode'](cmd, python_shell=False)
    return not retcode