"""
Manage groups on FreeBSD

.. important::
    If you feel that Salt should be using this module to manage groups on a
    minion, and it is using a different module (or gives an error similar to
    *'group.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import logging
import salt.utils.args
import salt.utils.data
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
    '\n    Set the user module if the kernel is FreeBSD or Dragonfly\n    '
    if __grains__['kernel'] in ('FreeBSD', 'DragonFly'):
        return __virtualname__
    return (False, 'The pw_group execution module cannot be loaded: system is not supported.')

def add(name, gid=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionchanged:: 3006.0\n\n    Add the specified group\n\n    name\n        Name of the new group\n\n    gid\n        Use GID for the new group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.add foo 3456\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    if salt.utils.data.is_true(kwargs.pop('system', False)):
        log.warning("pw_group module does not support the 'system' argument")
    if 'non_unique' in kwargs:
        log.warning('The non_unique parameter is not supported on this platform.')
    if kwargs:
        log.warning('Invalid kwargs passed to group.add')
    cmd = 'pw groupadd '
    if gid:
        cmd += '-g {} '.format(gid)
    cmd = '{} -n {}'.format(cmd, name)
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    return not ret['retcode']

def delete(name):
    if False:
        return 10
    "\n    Remove the named group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.delete foo\n    "
    ret = __salt__['cmd.run_all']('pw groupdel {}'.format(name), python_shell=False)
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
        while True:
            i = 10
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
        i = 10
        return i + 15
    "\n    Change the gid for a named group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.chgid foo 4376\n    "
    pre_gid = __salt__['file.group_to_gid'](name)
    if gid == pre_gid:
        return True
    cmd = 'pw groupmod {} -g {}'.format(name, gid)
    __salt__['cmd.run'](cmd, python_shell=False)
    post_gid = __salt__['file.group_to_gid'](name)
    if post_gid != pre_gid:
        return post_gid == gid
    return False

def adduser(name, username):
    if False:
        i = 10
        return i + 15
    "\n    Add a user in the group.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' group.adduser foo bar\n\n    Verifies if a valid username 'bar' as a member of an existing group 'foo',\n    if not then adds it.\n    "
    retcode = __salt__['cmd.retcode']('pw groupmod {} -m {}'.format(name, username), python_shell=False)
    return not retcode

def deluser(name, username):
    if False:
        i = 10
        return i + 15
    "\n    Remove a user from the group.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' group.deluser foo bar\n\n    Removes a member user 'bar' from a group 'foo'. If group is not present\n    then returns True.\n    "
    grp_info = __salt__['group.info'](name)
    if username not in grp_info['members']:
        return True
    retcode = __salt__['cmd.retcode']('pw groupmod {} -d {}'.format(name, username), python_shell=False)
    return not retcode

def members(name, members_list):
    if False:
        print('Hello World!')
    "\n    Replaces members of the group with a provided list.\n\n    .. versionadded:: 2015.5.4\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.members foo 'user1,user2,user3,...'\n\n    Replaces a membership list for a local group 'foo'.\n        foo:x:1234:user1,user2,user3,...\n    "
    retcode = __salt__['cmd.retcode']('pw groupmod {} -M {}'.format(name, members_list), python_shell=False)
    return not retcode