"""
Manage groups on Mac OS 10.7+
"""
import logging
import salt.utils.functools
import salt.utils.itertools
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError, SaltInvocationError
from salt.modules.mac_user import _dscl, _flush_dscl_cache
try:
    import grp
except ImportError:
    pass
log = logging.getLogger(__name__)
__virtualname__ = 'group'

def __virtual__():
    if False:
        while True:
            i = 10
    global _dscl, _flush_dscl_cache
    if __grains__.get('kernel') != 'Darwin' or __grains__['osrelease_info'] < (10, 7):
        return (False, 'The mac_group execution module cannot be loaded: only available on Darwin-based systems >= 10.7')
    _dscl = salt.utils.functools.namespaced_function(_dscl, globals())
    _flush_dscl_cache = salt.utils.functools.namespaced_function(_flush_dscl_cache, globals())
    return __virtualname__

def add(name, gid=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionchanged:: 3006.0\n\n    Add the specified group\n\n    name\n        Name of the new group\n\n    gid\n        Use GID for the new group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.add foo 3456\n    "
    if info(name):
        raise CommandExecutionError("Group '{}' already exists".format(name))
    if salt.utils.stringutils.contains_whitespace(name):
        raise SaltInvocationError('Group name cannot contain whitespace')
    if name.startswith('_'):
        raise SaltInvocationError('Salt will not create groups beginning with underscores')
    if gid is not None and (not isinstance(gid, int)):
        raise SaltInvocationError('gid must be an integer')
    if 'non_unique' in kwargs:
        log.warning('The non_unique parameter is not supported on this platform.')
    gid_list = _list_gids()
    if str(gid) in gid_list:
        raise CommandExecutionError("gid '{}' already exists".format(gid))
    cmd = ['dseditgroup', '-o', 'create']
    if gid:
        cmd.extend(['-i', gid])
    cmd.append(name)
    return __salt__['cmd.retcode'](cmd, python_shell=False) == 0

def _list_gids():
    if False:
        print('Hello World!')
    '\n    Return a list of gids in use\n    '
    output = __salt__['cmd.run'](['dscacheutil', '-q', 'group'], output_loglevel='quiet', python_shell=False)
    ret = set()
    for line in salt.utils.itertools.split(output, '\n'):
        if line.startswith('gid:'):
            ret.update(line.split()[1:])
    return sorted(ret)

def delete(name):
    if False:
        i = 10
        return i + 15
    "\n    Remove the named group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.delete foo\n    "
    if salt.utils.stringutils.contains_whitespace(name):
        raise SaltInvocationError('Group name cannot contain whitespace')
    if name.startswith('_'):
        raise SaltInvocationError('Salt will not remove groups beginning with underscores')
    if not info(name):
        return True
    cmd = ['dseditgroup', '-o', 'delete', name]
    return __salt__['cmd.retcode'](cmd, python_shell=False) == 0

def adduser(group, name):
    if False:
        while True:
            i = 10
    "\n    Add a user in the group.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' group.adduser foo bar\n\n    Verifies if a valid username 'bar' as a member of an existing group 'foo',\n    if not then adds it.\n    "
    cmd = 'dscl . -merge /Groups/{} GroupMembership {}'.format(group, name)
    return __salt__['cmd.retcode'](cmd) == 0

def deluser(group, name):
    if False:
        return 10
    "\n    Remove a user from the group\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' group.deluser foo bar\n\n    Removes a member user 'bar' from a group 'foo'. If group is not present\n    then returns True.\n    "
    cmd = 'dscl . -delete /Groups/{} GroupMembership {}'.format(group, name)
    return __salt__['cmd.retcode'](cmd) == 0

def members(name, members_list):
    if False:
        while True:
            i = 10
    "\n    Replaces members of the group with a provided list.\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.members foo 'user1,user2,user3,...'\n\n    Replaces a membership list for a local group 'foo'.\n    "
    retcode = 1
    grp_info = __salt__['group.info'](name)
    if grp_info and name in grp_info['name']:
        cmd = '/usr/bin/dscl . -delete /Groups/{} GroupMembership'.format(name)
        retcode = __salt__['cmd.retcode'](cmd) == 0
        for user in members_list.split(','):
            cmd = '/usr/bin/dscl . -merge /Groups/{} GroupMembership {}'.format(name, user)
            retcode = __salt__['cmd.retcode'](cmd)
            if not retcode == 0:
                break
            else:
                retcode = 0
    return retcode == 0

def info(name):
    if False:
        return 10
    "\n    Return information about a group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.info foo\n    "
    if salt.utils.stringutils.contains_whitespace(name):
        raise SaltInvocationError('Group name cannot contain whitespace')
    try:
        grinfo = next(iter((x for x in grp.getgrall() if x.gr_name == name)))
    except StopIteration:
        return {}
    else:
        return _format_info(grinfo)

def _format_info(data):
    if False:
        return 10
    '\n    Return formatted information in a pretty way.\n    '
    return {'name': data.gr_name, 'gid': data.gr_gid, 'passwd': data.gr_passwd, 'members': data.gr_mem}

def getent(refresh=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return info on all groups\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.getent\n    "
    if 'group.getent' in __context__ and (not refresh):
        return __context__['group.getent']
    ret = []
    for grinfo in grp.getgrall():
        if not grinfo.gr_name.startswith('_'):
            ret.append(_format_info(grinfo))
    __context__['group.getent'] = ret
    return ret

def chgid(name, gid):
    if False:
        for i in range(10):
            print('nop')
    "\n    Change the gid for a named group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.chgid foo 4376\n    "
    if not isinstance(gid, int):
        raise SaltInvocationError('gid must be an integer')
    pre_gid = __salt__['file.group_to_gid'](name)
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("Group '{}' does not exist".format(name))
    if gid == pre_info['gid']:
        return True
    cmd = ['dseditgroup', '-o', 'edit', '-i', gid, name]
    return __salt__['cmd.retcode'](cmd, python_shell=False) == 0