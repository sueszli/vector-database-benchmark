"""
Manage groups on Linux, OpenBSD and NetBSD

.. important::
    If you feel that Salt should be using this module to manage groups on a
    minion, and it is using a different module (or gives an error similar to
    *'group.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import functools
import logging
import os
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError
try:
    import grp
except ImportError:
    pass
log = logging.getLogger(__name__)
__virtualname__ = 'group'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Set the user module if the kernel is Linux or OpenBSD\n    '
    if __grains__['kernel'] in ('Linux', 'OpenBSD', 'NetBSD'):
        return __virtualname__
    return (False, 'The groupadd execution module cannot be loaded:  only available on Linux, OpenBSD and NetBSD')

def _which(cmd):
    if False:
        print('Hello World!')
    '\n    Utility function wrapper to error out early if a command is not found\n    '
    _cmd = salt.utils.path.which(cmd)
    if not _cmd:
        raise CommandExecutionError(f"Command '{cmd}' cannot be found")
    return _cmd

def add(name, gid=None, system=False, root=None, non_unique=False, local=False):
    if False:
        while True:
            i = 10
    "\n    .. versionchanged:: 3006.0\n\n    Add the specified group\n\n    name\n        Name of the new group\n\n    gid\n        Use GID for the new group\n\n    system\n        Create a system account\n\n    root\n        Directory to chroot into\n\n    non_unique\n        Allow creating groups with duplicate (non-unique) GIDs\n\n        .. versionadded:: 3006.0\n\n    local\n        Specifically add the group locally rather than through remote providers (e.g. LDAP)\n\n        .. versionadded:: 3007.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.add foo 3456\n    "
    cmd = [_which('lgroupadd' if local else 'groupadd')]
    if gid:
        cmd.append(f'-g {gid}')
        if non_unique and (not local):
            cmd.append('-o')
    if system and __grains__['kernel'] != 'OpenBSD':
        cmd.append('-r')
    if root is not None and (not local):
        cmd.extend(('-R', root))
    cmd.append(name)
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    return not ret['retcode']

def delete(name, root=None, local=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove the named group\n\n    name\n        Name group to delete\n\n    root\n        Directory to chroot into\n\n    local (Only on systems with lgroupdel available):\n        Ensure the group account is removed locally ignoring global\n        account management (default is False).\n\n        .. versionadded:: 3007.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.delete foo\n    "
    cmd = [_which('lgroupdel' if local else 'groupdel')]
    if root is not None and (not local):
        cmd.extend(('-R', root))
    cmd.append(name)
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    return not ret['retcode']

def info(name, root=None):
    if False:
        return 10
    "\n    Return information about a group\n\n    name\n        Name of the group\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.info foo\n    "
    if root is not None:
        getgrnam = functools.partial(_getgrnam, root=root)
    else:
        getgrnam = functools.partial(grp.getgrnam)
    try:
        grinfo = getgrnam(name)
    except KeyError:
        return {}
    else:
        return _format_info(grinfo)

def _format_info(data):
    if False:
        i = 10
        return i + 15
    '\n    Return formatted information in a pretty way.\n    '
    return {'name': data.gr_name, 'passwd': data.gr_passwd, 'gid': data.gr_gid, 'members': data.gr_mem}

def getent(refresh=False, root=None):
    if False:
        return 10
    "\n    Return info on all groups\n\n    refresh\n        Force a refresh of group information\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.getent\n    "
    if 'group.getent' in __context__ and (not refresh):
        return __context__['group.getent']
    ret = []
    if root is not None:
        getgrall = functools.partial(_getgrall, root=root)
    else:
        getgrall = functools.partial(grp.getgrall)
    for grinfo in getgrall():
        ret.append(_format_info(grinfo))
    __context__['group.getent'] = ret
    return ret

def _chattrib(name, key, value, param, root=None):
    if False:
        while True:
            i = 10
    '\n    Change an attribute for a named user\n    '
    pre_info = info(name, root=root)
    if not pre_info:
        return False
    if value == pre_info[key]:
        return True
    cmd = [_which('groupmod')]
    if root is not None:
        cmd.extend(('-R', root))
    cmd.extend((param, value, name))
    __salt__['cmd.run'](cmd, python_shell=False)
    return info(name, root=root).get(key) == value

def chgid(name, gid, root=None, non_unique=False):
    if False:
        return 10
    "\n    .. versionchanged:: 3006.0\n\n    Change the gid for a named group\n\n    name\n        Name of the group to modify\n\n    gid\n        Change the group ID to GID\n\n    root\n        Directory to chroot into\n\n    non_unique\n        Allow modifying groups with duplicate (non-unique) GIDs\n\n        .. versionadded:: 3006.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.chgid foo 4376\n    "
    param = '-g'
    if non_unique:
        param = '-og'
    return _chattrib(name, 'gid', gid, param, root=root)

def adduser(name, username, root=None):
    if False:
        while True:
            i = 10
    "\n    Add a user in the group.\n\n    name\n        Name of the group to modify\n\n    username\n        Username to add to the group\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' group.adduser foo bar\n\n    Verifies if a valid username 'bar' as a member of an existing group 'foo',\n    if not then adds it.\n    "
    on_suse_11 = __grains__.get('os_family') == 'Suse' and __grains__.get('osmajorrelease') == '11'
    if __grains__['kernel'] == 'Linux':
        if on_suse_11:
            cmd = [_which('usermod'), '-A', name, username]
        else:
            cmd = [_which('gpasswd'), '--add', username, name]
        if root is not None:
            cmd.extend(('--root', root))
    else:
        cmd = [_which('usermod'), '-G', name, username]
        if root is not None:
            cmd.extend(('-R', root))
    retcode = __salt__['cmd.retcode'](cmd, python_shell=False)
    return not retcode

def deluser(name, username, root=None):
    if False:
        i = 10
        return i + 15
    "\n    Remove a user from the group.\n\n    name\n        Name of the group to modify\n\n    username\n        Username to delete from the group\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n         salt '*' group.deluser foo bar\n\n    Removes a member user 'bar' from a group 'foo'. If group is not present\n    then returns True.\n    "
    on_suse_11 = __grains__.get('os_family') == 'Suse' and __grains__.get('osmajorrelease') == '11'
    grp_info = __salt__['group.info'](name)
    try:
        if username in grp_info['members']:
            if __grains__['kernel'] == 'Linux':
                if on_suse_11:
                    cmd = [_which('usermod'), '-R', name, username]
                else:
                    cmd = [_which('gpasswd'), '--del', username, name]
                if root is not None:
                    cmd.extend(('--root', root))
                retcode = __salt__['cmd.retcode'](cmd, python_shell=False)
            elif __grains__['kernel'] == 'OpenBSD':
                out = __salt__['cmd.run_stdout'](f'id -Gn {username}', python_shell=False)
                cmd = [_which('usermod'), '-S']
                cmd.append(','.join([g for g in out.split() if g != str(name)]))
                cmd.append(f'{username}')
                retcode = __salt__['cmd.retcode'](cmd, python_shell=False)
            else:
                log.error('group.deluser is not yet supported on this platform')
                return False
            return not retcode
        else:
            return True
    except CommandExecutionError:
        raise
    except Exception:
        return True

def members(name, members_list, root=None):
    if False:
        return 10
    "\n    Replaces members of the group with a provided list.\n\n    name\n        Name of the group to modify\n\n    members_list\n        Username list to set into the group\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' group.members foo 'user1,user2,user3,...'\n\n    Replaces a membership list for a local group 'foo'.\n        foo:x:1234:user1,user2,user3,...\n    "
    on_suse_11 = __grains__.get('os_family') == 'Suse' and __grains__.get('osmajorrelease') == '11'
    if __grains__['kernel'] == 'Linux':
        if on_suse_11:
            for old_member in __salt__['group.info'](name).get('members'):
                __salt__['cmd.run']('{} -R {} {}'.format(_which('groupmod'), old_member, name), python_shell=False)
            cmd = [_which('groupmod'), '-A', members_list, name]
        else:
            cmd = [_which('gpasswd'), '--members', members_list, name]
        if root is not None:
            cmd.extend(('--root', root))
        retcode = __salt__['cmd.retcode'](cmd, python_shell=False)
    elif __grains__['kernel'] == 'OpenBSD':
        retcode = 1
        grp_info = __salt__['group.info'](name)
        if grp_info and name in grp_info['name']:
            __salt__['cmd.run']('{} {}'.format(_which('groupdel'), name), python_shell=False)
            __salt__['cmd.run']('{} -g {} {}'.format(_which('groupadd'), grp_info['gid'], name), python_shell=False)
            for user in members_list.split(','):
                if user:
                    retcode = __salt__['cmd.retcode']([_which('usermod'), '-G', name, user], python_shell=False)
                    if not retcode == 0:
                        break
                else:
                    retcode = 0
    else:
        log.error('group.members is not yet supported on this platform')
        return False
    return not retcode

def _getgrnam(name, root=None):
    if False:
        i = 10
        return i + 15
    '\n    Alternative implementation for getgrnam, that use only /etc/group\n    '
    root = root or '/'
    passwd = os.path.join(root, 'etc/group')
    with salt.utils.files.fopen(passwd) as fp_:
        for line in fp_:
            line = salt.utils.stringutils.to_unicode(line)
            comps = line.strip().split(':')
            if len(comps) < 4:
                log.debug('Ignoring group line: %s', line)
                continue
            if comps[0] == name:
                comps[2] = int(comps[2])
                comps[3] = comps[3].split(',') if comps[3] else []
                return grp.struct_group(comps)
    raise KeyError(f'getgrnam(): name not found: {name}')

def _getgrall(root=None):
    if False:
        i = 10
        return i + 15
    '\n    Alternative implemetantion for getgrall, that use only /etc/group\n    '
    root = root or '/'
    passwd = os.path.join(root, 'etc/group')
    with salt.utils.files.fopen(passwd) as fp_:
        for line in fp_:
            line = salt.utils.stringutils.to_unicode(line)
            comps = line.strip().split(':')
            if len(comps) < 4:
                log.debug('Ignoring group line: %s', line)
                continue
            comps[2] = int(comps[2])
            comps[3] = comps[3].split(',') if comps[3] else []
            yield grp.struct_group(comps)