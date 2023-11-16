"""
Manage users with the useradd command

.. important::
    If you feel that Salt should be using this module to manage users on a
    minion, and it is using a different module (or gives an error similar to
    *'user.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import copy
import functools
import logging
import os
import salt.utils.data
import salt.utils.decorators.path
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
import salt.utils.user
from salt.exceptions import CommandExecutionError
try:
    import pwd
    HAS_PWD = True
except ImportError:
    HAS_PWD = False
log = logging.getLogger(__name__)
__virtualname__ = 'user'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set the user module if the kernel is Linux, OpenBSD, NetBSD or AIX\n    '
    if HAS_PWD and __grains__['kernel'] in ('Linux', 'OpenBSD', 'NetBSD', 'AIX'):
        return __virtualname__
    return (False, 'useradd execution module not loaded: either pwd python library not available or system not one of Linux, OpenBSD, NetBSD or AIX')

def _quote_username(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Usernames can only contain ascii chars, so make sure we return a str type\n    '
    if not isinstance(name, str):
        return str(name)
    else:
        return salt.utils.stringutils.to_str(name)

def _get_gecos(name, root=None):
    if False:
        i = 10
        return i + 15
    '\n    Retrieve GECOS field info and return it in dictionary form\n    '
    if root is not None and __grains__['kernel'] != 'AIX':
        getpwnam = functools.partial(_getpwnam, root=root)
    else:
        getpwnam = functools.partial(pwd.getpwnam)
    gecos_field = salt.utils.stringutils.to_unicode(getpwnam(_quote_username(name)).pw_gecos).split(',', 4)
    if not gecos_field:
        return {}
    else:
        while len(gecos_field) < 5:
            gecos_field.append('')
        return {'fullname': salt.utils.data.decode(gecos_field[0]), 'roomnumber': salt.utils.data.decode(gecos_field[1]), 'workphone': salt.utils.data.decode(gecos_field[2]), 'homephone': salt.utils.data.decode(gecos_field[3]), 'other': salt.utils.data.decode(gecos_field[4])}

def _build_gecos(gecos_dict):
    if False:
        print('Hello World!')
    '\n    Accepts a dictionary entry containing GECOS field names and their values,\n    and returns a full GECOS comment string, to be used with usermod.\n    '
    return '{},{},{},{},{}'.format(gecos_dict.get('fullname', ''), gecos_dict.get('roomnumber', ''), gecos_dict.get('workphone', ''), gecos_dict.get('homephone', ''), gecos_dict.get('other', '')).rstrip(',')

def _which(cmd):
    if False:
        while True:
            i = 10
    '\n    Utility function wrapper to error out early if a command is not found\n    '
    _cmd = salt.utils.path.which(cmd)
    if not _cmd:
        raise CommandExecutionError(f"Command '{cmd}' cannot be found")
    return _cmd

def _update_gecos(name, key, value, root=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Common code to change a user's GECOS information\n    "
    if value is None:
        value = ''
    elif not isinstance(value, str):
        value = str(value)
    else:
        value = salt.utils.stringutils.to_unicode(value)
    pre_info = _get_gecos(name, root=root)
    if not pre_info:
        return False
    if value == pre_info[key]:
        return True
    gecos_data = copy.deepcopy(pre_info)
    gecos_data[key] = value
    cmd = [_which('usermod')]
    if root is not None and __grains__['kernel'] != 'AIX':
        cmd.extend(('-R', root))
    cmd.extend(('-c', _build_gecos(gecos_data), name))
    __salt__['cmd.run'](cmd, python_shell=False)
    return _get_gecos(name, root=root).get(key) == value

def add(name, uid=None, gid=None, groups=None, home=None, shell=None, unique=True, system=False, fullname='', roomnumber='', workphone='', homephone='', other='', createhome=True, loginclass=None, nologinit=False, root=None, usergroup=None, local=False):
    if False:
        while True:
            i = 10
    "\n    Add a user to the minion\n\n    name\n        Username LOGIN to add\n\n    uid\n        User ID of the new account\n\n    gid\n        Name or ID of the primary group of the new account\n\n    groups\n        List of supplementary groups of the new account\n\n    home\n        Home directory of the new account\n\n    shell\n        Login shell of the new account\n\n    unique\n        If not True, the user account can have a non-unique UID\n\n    system\n        Create a system account\n\n    fullname\n        GECOS field for the full name\n\n    roomnumber\n        GECOS field for the room number\n\n    workphone\n        GECOS field for the work phone\n\n    homephone\n        GECOS field for the home phone\n\n    other\n        GECOS field for other information\n\n    createhome\n        Create the user's home directory\n\n    loginclass\n        Login class for the new account (OpenBSD)\n\n    nologinit\n        Do not add the user to the lastlog and faillog databases\n\n    root\n        Directory to chroot into\n\n    usergroup\n        Create and add the user to a new primary group of the same name\n\n    local (Only on systems with luseradd available)\n        Specifically add the user locally rather than possibly through remote providers (e.g. LDAP)\n\n        .. versionadded:: 3007.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.add name <uid> <gid> <groups> <home> <shell>\n    "
    cmd = [_which('luseradd' if local else 'useradd')]
    if shell:
        cmd.extend(['-s', shell])
    if uid not in (None, ''):
        cmd.extend(['-u', uid])
    if gid not in (None, ''):
        cmd.extend(['-g', gid])
    elif usergroup:
        if not local:
            cmd.append('-U')
            if __grains__['kernel'] != 'Linux':
                log.warning("'usergroup' is only supported on GNU/Linux hosts.")
    elif groups is not None and name in groups:
        defs_file = '/etc/login.defs'
        if __grains__['kernel'] != 'OpenBSD':
            try:
                with salt.utils.files.fopen(defs_file) as fp_:
                    for line in fp_:
                        line = salt.utils.stringutils.to_unicode(line)
                        if 'USERGROUPS_ENAB' not in line[:15]:
                            continue
                        if 'yes' in line:
                            cmd.extend(['-g', __salt__['file.group_to_gid'](name)])
                        break
            except OSError:
                log.debug('Error reading %s', defs_file, exc_info_on_loglevel=logging.DEBUG)
        else:
            usermgmt_file = '/etc/usermgmt.conf'
            try:
                with salt.utils.files.fopen(usermgmt_file) as fp_:
                    for line in fp_:
                        line = salt.utils.stringutils.to_unicode(line)
                        if 'group' not in line[:5]:
                            continue
                        cmd.extend(['-g', line.split()[-1]])
                        break
            except OSError:
                pass
    if usergroup is False:
        cmd.append('-n' if local else '-N')
    if createhome:
        if not local:
            cmd.append('-m')
    elif __grains__['kernel'] != 'NetBSD' and __grains__['kernel'] != 'OpenBSD':
        cmd.append('-M')
    if nologinit:
        cmd.append('-l')
    if home is not None:
        cmd.extend(['-d', home])
    if not unique and __grains__['kernel'] != 'AIX':
        cmd.append('-o')
    if system and __grains__['kernel'] != 'NetBSD' and (__grains__['kernel'] != 'OpenBSD'):
        cmd.append('-r')
    if __grains__['kernel'] == 'OpenBSD':
        if loginclass is not None:
            cmd.extend(['-L', loginclass])
    cmd.append(name)
    if root is not None and (not local) and (__grains__['kernel'] != 'AIX'):
        cmd.extend(('-R', root))
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] != 0:
        return False
    if groups:
        chgroups(name, groups, root=root)
    if fullname:
        chfullname(name, fullname, root=root)
    if roomnumber:
        chroomnumber(name, roomnumber, root=root)
    if workphone:
        chworkphone(name, workphone, root=root)
    if homephone:
        chhomephone(name, homephone, root=root)
    if other:
        chother(name, other, root=root)
    return True

def delete(name, remove=False, force=False, root=None, local=False):
    if False:
        return 10
    "\n    Remove a user from the minion\n\n    name\n        Username to delete\n\n    remove\n        Remove home directory and mail spool\n\n    force\n        Force some actions that would fail otherwise\n\n    root\n        Directory to chroot into\n\n    local (Only on systems with luserdel available):\n        Ensure the user account is removed locally ignoring global\n        account management (default is False).\n\n        .. versionadded:: 3007.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.delete name remove=True force=True\n    "
    cmd = [_which('luserdel' if local else 'userdel')]
    if remove:
        cmd.append('-r')
    if force and __grains__['kernel'] != 'OpenBSD' and (__grains__['kernel'] != 'AIX') and (not local):
        cmd.append('-f')
    cmd.append(name)
    if root is not None and __grains__['kernel'] != 'AIX' and (not local):
        cmd.extend(('-R', root))
    ret = __salt__['cmd.run_all'](cmd, python_shell=False)
    if ret['retcode'] == 0:
        return True
    if ret['retcode'] == 12:
        if __grains__['os_family'] not in ('Debian',):
            return False
        if 'var/mail' in ret['stderr'] or 'var/spool/mail' in ret['stderr']:
            log.debug('While the userdel exited with code 12, this is a known bug on debian based distributions. See http://goo.gl/HH3FzT')
            return True
    return False

def getent(refresh=False, root=None):
    if False:
        return 10
    "\n    Return the list of all info for all users\n\n    refresh\n        Force a refresh of user information\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.getent\n    "
    if 'user.getent' in __context__ and (not refresh):
        return __context__['user.getent']
    ret = []
    if root is not None and __grains__['kernel'] != 'AIX':
        getpwall = functools.partial(_getpwall, root=root)
    else:
        getpwall = functools.partial(pwd.getpwall)
    for data in getpwall():
        ret.append(_format_info(data))
    __context__['user.getent'] = ret
    return ret

def _chattrib(name, key, value, param, persist=False, root=None):
    if False:
        print('Hello World!')
    '\n    Change an attribute for a named user\n    '
    pre_info = info(name, root=root)
    if not pre_info:
        raise CommandExecutionError(f"User '{name}' does not exist")
    if value == pre_info[key]:
        return True
    cmd = [_which('usermod')]
    if root is not None and __grains__['kernel'] != 'AIX':
        cmd.extend(('-R', root))
    if persist and __grains__['kernel'] != 'OpenBSD':
        cmd.append('-m')
    cmd.extend((param, value, name))
    __salt__['cmd.run'](cmd, python_shell=False)
    return info(name, root=root).get(key) == value

def chuid(name, uid, root=None):
    if False:
        i = 10
        return i + 15
    "\n    Change the uid for a named user\n\n    name\n        User to modify\n\n    uid\n        New UID for the user account\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chuid foo 4376\n    "
    return _chattrib(name, 'uid', uid, '-u', root=root)

def chgid(name, gid, root=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Change the default group of the user\n\n    name\n        User to modify\n\n    gid\n        Force use GID as new primary group\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chgid foo 4376\n    "
    return _chattrib(name, 'gid', gid, '-g', root=root)

def chshell(name, shell, root=None):
    if False:
        i = 10
        return i + 15
    "\n    Change the default shell of the user\n\n    name\n        User to modify\n\n    shell\n        New login shell for the user account\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chshell foo /bin/zsh\n    "
    return _chattrib(name, 'shell', shell, '-s', root=root)

def chhome(name, home, persist=False, root=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Change the home directory of the user, pass True for persist to move files\n    to the new home directory if the old home directory exist.\n\n    name\n        User to modify\n\n    home\n        New home directory for the user account\n\n    persist\n        Move contents of the home directory to the new location\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chhome foo /home/users/foo True\n    "
    return _chattrib(name, 'home', home, '-d', persist=persist, root=root)

def chgroups(name, groups, append=False, root=None):
    if False:
        i = 10
        return i + 15
    "\n    Change the groups to which this user belongs\n\n    name\n        User to modify\n\n    groups\n        Groups to set for the user\n\n    append : False\n        If ``True``, append the specified group(s). Otherwise, this function\n        will replace the user's groups with the specified group(s).\n\n    root\n        Directory to chroot into\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' user.chgroups foo wheel,root\n        salt '*' user.chgroups foo wheel,root append=True\n    "
    if isinstance(groups, str):
        groups = groups.split(',')
    ugrps = set(list_groups(name))
    if ugrps == set(groups):
        return True
    cmd = [_which('usermod')]
    if __grains__['kernel'] != 'OpenBSD':
        if append and __grains__['kernel'] != 'AIX':
            cmd.append('-a')
        cmd.append('-G')
    elif append:
        cmd.append('-G')
    else:
        cmd.append('-S')
    if append and __grains__['kernel'] == 'AIX':
        cmd.extend([','.join(ugrps) + ',' + ','.join(groups), name])
    else:
        cmd.extend([','.join(groups), name])
    if root is not None and __grains__['kernel'] != 'AIX':
        cmd.extend(('-R', root))
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if __grains__['kernel'] != 'OpenBSD' and __grains__['kernel'] != 'AIX':
        if result['retcode'] != 0 and 'not found in' in result['stderr']:
            ret = True
            for group in groups:
                cmd = ['gpasswd', '-a', name, group]
                if __salt__['cmd.retcode'](cmd, python_shell=False) != 0:
                    ret = False
            return ret
    return result['retcode'] == 0

def chfullname(name, fullname, root=None):
    if False:
        print('Hello World!')
    '\n    Change the user\'s Full Name\n\n    name\n        User to modify\n\n    fullname\n        GECOS field for the full name\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' user.chfullname foo "Foo Bar"\n    '
    return _update_gecos(name, 'fullname', fullname, root=root)

def chroomnumber(name, roomnumber, root=None):
    if False:
        print('Hello World!')
    "\n    Change the user's Room Number\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chroomnumber foo 123\n    "
    return _update_gecos(name, 'roomnumber', roomnumber, root=root)

def chworkphone(name, workphone, root=None):
    if False:
        while True:
            i = 10
    "\n    Change the user's Work Phone\n\n    name\n        User to modify\n\n    workphone\n        GECOS field for the work phone\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chworkphone foo 7735550123\n    "
    return _update_gecos(name, 'workphone', workphone, root=root)

def chhomephone(name, homephone, root=None):
    if False:
        return 10
    "\n    Change the user's Home Phone\n\n    name\n        User to modify\n\n    homephone\n        GECOS field for the home phone\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chhomephone foo 7735551234\n    "
    return _update_gecos(name, 'homephone', homephone, root=root)

def chother(name, other, root=None):
    if False:
        while True:
            i = 10
    "\n    Change the user's other GECOS attribute\n\n    name\n        User to modify\n\n    other\n        GECOS field for other information\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chother foobar\n    "
    return _update_gecos(name, 'other', other, root=root)

def chloginclass(name, loginclass, root=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Change the default login class of the user\n\n    name\n        User to modify\n\n    loginclass\n        Login class for the new account\n\n    root\n        Directory to chroot into\n\n    .. note::\n        This function only applies to OpenBSD systems.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chloginclass foo staff\n    "
    if __grains__['kernel'] != 'OpenBSD':
        return False
    if loginclass == get_loginclass(name):
        return True
    cmd = [_which('usermod'), '-L', loginclass, name]
    if root is not None and __grains__['kernel'] != 'AIX':
        cmd.extend(('-R', root))
    __salt__['cmd.run'](cmd, python_shell=False)
    return get_loginclass(name) == loginclass

def info(name, root=None):
    if False:
        i = 10
        return i + 15
    "\n    Return user information\n\n    name\n        User to get the information\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.info root\n    "
    if root is not None and __grains__['kernel'] != 'AIX':
        getpwnam = functools.partial(_getpwnam, root=root)
    else:
        getpwnam = functools.partial(pwd.getpwnam)
    try:
        data = getpwnam(_quote_username(name))
    except KeyError:
        return {}
    else:
        return _format_info(data)

def get_loginclass(name):
    if False:
        return 10
    "\n    Get the login class of the user\n\n    name\n        User to get the information\n\n    .. note::\n        This function only applies to OpenBSD systems.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.get_loginclass foo\n    "
    if __grains__['kernel'] != 'OpenBSD':
        return False
    userinfo = __salt__['cmd.run_stdout'](['userinfo', name], python_shell=False)
    for line in userinfo.splitlines():
        if line.startswith('class'):
            try:
                ret = line.split(None, 1)[1]
                break
            except (ValueError, IndexError):
                continue
    else:
        ret = ''
    return ret

def _format_info(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return user information in a pretty way\n    '
    gecos_field = salt.utils.stringutils.to_unicode(data.pw_gecos).split(',', 4)
    while len(gecos_field) < 5:
        gecos_field.append('')
    return {'gid': data.pw_gid, 'groups': list_groups(data.pw_name), 'home': data.pw_dir, 'name': data.pw_name, 'passwd': data.pw_passwd, 'shell': data.pw_shell, 'uid': data.pw_uid, 'fullname': gecos_field[0], 'roomnumber': gecos_field[1], 'workphone': gecos_field[2], 'homephone': gecos_field[3], 'other': gecos_field[4]}

@salt.utils.decorators.path.which('id')
def primary_group(name):
    if False:
        i = 10
        return i + 15
    "\n    Return the primary group of the named user\n\n    .. versionadded:: 2016.3.0\n\n    name\n        User to get the information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.primary_group saltadmin\n    "
    return __salt__['cmd.run'](['id', '-g', '-n', name])

def list_groups(name):
    if False:
        return 10
    "\n    Return a list of groups the named user belongs to\n\n    name\n        User to get the information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.list_groups foo\n    "
    return salt.utils.user.get_group_list(name)

def list_users(root=None):
    if False:
        print('Hello World!')
    "\n    Return a list of all users\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.list_users\n    "
    if root is not None and __grains__['kernel'] != 'AIX':
        getpwall = functools.partial(_getpwall, root=root)
    else:
        getpwall = functools.partial(pwd.getpwall)
    return sorted((user.pw_name for user in getpwall()))

def rename(name, new_name, root=None):
    if False:
        i = 10
        return i + 15
    "\n    Change the username for a named user\n\n    name\n        User to modify\n\n    new_name\n        New value of the login name\n\n    root\n        Directory to chroot into\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.rename name new_name\n    "
    if info(new_name, root=root):
        raise CommandExecutionError(f"User '{new_name}' already exists")
    return _chattrib(name, 'name', new_name, '-l', root=root)

def _getpwnam(name, root=None):
    if False:
        i = 10
        return i + 15
    '\n    Alternative implementation for getpwnam, that use only /etc/passwd\n    '
    root = '/' if not root else root
    passwd = os.path.join(root, 'etc/passwd')
    with salt.utils.files.fopen(passwd) as fp_:
        for line in fp_:
            line = salt.utils.stringutils.to_unicode(line)
            comps = line.strip().split(':')
            if comps[0] == name:
                (comps[2], comps[3]) = (int(comps[2]), int(comps[3]))
                return pwd.struct_passwd(comps)
    raise KeyError

def _getpwall(root=None):
    if False:
        return 10
    '\n    Alternative implemetantion for getpwall, that use only /etc/passwd\n    '
    root = '/' if not root else root
    passwd = os.path.join(root, 'etc/passwd')
    with salt.utils.files.fopen(passwd) as fp_:
        for line in fp_:
            line = salt.utils.stringutils.to_unicode(line)
            comps = line.strip().split(':')
            (comps[2], comps[3]) = (int(comps[2]), int(comps[3]))
            yield pwd.struct_passwd(comps)