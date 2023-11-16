"""
Manage users on Mac OS 10.7+

.. important::
    If you feel that Salt should be using this module to manage users on a
    minion, and it is using a different module (or gives an error similar to
    *'user.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import logging
import time
import salt.utils.args
import salt.utils.data
import salt.utils.decorators.path
import salt.utils.files
import salt.utils.stringutils
import salt.utils.user
from salt.exceptions import CommandExecutionError, SaltInvocationError
try:
    import pwd
except ImportError:
    pass
log = logging.getLogger(__name__)
__virtualname__ = 'user'

def __virtual__():
    if False:
        print('Hello World!')
    if __grains__.get('kernel') != 'Darwin' or __grains__['osrelease_info'] < (10, 7):
        return (False, 'Only available on Mac OS 10.7+ systems')
    else:
        return __virtualname__

def _flush_dscl_cache():
    if False:
        while True:
            i = 10
    '\n    Flush dscl cache\n    '
    __salt__['cmd.run'](['dscacheutil', '-flushcache'], python_shell=False)

def _dscl(cmd, ctype='create'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run a dscl -create command\n    '
    if __grains__['osrelease_info'] < (10, 8):
        (source, noderoot) = ('.', '')
    else:
        (source, noderoot) = ('localhost', '/Local/Default')
    if noderoot:
        cmd[0] = noderoot + cmd[0]
    return __salt__['cmd.run_all'](['dscl', source, '-' + ctype] + cmd, output_loglevel='quiet' if ctype == 'passwd' else 'debug', python_shell=False)

def _first_avail_uid():
    if False:
        while True:
            i = 10
    uids = {x.pw_uid for x in pwd.getpwall()}
    for idx in range(501, 2 ** 24):
        if idx not in uids:
            return idx

def add(name, uid=None, gid=None, groups=None, home=None, shell=None, fullname=None, createhome=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Add a user to the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.add name <uid> <gid> <groups> <home> <shell>\n    "
    if info(name):
        raise CommandExecutionError("User '{}' already exists".format(name))
    if salt.utils.stringutils.contains_whitespace(name):
        raise SaltInvocationError('Username cannot contain whitespace')
    if uid is None:
        uid = _first_avail_uid()
    if gid is None:
        gid = 20
    if home is None:
        home = '/Users/{}'.format(name)
    if shell is None:
        shell = '/bin/bash'
    if fullname is None:
        fullname = ''
    if not isinstance(uid, int):
        raise SaltInvocationError('uid must be an integer')
    if not isinstance(gid, int):
        raise SaltInvocationError('gid must be an integer')
    name_path = '/Users/{}'.format(name)
    _dscl([name_path, 'UniqueID', uid])
    _dscl([name_path, 'PrimaryGroupID', gid])
    _dscl([name_path, 'UserShell', shell])
    _dscl([name_path, 'NFSHomeDirectory', home])
    _dscl([name_path, 'RealName', fullname])
    if createhome:
        __salt__['file.mkdir'](home, user=uid, group=gid)
    time.sleep(1)
    if groups:
        chgroups(name, groups)
    return True

def delete(name, remove=False, force=False):
    if False:
        i = 10
        return i + 15
    "\n    Remove a user from the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.delete name remove=True force=True\n    "
    if salt.utils.stringutils.contains_whitespace(name):
        raise SaltInvocationError('Username cannot contain whitespace')
    user_info = info(name)
    if not user_info:
        return True
    if force:
        log.warning('force option is unsupported on MacOS, ignoring')
    chgroups(name, ())
    ret = _dscl(['/Users/{}'.format(name)], ctype='delete')['retcode'] == 0
    if ret and remove:
        __salt__['file.remove'](user_info['home'])
    return ret

def getent(refresh=False):
    if False:
        i = 10
        return i + 15
    "\n    Return the list of all info for all users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.getent\n    "
    if 'user.getent' in __context__ and (not refresh):
        return __context__['user.getent']
    ret = []
    for data in pwd.getpwall():
        ret.append(_format_info(data))
    __context__['user.getent'] = ret
    return ret

def chuid(name, uid):
    if False:
        print('Hello World!')
    "\n    Change the uid for a named user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chuid foo 4376\n    "
    if not isinstance(uid, int):
        raise SaltInvocationError('uid must be an integer')
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if uid == pre_info['uid']:
        return True
    _dscl(['/Users/{}'.format(name), 'UniqueID', pre_info['uid'], uid], ctype='change')
    time.sleep(1)
    return info(name).get('uid') == uid

def chgid(name, gid):
    if False:
        while True:
            i = 10
    "\n    Change the default group of the user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chgid foo 4376\n    "
    if not isinstance(gid, int):
        raise SaltInvocationError('gid must be an integer')
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if gid == pre_info['gid']:
        return True
    _dscl(['/Users/{}'.format(name), 'PrimaryGroupID', pre_info['gid'], gid], ctype='change')
    time.sleep(1)
    return info(name).get('gid') == gid

def chshell(name, shell):
    if False:
        return 10
    "\n    Change the default shell of the user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chshell foo /bin/zsh\n    "
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if shell == pre_info['shell']:
        return True
    _dscl(['/Users/{}'.format(name), 'UserShell', pre_info['shell'], shell], ctype='change')
    time.sleep(1)
    return info(name).get('shell') == shell

def chhome(name, home, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Change the home directory of the user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chhome foo /Users/foo\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    persist = kwargs.pop('persist', False)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    if persist:
        log.info("Ignoring unsupported 'persist' argument to user.chhome")
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if home == pre_info['home']:
        return True
    _dscl(['/Users/{}'.format(name), 'NFSHomeDirectory', pre_info['home'], home], ctype='change')
    time.sleep(1)
    return info(name).get('home') == home

def chfullname(name, fullname):
    if False:
        while True:
            i = 10
    "\n    Change the user's Full Name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chfullname foo 'Foo Bar'\n    "
    fullname = salt.utils.data.decode(fullname)
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    pre_info['fullname'] = salt.utils.data.decode(pre_info['fullname'])
    if fullname == pre_info['fullname']:
        return True
    _dscl(['/Users/{}'.format(name), 'RealName', fullname], ctype='create')
    time.sleep(1)
    current = salt.utils.data.decode(info(name).get('fullname'))
    return current == fullname

def chgroups(name, groups, append=False):
    if False:
        i = 10
        return i + 15
    "\n    Change the groups to which the user belongs. Note that the user's primary\n    group does not have to be one of the groups passed, membership in the\n    user's primary group is automatically assumed.\n\n    groups\n        Groups to which the user should belong, can be passed either as a\n        python list or a comma-separated string\n\n    append\n        Instead of removing user from groups not included in the ``groups``\n        parameter, just add user to any groups for which they are not members\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chgroups foo wheel,root\n    "
    uinfo = info(name)
    if not uinfo:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if isinstance(groups, str):
        groups = groups.split(',')
    bad_groups = [x for x in groups if salt.utils.stringutils.contains_whitespace(x)]
    if bad_groups:
        raise SaltInvocationError('Invalid group name(s): {}'.format(', '.join(bad_groups)))
    ugrps = set(list_groups(name))
    desired = {str(x) for x in groups if bool(str(x))}
    primary_group = __salt__['file.gid_to_group'](uinfo['gid'])
    if primary_group:
        desired.add(primary_group)
    if ugrps == desired:
        return True
    for group in desired - ugrps:
        _dscl(['/Groups/{}'.format(group), 'GroupMembership', name], ctype='append')
    if not append:
        for group in ugrps - desired:
            _dscl(['/Groups/{}'.format(group), 'GroupMembership', name], ctype='delete')
    time.sleep(1)
    return set(list_groups(name)) == desired

def info(name):
    if False:
        i = 10
        return i + 15
    "\n    Return user information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.info root\n    "
    try:
        data = next(iter((x for x in pwd.getpwall() if x.pw_name == name)))
    except StopIteration:
        return {}
    else:
        return _format_info(data)

def _format_info(data):
    if False:
        print('Hello World!')
    '\n    Return user information in a pretty way\n    '
    return {'gid': data.pw_gid, 'groups': list_groups(data.pw_name), 'home': data.pw_dir, 'name': data.pw_name, 'shell': data.pw_shell, 'uid': data.pw_uid, 'fullname': data.pw_gecos}

@salt.utils.decorators.path.which('id')
def primary_group(name):
    if False:
        return 10
    "\n    Return the primary group of the named user\n\n    .. versionadded:: 2016.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.primary_group saltadmin\n    "
    return __salt__['cmd.run'](['id', '-g', '-n', name])

def list_groups(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of groups the named user belongs to.\n\n    name\n\n        The name of the user for which to list groups. Starting in Salt 2016.11.0,\n        all groups for the user, including groups beginning with an underscore\n        will be listed.\n\n        .. versionchanged:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.list_groups foo\n    "
    groups = [group for group in salt.utils.user.get_group_list(name)]
    return groups

def list_users():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of all users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.list_users\n    "
    users = _dscl(['/users'], 'list')['stdout']
    return users.split()

def rename(name, new_name):
    if False:
        print('Hello World!')
    "\n    Change the username for a named user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.rename name new_name\n    "
    current_info = info(name)
    if not current_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    new_info = info(new_name)
    if new_info:
        raise CommandExecutionError("User '{}' already exists".format(new_name))
    _dscl(['/Users/{}'.format(name), 'RecordName', name, new_name], ctype='change')
    time.sleep(1)
    return info(new_name).get('RecordName') == new_name

def get_auto_login():
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2016.3.0\n\n    Gets the current setting for Auto Login\n\n    :return: If enabled, returns the user name, otherwise returns False\n    :rtype: str, bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.get_auto_login\n    "
    cmd = ['defaults', 'read', '/Library/Preferences/com.apple.loginwindow.plist', 'autoLoginUser']
    ret = __salt__['cmd.run_all'](cmd, ignore_retcode=True)
    return False if ret['retcode'] else ret['stdout']

def _kcpassword(password):
    if False:
        return 10
    '\n    Internal function for obfuscating the password used for AutoLogin\n    This is later written as the contents of the ``/etc/kcpassword`` file\n\n    .. versionadded:: 2017.7.3\n\n    Adapted from:\n    https://github.com/timsutton/osx-vm-templates/blob/master/scripts/support/set_kcpassword.py\n\n    Args:\n\n        password(str):\n            The password to obfuscate\n\n    Returns:\n        str: The obfuscated password\n    '
    key = [125, 137, 82, 35, 210, 188, 221, 234, 163, 185, 31]
    key_len = len(key) + 1
    password = list(map(ord, password)) + [0]
    remainder = len(password) % key_len
    if remainder > 0:
        password = password + [0] * (key_len - remainder)
    for chunk_index in range(0, len(password), len(key)):
        key_index = 0
        for password_index in range(chunk_index, min(chunk_index + len(key), len(password))):
            password[password_index] = password[password_index] ^ key[key_index]
            key_index += 1
    return bytes(password)

def enable_auto_login(name, password):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.3.0\n\n    Configures the machine to auto login with the specified user\n\n    Args:\n\n        name (str): The user account use for auto login\n\n        password (str): The password to user for auto login\n\n            .. versionadded:: 2017.7.3\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.enable_auto_login stevej\n    "
    cmd = ['defaults', 'write', '/Library/Preferences/com.apple.loginwindow.plist', 'autoLoginUser', name]
    __salt__['cmd.run'](cmd)
    current = get_auto_login()
    o_password = _kcpassword(password=password)
    with salt.utils.files.set_umask(63):
        with salt.utils.files.fopen('/etc/kcpassword', 'wb') as fd:
            fd.write(o_password)
    return current if isinstance(current, bool) else current.lower() == name.lower()

def disable_auto_login():
    if False:
        return 10
    "\n    .. versionadded:: 2016.3.0\n\n    Disables auto login on the machine\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.disable_auto_login\n    "
    cmd = 'rm -f /etc/kcpassword'
    __salt__['cmd.run'](cmd)
    cmd = ['defaults', 'delete', '/Library/Preferences/com.apple.loginwindow.plist', 'autoLoginUser']
    __salt__['cmd.run'](cmd)
    return True if not get_auto_login() else False