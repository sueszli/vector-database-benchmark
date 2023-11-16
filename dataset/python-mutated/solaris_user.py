"""
Manage users with the useradd command

.. important::

    If you feel that Salt should be using this module to manage users on a
    minion, and it is using a different module (or gives an error similar to
    *'user.info' is not available*), see :ref:`here
    <module-provider-override>`.

"""
import copy
import logging
import salt.utils.data
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
        i = 10
        return i + 15
    '\n    Set the user module if the kernel is SunOS\n    '
    if __grains__['kernel'] == 'SunOS' and HAS_PWD:
        return __virtualname__
    return (False, 'The solaris_user execution module failed to load: only available on Solaris systems with pwd module installed.')

def _get_gecos(name):
    if False:
        i = 10
        return i + 15
    '\n    Retrieve GECOS field info and return it in dictionary form\n    '
    gecos_field = pwd.getpwnam(name).pw_gecos.split(',', 3)
    if not gecos_field:
        return {}
    else:
        while len(gecos_field) < 4:
            gecos_field.append('')
        return {'fullname': str(gecos_field[0]), 'roomnumber': str(gecos_field[1]), 'workphone': str(gecos_field[2]), 'homephone': str(gecos_field[3])}

def _build_gecos(gecos_dict):
    if False:
        for i in range(10):
            print('nop')
    '\n    Accepts a dictionary entry containing GECOS field names and their values,\n    and returns a full GECOS comment string, to be used with usermod.\n    '
    return '{},{},{},{}'.format(gecos_dict.get('fullname', ''), gecos_dict.get('roomnumber', ''), gecos_dict.get('workphone', ''), gecos_dict.get('homephone', ''))

def _update_gecos(name, key, value):
    if False:
        print('Hello World!')
    "\n    Common code to change a user's GECOS information\n    "
    if not isinstance(value, str):
        value = str(value)
    pre_info = _get_gecos(name)
    if not pre_info:
        return False
    if value == pre_info[key]:
        return True
    gecos_data = copy.deepcopy(pre_info)
    gecos_data[key] = value
    cmd = ['usermod', '-c', _build_gecos(gecos_data), name]
    __salt__['cmd.run'](cmd, python_shell=False)
    post_info = info(name)
    return _get_gecos(name).get(key) == value

def add(name, uid=None, gid=None, groups=None, home=None, shell=None, unique=True, fullname='', roomnumber='', workphone='', homephone='', createhome=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Add a user to the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.add name <uid> <gid> <groups> <home> <shell>\n    "
    if salt.utils.data.is_true(kwargs.pop('system', False)):
        log.warning("solaris_user module does not support the 'system' argument")
    if kwargs:
        log.warning('Invalid kwargs passed to user.add')
    if isinstance(groups, str):
        groups = groups.split(',')
    cmd = ['useradd']
    if shell:
        cmd.extend(['-s', shell])
    if uid:
        cmd.extend(['-u', uid])
    if gid:
        cmd.extend(['-g', gid])
    if groups:
        cmd.extend(['-G', ','.join(groups)])
    if createhome:
        cmd.append('-m')
    if home is not None:
        cmd.extend(['-d', home])
    if not unique:
        cmd.append('-o')
    cmd.append(name)
    if __salt__['cmd.retcode'](cmd, python_shell=False) != 0:
        return False
    else:
        if fullname:
            chfullname(name, fullname)
        if roomnumber:
            chroomnumber(name, roomnumber)
        if workphone:
            chworkphone(name, workphone)
        if homephone:
            chhomephone(name, homephone)
        return True

def delete(name, remove=False, force=False):
    if False:
        print('Hello World!')
    "\n    Remove a user from the minion\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.delete name remove=True force=True\n    "
    if salt.utils.data.is_true(force):
        log.warning('userdel does not support force-deleting user while user is logged in')
    cmd = ['userdel']
    if remove:
        cmd.append('-r')
    cmd.append(name)
    return __salt__['cmd.retcode'](cmd, python_shell=False) == 0

def getent(refresh=False):
    if False:
        print('Hello World!')
    "\n    Return the list of all info for all users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.getent\n    "
    if 'user.getent' in __context__ and (not refresh):
        return __context__['user.getent']
    ret = []
    for data in pwd.getpwall():
        ret.append(info(data.pw_name))
    __context__['user.getent'] = ret
    return ret

def chuid(name, uid):
    if False:
        while True:
            i = 10
    "\n    Change the uid for a named user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chuid foo 4376\n    "
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if uid == pre_info['uid']:
        return True
    cmd = ['usermod', '-u', uid, name]
    __salt__['cmd.run'](cmd, python_shell=False)
    return info(name).get('uid') == uid

def chgid(name, gid):
    if False:
        print('Hello World!')
    "\n    Change the default group of the user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chgid foo 4376\n    "
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if gid == pre_info['gid']:
        return True
    cmd = ['usermod', '-g', gid, name]
    __salt__['cmd.run'](cmd, python_shell=False)
    return info(name).get('gid') == gid

def chshell(name, shell):
    if False:
        while True:
            i = 10
    "\n    Change the default shell of the user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chshell foo /bin/zsh\n    "
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if shell == pre_info['shell']:
        return True
    cmd = ['usermod', '-s', shell, name]
    __salt__['cmd.run'](cmd, python_shell=False)
    return info(name).get('shell') == shell

def chhome(name, home, persist=False):
    if False:
        print('Hello World!')
    "\n    Set a new home directory for an existing user\n\n    name\n        Username to modify\n\n    home\n        New home directory to set\n\n    persist : False\n        Set to ``True`` to prevent configuration files in the new home\n        directory from being overwritten by the files from the skeleton\n        directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chhome foo /home/users/foo True\n    "
    pre_info = info(name)
    if not pre_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    if home == pre_info['home']:
        return True
    cmd = ['usermod', '-d', home]
    if persist:
        cmd.append('-m')
    cmd.append(name)
    __salt__['cmd.run'](cmd, python_shell=False)
    return info(name).get('home') == home

def chgroups(name, groups, append=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Change the groups to which a user belongs\n\n    name\n        Username to modify\n\n    groups\n        List of groups to set for the user. Can be passed as a comma-separated\n        list or a Python list.\n\n    append : False\n        Set to ``True`` to append these groups to the user's existing list of\n        groups. Otherwise, the specified groups will replace any existing\n        groups for the user.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chgroups foo wheel,root True\n    "
    if isinstance(groups, str):
        groups = groups.split(',')
    ugrps = set(list_groups(name))
    if ugrps == set(groups):
        return True
    if append:
        groups.update(ugrps)
    cmd = ['usermod', '-G', ','.join(groups), name]
    return __salt__['cmd.retcode'](cmd, python_shell=False) == 0

def chfullname(name, fullname):
    if False:
        print('Hello World!')
    '\n    Change the user\'s Full Name\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' user.chfullname foo "Foo Bar"\n    '
    return _update_gecos(name, 'fullname', fullname)

def chroomnumber(name, roomnumber):
    if False:
        print('Hello World!')
    "\n    Change the user's Room Number\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chroomnumber foo 123\n    "
    return _update_gecos(name, 'roomnumber', roomnumber)

def chworkphone(name, workphone):
    if False:
        print('Hello World!')
    '\n    Change the user\'s Work Phone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' user.chworkphone foo "7735550123"\n    '
    return _update_gecos(name, 'workphone', workphone)

def chhomephone(name, homephone):
    if False:
        return 10
    '\n    Change the user\'s Home Phone\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' user.chhomephone foo "7735551234"\n    '
    return _update_gecos(name, 'homephone', homephone)

def info(name):
    if False:
        return 10
    "\n    Return user information\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.info root\n    "
    ret = {}
    try:
        data = pwd.getpwnam(name)
        ret['gid'] = data.pw_gid
        ret['groups'] = list_groups(name)
        ret['home'] = data.pw_dir
        ret['name'] = data.pw_name
        ret['passwd'] = data.pw_passwd
        ret['shell'] = data.pw_shell
        ret['uid'] = data.pw_uid
        gecos_field = data.pw_gecos.split(',', 3)
        while len(gecos_field) < 4:
            gecos_field.append('')
        ret['fullname'] = gecos_field[0]
        ret['roomnumber'] = gecos_field[1]
        ret['workphone'] = gecos_field[2]
        ret['homephone'] = gecos_field[3]
    except KeyError:
        return {}
    return ret

def list_groups(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of groups the named user belongs to\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.list_groups foo\n    "
    return salt.utils.user.get_group_list(name)

def list_users():
    if False:
        while True:
            i = 10
    "\n    Return a list of all users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.list_users\n    "
    return sorted((user.pw_name for user in pwd.getpwall()))

def rename(name, new_name):
    if False:
        return 10
    "\n    Change the username for a named user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.rename name new_name\n    "
    current_info = info(name)
    if not current_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    new_info = info(new_name)
    if new_info:
        raise CommandExecutionError("User '{}' already exists".format(new_name))
    cmd = ['usermod', '-l', new_name, name]
    __salt__['cmd.run'](cmd, python_shell=False)
    return info(new_name).get('name') == new_name