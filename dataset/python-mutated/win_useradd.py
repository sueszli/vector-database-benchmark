"""
Module for managing Windows Users.

.. important::
    If you feel that Salt should be using this module to manage users on a
    minion, and it is using a different module (or gives an error similar to
    *'user.info' is not available*), see :ref:`here
    <module-provider-override>`.

:depends:
        - pywintypes
        - win32api
        - win32con
        - win32net
        - win32netcon
        - win32profile
        - win32security
        - win32ts
        - wmi

.. note::
    This currently only works with local user accounts, not domain accounts
"""
import logging
import shlex
import time
from datetime import datetime
import salt.utils.args
import salt.utils.dateutils
import salt.utils.platform
import salt.utils.winapi
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
try:
    import pywintypes
    import win32api
    import win32con
    import win32net
    import win32netcon
    import win32profile
    import win32security
    import win32ts
    import wmi
    HAS_WIN32NET_MODS = True
except ImportError:
    HAS_WIN32NET_MODS = False
__virtualname__ = 'user'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Requires Windows and Windows Modules\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Module win_useradd: Windows Only')
    if not HAS_WIN32NET_MODS:
        return (False, 'Module win_useradd: Missing Win32 Modules')
    return __virtualname__

def add(name, password=None, fullname=None, description=None, groups=None, home=None, homedrive=None, profile=None, logonscript=None):
    if False:
        while True:
            i = 10
    "\n    Add a user to the minion.\n\n    Args:\n        name (str): User name\n\n        password (str, optional): User's password in plain text.\n\n        fullname (str, optional): The user's full name.\n\n        description (str, optional): A brief description of the user account.\n\n        groups (str, optional): A list of groups to add the user to.\n            (see chgroups)\n\n        home (str, optional): The path to the user's home directory.\n\n        homedrive (str, optional): The drive letter to assign to the home\n            directory. Must be the Drive Letter followed by a colon. ie: U:\n\n        profile (str, optional): An explicit path to a profile. Can be a UNC or\n            a folder on the system. If left blank, windows uses its default\n            profile directory.\n\n        logonscript (str, optional): Path to a login script to run when the user\n            logs on.\n\n    Returns:\n        bool: True if successful. False is unsuccessful.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.add name password\n    "
    user_info = {}
    if name:
        user_info['name'] = name
    else:
        return False
    user_info['password'] = password
    user_info['priv'] = win32netcon.USER_PRIV_USER
    user_info['home_dir'] = home
    user_info['comment'] = description
    user_info['flags'] = win32netcon.UF_SCRIPT
    user_info['script_path'] = logonscript
    try:
        win32net.NetUserAdd(None, 1, user_info)
    except win32net.error as exc:
        log.error('Failed to create user %s', name)
        log.error('nbr: %s', exc.winerror)
        log.error('ctx: %s', exc.funcname)
        log.error('msg: %s', exc.strerror)
        return exc.strerror
    update(name=name, homedrive=homedrive, profile=profile, fullname=fullname)
    ret = chgroups(name, groups) if groups else True
    return ret

def update(name, password=None, fullname=None, description=None, home=None, homedrive=None, logonscript=None, profile=None, expiration_date=None, expired=None, account_disabled=None, unlock_account=None, password_never_expires=None, disallow_change_password=None):
    if False:
        while True:
            i = 10
    "\n    Updates settings for the windows user. Name is the only required parameter.\n    Settings will only be changed if the parameter is passed a value.\n\n    .. versionadded:: 2015.8.0\n\n    Args:\n        name (str): The user name to update.\n\n        password (str, optional): New user password in plain text.\n\n        fullname (str, optional): The user's full name.\n\n        description (str, optional): A brief description of the user account.\n\n        home (str, optional): The path to the user's home directory.\n\n        homedrive (str, optional): The drive letter to assign to the home\n            directory. Must be the Drive Letter followed by a colon. ie: U:\n\n        logonscript (str, optional): The path to the logon script.\n\n        profile (str, optional): The path to the user's profile directory.\n\n        expiration_date (date, optional): The date and time when the account\n            expires. Can be a valid date/time string. To set to never expire\n            pass the string 'Never'.\n\n        expired (bool, optional): Pass `True` to expire the account. The user\n            will be prompted to change their password at the next logon. Pass\n            `False` to mark the account as 'not expired'. You can't use this to\n            negate the expiration if the expiration was caused by the account\n            expiring. You'll have to change the `expiration_date` as well.\n\n        account_disabled (bool, optional): True disables the account. False\n            enables the account.\n\n        unlock_account (bool, optional): True unlocks a locked user account.\n            False is ignored.\n\n        password_never_expires (bool, optional): True sets the password to never\n            expire. False allows the password to expire.\n\n        disallow_change_password (bool, optional): True blocks the user from\n            changing the password. False allows the user to change the password.\n\n    Returns:\n        bool: True if successful. False is unsuccessful.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.update bob password=secret profile=C:\\Users\\Bob\n                 home=\\server\\homeshare\\bob homedrive=U:\n    "
    try:
        user_info = win32net.NetUserGetInfo(None, name, 4)
    except win32net.error as exc:
        log.error('Failed to update user %s', name)
        log.error('nbr: %s', exc.winerror)
        log.error('ctx: %s', exc.funcname)
        log.error('msg: %s', exc.strerror)
        return exc.strerror
    if password:
        user_info['password'] = password
    if home:
        user_info['home_dir'] = home
    if homedrive:
        user_info['home_dir_drive'] = homedrive
    if description:
        user_info['comment'] = description
    if logonscript:
        user_info['script_path'] = logonscript
    if fullname:
        user_info['full_name'] = fullname
    if profile:
        user_info['profile'] = profile
    if expiration_date:
        if expiration_date == 'Never':
            user_info['acct_expires'] = win32netcon.TIMEQ_FOREVER
        else:
            try:
                dt_obj = salt.utils.dateutils.date_cast(expiration_date)
            except (ValueError, RuntimeError):
                return 'Invalid Date/Time Format: {}'.format(expiration_date)
            user_info['acct_expires'] = time.mktime(dt_obj.timetuple())
    if expired is not None:
        if expired:
            user_info['password_expired'] = 1
        else:
            user_info['password_expired'] = 0
    if account_disabled is not None:
        if account_disabled:
            user_info['flags'] |= win32netcon.UF_ACCOUNTDISABLE
        else:
            user_info['flags'] &= ~win32netcon.UF_ACCOUNTDISABLE
    if unlock_account is not None:
        if unlock_account:
            user_info['flags'] &= ~win32netcon.UF_LOCKOUT
    if password_never_expires is not None:
        if password_never_expires:
            user_info['flags'] |= win32netcon.UF_DONT_EXPIRE_PASSWD
        else:
            user_info['flags'] &= ~win32netcon.UF_DONT_EXPIRE_PASSWD
    if disallow_change_password is not None:
        if disallow_change_password:
            user_info['flags'] |= win32netcon.UF_PASSWD_CANT_CHANGE
        else:
            user_info['flags'] &= ~win32netcon.UF_PASSWD_CANT_CHANGE
    try:
        win32net.NetUserSetInfo(None, name, 4, user_info)
    except win32net.error as exc:
        log.error('Failed to update user %s', name)
        log.error('nbr: %s', exc.winerror)
        log.error('ctx: %s', exc.funcname)
        log.error('msg: %s', exc.strerror)
        return exc.strerror
    return True

def delete(name, purge=False, force=False):
    if False:
        print('Hello World!')
    "\n    Remove a user from the minion\n\n    Args:\n        name (str): The name of the user to delete\n\n        purge (bool, optional): Boolean value indicating that the user profile\n            should also be removed when the user account is deleted. If set to\n            True the profile will be removed. Default is False.\n\n        force (bool, optional): Boolean value indicating that the user account\n            should be deleted even if the user is logged in. True will log the\n            user out and delete user.\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.delete name\n    "
    try:
        user_info = win32net.NetUserGetInfo(None, name, 4)
    except win32net.error as exc:
        log.error('User not found: %s', name)
        log.error('nbr: %s', exc.winerror)
        log.error('ctx: %s', exc.funcname)
        log.error('msg: %s', exc.strerror)
        return exc.strerror
    try:
        sess_list = win32ts.WTSEnumerateSessions()
    except win32ts.error as exc:
        log.error('No logged in users found')
        log.error('nbr: %s', exc.winerror)
        log.error('ctx: %s', exc.funcname)
        log.error('msg: %s', exc.strerror)
    logged_in = False
    session_id = None
    for sess in sess_list:
        if win32ts.WTSQuerySessionInformation(None, sess['SessionId'], win32ts.WTSUserName) == name:
            session_id = sess['SessionId']
            logged_in = True
    if logged_in:
        if force:
            try:
                win32ts.WTSLogoffSession(win32ts.WTS_CURRENT_SERVER_HANDLE, session_id, True)
            except win32ts.error as exc:
                log.error('User not found: %s', name)
                log.error('nbr: %s', exc.winerror)
                log.error('ctx: %s', exc.funcname)
                log.error('msg: %s', exc.strerror)
                return exc.strerror
        else:
            log.error('User %s is currently logged in.', name)
            return False
    if purge:
        try:
            sid = getUserSid(name)
            win32profile.DeleteProfile(sid)
        except pywintypes.error as exc:
            (number, context, message) = exc.args
            if number == 2:
                pass
            else:
                log.error('Failed to remove profile for %s', name)
                log.error('nbr: %s', exc.winerror)
                log.error('ctx: %s', exc.funcname)
                log.error('msg: %s', exc.strerror)
                return exc.strerror
    try:
        win32net.NetUserDel(None, name)
    except win32net.error as exc:
        log.error('Failed to delete user %s', name)
        log.error('nbr: %s', exc.winerror)
        log.error('ctx: %s', exc.funcname)
        log.error('msg: %s', exc.strerror)
        return exc.strerror
    return True

def getUserSid(username):
    if False:
        while True:
            i = 10
    "\n    Get the Security ID for the user\n\n    Args:\n        username (str): The user name for which to look up the SID\n\n    Returns:\n        str: The user SID\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.getUserSid jsnuffy\n    "
    domain = win32api.GetComputerName()
    if username.find('\\') != -1:
        domain = username.split('\\')[0]
        username = username.split('\\')[-1]
    domain = domain.upper()
    return win32security.ConvertSidToStringSid(win32security.LookupAccountName(None, domain + '\\' + username)[0])

def setpassword(name, password):
    if False:
        while True:
            i = 10
    "\n    Set the user's password\n\n    Args:\n        name (str): The user name for which to set the password\n\n        password (str): The new password\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.setpassword jsnuffy sup3rs3cr3t\n    "
    return update(name=name, password=password)

def addgroup(name, group):
    if False:
        while True:
            i = 10
    "\n    Add user to a group\n\n    Args:\n        name (str): The user name to add to the group\n\n        group (str): The name of the group to which to add the user\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.addgroup jsnuffy 'Power Users'\n    "
    name = shlex.quote(name)
    group = shlex.quote(group).lstrip("'").rstrip("'")
    user = info(name)
    if not user:
        return False
    if group in user['groups']:
        return True
    cmd = 'net localgroup "{}" {} /add'.format(group, name)
    ret = __salt__['cmd.run_all'](cmd, python_shell=True)
    return ret['retcode'] == 0

def removegroup(name, group):
    if False:
        i = 10
        return i + 15
    "\n    Remove user from a group\n\n    Args:\n        name (str): The user name to remove from the group\n\n        group (str): The name of the group from which to remove the user\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.removegroup jsnuffy 'Power Users'\n    "
    name = shlex.quote(name)
    group = shlex.quote(group).lstrip("'").rstrip("'")
    user = info(name)
    if not user:
        return False
    if group not in user['groups']:
        return True
    cmd = 'net localgroup "{}" {} /delete'.format(group, name)
    ret = __salt__['cmd.run_all'](cmd, python_shell=True)
    return ret['retcode'] == 0

def chhome(name, home, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Change the home directory of the user, pass True for persist to move files\n    to the new home directory if the old home directory exist.\n\n    Args:\n        name (str): The name of the user whose home directory you wish to change\n\n        home (str): The new location of the home directory\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chhome foo \\\\fileserver\\home\\foo True\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    persist = kwargs.pop('persist', False)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    if persist:
        log.info("Ignoring unsupported 'persist' argument to user.chhome")
    pre_info = info(name)
    if not pre_info:
        return False
    if home == pre_info['home']:
        return True
    if not update(name=name, home=home):
        return False
    post_info = info(name)
    if post_info['home'] != pre_info['home']:
        return post_info['home'] == home
    return False

def chprofile(name, profile):
    if False:
        print('Hello World!')
    "\n    Change the profile directory of the user\n\n    Args:\n        name (str): The name of the user whose profile you wish to change\n\n        profile (str): The new location of the profile\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chprofile foo \\\\fileserver\\profiles\\foo\n    "
    return update(name=name, profile=profile)

def chfullname(name, fullname):
    if False:
        return 10
    "\n    Change the full name of the user\n\n    Args:\n        name (str): The user name for which to change the full name\n\n        fullname (str): The new value for the full name\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chfullname user 'First Last'\n    "
    return update(name=name, fullname=fullname)

def chgroups(name, groups, append=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Change the groups this user belongs to, add append=False to make the user a\n    member of only the specified groups\n\n    Args:\n        name (str): The user name for which to change groups\n\n        groups (str, list): A single group or a list of groups to assign to the\n            user. For multiple groups this can be a comma delimited string or a\n            list.\n\n        append (bool, optional): True adds the passed groups to the user's\n            current groups. False sets the user's groups to the passed groups\n            only. Default is True.\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.chgroups jsnuffy Administrators,Users True\n    "
    if isinstance(groups, str):
        groups = groups.split(',')
    groups = [x.strip(' *') for x in groups]
    ugrps = set(list_groups(name))
    if ugrps == set(groups):
        return True
    name = shlex.quote(name)
    if not append:
        for group in ugrps:
            group = shlex.quote(group).lstrip("'").rstrip("'")
            if group not in groups:
                cmd = 'net localgroup "{}" {} /delete'.format(group, name)
                __salt__['cmd.run_all'](cmd, python_shell=True)
    for group in groups:
        if group in ugrps:
            continue
        group = shlex.quote(group).lstrip("'").rstrip("'")
        cmd = 'net localgroup "{}" {} /add'.format(group, name)
        out = __salt__['cmd.run_all'](cmd, python_shell=True)
        if out['retcode'] != 0:
            log.error(out['stdout'])
            return False
    agrps = set(list_groups(name))
    return len(ugrps - agrps) == 0

def info(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return user information\n\n    Args:\n        name (str): Username for which to display information\n\n    Returns:\n        dict: A dictionary containing user information\n            - fullname\n            - username\n            - SID\n            - passwd (will always return None)\n            - comment (same as description, left here for backwards compatibility)\n            - description\n            - active\n            - logonscript\n            - profile\n            - home\n            - homedrive\n            - groups\n            - password_changed\n            - successful_logon_attempts\n            - failed_logon_attempts\n            - last_logon\n            - account_disabled\n            - account_locked\n            - password_never_expires\n            - disallow_change_password\n            - gid\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.info jsnuffy\n    "
    ret = {}
    items = {}
    try:
        items = win32net.NetUserGetInfo(None, name, 4)
    except win32net.error:
        pass
    if items:
        groups = []
        try:
            groups = win32net.NetUserGetLocalGroups(None, name)
        except win32net.error:
            pass
        ret['fullname'] = items['full_name']
        ret['name'] = items['name']
        ret['uid'] = win32security.ConvertSidToStringSid(items['user_sid'])
        ret['passwd'] = items['password']
        ret['comment'] = items['comment']
        ret['description'] = items['comment']
        ret['active'] = not bool(items['flags'] & win32netcon.UF_ACCOUNTDISABLE)
        ret['logonscript'] = items['script_path']
        ret['profile'] = items['profile']
        ret['failed_logon_attempts'] = items['bad_pw_count']
        ret['successful_logon_attempts'] = items['num_logons']
        secs = time.mktime(datetime.now().timetuple()) - items['password_age']
        ret['password_changed'] = datetime.fromtimestamp(secs).strftime('%Y-%m-%d %H:%M:%S')
        if items['last_logon'] == 0:
            ret['last_logon'] = 'Never'
        else:
            ret['last_logon'] = datetime.fromtimestamp(items['last_logon']).strftime('%Y-%m-%d %H:%M:%S')
        ret['expiration_date'] = datetime.fromtimestamp(items['acct_expires']).strftime('%Y-%m-%d %H:%M:%S')
        ret['expired'] = items['password_expired'] == 1
        if not ret['profile']:
            ret['profile'] = _get_userprofile_from_registry(name, ret['uid'])
        ret['home'] = items['home_dir']
        ret['homedrive'] = items['home_dir_drive']
        if not ret['home']:
            ret['home'] = ret['profile']
        ret['groups'] = groups
        if items['flags'] & win32netcon.UF_DONT_EXPIRE_PASSWD == 0:
            ret['password_never_expires'] = False
        else:
            ret['password_never_expires'] = True
        if items['flags'] & win32netcon.UF_ACCOUNTDISABLE == 0:
            ret['account_disabled'] = False
        else:
            ret['account_disabled'] = True
        if items['flags'] & win32netcon.UF_LOCKOUT == 0:
            ret['account_locked'] = False
        else:
            ret['account_locked'] = True
        if items['flags'] & win32netcon.UF_PASSWD_CANT_CHANGE == 0:
            ret['disallow_change_password'] = False
        else:
            ret['disallow_change_password'] = True
        ret['gid'] = ''
        return ret
    else:
        return {}

def _get_userprofile_from_registry(user, sid):
    if False:
        for i in range(10):
            print('nop')
    "\n    In case net user doesn't return the userprofile we can get it from the\n    registry\n\n    Args:\n        user (str): The user name, used in debug message\n\n        sid (str): The sid to lookup in the registry\n\n    Returns:\n        str: Profile directory\n    "
    profile_dir = __utils__['reg.read_value']('HKEY_LOCAL_MACHINE', 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\ProfileList\\{}'.format(sid), 'ProfileImagePath')['vdata']
    log.debug('user %s with sid=%s profile is located at "%s"', user, sid, profile_dir)
    return profile_dir

def list_groups(name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of groups the named user belongs to\n\n    Args:\n        name (str): The user name for which to list groups\n\n    Returns:\n        list: A list of groups to which the user belongs\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.list_groups foo\n    "
    ugrp = set()
    try:
        user = info(name)['groups']
    except KeyError:
        return False
    for group in user:
        ugrp.add(group.strip(' *'))
    return sorted(list(ugrp))

def getent(refresh=False):
    if False:
        i = 10
        return i + 15
    "\n    Return the list of all info for all users\n\n    Args:\n        refresh (bool, optional): Refresh the cached user information. Useful\n            when used from within a state function. Default is False.\n\n    Returns:\n        dict: A dictionary containing information about all users on the system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.getent\n    "
    if 'user.getent' in __context__ and (not refresh):
        return __context__['user.getent']
    ret = []
    for user in __salt__['user.list_users']():
        stuff = {}
        user_info = __salt__['user.info'](user)
        stuff['gid'] = ''
        stuff['groups'] = user_info['groups']
        stuff['home'] = user_info['home']
        stuff['name'] = user_info['name']
        stuff['passwd'] = user_info['passwd']
        stuff['shell'] = ''
        stuff['uid'] = user_info['uid']
        ret.append(stuff)
    __context__['user.getent'] = ret
    return ret

def list_users():
    if False:
        i = 10
        return i + 15
    "\n    Return a list of all users on Windows\n\n    Returns:\n        list: A list of all users on the system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.list_users\n    "
    res = 0
    user_list = []
    dowhile = True
    try:
        while res or dowhile:
            dowhile = False
            (users, _, res) = win32net.NetUserEnum(None, 0, win32netcon.FILTER_NORMAL_ACCOUNT, res, win32netcon.MAX_PREFERRED_LENGTH)
            for user in users:
                user_list.append(user['name'])
        return user_list
    except win32net.error:
        pass

def rename(name, new_name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Change the username for a named user\n\n    Args:\n        name (str): The user name to change\n\n        new_name (str): The new name for the current user\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.rename jsnuffy jshmoe\n    "
    current_info = info(name)
    if not current_info:
        raise CommandExecutionError("User '{}' does not exist".format(name))
    new_info = info(new_name)
    if new_info:
        raise CommandExecutionError("User '{}' already exists".format(new_name))
    with salt.utils.winapi.Com():
        c = wmi.WMI(find_classes=0)
        try:
            user = c.Win32_UserAccount(Name=name)[0]
        except IndexError:
            raise CommandExecutionError("User '{}' does not exist".format(name))
        result = user.Rename(new_name)[0]
        if not result == 0:
            error_dict = {0: 'Success', 1: 'Instance not found', 2: 'Instance required', 3: 'Invalid parameter', 4: 'User not found', 5: 'Domain not found', 6: 'Operation is allowed only on the primary domain controller of the domain', 7: 'Operation is not allowed on the last administrative account', 8: 'Operation is not allowed on specified special groups: user, admin, local, or guest', 9: 'Other API error', 10: 'Internal error'}
            raise CommandExecutionError("There was an error renaming '{}' to '{}'. Error: {}".format(name, new_name, error_dict[result]))
    return info(new_name).get('name') == new_name

def current(sam=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the username that salt-minion is running under. If salt-minion is\n    running as a service it should return the Local System account. If salt is\n    running from a command prompt it should return the username that started the\n    command prompt.\n\n    .. versionadded:: 2015.5.6\n\n    Args:\n        sam (bool, optional): False returns just the username without any domain\n            notation. True returns the domain with the username in the SAM\n            format. Ie: ``domain\\username``\n\n    Returns:\n        str: Returns username\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' user.current\n    "
    try:
        if sam:
            user_name = win32api.GetUserNameEx(win32con.NameSamCompatible)
        else:
            user_name = win32api.GetUserName()
    except pywintypes.error as exc:
        log.error('Failed to get current user')
        log.error('nbr: %s', exc.winerror)
        log.error('ctx: %s', exc.funcname)
        log.error('msg: %s', exc.strerror)
        raise CommandExecutionError('Failed to get current user', info=exc)
    if not user_name:
        raise CommandExecutionError('Failed to get current user')
    return user_name