"""
Various functions to be used by windows during start up and to monkey patch
missing functions in other modules.
"""
import ctypes
import platform
import re
from salt.exceptions import CommandExecutionError
try:
    import psutil
    import pywintypes
    import win32api
    import win32net
    import win32security
    from win32con import HWND_BROADCAST, SMTO_ABORTIFHUNG, WM_SETTINGCHANGE
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if Win32 Libraries are installed\n    '
    if not HAS_WIN32:
        return (False, 'This utility requires pywin32')
    return 'win_functions'

def get_parent_pid():
    if False:
        print('Hello World!')
    '\n    This is a monkey patch for os.getppid. Used in:\n    - salt.utils.parsers\n\n    Returns:\n        int: The parent process id\n    '
    return psutil.Process().ppid()

def is_admin(name):
    if False:
        return 10
    '\n    Is the passed user a member of the Administrators group\n\n    Args:\n        name (str): The name to check\n\n    Returns:\n        bool: True if user is a member of the Administrators group, False\n        otherwise\n    '
    groups = get_user_groups(name, True)
    for group in groups:
        if group in ('S-1-5-32-544', 'S-1-5-18'):
            return True
    return False

def get_user_groups(name, sid=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the groups to which a user belongs\n\n    Args:\n        name (str): The user name to query\n        sid (bool): True will return a list of SIDs, False will return a list of\n        group names\n\n    Returns:\n        list: A list of group names or sids\n    '
    groups = []
    if name.upper() == 'SYSTEM':
        groups = ['SYSTEM']
    else:
        try:
            groups = win32net.NetUserGetLocalGroups(None, name)
        except (win32net.error, pywintypes.error) as exc:
            if exc.winerror in (5, 1722, 2453, 1927, 1355):
                groups = win32net.NetUserGetLocalGroups(None, name, 0)
            else:
                try:
                    groups = win32net.NetUserGetGroups(None, name)
                except win32net.error as exc:
                    if exc.winerror in (5, 1722, 2453, 1927, 1355):
                        groups = win32net.NetUserGetLocalGroups(None, name, 0)
                except pywintypes.error:
                    if exc.winerror in (5, 1722, 2453, 1927, 1355):
                        groups = win32net.NetUserGetLocalGroups(None, name, 1)
                    else:
                        raise
    if not sid:
        return groups
    ret_groups = []
    for group in groups:
        ret_groups.append(get_sid_from_name(group))
    return ret_groups

def get_sid_from_name(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    This is a tool for getting a sid from a name. The name can be any object.\n    Usually a user or a group\n\n    Args:\n        name (str): The name of the user or group for which to get the sid\n\n    Returns:\n        str: The corresponding SID\n    '
    if name is None:
        name = 'NULL SID'
    try:
        sid = win32security.LookupAccountName(None, name)[0]
    except pywintypes.error as exc:
        raise CommandExecutionError('User {} not found: {}'.format(name, exc))
    return win32security.ConvertSidToStringSid(sid)

def get_current_user(with_domain=True):
    if False:
        while True:
            i = 10
    '\n    Gets the user executing the process\n\n    Args:\n\n        with_domain (bool):\n            ``True`` will prepend the user name with the machine name or domain\n            separated by a backslash\n\n    Returns:\n        str: The user name\n    '
    try:
        user_name = win32api.GetUserNameEx(win32api.NameSamCompatible)
        if user_name[-1] == '$':
            test_user = win32api.GetUserName()
            if test_user == 'SYSTEM':
                user_name = 'SYSTEM'
            elif get_sid_from_name(test_user) == 'S-1-5-18':
                user_name = 'SYSTEM'
        elif not with_domain:
            user_name = win32api.GetUserName()
    except pywintypes.error as exc:
        raise CommandExecutionError('Failed to get current user: {}'.format(exc))
    if not user_name:
        return False
    return user_name

def get_sam_name(username):
    if False:
        while True:
            i = 10
    "\n    Gets the SAM name for a user. It basically prefixes a username without a\n    backslash with the computer name. If the user does not exist, a SAM\n    compatible name will be returned using the local hostname as the domain.\n\n    i.e. salt.utils.get_same_name('Administrator') would return 'DOMAIN.COM\\Administrator'\n\n    .. note:: Long computer names are truncated to 15 characters\n    "
    try:
        sid_obj = win32security.LookupAccountName(None, username)[0]
    except pywintypes.error:
        return '\\'.join([platform.node()[:15].upper(), username])
    (username, domain, _) = win32security.LookupAccountSid(None, sid_obj)
    return '\\'.join([domain, username])

def enable_ctrl_logoff_handler():
    if False:
        while True:
            i = 10
    '\n    Set the control handler on the console\n    '
    if HAS_WIN32:
        ctrl_logoff_event = 5
        win32api.SetConsoleCtrlHandler(lambda event: True if event == ctrl_logoff_event else False, 1)

def escape_argument(arg, escape=True):
    if False:
        i = 10
        return i + 15
    '\n    Escape the argument for the cmd.exe shell.\n    See http://blogs.msdn.com/b/twistylittlepassagesallalike/archive/2011/04/23/everyone-quotes-arguments-the-wrong-way.aspx\n\n    First we escape the quote chars to produce a argument suitable for\n    CommandLineToArgvW. We don\'t need to do this for simple arguments.\n\n    Args:\n        arg (str): a single command line argument to escape for the cmd.exe shell\n\n    Kwargs:\n        escape (bool): True will call the escape_for_cmd_exe() function\n                       which escapes the characters \'()%!^"<>&|\'. False\n                       will not call the function and only quotes the cmd\n\n    Returns:\n        str: an escaped string suitable to be passed as a program argument to the cmd.exe shell\n    '
    if not arg or re.search('(["\\s])', arg):
        arg = '"' + arg.replace('"', '\\"') + '"'
    if not escape:
        return arg
    return escape_for_cmd_exe(arg)

def escape_for_cmd_exe(arg):
    if False:
        print('Hello World!')
    '\n    Escape an argument string to be suitable to be passed to\n    cmd.exe on Windows\n\n    This method takes an argument that is expected to already be properly\n    escaped for the receiving program to be properly parsed. This argument\n    will be further escaped to pass the interpolation performed by cmd.exe\n    unchanged.\n\n    Any meta-characters will be escaped, removing the ability to e.g. use\n    redirects or variables.\n\n    Args:\n        arg (str): a single command line argument to escape for cmd.exe\n\n    Returns:\n        str: an escaped string suitable to be passed as a program argument to cmd.exe\n    '
    meta_chars = '()%!^"<>&|'
    meta_re = re.compile('(' + '|'.join((re.escape(char) for char in list(meta_chars))) + ')')
    meta_map = {char: '^{}'.format(char) for char in meta_chars}

    def escape_meta_chars(m):
        if False:
            i = 10
            return i + 15
        char = m.group(1)
        return meta_map[char]
    return meta_re.sub(escape_meta_chars, arg)

def broadcast_setting_change(message='Environment'):
    if False:
        i = 10
        return i + 15
    '\n    Send a WM_SETTINGCHANGE Broadcast to all Windows\n\n    Args:\n\n        message (str):\n            A string value representing the portion of the system that has been\n            updated and needs to be refreshed. Default is ``Environment``. These\n            are some common values:\n\n            - "Environment" : to effect a change in the environment variables\n            - "intl" : to effect a change in locale settings\n            - "Policy" : to effect a change in Group Policy Settings\n            - a leaf node in the registry\n            - the name of a section in the ``Win.ini`` file\n\n            See lParam within msdn docs for\n            `WM_SETTINGCHANGE <https://msdn.microsoft.com/en-us/library/ms725497%28VS.85%29.aspx>`_\n            for more information on Broadcasting Messages.\n\n            See GWL_WNDPROC within msdn docs for\n            `SetWindowLong <https://msdn.microsoft.com/en-us/library/windows/desktop/ms633591(v=vs.85).aspx>`_\n            for information on how to retrieve those messages.\n\n    .. note::\n        This will only affect new processes that aren\'t launched by services. To\n        apply changes to the path or registry to services, the host must be\n        restarted. The ``salt-minion``, if running as a service, will not see\n        changes to the environment until the system is restarted. Services\n        inherit their environment from ``services.exe`` which does not respond\n        to messaging events. See\n        `MSDN Documentation <https://support.microsoft.com/en-us/help/821761/changes-that-you-make-to-environment-variables-do-not-affect-services>`_\n        for more information.\n\n    CLI Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_functions\n        salt.utils.win_functions.broadcast_setting_change(\'Environment\')\n    '
    broadcast_message = ctypes.create_unicode_buffer(message)
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    result = user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, broadcast_message, SMTO_ABORTIFHUNG, 5000, 0)
    return result == 1

def guid_to_squid(guid):
    if False:
        while True:
            i = 10
    "\n    Converts a GUID   to a compressed guid (SQUID)\n\n    Each Guid has 5 parts separated by '-'. For the first three each one will be\n    totally reversed, and for the remaining two each one will be reversed by\n    every other character. Then the final compressed Guid will be constructed by\n    concatenating all the reversed parts without '-'.\n\n    .. Example::\n\n        Input:                  2BE0FA87-5B36-43CF-95C8-C68D6673FB94\n        Reversed:               78AF0EB2-63B5-FC34-598C-6CD86637BF49\n        Final Compressed Guid:  78AF0EB263B5FC34598C6CD86637BF49\n\n    Args:\n\n        guid (str): A valid GUID\n\n    Returns:\n        str: A valid compressed GUID (SQUID)\n    "
    guid_pattern = re.compile('^\\{(\\w{8})-(\\w{4})-(\\w{4})-(\\w\\w)(\\w\\w)-(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)\\}$')
    guid_match = guid_pattern.match(guid)
    squid = ''
    if guid_match is not None:
        for index in range(1, 12):
            squid += guid_match.group(index)[::-1]
    return squid

def squid_to_guid(squid):
    if False:
        i = 10
        return i + 15
    '\n    Converts a compressed GUID (SQUID) back into a GUID\n\n    Args:\n\n        squid (str): A valid compressed GUID\n\n    Returns:\n        str: A valid GUID\n    '
    squid_pattern = re.compile('^(\\w{8})(\\w{4})(\\w{4})(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)(\\w\\w)$')
    squid_match = squid_pattern.match(squid)
    guid = ''
    if squid_match is not None:
        guid = '{' + squid_match.group(1)[::-1] + '-' + squid_match.group(2)[::-1] + '-' + squid_match.group(3)[::-1] + '-' + squid_match.group(4)[::-1] + squid_match.group(5)[::-1] + '-'
        for index in range(6, 12):
            guid += squid_match.group(index)[::-1]
        guid += '}'
    return guid