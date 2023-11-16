"""
System module for sleeping, restarting, and shutting down the system on Mac OS X

.. versionadded:: 2016.3.0

.. warning::
    Using this module will enable ``atrun`` on the system if it is disabled.
"""
import getpass
import shlex
import salt.utils.platform
from salt.exceptions import CommandExecutionError, SaltInvocationError
__virtualname__ = 'system'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only for MacOS with atrun enabled\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'The mac_system module could not be loaded: module only works on MacOS systems.')
    if getpass.getuser() != 'root':
        return (False, 'The mac_system module is not useful for non-root users.')
    if not _atrun_enabled():
        if not _enable_atrun():
            return (False, 'atrun could not be enabled on this system')
    return __virtualname__

def _atrun_enabled():
    if False:
        i = 10
        return i + 15
    '\n    Check to see if atrun is running and enabled on the system\n    '
    try:
        return __salt__['service.list']('com.apple.atrun')
    except CommandExecutionError:
        return False

def _enable_atrun():
    if False:
        return 10
    '\n    Enable and start the atrun daemon\n    '
    name = 'com.apple.atrun'
    try:
        __salt__['service.enable'](name)
        __salt__['service.start'](name)
    except CommandExecutionError:
        return False
    return _atrun_enabled()

def _execute_command(cmd, at_time=None):
    if False:
        while True:
            i = 10
    '\n    Helper function to execute the command\n\n    :param str cmd: the command to run\n\n    :param str at_time: If passed, the cmd will be scheduled.\n\n    Returns: bool\n    '
    if at_time:
        cmd = "echo '{}' | at {}".format(cmd, shlex.quote(at_time))
    return not bool(__salt__['cmd.retcode'](cmd, python_shell=True))

def halt(at_time=None):
    if False:
        print('Hello World!')
    '\n    Halt a running system\n\n    :param str at_time: Any valid `at` expression. For example, some valid at\n        expressions could be:\n\n        - noon\n        - midnight\n        - fri\n        - 9:00 AM\n        - 2:30 PM tomorrow\n        - now + 10 minutes\n\n    .. note::\n        If you pass a time only, with no \'AM/PM\' designation, you have to\n        double quote the parameter on the command line. For example: \'"14:00"\'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.halt\n        salt \'*\' system.halt \'now + 10 minutes\'\n    '
    cmd = 'shutdown -h now'
    return _execute_command(cmd, at_time)

def sleep(at_time=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sleep the system. If a user is active on the system it will likely fail to\n    sleep.\n\n    :param str at_time: Any valid `at` expression. For example, some valid at\n        expressions could be:\n\n        - noon\n        - midnight\n        - fri\n        - 9:00 AM\n        - 2:30 PM tomorrow\n        - now + 10 minutes\n\n    .. note::\n        If you pass a time only, with no \'AM/PM\' designation, you have to\n        double quote the parameter on the command line. For example: \'"14:00"\'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.sleep\n        salt \'*\' system.sleep \'10:00 PM\'\n    '
    cmd = 'shutdown -s now'
    return _execute_command(cmd, at_time)

def restart(at_time=None):
    if False:
        print('Hello World!')
    '\n    Restart the system\n\n    :param str at_time: Any valid `at` expression. For example, some valid at\n        expressions could be:\n\n        - noon\n        - midnight\n        - fri\n        - 9:00 AM\n        - 2:30 PM tomorrow\n        - now + 10 minutes\n\n    .. note::\n        If you pass a time only, with no \'AM/PM\' designation, you have to\n        double quote the parameter on the command line. For example: \'"14:00"\'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.restart\n        salt \'*\' system.restart \'12:00 PM fri\'\n    '
    cmd = 'shutdown -r now'
    return _execute_command(cmd, at_time)

def shutdown(at_time=None):
    if False:
        print('Hello World!')
    '\n    Shutdown the system\n\n    :param str at_time: Any valid `at` expression. For example, some valid at\n        expressions could be:\n\n        - noon\n        - midnight\n        - fri\n        - 9:00 AM\n        - 2:30 PM tomorrow\n        - now + 10 minutes\n\n    .. note::\n        If you pass a time only, with no \'AM/PM\' designation, you have to\n        double quote the parameter on the command line. For example: \'"14:00"\'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.shutdown\n        salt \'*\' system.shutdown \'now + 1 hour\'\n    '
    return halt(at_time)

def get_remote_login():
    if False:
        for i in range(10):
            print('nop')
    "\n    Displays whether remote login (SSH) is on or off.\n\n    :return: True if remote login is on, False if off\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_remote_login\n    "
    ret = __utils__['mac_utils.execute_return_result']('systemsetup -getremotelogin')
    enabled = __utils__['mac_utils.validate_enabled'](__utils__['mac_utils.parse_return'](ret))
    return enabled == 'on'

def set_remote_login(enable):
    if False:
        print('Hello World!')
    '\n    Set the remote login (SSH) to either on or off.\n\n    :param bool enable: True to enable, False to disable. "On" and "Off" are\n        also acceptable values. Additionally you can pass 1 and 0 to represent\n        True and False respectively\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.set_remote_login True\n    '
    state = __utils__['mac_utils.validate_enabled'](enable)
    cmd = 'systemsetup -f -setremotelogin {}'.format(state)
    __utils__['mac_utils.execute_return_success'](cmd)
    return __utils__['mac_utils.confirm_updated'](state, get_remote_login, normalize_ret=True)

def get_remote_events():
    if False:
        return 10
    "\n    Displays whether remote apple events are on or off.\n\n    :return: True if remote apple events are on, False if off\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_remote_events\n    "
    ret = __utils__['mac_utils.execute_return_result']('systemsetup -getremoteappleevents')
    enabled = __utils__['mac_utils.validate_enabled'](__utils__['mac_utils.parse_return'](ret))
    return enabled == 'on'

def set_remote_events(enable):
    if False:
        return 10
    '\n    Set whether the server responds to events sent by other computers (such as\n    AppleScripts)\n\n    :param bool enable: True to enable, False to disable. "On" and "Off" are\n        also acceptable values. Additionally you can pass 1 and 0 to represent\n        True and False respectively\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.set_remote_events On\n    '
    state = __utils__['mac_utils.validate_enabled'](enable)
    cmd = 'systemsetup -setremoteappleevents {}'.format(state)
    __utils__['mac_utils.execute_return_success'](cmd)
    return __utils__['mac_utils.confirm_updated'](state, get_remote_events, normalize_ret=True)

def get_computer_name():
    if False:
        print('Hello World!')
    "\n    Gets the computer name\n\n    :return: The computer name\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_computer_name\n    "
    ret = __utils__['mac_utils.execute_return_result']('scutil --get ComputerName')
    return __utils__['mac_utils.parse_return'](ret)

def set_computer_name(name):
    if False:
        i = 10
        return i + 15
    '\n    Set the computer name\n\n    :param str name: The new computer name\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.set_computer_name "Mike\'s Mac"\n    '
    cmd = 'scutil --set ComputerName "{}"'.format(name)
    __utils__['mac_utils.execute_return_success'](cmd)
    return __utils__['mac_utils.confirm_updated'](name, get_computer_name)

def get_subnet_name():
    if False:
        while True:
            i = 10
    "\n    Gets the local subnet name\n\n    :return: The local subnet name\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_subnet_name\n    "
    ret = __utils__['mac_utils.execute_return_result']('systemsetup -getlocalsubnetname')
    return __utils__['mac_utils.parse_return'](ret)

def set_subnet_name(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the local subnet name\n\n    :param str name: The new local subnet name\n\n    .. note::\n       Spaces are changed to dashes. Other special characters are removed.\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        The following will be set as \'Mikes-Mac\'\n        salt \'*\' system.set_subnet_name "Mike\'s Mac"\n    '
    cmd = 'systemsetup -setlocalsubnetname "{}"'.format(name)
    __utils__['mac_utils.execute_return_success'](cmd)
    return __utils__['mac_utils.confirm_updated'](name, get_subnet_name)

def get_startup_disk():
    if False:
        while True:
            i = 10
    "\n    Displays the current startup disk\n\n    :return: The current startup disk\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_startup_disk\n    "
    ret = __utils__['mac_utils.execute_return_result']('systemsetup -getstartupdisk')
    return __utils__['mac_utils.parse_return'](ret)

def list_startup_disks():
    if False:
        while True:
            i = 10
    "\n    List all valid startup disks on the system.\n\n    :return: A list of valid startup disks\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.list_startup_disks\n    "
    ret = __utils__['mac_utils.execute_return_result']('systemsetup -liststartupdisks')
    return ret.splitlines()

def set_startup_disk(path):
    if False:
        print('Hello World!')
    "\n    Set the current startup disk to the indicated path. Use\n    ``system.list_startup_disks`` to find valid startup disks on the system.\n\n    :param str path: The valid startup disk path\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.set_startup_disk /System/Library/CoreServices\n    "
    if path not in list_startup_disks():
        msg = 'Invalid value passed for path.\nMust be a valid startup disk as found in system.list_startup_disks.\nPassed: {}'.format(path)
        raise SaltInvocationError(msg)
    cmd = 'systemsetup -setstartupdisk {}'.format(path)
    __utils__['mac_utils.execute_return_result'](cmd)
    return __utils__['mac_utils.confirm_updated'](path, get_startup_disk)

def get_restart_delay():
    if False:
        i = 10
        return i + 15
    "\n    Get the number of seconds after which the computer will start up after a\n    power failure.\n\n    :return: A string value representing the number of seconds the system will\n        delay restart after power loss\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_restart_delay\n    "
    ret = __utils__['mac_utils.execute_return_result']('systemsetup -getwaitforstartupafterpowerfailure')
    return __utils__['mac_utils.parse_return'](ret)

def set_restart_delay(seconds):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the number of seconds after which the computer will start up after a\n    power failure.\n\n    .. warning::\n\n        This command fails with the following error:\n\n        ``Error, IOServiceOpen returned 0x10000003``\n\n        The setting is not updated. This is an apple bug. It seems like it may\n        only work on certain versions of Mac Server X. This article explains the\n        issue in more detail, though it is quite old.\n\n        http://lists.apple.com/archives/macos-x-server/2006/Jul/msg00967.html\n\n    :param int seconds: The number of seconds. Must be a multiple of 30\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.set_restart_delay 180\n    "
    if seconds % 30 != 0:
        msg = 'Invalid value passed for seconds.\nMust be a multiple of 30.\nPassed: {}'.format(seconds)
        raise SaltInvocationError(msg)
    cmd = 'systemsetup -setwaitforstartupafterpowerfailure {}'.format(seconds)
    __utils__['mac_utils.execute_return_success'](cmd)
    return __utils__['mac_utils.confirm_updated'](seconds, get_restart_delay)

def get_disable_keyboard_on_lock():
    if False:
        print('Hello World!')
    "\n    Get whether or not the keyboard should be disabled when the X Serve enclosure\n    lock is engaged.\n\n    :return: True if disable keyboard on lock is on, False if off\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_disable_keyboard_on_lock\n    "
    ret = __utils__['mac_utils.execute_return_result']('systemsetup -getdisablekeyboardwhenenclosurelockisengaged')
    enabled = __utils__['mac_utils.validate_enabled'](__utils__['mac_utils.parse_return'](ret))
    return enabled == 'on'

def set_disable_keyboard_on_lock(enable):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get whether or not the keyboard should be disabled when the X Serve\n    enclosure lock is engaged.\n\n    :param bool enable: True to enable, False to disable. "On" and "Off" are\n        also acceptable values. Additionally you can pass 1 and 0 to represent\n        True and False respectively\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.set_disable_keyboard_on_lock False\n    '
    state = __utils__['mac_utils.validate_enabled'](enable)
    cmd = 'systemsetup -setdisablekeyboardwhenenclosurelockisengaged {}'.format(state)
    __utils__['mac_utils.execute_return_success'](cmd)
    return __utils__['mac_utils.confirm_updated'](state, get_disable_keyboard_on_lock, normalize_ret=True)

def get_boot_arch():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the kernel architecture setting from ``com.apple.Boot.plist``\n\n    :return: A string value representing the boot architecture setting\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_boot_arch\n    "
    ret = __utils__['mac_utils.execute_return_result']('systemsetup -getkernelbootarchitecturesetting')
    arch = __utils__['mac_utils.parse_return'](ret)
    if 'default' in arch:
        return 'default'
    elif 'i386' in arch:
        return 'i386'
    elif 'x86_64' in arch:
        return 'x86_64'
    return 'unknown'

def set_boot_arch(arch='default'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the kernel to boot in 32 or 64 bit mode on next boot.\n\n    .. note::\n        When this function fails with the error ``changes to kernel\n        architecture failed to save!``, then the boot arch is not updated.\n        This is either an Apple bug, not available on the test system, or a\n        result of system files being locked down in macOS (SIP Protection).\n\n    :param str arch: A string representing the desired architecture. If no\n        value is passed, default is assumed. Valid values include:\n\n        - i386\n        - x86_64\n        - default\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.set_boot_arch i386\n    "
    if arch not in ['i386', 'x86_64', 'default']:
        msg = 'Invalid value passed for arch.\nMust be i386, x86_64, or default.\nPassed: {}'.format(arch)
        raise SaltInvocationError(msg)
    cmd = 'systemsetup -setkernelbootarchitecture {}'.format(arch)
    __utils__['mac_utils.execute_return_success'](cmd)
    return __utils__['mac_utils.confirm_updated'](arch, get_boot_arch)