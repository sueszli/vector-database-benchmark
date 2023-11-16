"""
Win System Utils

Functions shared with salt.modules.win_system and salt.grains.pending_reboot

.. versionadded:: 3001
"""
import logging
import salt.utils.win_reg
import salt.utils.win_update
try:
    import win32api
    import win32con
    HAS_WIN32_MODS = True
except ImportError:
    HAS_WIN32_MODS = False
log = logging.getLogger(__name__)
__virtualname__ = 'win_system'
MINION_VOLATILE_KEY = 'SYSTEM\\CurrentControlSet\\Services\\salt-minion\\Volatile-Data'
REBOOT_REQUIRED_NAME = 'Reboot required'

def __virtual__():
    if False:
        return 10
    '\n    Only works on Windows systems\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'win_system salt util failed to load: The util will only run on Windows systems')
    if not HAS_WIN32_MODS:
        return (False, 'win_system salt util failed to load: The util will only run on Windows systems')
    return __virtualname__

def get_computer_name():
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the Windows computer name. Uses the win32api to get the current computer\n    name.\n\n    .. versionadded:: 3001\n\n    Returns:\n        str: Returns the computer name if found. Otherwise returns ``False``.\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_computer_name()\n    '
    name = win32api.GetComputerNameEx(win32con.ComputerNamePhysicalDnsHostname)
    return name if name else False

def get_pending_computer_name():
    if False:
        print('Hello World!')
    '\n    Get a pending computer name. If the computer name has been changed, and the\n    change is pending a system reboot, this function will return the pending\n    computer name. Otherwise, ``None`` will be returned. If there was an error\n    retrieving the pending computer name, ``False`` will be returned, and an\n    error message will be logged to the minion log.\n\n    .. versionadded:: 3001\n\n    Returns:\n        str:\n            Returns the pending name if pending restart. Returns ``None`` if not\n            pending restart.\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_computer_name()\n    '
    current = get_computer_name()
    try:
        pending = salt.utils.win_reg.read_value(hive='HKLM', key='SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters', vname='NV Hostname')['vdata']
    except TypeError:
        return None
    if pending:
        return pending if pending.lower() != current.lower() else None

def get_pending_component_servicing():
    if False:
        for i in range(10):
            print('nop')
    "\n    Determine whether there are pending Component Based Servicing tasks that\n    require a reboot.\n\n    If any the following registry keys exist then a reboot is pending:\n\n    ``HKLM:\\\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootPending``\n    ``HKLM:\\\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootInProgress``\n    ``HKLM:\\\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\PackagesPending``\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if there are pending Component Based Servicing tasks,\n        otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.get_pending_component_servicing\n    "
    base_key = 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing'
    sub_keys = ('RebootPending', 'RebootInProgress', 'PackagesPending')
    for sub_key in sub_keys:
        key = '\\'.join((base_key, sub_key))
        if salt.utils.win_reg.key_exists(hive='HKLM', key=key):
            return True
    return False

def get_pending_domain_join():
    if False:
        while True:
            i = 10
    '\n    Determine whether there is a pending domain join action that requires a\n    reboot.\n\n    If any the following registry keys exist then a reboot is pending:\n\n    ``HKLM:\\\\SYSTEM\\CurrentControlSet\\Services\\Netlogon\\AvoidSpnSet``\n    ``HKLM:\\\\SYSTEM\\CurrentControlSet\\Services\\Netlogon\\JoinDomain``\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if there is a pending domain join action, otherwise\n        ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_domain_join()\n    '
    base_key = 'SYSTEM\\CurrentControlSet\\Services\\Netlogon'
    sub_keys = ('AvoidSpnSet', 'JoinDomain')
    for sub_key in sub_keys:
        key = '\\'.join((base_key, sub_key))
        if salt.utils.win_reg.key_exists(hive='HKLM', key=key):
            return True
    return False

def get_pending_file_rename():
    if False:
        i = 10
        return i + 15
    '\n    Determine whether there are pending file rename operations that require a\n    reboot.\n\n    A reboot is pending if any of the following value names exist and have value\n    data set:\n\n    - ``PendingFileRenameOperations``\n    - ``PendingFileRenameOperations2``\n\n    in the following registry key:\n\n    ``HKLM:\\\\SYSTEM\\CurrentControlSet\\Control\\Session Manager``\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if there are pending file rename operations, otherwise\n        ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_file_rename()\n    '
    vnames = ('PendingFileRenameOperations', 'PendingFileRenameOperations2')
    key = 'SYSTEM\\CurrentControlSet\\Control\\Session Manager'
    for vname in vnames:
        reg_ret = salt.utils.win_reg.read_value(hive='HKLM', key=key, vname=vname)
        if reg_ret['success']:
            if reg_ret['vdata'] and reg_ret['vdata'] != '(value not set)':
                return True
    return False

def get_pending_servermanager():
    if False:
        while True:
            i = 10
    '\n    Determine whether there are pending Server Manager tasks that require a\n    reboot.\n\n    A reboot is pending if the ``CurrentRebootAttempts`` value name exists and\n    has an integer value. The value name resides in the following registry key:\n\n    ``HKLM:\\\\SOFTWARE\\Microsoft\\ServerManager``\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if there are pending Server Manager tasks, otherwise\n        ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_servermanager()\n    '
    vname = 'CurrentRebootAttempts'
    key = 'SOFTWARE\\Microsoft\\ServerManager'
    reg_ret = salt.utils.win_reg.read_value(hive='HKLM', key=key, vname=vname)
    if reg_ret['success']:
        try:
            if int(reg_ret['vdata']) > 0:
                return True
        except ValueError:
            pass
    return False

def get_pending_dvd_reboot():
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine whether the DVD Reboot flag is set.\n\n    The system requires a reboot if the ``DVDRebootSignal`` value name exists\n    at the following registry location:\n\n    ``HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnce``\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if the above condition is met, otherwise ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_dvd_reboot()\n    '
    return salt.utils.win_reg.value_exists(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnce', vname='DVDRebootSignal')

def get_pending_update():
    if False:
        return 10
    '\n    Determine whether there are pending updates that require a reboot.\n\n    If either of the following registry keys exists, a reboot is pending:\n\n    ``HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update\\RebootRequired``\n    ``HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update\\PostRebootReporting``\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if any of the above conditions are met, otherwise\n        ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_update()\n    '
    base_key = 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update'
    sub_keys = ('RebootRequired', 'PostRebootReporting')
    for sub_key in sub_keys:
        key = '\\'.join((base_key, sub_key))
        if salt.utils.win_reg.key_exists(hive='HKLM', key=key):
            return True
    return False

def get_reboot_required_witnessed():
    if False:
        print('Hello World!')
    '\n    Determine if at any time during the current boot session the salt minion\n    witnessed an event indicating that a reboot is required.\n\n    This function will return ``True`` if an install completed with exit\n    code 3010 during the current boot session and can be extended where\n    appropriate in the future.\n\n    If the ``Reboot required`` value name exists in the following location and\n    has a value of ``1`` then the system is pending reboot:\n\n    ``HKLM:\\\\SYSTEM\\CurrentControlSet\\Services\\salt-minion\\Volatile-Data``\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if the ``Requires reboot`` registry flag is set to ``1``,\n        otherwise ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_reboot_required_witnessed()\n\n    '
    value_dict = salt.utils.win_reg.read_value(hive='HKLM', key=MINION_VOLATILE_KEY, vname=REBOOT_REQUIRED_NAME)
    return value_dict['vdata'] == 1

def set_reboot_required_witnessed():
    if False:
        for i in range(10):
            print('nop')
    "\n    This function is used to remember that an event indicating that a reboot is\n    required was witnessed. This function relies on the salt-minion's ability to\n    create the following volatile registry key in the *HKLM* hive:\n\n       *SYSTEM\\CurrentControlSet\\Services\\salt-minion\\Volatile-Data*\n\n    Because this registry key is volatile, it will not persist beyond the\n    current boot session. Also, in the scope of this key, the name *'Reboot\n    required'* will be assigned the value of *1*.\n\n    For the time being, this function is being used whenever an install\n    completes with exit code 3010 and can be extended where appropriate in the\n    future.\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.set_reboot_required_witnessed()\n    "
    return salt.utils.win_reg.set_value(hive='HKLM', key=MINION_VOLATILE_KEY, volatile=True, vname=REBOOT_REQUIRED_NAME, vdata=1, vtype='REG_DWORD')

def get_pending_update_exe_volatile():
    if False:
        print('Hello World!')
    '\n    Determine whether there is a volatile update exe that requires a reboot.\n\n    Checks ``HKLM:\\Microsoft\\Updates``. If the ``UpdateExeVolatile`` value\n    name is anything other than 0 there is a reboot pending\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if there is a volatile exe, otherwise ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_update_exe_volatile()\n    '
    key = 'SOFTWARE\\Microsoft\\Updates'
    reg_ret = salt.utils.win_reg.read_value(hive='HKLM', key=key, vname='UpdateExeVolatile')
    if reg_ret['success']:
        try:
            if int(reg_ret['vdata']) != 0:
                return True
        except ValueError:
            pass
    return False

def get_pending_windows_update():
    if False:
        i = 10
        return i + 15
    '\n    Check the Windows Update system for a pending reboot state.\n\n    This leverages the Windows Update System to determine if the system is\n    pending a reboot.\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if the Windows Update system reports a pending update,\n        otherwise ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_windows_update()\n    '
    return salt.utils.win_update.needs_reboot()

def get_pending_reboot():
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine whether there is a reboot pending.\n\n    .. versionadded:: 3001\n\n    Returns:\n        bool: ``True`` if the system is pending reboot, otherwise ``False``\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_reboot()\n    '
    checks = (get_pending_update, get_pending_windows_update, get_pending_update_exe_volatile, get_pending_file_rename, get_pending_servermanager, get_pending_component_servicing, get_pending_dvd_reboot, get_reboot_required_witnessed, get_pending_computer_name, get_pending_domain_join)
    for check in checks:
        if check():
            return True
    return False

def get_pending_reboot_details():
    if False:
        return 10
    '\n    Determine which check is signalling that the system is pending a reboot.\n    Useful in determining why your system is signalling that it needs a reboot.\n\n    .. versionadded:: 3001\n\n    Returns:\n        dict: A dictionary of the results of each function that checks for a\n        pending reboot\n\n    Example:\n\n    .. code-block:: python\n\n        import salt.utils.win_system\n        salt.utils.win_system.get_pending_reboot_details()\n    '
    return {'Pending Component Servicing': get_pending_component_servicing(), 'Pending Computer Rename': get_pending_computer_name() is not None, 'Pending DVD Reboot': get_pending_dvd_reboot(), 'Pending File Rename': get_pending_file_rename(), 'Pending Join Domain': get_pending_domain_join(), 'Pending ServerManager': get_pending_servermanager(), 'Pending Update': get_pending_update(), 'Pending Windows Update': get_pending_windows_update(), 'Reboot Required Witnessed': get_reboot_required_witnessed(), 'Volatile Update Exe': get_pending_update_exe_volatile()}