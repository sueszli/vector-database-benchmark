"""
Module for editing power settings on macOS

 .. versionadded:: 2016.3.0
"""
import salt.utils.mac_utils
import salt.utils.platform
from salt.exceptions import SaltInvocationError
__virtualname__ = 'power'

def __virtual__():
    if False:
        return 10
    '\n    Only for macOS\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'The mac_power module could not be loaded: module only works on macOS systems.')
    return __virtualname__

def _validate_sleep(minutes):
    if False:
        return 10
    '\n    Helper function that validates the minutes parameter. Can be any number\n    between 1 and 180. Can also be the string values "Never" and "Off".\n\n    Because "On" and "Off" get converted to boolean values on the command line\n    it will error if "On" is passed\n\n    Returns: The value to be passed to the command\n    '
    if isinstance(minutes, str):
        if minutes.lower() in ['never', 'off']:
            return 'Never'
        else:
            msg = 'Invalid String Value for Minutes.\nString values must be "Never" or "Off".\nPassed: {}'.format(minutes)
            raise SaltInvocationError(msg)
    elif isinstance(minutes, bool):
        if minutes:
            msg = 'Invalid Boolean Value for Minutes.\nBoolean value "On" or "True" is not allowed.\nSalt CLI converts "On" to boolean True.\nPassed: {}'.format(minutes)
            raise SaltInvocationError(msg)
        else:
            return 'Never'
    elif isinstance(minutes, int):
        if minutes in range(1, 181):
            return minutes
        else:
            msg = 'Invalid Integer Value for Minutes.\nInteger values must be between 1 and 180.\nPassed: {}'.format(minutes)
            raise SaltInvocationError(msg)
    else:
        msg = 'Unknown Variable Type Passed for Minutes.\nPassed: {}'.format(minutes)
        raise SaltInvocationError(msg)

def get_sleep():
    if False:
        while True:
            i = 10
    "\n    Displays the amount of idle time until the machine sleeps. Settings for\n    Computer, Display, and Hard Disk are displayed.\n\n    :return: A dictionary containing the sleep status for Computer, Display, and\n        Hard Disk\n\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' power.get_sleep\n    "
    return {'Computer': get_computer_sleep(), 'Display': get_display_sleep(), 'Hard Disk': get_harddisk_sleep()}

def set_sleep(minutes):
    if False:
        return 10
    '\n    Sets the amount of idle time until the machine sleeps. Sets the same value\n    for Computer, Display, and Hard Disk. Pass "Never" or "Off" for computers\n    that should never sleep.\n\n    :param minutes: Can be an integer between 1 and 180 or "Never" or "Off"\n    :ptype: int, str\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_sleep 120\n        salt \'*\' power.set_sleep never\n    '
    value = _validate_sleep(minutes)
    cmd = 'systemsetup -setsleep {}'.format(value)
    salt.utils.mac_utils.execute_return_success(cmd)
    state = []
    for check in (get_computer_sleep, get_display_sleep, get_harddisk_sleep):
        state.append(salt.utils.mac_utils.confirm_updated(value, check))
    return all(state)

def get_computer_sleep():
    if False:
        for i in range(10):
            print('nop')
    "\n    Display the amount of idle time until the computer sleeps.\n\n    :return: A string representing the sleep settings for the computer\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' power.get_computer_sleep\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getcomputersleep')
    return salt.utils.mac_utils.parse_return(ret)

def set_computer_sleep(minutes):
    if False:
        print('Hello World!')
    '\n    Set the amount of idle time until the computer sleeps. Pass "Never" of "Off"\n    to never sleep.\n\n    :param minutes: Can be an integer between 1 and 180 or "Never" or "Off"\n    :ptype: int, str\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_computer_sleep 120\n        salt \'*\' power.set_computer_sleep off\n    '
    value = _validate_sleep(minutes)
    cmd = 'systemsetup -setcomputersleep {}'.format(value)
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.confirm_updated(str(value), get_computer_sleep)

def get_display_sleep():
    if False:
        for i in range(10):
            print('nop')
    "\n    Display the amount of idle time until the display sleeps.\n\n    :return: A string representing the sleep settings for the displey\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' power.get_display_sleep\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getdisplaysleep')
    return salt.utils.mac_utils.parse_return(ret)

def set_display_sleep(minutes):
    if False:
        print('Hello World!')
    '\n    Set the amount of idle time until the display sleeps. Pass "Never" of "Off"\n    to never sleep.\n\n    :param minutes: Can be an integer between 1 and 180 or "Never" or "Off"\n    :ptype: int, str\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_display_sleep 120\n        salt \'*\' power.set_display_sleep off\n    '
    value = _validate_sleep(minutes)
    cmd = 'systemsetup -setdisplaysleep {}'.format(value)
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.confirm_updated(str(value), get_display_sleep)

def get_harddisk_sleep():
    if False:
        return 10
    "\n    Display the amount of idle time until the hard disk sleeps.\n\n    :return: A string representing the sleep settings for the hard disk\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' power.get_harddisk_sleep\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getharddisksleep')
    return salt.utils.mac_utils.parse_return(ret)

def set_harddisk_sleep(minutes):
    if False:
        while True:
            i = 10
    '\n    Set the amount of idle time until the harddisk sleeps. Pass "Never" of "Off"\n    to never sleep.\n\n    :param minutes: Can be an integer between 1 and 180 or "Never" or "Off"\n    :ptype: int, str\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_harddisk_sleep 120\n        salt \'*\' power.set_harddisk_sleep off\n    '
    value = _validate_sleep(minutes)
    cmd = 'systemsetup -setharddisksleep {}'.format(value)
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.confirm_updated(str(value), get_harddisk_sleep)

def get_wake_on_modem():
    if False:
        i = 10
        return i + 15
    '\n    Displays whether \'wake on modem\' is on or off if supported\n\n    :return: A string value representing the "wake on modem" settings\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.get_wake_on_modem\n    '
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getwakeonmodem')
    return salt.utils.mac_utils.validate_enabled(salt.utils.mac_utils.parse_return(ret)) == 'on'

def set_wake_on_modem(enabled):
    if False:
        while True:
            i = 10
    '\n    Set whether or not the computer will wake from sleep when modem activity is\n    detected.\n\n    :param bool enabled: True to enable, False to disable. "On" and "Off" are\n        also acceptable values. Additionally you can pass 1 and 0 to represent\n        True and False respectively\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_wake_on_modem True\n    '
    state = salt.utils.mac_utils.validate_enabled(enabled)
    cmd = 'systemsetup -setwakeonmodem {}'.format(state)
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.confirm_updated(state, get_wake_on_modem)

def get_wake_on_network():
    if False:
        print('Hello World!')
    '\n    Displays whether \'wake on network\' is on or off if supported\n\n    :return: A string value representing the "wake on network" settings\n    :rtype: string\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.get_wake_on_network\n    '
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getwakeonnetworkaccess')
    return salt.utils.mac_utils.validate_enabled(salt.utils.mac_utils.parse_return(ret)) == 'on'

def set_wake_on_network(enabled):
    if False:
        while True:
            i = 10
    '\n    Set whether or not the computer will wake from sleep when network activity\n    is detected.\n\n    :param bool enabled: True to enable, False to disable. "On" and "Off" are\n        also acceptable values. Additionally you can pass 1 and 0 to represent\n        True and False respectively\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_wake_on_network True\n    '
    state = salt.utils.mac_utils.validate_enabled(enabled)
    cmd = 'systemsetup -setwakeonnetworkaccess {}'.format(state)
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.confirm_updated(state, get_wake_on_network)

def get_restart_power_failure():
    if False:
        print('Hello World!')
    '\n    Displays whether \'restart on power failure\' is on or off if supported\n\n    :return: A string value representing the "restart on power failure" settings\n    :rtype: string\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.get_restart_power_failure\n    '
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getrestartpowerfailure')
    return salt.utils.mac_utils.validate_enabled(salt.utils.mac_utils.parse_return(ret)) == 'on'

def set_restart_power_failure(enabled):
    if False:
        print('Hello World!')
    '\n    Set whether or not the computer will automatically restart after a power\n    failure.\n\n    :param bool enabled: True to enable, False to disable. "On" and "Off" are\n        also acceptable values. Additionally you can pass 1 and 0 to represent\n        True and False respectively\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_restart_power_failure True\n    '
    state = salt.utils.mac_utils.validate_enabled(enabled)
    cmd = 'systemsetup -setrestartpowerfailure {}'.format(state)
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.confirm_updated(state, get_restart_power_failure)

def get_restart_freeze():
    if False:
        i = 10
        return i + 15
    '\n    Displays whether \'restart on freeze\' is on or off if supported\n\n    :return: A string value representing the "restart on freeze" settings\n    :rtype: string\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.get_restart_freeze\n    '
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getrestartfreeze')
    return salt.utils.mac_utils.validate_enabled(salt.utils.mac_utils.parse_return(ret)) == 'on'

def set_restart_freeze(enabled):
    if False:
        print('Hello World!')
    '\n    Specifies whether the server restarts automatically after a system freeze.\n    This setting doesn\'t seem to be editable. The command completes successfully\n    but the setting isn\'t actually updated. This is probably a macOS. The\n    functions remains in case they ever fix the bug.\n\n    :param bool enabled: True to enable, False to disable. "On" and "Off" are\n        also acceptable values. Additionally you can pass 1 and 0 to represent\n        True and False respectively\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_restart_freeze True\n    '
    state = salt.utils.mac_utils.validate_enabled(enabled)
    cmd = 'systemsetup -setrestartfreeze {}'.format(state)
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.confirm_updated(state, get_restart_freeze, True)

def get_sleep_on_power_button():
    if False:
        for i in range(10):
            print('nop')
    '\n    Displays whether \'allow power button to sleep computer\' is on or off if\n    supported\n\n    :return: A string value representing the "allow power button to sleep\n        computer" settings\n\n    :rtype: string\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.get_sleep_on_power_button\n    '
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getallowpowerbuttontosleepcomputer')
    return salt.utils.mac_utils.validate_enabled(salt.utils.mac_utils.parse_return(ret)) == 'on'

def set_sleep_on_power_button(enabled):
    if False:
        return 10
    '\n    Set whether or not the power button can sleep the computer.\n\n    :param bool enabled: True to enable, False to disable. "On" and "Off" are\n        also acceptable values. Additionally you can pass 1 and 0 to represent\n        True and False respectively\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' power.set_sleep_on_power_button True\n    '
    state = salt.utils.mac_utils.validate_enabled(enabled)
    cmd = 'systemsetup -setallowpowerbuttontosleepcomputer {}'.format(state)
    salt.utils.mac_utils.execute_return_success(cmd)
    return salt.utils.mac_utils.confirm_updated(state, get_sleep_on_power_button)