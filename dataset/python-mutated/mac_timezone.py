"""
Module for editing date/time settings on macOS

 .. versionadded:: 2016.3.0
"""
from datetime import datetime
import salt.utils.mac_utils
import salt.utils.platform
from salt.exceptions import SaltInvocationError
__virtualname__ = 'timezone'

def __virtual__():
    if False:
        return 10
    '\n    Only for macOS\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'The mac_timezone module could not be loaded: module only works on macOS systems.')
    return __virtualname__

def _get_date_time_format(dt_string):
    if False:
        while True:
            i = 10
    '\n    Function that detects the date/time format for the string passed.\n\n    :param str dt_string:\n        A date/time string\n\n    :return: The format of the passed dt_string\n    :rtype: str\n\n    :raises: SaltInvocationError on Invalid Date/Time string\n    '
    valid_formats = ['%H:%M', '%H:%M:%S', '%m:%d:%y', '%m:%d:%Y', '%m/%d/%y', '%m/%d/%Y']
    for dt_format in valid_formats:
        try:
            datetime.strptime(dt_string, dt_format)
            return dt_format
        except ValueError:
            continue
    msg = 'Invalid Date/Time Format: {}'.format(dt_string)
    raise SaltInvocationError(msg)

def get_date():
    if False:
        print('Hello World!')
    "\n    Displays the current date\n\n    :return: the system date\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.get_date\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getdate')
    return salt.utils.mac_utils.parse_return(ret)

def set_date(date):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the current month, day, and year\n\n    :param str date: The date to set. Valid date formats are:\n\n        - %m:%d:%y\n        - %m:%d:%Y\n        - %m/%d/%y\n        - %m/%d/%Y\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    :raises: SaltInvocationError on Invalid Date format\n    :raises: CommandExecutionError on failure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.set_date 1/13/2016\n    "
    date_format = _get_date_time_format(date)
    dt_obj = datetime.strptime(date, date_format)
    cmd = 'systemsetup -setdate {}'.format(dt_obj.strftime('%m:%d:%Y'))
    return salt.utils.mac_utils.execute_return_success(cmd)

def get_time():
    if False:
        return 10
    "\n    Get the current system time.\n\n    :return: The current time in 24 hour format\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.get_time\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -gettime')
    return salt.utils.mac_utils.parse_return(ret)

def set_time(time):
    if False:
        while True:
            i = 10
    '\n    Sets the current time. Must be in 24 hour format.\n\n    :param str time: The time to set in 24 hour format.  The value must be\n        double quoted. ie: \'"17:46"\'\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    :raises: SaltInvocationError on Invalid Time format\n    :raises: CommandExecutionError on failure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' timezone.set_time \'"17:34"\'\n    '
    time_format = _get_date_time_format(time)
    dt_obj = datetime.strptime(time, time_format)
    cmd = 'systemsetup -settime {}'.format(dt_obj.strftime('%H:%M:%S'))
    return salt.utils.mac_utils.execute_return_success(cmd)

def get_zone():
    if False:
        i = 10
        return i + 15
    "\n    Displays the current time zone\n\n    :return: The current time zone\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.get_zone\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -gettimezone')
    return salt.utils.mac_utils.parse_return(ret)

def get_zonecode():
    if False:
        i = 10
        return i + 15
    "\n    Displays the current time zone abbreviated code\n\n    :return: The current time zone code\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.get_zonecode\n    "
    return salt.utils.mac_utils.execute_return_result('date +%Z')

def get_offset():
    if False:
        while True:
            i = 10
    "\n    Displays the current time zone offset\n\n    :return: The current time zone offset\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.get_offset\n    "
    return salt.utils.mac_utils.execute_return_result('date +%z')

def list_zones():
    if False:
        return 10
    "\n    Displays a list of available time zones. Use this list when setting a\n    time zone using ``timezone.set_zone``\n\n    :return: a list of time zones\n    :rtype: list\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.list_zones\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -listtimezones')
    zones = salt.utils.mac_utils.parse_return(ret)
    return [x.strip() for x in zones.splitlines()]

def set_zone(time_zone):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the local time zone. Use ``timezone.list_zones`` to list valid time_zone\n    arguments\n\n    :param str time_zone: The time zone to apply\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    :raises: SaltInvocationError on Invalid Timezone\n    :raises: CommandExecutionError on failure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.set_zone America/Denver\n    "
    if time_zone not in list_zones():
        raise SaltInvocationError('Invalid Timezone: {}'.format(time_zone))
    salt.utils.mac_utils.execute_return_success('systemsetup -settimezone {}'.format(time_zone))
    return time_zone in get_zone()

def zone_compare(time_zone):
    if False:
        while True:
            i = 10
    "\n    Compares the given timezone name with the system timezone name.\n\n    :return: True if they are the same, False if not\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.zone_compare America/Boise\n    "
    return time_zone == get_zone()

def get_using_network_time():
    if False:
        for i in range(10):
            print('nop')
    "\n    Display whether network time is on or off\n\n    :return: True if network time is on, False if off\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.get_using_network_time\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getusingnetworktime')
    return salt.utils.mac_utils.validate_enabled(salt.utils.mac_utils.parse_return(ret)) == 'on'

def set_using_network_time(enable):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set whether network time is on or off.\n\n    :param enable: True to enable, False to disable. Can also use 'on' or 'off'\n    :type: str bool\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    :raises: CommandExecutionError on failure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.set_using_network_time True\n    "
    state = salt.utils.mac_utils.validate_enabled(enable)
    cmd = 'systemsetup -setusingnetworktime {}'.format(state)
    salt.utils.mac_utils.execute_return_success(cmd)
    return state == salt.utils.mac_utils.validate_enabled(get_using_network_time())

def get_time_server():
    if False:
        print('Hello World!')
    "\n    Display the currently set network time server.\n\n    :return: the network time server\n    :rtype: str\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.get_time_server\n    "
    ret = salt.utils.mac_utils.execute_return_result('systemsetup -getnetworktimeserver')
    return salt.utils.mac_utils.parse_return(ret)

def set_time_server(time_server='time.apple.com'):
    if False:
        while True:
            i = 10
    "\n    Designates a network time server. Enter the IP address or DNS name for the\n    network time server.\n\n    :param time_server: IP or DNS name of the network time server. If nothing\n        is passed the time server will be set to the macOS default of\n        'time.apple.com'\n    :type: str\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    :raises: CommandExecutionError on failure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.set_time_server time.acme.com\n    "
    cmd = 'systemsetup -setnetworktimeserver {}'.format(time_server)
    salt.utils.mac_utils.execute_return_success(cmd)
    return time_server in get_time_server()

def get_hwclock():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get current hardware clock setting (UTC or localtime)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.get_hwclock\n    "
    return False

def set_hwclock(clock):
    if False:
        print('Hello World!')
    "\n    Sets the hardware clock to be either UTC or localtime\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' timezone.set_hwclock UTC\n    "
    return False