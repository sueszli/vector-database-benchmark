"""
Manage macOS local directory passwords and policies

.. versionadded:: 2016.3.0

Note that it is usually better to apply password policies through the creation
of a configuration profile.
"""
import logging
from datetime import datetime
import salt.utils.mac_utils
import salt.utils.platform
from salt.exceptions import CommandExecutionError
try:
    import pwd
    HAS_PWD = True
except ImportError:
    HAS_PWD = False
log = logging.getLogger(__name__)
__virtualname__ = 'shadow'

def __virtual__():
    if False:
        i = 10
        return i + 15
    if not salt.utils.platform.is_darwin():
        return (False, 'Not macOS')
    if HAS_PWD:
        return __virtualname__
    else:
        return (False, 'The pwd module failed to load.')

def _get_account_policy(name):
    if False:
        return 10
    '\n    Get the entire accountPolicy and return it as a dictionary. For use by this\n    module only\n\n    :param str name: The user name\n\n    :return: a dictionary containing all values for the accountPolicy\n    :rtype: dict\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n    '
    cmd = 'pwpolicy -u {} -getpolicy'.format(name)
    try:
        ret = salt.utils.mac_utils.execute_return_result(cmd)
    except CommandExecutionError as exc:
        if 'Error: user <{}> not found'.format(name) in exc.strerror:
            raise CommandExecutionError('User not found: {}'.format(name))
        raise CommandExecutionError('Unknown error: {}'.format(exc.strerror))
    try:
        policy_list = ret.split('\n')[1].split(' ')
        policy_dict = {}
        for policy in policy_list:
            if '=' in policy:
                (key, value) = policy.split('=')
                policy_dict[key] = value
        return policy_dict
    except IndexError:
        return {}

def _set_account_policy(name, policy):
    if False:
        while True:
            i = 10
    '\n    Set a value in the user accountPolicy. For use by this module only\n\n    :param str name: The user name\n    :param str policy: The policy to apply\n\n    :return: True if success, otherwise False\n    :rtype: bool\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n    '
    cmd = 'pwpolicy -u {} -setpolicy "{}"'.format(name, policy)
    try:
        return salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        if 'Error: user <{}> not found'.format(name) in exc.strerror:
            raise CommandExecutionError('User not found: {}'.format(name))
        raise CommandExecutionError('Unknown error: {}'.format(exc.strerror))

def _get_account_policy_data_value(name, key):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the value for a key in the accountPolicy section of the user's plist\n    file. For use by this module only\n\n    :param str name: The username\n    :param str key: The accountPolicy key\n\n    :return: The value contained within the key\n    :rtype: str\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n    "
    cmd = 'dscl . -readpl /Users/{} accountPolicyData {}'.format(name, key)
    try:
        ret = salt.utils.mac_utils.execute_return_result(cmd)
    except CommandExecutionError as exc:
        if 'eDSUnknownNodeName' in exc.strerror:
            raise CommandExecutionError('User not found: {}'.format(name))
        raise CommandExecutionError('Unknown error: {}'.format(exc.strerror))
    return ret

def _convert_to_datetime(unix_timestamp):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts a unix timestamp to a human readable date/time\n\n    :param float unix_timestamp: A unix timestamp\n\n    :return: A date/time in the format YYYY-mm-dd HH:MM:SS\n    :rtype: str\n    '
    try:
        unix_timestamp = float(unix_timestamp)
        return datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return 'Invalid Timestamp'

def info(name):
    if False:
        return 10
    "\n    Return information for the specified user\n\n    :param str name: The username\n\n    :return: A dictionary containing the user's shadow information\n    :rtype: dict\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.info admin\n    "
    try:
        data = pwd.getpwnam(name)
        return {'name': data.pw_name, 'passwd': data.pw_passwd, 'account_created': get_account_created(name), 'login_failed_count': get_login_failed_count(name), 'login_failed_last': get_login_failed_last(name), 'lstchg': get_last_change(name), 'max': get_maxdays(name), 'expire': get_expire(name), 'change': get_change(name), 'min': 'Unavailable', 'warn': 'Unavailable', 'inact': 'Unavailable'}
    except KeyError:
        log.debug('User not found: %s', name)
        return {'name': '', 'passwd': '', 'account_created': '', 'login_failed_count': '', 'login_failed_last': '', 'lstchg': '', 'max': '', 'expire': '', 'change': '', 'min': '', 'warn': '', 'inact': ''}

def get_account_created(name):
    if False:
        while True:
            i = 10
    "\n    Get the date/time the account was created\n\n    :param str name: The username of the account\n\n    :return: The date/time the account was created (yyyy-mm-dd hh:mm:ss)\n    :rtype: str\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.get_account_created admin\n    "
    ret = _get_account_policy_data_value(name, 'creationTime')
    unix_timestamp = salt.utils.mac_utils.parse_return(ret)
    date_text = _convert_to_datetime(unix_timestamp)
    return date_text

def get_last_change(name):
    if False:
        return 10
    "\n    Get the date/time the account was changed\n\n    :param str name: The username of the account\n\n    :return: The date/time the account was modified (yyyy-mm-dd hh:mm:ss)\n    :rtype: str\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.get_last_change admin\n    "
    ret = _get_account_policy_data_value(name, 'passwordLastSetTime')
    unix_timestamp = salt.utils.mac_utils.parse_return(ret)
    date_text = _convert_to_datetime(unix_timestamp)
    return date_text

def get_login_failed_count(name):
    if False:
        print('Hello World!')
    "\n    Get the number of failed login attempts\n\n    :param str name: The username of the account\n\n    :return: The number of failed login attempts\n    :rtype: int\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.get_login_failed_count admin\n    "
    ret = _get_account_policy_data_value(name, 'failedLoginCount')
    return salt.utils.mac_utils.parse_return(ret)

def get_login_failed_last(name):
    if False:
        while True:
            i = 10
    "\n    Get the date/time of the last failed login attempt\n\n    :param str name: The username of the account\n\n    :return: The date/time of the last failed login attempt on this account\n        (yyyy-mm-dd hh:mm:ss)\n    :rtype: str\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.get_login_failed_last admin\n    "
    ret = _get_account_policy_data_value(name, 'failedLoginTimestamp')
    unix_timestamp = salt.utils.mac_utils.parse_return(ret)
    date_text = _convert_to_datetime(unix_timestamp)
    return date_text

def set_maxdays(name, days):
    if False:
        while True:
            i = 10
    "\n    Set the maximum age of the password in days\n\n    :param str name: The username of the account\n\n    :param int days: The maximum age of the account in days\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_maxdays admin 90\n    "
    minutes = days * 24 * 60
    _set_account_policy(name, 'maxMinutesUntilChangePassword={}'.format(minutes))
    return get_maxdays(name) == days

def get_maxdays(name):
    if False:
        i = 10
        return i + 15
    "\n    Get the maximum age of the password\n\n    :param str name: The username of the account\n\n    :return: The maximum age of the password in days\n    :rtype: int\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.get_maxdays admin 90\n    "
    policies = _get_account_policy(name)
    if 'maxMinutesUntilChangePassword' in policies:
        max_minutes = policies['maxMinutesUntilChangePassword']
        return int(max_minutes) / 24 / 60
    return 0

def set_mindays(name, days):
    if False:
        while True:
            i = 10
    "\n    Set the minimum password age in days. Not available in macOS.\n\n    :param str name: The user name\n\n    :param int days: The number of days\n\n    :return: Will always return False until macOS supports this feature.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_mindays admin 90\n    "
    return False

def set_inactdays(name, days):
    if False:
        print('Hello World!')
    "\n    Set the number if inactive days before the account is locked. Not available\n    in macOS\n\n    :param str name: The user name\n\n    :param int days: The number of days\n\n    :return: Will always return False until macOS supports this feature.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_inactdays admin 90\n    "
    return False

def set_warndays(name, days):
    if False:
        while True:
            i = 10
    "\n    Set the number of days before the password expires that the user will start\n    to see a warning. Not available in macOS\n\n    :param str name: The user name\n\n    :param int days: The number of days\n\n    :return: Will always return False until macOS supports this feature.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_warndays admin 90\n    "
    return False

def set_change(name, date):
    if False:
        print('Hello World!')
    "\n    Sets the date on which the password expires. The user will be required to\n    change their password. Format is mm/dd/yyyy\n\n    :param str name: The name of the user account\n\n    :param date date: The date the password will expire. Must be in mm/dd/yyyy\n        format.\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_change username 09/21/2016\n    "
    _set_account_policy(name, 'usingExpirationDate=1 expirationDateGMT={}'.format(date))
    return get_change(name) == date

def get_change(name):
    if False:
        while True:
            i = 10
    "\n    Gets the date on which the password expires\n\n    :param str name: The name of the user account\n\n    :return: The date the password will expire\n    :rtype: str\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.get_change username\n    "
    policies = _get_account_policy(name)
    if 'expirationDateGMT' in policies:
        return policies['expirationDateGMT']
    return 'Value not set'

def set_expire(name, date):
    if False:
        i = 10
        return i + 15
    "\n    Sets the date on which the account expires. The user will not be able to\n    login after this date. Date format is mm/dd/yyyy\n\n    :param str name: The name of the user account\n\n    :param datetime date: The date the account will expire. Format must be\n        mm/dd/yyyy.\n\n    :return: True if successful, False if not\n    :rtype: bool\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_expire username 07/23/2015\n    "
    _set_account_policy(name, 'usingHardExpirationDate=1 hardExpireDateGMT={}'.format(date))
    return get_expire(name) == date

def get_expire(name):
    if False:
        return 10
    "\n    Gets the date on which the account expires\n\n    :param str name: The name of the user account\n\n    :return: The date the account expires\n    :rtype: str\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.get_expire username\n    "
    policies = _get_account_policy(name)
    if 'hardExpireDateGMT' in policies:
        return policies['hardExpireDateGMT']
    return 'Value not set'

def del_password(name):
    if False:
        return 10
    "\n    Deletes the account password\n\n    :param str name: The user name of the account\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.del_password username\n    "
    cmd = "dscl . -passwd /Users/{} ''".format(name)
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        if 'eDSUnknownNodeName' in exc.strerror:
            raise CommandExecutionError('User not found: {}'.format(name))
        raise CommandExecutionError('Unknown error: {}'.format(exc.strerror))
    cmd = "dscl . -create /Users/{} Password '*'".format(name)
    salt.utils.mac_utils.execute_return_success(cmd)
    return info(name)['passwd'] == '*'

def set_password(name, password):
    if False:
        i = 10
        return i + 15
    "\n    Set the password for a named user (insecure, the password will be in the\n    process list while the command is running)\n\n    :param str name: The name of the local user, which is assumed to be in the\n        local directory service\n\n    :param str password: The plaintext password to set\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    :raises: CommandExecutionError on user not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' mac_shadow.set_password macuser macpassword\n    "
    cmd = "dscl . -passwd /Users/{} '{}'".format(name, password)
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        if 'eDSUnknownNodeName' in exc.strerror:
            raise CommandExecutionError('User not found: {}'.format(name))
        raise CommandExecutionError('Unknown error: {}'.format(exc.strerror))
    return True