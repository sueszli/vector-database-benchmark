"""
Manage the shadow file

.. important::
    If you feel that Salt should be using this module to manage passwords on a
    minion, and it is using a different module (or gives an error similar to
    *'shadow.info' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import salt.utils.platform
__virtualname__ = 'shadow'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only works on Windows systems\n    '
    if salt.utils.platform.is_windows():
        return __virtualname__
    return (False, 'Module win_shadow: module only works on Windows systems.')

def info(name):
    if False:
        i = 10
        return i + 15
    "\n    Return information for the specified user\n    This is just returns dummy data so that salt states can work.\n\n    :param str name: The name of the user account to show.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.info root\n    "
    info = __salt__['user.info'](name=name)
    ret = {'name': name, 'passwd': '', 'lstchg': '', 'min': '', 'max': '', 'warn': '', 'inact': '', 'expire': ''}
    if info:
        ret = {'name': info['name'], 'passwd': 'Unavailable', 'lstchg': info['password_changed'], 'min': '', 'max': '', 'warn': '', 'inact': '', 'expire': info['expiration_date']}
    return ret

def set_expire(name, expire):
    if False:
        i = 10
        return i + 15
    "\n    Set the expiration date for a user account.\n\n    :param name: The name of the user account to edit.\n\n    :param expire: The date the account will expire.\n\n    :return: True if successful. False if unsuccessful.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_expire <username> 2016/7/1\n    "
    return __salt__['user.update'](name, expiration_date=expire)

def require_password_change(name):
    if False:
        while True:
            i = 10
    "\n    Require the user to change their password the next time they log in.\n\n    :param name: The name of the user account to require a password change.\n\n    :return: True if successful. False if unsuccessful.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.require_password_change <username>\n    "
    return __salt__['user.update'](name, expired=True)

def unlock_account(name):
    if False:
        return 10
    "\n    Unlocks a user account.\n\n    :param name: The name of the user account to unlock.\n\n    :return: True if successful. False if unsuccessful.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.unlock_account <username>\n    "
    return __salt__['user.update'](name, unlock_account=True)

def set_password(name, password):
    if False:
        i = 10
        return i + 15
    "\n    Set the password for a named user.\n\n    :param str name: The name of the user account\n\n    :param str password: The new password\n\n    :return: True if successful. False if unsuccessful.\n    :rtype: bool\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' shadow.set_password root mysecretpassword\n    "
    return __salt__['user.update'](name=name, password=password)