"""
A salt module for modifying the audit policies on the machine

Though this module does not set group policy for auditing, it displays how all
auditing configuration is applied on the machine, either set directly or via
local or domain group policy.

.. versionadded:: 2018.3.4
.. versionadded:: 2019.2.1

This module allows you to view and modify the audit settings as they are applied
on the machine. The audit settings are broken down into nine categories:

- Account Logon
- Account Management
- Detailed Tracking
- DS Access
- Logon/Logoff
- Object Access
- Policy Change
- Privilege Use
- System

The ``get_settings`` function will return the subcategories for all nine of
the above categories in one dictionary along with their auditing status.

To modify a setting you only need to specify the subcategory name and the value
you wish to set. Valid settings are:

- No Auditing
- Success
- Failure
- Success and Failure

CLI Example:

.. code-block:: bash

    # Get current state of all audit settings
    salt * auditpol.get_settings

    # Get the current state of all audit settings in the "Account Logon"
    # category
    salt * auditpol.get_settings category="Account Logon"

    # Get current state of the "Credential Validation" setting
    salt * auditpol.get_setting name="Credential Validation"

    # Set the state of the "Credential Validation" setting to Success and
    # Failure
    salt * auditpol.set_setting name="Credential Validation" value="Success and Failure"

    # Set the state of the "Credential Validation" setting to No Auditing
    salt * auditpol.set_setting name="Credential Validation" value="No Auditing"
"""
import salt.utils.platform
__virtualname__ = 'auditpol'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only works on Windows systems\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Module win_auditpol: module only available on Windows')
    return __virtualname__

def get_settings(category='All'):
    if False:
        i = 10
        return i + 15
    '\n    Get the current configuration for all audit settings specified in the\n    category\n\n    Args:\n        category (str):\n            One of the nine categories to return. Can also be ``All`` to return\n            the settings for all categories. Valid options are:\n\n            - Account Logon\n            - Account Management\n            - Detailed Tracking\n            - DS Access\n            - Logon/Logoff\n            - Object Access\n            - Policy Change\n            - Privilege Use\n            - System\n            - All\n\n            Default value is ``All``\n\n    Returns:\n        dict: A dictionary containing all subcategories for the specified\n            category along with their current configuration\n\n    Raises:\n        KeyError: On invalid category\n        CommandExecutionError: If an error is encountered retrieving the settings\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Get current state of all audit settings\n        salt * auditipol.get_settings\n\n        # Get the current state of all audit settings in the "Account Logon"\n        # category\n        salt * auditpol.get_settings "Account Logon"\n    '
    return __utils__['auditpol.get_settings'](category=category)

def get_setting(name):
    if False:
        i = 10
        return i + 15
    '\n    Get the current configuration for the named audit setting\n\n    Args:\n        name (str): The name of the setting to retrieve\n\n    Returns:\n        str: The current configuration for the named setting\n\n    Raises:\n        KeyError: On invalid setting name\n        CommandExecutionError: If an error is encountered retrieving the settings\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Get current state of the "Credential Validation" setting\n        salt * auditpol.get_setting "Credential Validation"\n    '
    return __utils__['auditpol.get_setting'](name=name)

def set_setting(name, value):
    if False:
        i = 10
        return i + 15
    '\n    Set the configuration for the named audit setting\n\n    Args:\n\n        name (str):\n            The name of the setting to configure\n\n        value (str):\n            The configuration for the named value. Valid options are:\n\n            - No Auditing\n            - Success\n            - Failure\n            - Success and Failure\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        KeyError: On invalid ``name`` or ``value``\n        CommandExecutionError: If an error is encountered modifying the setting\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Set the state of the "Credential Validation" setting to Success and\n        # Failure\n        salt * auditpol.set_setting "Credential Validation" "Success and Failure"\n\n        # Set the state of the "Credential Validation" setting to No Auditing\n        salt * auditpol.set_setting "Credential Validation" "No Auditing"\n    '
    return __utils__['auditpol.set_setting'](name=name, value=value)