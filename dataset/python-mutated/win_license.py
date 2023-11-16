"""
This module allows you to manage windows licensing via slmgr.vbs

.. code-block:: bash

    salt '*' license.install XXXXX-XXXXX-XXXXX-XXXXX-XXXXX
"""
import logging
import re
import salt.utils.platform
log = logging.getLogger(__name__)
__virtualname__ = 'license'

def __virtual__():
    if False:
        return 10
    '\n    Only work on Windows\n    '
    if salt.utils.platform.is_windows():
        return __virtualname__
    return (False, 'Module win_license: module only works on Windows systems.')

def installed(product_key):
    if False:
        while True:
            i = 10
    "\n    Check to see if the product key is already installed.\n\n    Note: This is not 100% accurate as we can only see the last\n     5 digits of the license.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' license.installed XXXXX-XXXXX-XXXXX-XXXXX-XXXXX\n    "
    cmd = 'cscript C:\\Windows\\System32\\slmgr.vbs /dli'
    out = __salt__['cmd.run'](cmd)
    return product_key[-5:] in out

def install(product_key):
    if False:
        return 10
    "\n    Install the given product key\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' license.install XXXXX-XXXXX-XXXXX-XXXXX-XXXXX\n    "
    cmd = 'cscript C:\\Windows\\System32\\slmgr.vbs /ipk {}'.format(product_key)
    return __salt__['cmd.run'](cmd)

def uninstall():
    if False:
        i = 10
        return i + 15
    "\n    Uninstall the current product key\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' license.uninstall\n    "
    cmd = 'cscript C:\\Windows\\System32\\slmgr.vbs /upk'
    return __salt__['cmd.run'](cmd)

def activate():
    if False:
        return 10
    "\n    Attempt to activate the current machine via Windows Activation\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' license.activate\n    "
    cmd = 'cscript C:\\Windows\\System32\\slmgr.vbs /ato'
    return __salt__['cmd.run'](cmd)

def licensed():
    if False:
        while True:
            i = 10
    "\n    Return true if the current machine is licensed correctly\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' license.licensed\n    "
    cmd = 'cscript C:\\Windows\\System32\\slmgr.vbs /dli'
    out = __salt__['cmd.run'](cmd)
    return 'License Status: Licensed' in out

def info():
    if False:
        print('Hello World!')
    "\n    Return information about the license, if the license is not\n    correctly activated this will return None.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' license.info\n    "
    cmd = 'cscript C:\\Windows\\System32\\slmgr.vbs /dli'
    out = __salt__['cmd.run'](cmd)
    match = re.search('Name: (.*)\\r\\nDescription: (.*)\\r\\nPartial Product Key: (.*)\\r\\nLicense Status: (.*)', out, re.MULTILINE)
    if match is not None:
        groups = match.groups()
        return {'name': groups[0], 'description': groups[1], 'partial_key': groups[2], 'licensed': 'Licensed' in groups[3]}
    return None