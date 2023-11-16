"""
Microsoft Update files management via wusa.exe

:maintainer:    Thomas Lemarchand
:platform:      Windows
:depends:       PowerShell

.. versionadded:: 2018.3.4
"""
import logging
import os
import salt.utils.platform
import salt.utils.win_pwsh
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'wusa'
__func_alias__ = {'list_': 'list'}

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Load only on Windows\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Only available on Windows systems')
    powershell_info = __salt__['cmd.shell_info'](shell='powershell', list_modules=False)
    if not powershell_info['installed']:
        return (False, 'PowerShell not available')
    return __virtualname__

def is_installed(name):
    if False:
        i = 10
        return i + 15
    "\n    Check if a specific KB is installed.\n\n    Args:\n\n        name (str):\n            The name of the KB to check\n\n    Returns:\n        bool: ``True`` if installed, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' wusa.is_installed KB123456\n    "
    return __salt__['cmd.retcode'](cmd=f'Get-HotFix -Id {name}', shell='powershell', ignore_retcode=True) == 0

def install(path, restart=False):
    if False:
        while True:
            i = 10
    "\n    Install a KB from a .msu file.\n\n    Args:\n\n        path (str):\n            The full path to the msu file to install\n\n        restart (bool):\n            ``True`` to force a restart if required by the installation. Adds\n            the ``/forcerestart`` switch to the ``wusa.exe`` command. ``False``\n            will add the ``/norestart`` switch instead. Default is ``False``\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    Raise:\n        CommandExecutionError: If the package is already installed or an error\n            is encountered\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' wusa.install C:/temp/KB123456.msu\n    "
    cmd = ['wusa.exe', path, '/quiet']
    if restart:
        cmd.append('/forcerestart')
    else:
        cmd.append('/norestart')
    ret_code = __salt__['cmd.retcode'](cmd, ignore_retcode=True)
    file_name = os.path.basename(path)
    errors = {2359302: f'{file_name} is already installed', 3010: f'{file_name} correctly installed but server reboot is needed to complete installation', 87: 'Unknown error'}
    if ret_code in errors:
        raise CommandExecutionError(errors[ret_code], ret_code)
    elif ret_code:
        raise CommandExecutionError(f'Unknown error: {ret_code}')
    return True

def uninstall(path, restart=False):
    if False:
        i = 10
        return i + 15
    "\n    Uninstall a specific KB.\n\n    Args:\n\n        path (str):\n            The full path to the msu file to uninstall. This can also be just\n            the name of the KB to uninstall\n\n        restart (bool):\n            ``True`` to force a restart if required by the installation. Adds\n            the ``/forcerestart`` switch to the ``wusa.exe`` command. ``False``\n            will add the ``/norestart`` switch instead. Default is ``False``\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    Raises:\n        CommandExecutionError: If an error is encountered\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' wusa.uninstall KB123456\n\n        # or\n\n        salt '*' wusa.uninstall C:/temp/KB123456.msu\n    "
    cmd = ['wusa.exe', '/uninstall', '/quiet']
    kb = os.path.splitext(os.path.basename(path))[0]
    if os.path.exists(path):
        cmd.append(path)
    else:
        cmd.append('/kb:{}'.format(kb[2:] if kb.lower().startswith('kb') else kb))
    if restart:
        cmd.append('/forcerestart')
    else:
        cmd.append('/norestart')
    ret_code = __salt__['cmd.retcode'](cmd, ignore_retcode=True)
    errors = {-2145116156: f'{kb} does not support uninstall', 2359303: f'{kb} not installed', 87: 'Unknown error. Try specifying an .msu file'}
    if ret_code in errors:
        raise CommandExecutionError(errors[ret_code], ret_code)
    elif ret_code:
        raise CommandExecutionError(f'Unknown error: {ret_code}')
    return True

def list_():
    if False:
        print('Hello World!')
    "\n    Get a list of updates installed on the machine\n\n    Returns:\n        list: A list of installed updates\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' wusa.list\n    "
    kbs = []
    ret = salt.utils.win_pwsh.run_dict('Get-HotFix | Select HotFixID')
    for item in ret:
        kbs.append(item['HotFixID'])
    return kbs