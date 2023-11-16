"""
Manage Windows features via the ServerManager powershell module. Can list
available and installed roles/features. Can install and remove roles/features.

:maintainer:    Shane Lee <slee@saltstack.com>
:platform:      Windows Server 2008R2 or greater
:depends:       PowerShell module ``ServerManager``
"""
import logging
import shlex
import salt.utils.json
import salt.utils.platform
import salt.utils.powershell
import salt.utils.versions
import salt.utils.win_pwsh
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'win_servermanager'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Load only on windows with servermanager module\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Module win_servermanager: module only works on Windows systems.')
    if salt.utils.versions.version_cmp(__grains__['osversion'], '6.1.7600') == -1:
        return (False, 'Failed to load win_servermanager module: Requires Remote Server Administration Tools which is only available on Windows 2008 R2 and later.')
    if not salt.utils.powershell.module_exists('ServerManager'):
        return (False, 'Failed to load win_servermanager module: ServerManager module not available. May need to install Remote Server Administration Tools.')
    return __virtualname__

def list_available():
    if False:
        while True:
            i = 10
    "\n    List available features to install\n\n    Returns:\n        str: A list of available features as returned by the\n        ``Get-WindowsFeature`` PowerShell command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_servermanager.list_available\n    "
    cmd = 'Import-Module ServerManager; Get-WindowsFeature -ErrorAction SilentlyContinue -WarningAction SilentlyContinue'
    return __salt__['cmd.shell'](cmd, shell='powershell')

def list_installed():
    if False:
        return 10
    "\n    List installed features. Supported on Windows Server 2008 and Windows 8 and\n    newer.\n\n    Returns:\n        dict: A dictionary of installed features\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' win_servermanager.list_installed\n    "
    cmd = 'Get-WindowsFeature -ErrorAction SilentlyContinue -WarningAction SilentlyContinue | Select DisplayName,Name,Installed'
    features = salt.utils.win_pwsh.run_dict(cmd)
    ret = {}
    for entry in features:
        if entry['Installed']:
            ret[entry['Name']] = entry['DisplayName']
    return ret

def install(feature, recurse=False, restart=False, source=None, exclude=None):
    if False:
        return 10
    '\n    Install a feature\n\n    .. note::\n        Some features require reboot after un/installation, if so until the\n        server is restarted other features can not be installed!\n\n    .. note::\n        Some features take a long time to complete un/installation, set -t with\n        a long timeout\n\n    Args:\n\n        feature (str, list):\n            The name of the feature(s) to install. This can be a single feature,\n            a string of features in a comma delimited list (no spaces), or a\n            list of features.\n\n            .. versionadded:: 2018.3.0\n                Added the ability to pass a list of features to be installed.\n\n        recurse (Options[bool]):\n            Install all sub-features. Default is False\n\n        restart (Optional[bool]):\n            Restarts the computer when installation is complete, if required by\n            the role/feature installed. Will also trigger a reboot if an item\n            in ``exclude`` requires a reboot to be properly removed. Default is\n            False\n\n        source (Optional[str]):\n            Path to the source files if missing from the target system. None\n            means that the system will use windows update services to find the\n            required files. Default is None\n\n        exclude (Optional[str]):\n            The name of the feature to exclude when installing the named\n            feature. This can be a single feature, a string of features in a\n            comma-delimited list (no spaces), or a list of features.\n\n            .. warning::\n                As there is no exclude option for the ``Add-WindowsFeature``\n                or ``Install-WindowsFeature`` PowerShell commands the features\n                named in ``exclude`` will be installed with other sub-features\n                and will then be removed. **If the feature named in ``exclude``\n                is not a sub-feature of one of the installed items it will still\n                be removed.**\n\n    Returns:\n        dict: A dictionary containing the results of the install\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Install the Telnet Client passing a single string\n        salt \'*\' win_servermanager.install Telnet-Client\n\n        # Install the TFTP Client and the SNMP Service passing a comma-delimited\n        # string. Install all sub-features\n        salt \'*\' win_servermanager.install TFTP-Client,SNMP-Service recurse=True\n\n        # Install the TFTP Client from d:\\side-by-side\n        salt \'*\' win_servermanager.install TFTP-Client source=d:\\\\side-by-side\n\n        # Install the XPS Viewer, SNMP Service, and Remote Access passing a\n        # list. Install all sub-features, but exclude the Web Server\n        salt \'*\' win_servermanager.install "[\'XPS-Viewer\', \'SNMP-Service\', \'RemoteAccess\']" True recurse=True exclude="Web-Server"\n    '
    if isinstance(feature, list):
        feature = ','.join(feature)
    command = 'Add-WindowsFeature'
    management_tools = ''
    if salt.utils.versions.version_cmp(__grains__['osversion'], '6.2') >= 0:
        command = 'Install-WindowsFeature'
        management_tools = '-IncludeManagementTools'
    cmd = '{} -Name {} {} {} {} -WarningAction SilentlyContinue'.format(command, shlex.quote(feature), management_tools, '-IncludeAllSubFeature' if recurse else '', '' if source is None else f'-Source {source}')
    out = salt.utils.win_pwsh.run_dict(cmd)
    if exclude is not None:
        removed = remove(exclude)
    if out['FeatureResult']:
        ret = {'ExitCode': out['ExitCode'], 'RestartNeeded': False, 'Restarted': False, 'Features': {}, 'Success': out['Success']}
        for item in out['FeatureResult']:
            ret['Features'][item['Name']] = {'DisplayName': item['DisplayName'], 'Message': item['Message'], 'RestartNeeded': item['RestartNeeded'], 'SkipReason': item['SkipReason'], 'Success': item['Success']}
            if item['RestartNeeded']:
                ret['RestartNeeded'] = True
        for item in feature.split(','):
            if item not in ret['Features']:
                ret['Features'][item] = {'Message': 'Already installed'}
        if exclude is not None:
            for item in removed['Features']:
                if item in ret['Features']:
                    ret['Features'][item] = {'Message': 'Removed after installation (exclude)', 'DisplayName': removed['Features'][item]['DisplayName'], 'RestartNeeded': removed['Features'][item]['RestartNeeded'], 'SkipReason': removed['Features'][item]['SkipReason'], 'Success': removed['Features'][item]['Success']}
                    if removed['Features'][item]['RestartNeeded']:
                        ret['RestartNeeded'] = True
        if restart:
            if ret['RestartNeeded']:
                if __salt__['system.reboot'](in_seconds=True):
                    ret['Restarted'] = True
        return ret
    else:
        ret = {'ExitCode': out['ExitCode'], 'Features': {}, 'RestartNeeded': False, 'Restarted': False, 'Success': out['Success']}
        for item in feature.split(','):
            ret['Features'][item] = {'Message': 'Already installed'}
        return ret

def remove(feature, remove_payload=False, restart=False):
    if False:
        while True:
            i = 10
    "\n    Remove an installed feature\n\n    .. note::\n        Some features require a reboot after installation/uninstallation. If\n        one of these features are modified, then other features cannot be\n        installed until the server is restarted. Additionally, some features\n        take a while to complete installation/uninstallation, so it is a good\n        idea to use the ``-t`` option to set a longer timeout.\n\n    Args:\n\n        feature (str, list):\n            The name of the feature(s) to remove. This can be a single feature,\n            a string of features in a comma delimited list (no spaces), or a\n            list of features.\n\n            .. versionadded:: 2018.3.0\n                Added the ability to pass a list of features to be removed.\n\n        remove_payload (Optional[bool]):\n            True will cause the feature to be removed from the side-by-side\n            store (``%SystemDrive%:\\Windows\\WinSxS``). Default is False\n\n        restart (Optional[bool]):\n            Restarts the computer when uninstall is complete, if required by the\n            role/feature removed. Default is False\n\n    Returns:\n        dict: A dictionary containing the results of the uninstall\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt -t 600 '*' win_servermanager.remove Telnet-Client\n    "
    if isinstance(feature, list):
        feature = ','.join(feature)
    command = 'Remove-WindowsFeature'
    management_tools = ''
    _remove_payload = ''
    if salt.utils.versions.version_cmp(__grains__['osversion'], '6.2') >= 0:
        command = 'Uninstall-WindowsFeature'
        management_tools = '-IncludeManagementTools'
        if remove_payload:
            _remove_payload = '-Remove'
    cmd = '{} -Name {} {} {} {} -WarningAction SilentlyContinue'.format(command, shlex.quote(feature), management_tools, _remove_payload, '-Restart' if restart else '')
    try:
        out = salt.utils.win_pwsh.run_dict(cmd)
    except CommandExecutionError as exc:
        if 'ArgumentNotValid' in exc.message:
            raise CommandExecutionError('Invalid Feature Name', info=exc.info)
        raise
    if out['FeatureResult']:
        ret = {'ExitCode': out['ExitCode'], 'RestartNeeded': False, 'Restarted': False, 'Features': {}, 'Success': out['Success']}
        for item in out['FeatureResult']:
            ret['Features'][item['Name']] = {'DisplayName': item['DisplayName'], 'Message': item['Message'], 'RestartNeeded': item['RestartNeeded'], 'SkipReason': item['SkipReason'], 'Success': item['Success']}
        for item in feature.split(','):
            if item not in ret['Features']:
                ret['Features'][item] = {'Message': 'Not installed'}
        return ret
    else:
        ret = {'ExitCode': out['ExitCode'], 'Features': {}, 'RestartNeeded': False, 'Restarted': False, 'Success': out['Success']}
        for item in feature.split(','):
            ret['Features'][item] = {'Message': 'Not installed'}
        return ret