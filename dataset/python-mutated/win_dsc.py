"""
Module for working with Windows PowerShell DSC (Desired State Configuration)

This module is Alpha

This module applies DSC Configurations in the form of PowerShell scripts or
MOF (Managed Object Format) schema files.

Use the ``psget`` module to manage PowerShell resources.

The idea is to leverage Salt to push DSC configuration scripts or MOF files to
the Minion.

:depends:
    - PowerShell 5.0
"""
import logging
import os
import salt.utils.json
import salt.utils.platform
import salt.utils.versions
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)
__virtualname__ = 'dsc'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Set the system module of the kernel is Windows\n    '
    if not salt.utils.platform.is_windows():
        log.debug('DSC: Only available on Windows systems')
        return (False, 'DSC: Only available on Windows systems')
    powershell_info = __salt__['cmd.shell_info']('powershell')
    if not powershell_info['installed']:
        log.debug('DSC: Requires PowerShell')
        return (False, 'DSC: Requires PowerShell')
    if salt.utils.versions.compare(powershell_info['version'], '<', '5.0'):
        log.debug('DSC: Requires PowerShell 5 or later')
        return (False, 'DSC: Requires PowerShell 5 or later')
    return __virtualname__

def _pshell(cmd, cwd=None, json_depth=2, ignore_retcode=False):
    if False:
        return 10
    '\n    Execute the desired PowerShell command and ensure that it returns data\n    in json format and load that into python. Either return a dict or raise a\n    CommandExecutionError.\n    '
    if 'convertto-json' not in cmd.lower():
        cmd = '{} | ConvertTo-Json -Depth {}'.format(cmd, json_depth)
    log.debug('DSC: %s', cmd)
    results = __salt__['cmd.run_all'](cmd, shell='powershell', cwd=cwd, python_shell=True, ignore_retcode=ignore_retcode)
    if 'pid' in results:
        del results['pid']
    if 'retcode' not in results or results['retcode'] != 0:
        raise CommandExecutionError('Issue executing PowerShell {}'.format(cmd), info=results)
    if results['stdout'] == '':
        results['stdout'] = '{}'
    try:
        ret = salt.utils.json.loads(results['stdout'], strict=False)
    except ValueError:
        raise CommandExecutionError('No JSON results from PowerShell', info=results)
    log.info('DSC: Returning "%s"', ret)
    return ret

def run_config(path, source=None, config_name=None, config_data=None, config_data_source=None, script_parameters=None, salt_env='base'):
    if False:
        while True:
            i = 10
    '\n    Compile a DSC Configuration in the form of a PowerShell script (.ps1) and\n    apply it. The PowerShell script can be cached from the master using the\n    ``source`` option. If there is more than one config within the PowerShell\n    script, the desired configuration can be applied by passing the name in the\n    ``config`` option.\n\n    This command would be the equivalent of running ``dsc.compile_config``\n    followed by ``dsc.apply_config``.\n\n    Args:\n\n        path (str): The local path to the PowerShell script that contains the\n            DSC Configuration. Required.\n\n        source (str): The path to the script on ``file_roots`` to cache at the\n            location specified by ``path``. The source file will be cached\n            locally and then executed. If source is not passed, the config\n            script located at ``path`` will be compiled. Optional.\n\n        config_name (str): The name of the Configuration within the script to\n            apply. If the script contains multiple configurations within the\n            file a ``config_name`` must be specified. If the ``config_name`` is\n            not specified, the name of the file will be used as the\n            ``config_name`` to run. Optional.\n\n        config_data (str): Configuration data in the form of a hash table that\n            will be passed to the ``ConfigurationData`` parameter when the\n            ``config_name`` is compiled. This can be the path to a ``.psd1``\n            file containing the proper hash table or the PowerShell code to\n            create the hash table.\n\n            .. versionadded:: 2017.7.0\n\n        config_data_source (str): The path to the ``.psd1`` file on\n            ``file_roots`` to cache at the location specified by\n            ``config_data``. If this is specified, ``config_data`` must be a\n            local path instead of a hash table.\n\n            .. versionadded:: 2017.7.0\n\n        script_parameters (str): Any additional parameters expected by the\n            configuration script. These must be defined in the script itself.\n            Note that these are passed to the script (the outermost scope), and\n            not to the dsc configuration inside the script (the inner scope).\n\n            .. versionadded:: 2017.7.0\n\n        salt_env (str): The salt environment to use when copying the source.\n            Default is \'base\'\n\n    Returns:\n        bool: True if successfully compiled and applied, otherwise False\n\n    CLI Example:\n\n    To compile a config from a script that already exists on the system:\n\n    .. code-block:: bash\n\n        salt \'*\' dsc.run_config C:\\\\DSC\\\\WebsiteConfig.ps1\n\n    To cache a config script to the system from the master and compile it:\n\n    .. code-block:: bash\n\n        salt \'*\' dsc.run_config C:\\\\DSC\\\\WebsiteConfig.ps1 salt://dsc/configs/WebsiteConfig.ps1\n\n    To cache a config script to the system from the master and compile it, passing in `script_parameters`:\n\n    .. code-block:: bash\n\n        salt \'*\' dsc.run_config path=C:\\\\DSC\\\\WebsiteConfig.ps1 source=salt://dsc/configs/WebsiteConfig.ps1 script_parameters="-hostname \'my-computer\' -ip \'192.168.1.10\' -DnsArray \'192.168.1.3\',\'192.168.1.4\',\'1.1.1.1\'"\n    '
    ret = compile_config(path=path, source=source, config_name=config_name, config_data=config_data, config_data_source=config_data_source, script_parameters=script_parameters, salt_env=salt_env)
    if ret.get('Exists'):
        config_path = os.path.dirname(ret['FullName'])
        return apply_config(config_path)
    else:
        return False

def compile_config(path, source=None, config_name=None, config_data=None, config_data_source=None, script_parameters=None, salt_env='base'):
    if False:
        while True:
            i = 10
    "\n    Compile a config from a PowerShell script (``.ps1``)\n\n    Args:\n\n        path (str): Path (local) to the script that will create the ``.mof``\n            configuration file. If no source is passed, the file must exist\n            locally. Required.\n\n        source (str): Path to the script on ``file_roots`` to cache at the\n            location specified by ``path``. The source file will be cached\n            locally and then executed. If source is not passed, the config\n            script located at ``path`` will be compiled. Optional.\n\n        config_name (str): The name of the Configuration within the script to\n            apply. If the script contains multiple configurations within the\n            file a ``config_name`` must be specified. If the ``config_name`` is\n            not specified, the name of the file will be used as the\n            ``config_name`` to run. Optional.\n\n        config_data (str): Configuration data in the form of a hash table that\n            will be passed to the ``ConfigurationData`` parameter when the\n            ``config_name`` is compiled. This can be the path to a ``.psd1``\n            file containing the proper hash table or the PowerShell code to\n            create the hash table.\n\n            .. versionadded:: 2017.7.0\n\n        config_data_source (str): The path to the ``.psd1`` file on\n            ``file_roots`` to cache at the location specified by\n            ``config_data``. If this is specified, ``config_data`` must be a\n            local path instead of a hash table.\n\n            .. versionadded:: 2017.7.0\n\n        script_parameters (str): Any additional parameters expected by the\n            configuration script. These must be defined in the script itself.\n\n            .. versionadded:: 2017.7.0\n\n        salt_env (str): The salt environment to use when copying the source.\n            Default is 'base'\n\n    Returns:\n        dict: A dictionary containing the results of the compilation\n\n    CLI Example:\n\n    To compile a config from a script that already exists on the system:\n\n    .. code-block:: bash\n\n        salt '*' dsc.compile_config C:\\\\DSC\\\\WebsiteConfig.ps1\n\n    To cache a config script to the system from the master and compile it:\n\n    .. code-block:: bash\n\n        salt '*' dsc.compile_config C:\\\\DSC\\\\WebsiteConfig.ps1 salt://dsc/configs/WebsiteConfig.ps1\n    "
    if source:
        log.info('DSC: Caching %s', source)
        cached_files = __salt__['cp.get_file'](path=source, dest=path, saltenv=salt_env, makedirs=True)
        if not cached_files:
            error = 'Failed to cache {}'.format(source)
            log.error('DSC: %s', error)
            raise CommandExecutionError(error)
    if config_data_source:
        log.info('DSC: Caching %s', config_data_source)
        cached_files = __salt__['cp.get_file'](path=config_data_source, dest=config_data, saltenv=salt_env, makedirs=True)
        if not cached_files:
            error = 'Failed to cache {}'.format(config_data_source)
            log.error('DSC: %s', error)
            raise CommandExecutionError(error)
    if not os.path.exists(path):
        error = '{} not found'.format(path)
        log.error('DSC: %s', error)
        raise CommandExecutionError(error)
    if config_name is None:
        config_name = os.path.splitext(os.path.basename(path))[0]
    cwd = os.path.dirname(path)
    cmd = [path]
    if script_parameters:
        cmd.append(script_parameters)
    cmd.append('| Where-Object FullName -match \'(?<!\\.meta)\\.mof$\' | Select-Object -Property FullName, Extension, Exists, @{Name="LastWriteTime";Expression={Get-Date ($_.LastWriteTime) -Format g}}')
    cmd = ' '.join(cmd)
    ret = _pshell(cmd, cwd)
    if ret:
        if ret.get('Exists'):
            log.info('DSC: Compile Config: %s', ret)
            return ret
    cmd = ['.', path]
    if script_parameters:
        cmd.append(script_parameters)
    cmd.extend([';', config_name])
    if config_data:
        cmd.extend(['-ConfigurationData', config_data])
    cmd.append('| Where-Object FullName -match \'(?<!\\.meta)\\.mof$\' | Select-Object -Property FullName, Extension, Exists, @{Name="LastWriteTime";Expression={Get-Date ($_.LastWriteTime) -Format g}}')
    cmd = ' '.join(cmd)
    ret = _pshell(cmd, cwd)
    if ret:
        if ret.get('Exists'):
            log.info('DSC: Compile Config: %s', ret)
            return ret
    error = 'Failed to compile config: {}'.format(path)
    error += '\nReturned: {}'.format(ret)
    log.error('DSC: %s', error)
    raise CommandExecutionError(error)

def apply_config(path, source=None, salt_env='base'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run an compiled DSC configuration (a folder containing a .mof file). The\n    folder can be cached from the salt master using the ``source`` option.\n\n    Args:\n\n        path (str): Local path to the directory that contains the .mof\n            configuration file to apply. Required.\n\n        source (str): Path to the directory that contains the .mof file on the\n            ``file_roots``. The source directory will be copied to the path\n            directory and then executed. If the path and source directories\n            differ, the source directory will be applied. If source is not\n            passed, the config located at ``path`` will be applied. Optional.\n\n        salt_env (str): The salt environment to use when copying your source.\n            Default is 'base'\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    To apply a config that already exists on the system\n\n    .. code-block:: bash\n\n        salt '*' dsc.apply_config C:\\\\DSC\\\\WebSiteConfiguration\n\n    To cache a configuration from the master and apply it:\n\n    .. code-block:: bash\n\n        salt '*' dsc.apply_config C:\\\\DSC\\\\WebSiteConfiguration salt://dsc/configs/WebSiteConfiguration\n\n    "
    config = path
    if source:
        path_name = os.path.basename(os.path.normpath(path))
        source_name = os.path.basename(os.path.normpath(source))
        if path_name.lower() != source_name.lower():
            path = '{}\\{}'.format(path, source_name)
            log.debug('DSC: %s appended to the path.', source_name)
        dest_path = os.path.dirname(os.path.normpath(path))
        log.info('DSC: Caching %s', source)
        cached_files = __salt__['cp.get_dir'](source, dest_path, salt_env)
        if not cached_files:
            error = 'Failed to copy {}'.format(source)
            log.error('DSC: %s', error)
            raise CommandExecutionError(error)
        else:
            config = os.path.dirname(cached_files[0])
    if not os.path.exists(config):
        error = '{} not found'.format(config)
        log.error('DSC: %s', error)
        raise CommandExecutionError(error)
    cmd = 'Start-DscConfiguration -Path "{}" -Wait -Force'.format(config)
    _pshell(cmd)
    cmd = '$status = Get-DscConfigurationStatus; $status.Status'
    ret = _pshell(cmd)
    log.info('DSC: Apply Config: %s', ret)
    return ret == 'Success' or ret == {}

def get_config():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the current DSC Configuration\n\n    Returns:\n        dict: A dictionary representing the DSC Configuration on the machine\n\n    Raises:\n        CommandExecutionError: On failure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dsc.get_config\n    "
    cmd = 'Get-DscConfiguration | Select-Object * -ExcludeProperty Cim*'
    try:
        raw_config = _pshell(cmd, ignore_retcode=True)
    except CommandExecutionError as exc:
        if 'Current configuration does not exist' in exc.info['stderr']:
            raise CommandExecutionError('Not Configured')
        raise
    config = dict()
    if raw_config:
        if 'ConfigurationName' in raw_config:
            config_name = raw_config.pop('ConfigurationName')
            resource_id = raw_config.pop('ResourceId')
            config.setdefault(config_name, {resource_id: raw_config})
        else:
            for item in raw_config:
                if 'ConfigurationName' in item:
                    config_name = item.pop('ConfigurationName')
                    resource_id = item.pop('ResourceId')
                    config.setdefault(config_name, {})
                    config[config_name].setdefault(resource_id, item)
    if not config:
        raise CommandExecutionError('Unable to parse config')
    return config

def remove_config(reset=False):
    if False:
        print('Hello World!')
    "\n    Remove the current DSC Configuration. Removes current, pending, and previous\n    dsc configurations.\n\n    .. versionadded:: 2017.7.5\n\n    Args:\n        reset (bool):\n            Attempts to reset the DSC configuration by removing the following\n            from ``C:\\Windows\\System32\\Configuration``:\n\n            - File: DSCStatusHistory.mof\n            - File: DSCEngineCache.mof\n            - Dir: ConfigurationStatus\n\n            Default is False\n\n            .. warning::\n                ``remove_config`` may fail to reset the DSC environment if any\n                of the files in the ``ConfigurationStatus`` directory are in\n                use. If you wait a few minutes and run again, it may complete\n                successfully.\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        CommandExecutionError: On failure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dsc.remove_config True\n    "
    cmd = 'Stop-DscConfiguration'
    log.info('DSC: Stopping Running Configuration')
    try:
        _pshell(cmd)
    except CommandExecutionError as exc:
        if exc.info['retcode'] != 0:
            raise CommandExecutionError('Failed to Stop DSC Configuration', info=exc.info)
        log.info('DSC: %s', exc.info['stdout'])
    cmd = 'Remove-DscConfigurationDocument -Stage Current, Pending, Previous -Force'
    log.info('DSC: Removing Configuration')
    try:
        _pshell(cmd)
    except CommandExecutionError as exc:
        if exc.info['retcode'] != 0:
            raise CommandExecutionError('Failed to remove DSC Configuration', info=exc.info)
        log.info('DSC: %s', exc.info['stdout'])
    if not reset:
        return True

    def _remove_fs_obj(path):
        if False:
            for i in range(10):
                print('nop')
        if os.path.exists(path):
            log.info('DSC: Removing %s', path)
            if not __salt__['file.remove'](path):
                error = 'Failed to remove {}'.format(path)
                log.error('DSC: %s', error)
                raise CommandExecutionError(error)
    dsc_config_dir = '{}\\System32\\Configuration'.format(os.getenv('SystemRoot', 'C:\\Windows'))
    _remove_fs_obj('{}\\DSCStatusHistory.mof'.format(dsc_config_dir))
    _remove_fs_obj('{}\\DSCEngineCache.mof'.format(dsc_config_dir))
    _remove_fs_obj('{}\\ConfigurationStatus'.format(dsc_config_dir))
    return True

def restore_config():
    if False:
        return 10
    "\n    Reapplies the previous configuration.\n\n    .. versionadded:: 2017.7.5\n\n    .. note::\n        The current configuration will be come the previous configuration. If\n        run a second time back-to-back it is like toggling between two configs.\n\n    Returns:\n        bool: True if successfully restored\n\n    Raises:\n        CommandExecutionError: On failure\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dsc.restore_config\n    "
    cmd = 'Restore-DscConfiguration'
    try:
        _pshell(cmd, ignore_retcode=True)
    except CommandExecutionError as exc:
        if 'A previous configuration does not exist' in exc.info['stderr']:
            raise CommandExecutionError('Previous Configuration Not Found')
        raise
    return True

def test_config():
    if False:
        return 10
    "\n    Tests the current applied DSC Configuration\n\n    Returns:\n        bool: True if successfully applied, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dsc.test_config\n    "
    cmd = 'Test-DscConfiguration'
    try:
        _pshell(cmd, ignore_retcode=True)
    except CommandExecutionError as exc:
        if 'Current configuration does not exist' in exc.info['stderr']:
            raise CommandExecutionError('Not Configured')
        raise
    return True

def get_config_status():
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the status of the current DSC Configuration\n\n    Returns:\n        dict: A dictionary representing the status of the current DSC\n            Configuration on the machine\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dsc.get_config_status\n    "
    cmd = 'Get-DscConfigurationStatus | Select-Object -Property HostName, Status, MetaData, @{Name="StartDate";Expression={Get-Date ($_.StartDate) -Format g}}, Type, Mode, RebootRequested, NumberofResources'
    try:
        return _pshell(cmd, ignore_retcode=True)
    except CommandExecutionError as exc:
        if 'No status information available' in exc.info['stderr']:
            raise CommandExecutionError('Not Configured')
        raise

def get_lcm_config():
    if False:
        return 10
    "\n    Get the current Local Configuration Manager settings\n\n    Returns:\n        dict: A dictionary representing the Local Configuration Manager settings\n            on the machine\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dsc.get_lcm_config\n    "
    cmd = 'Get-DscLocalConfigurationManager | Select-Object -Property ConfigurationModeFrequencyMins, LCMState, RebootNodeIfNeeded, ConfigurationMode, ActionAfterReboot, RefreshMode, CertificateID, ConfigurationID, RefreshFrequencyMins, AllowModuleOverwrite, DebugMode, StatusRetentionTimeInDays '
    return _pshell(cmd)

def set_lcm_config(config_mode=None, config_mode_freq=None, refresh_freq=None, reboot_if_needed=None, action_after_reboot=None, refresh_mode=None, certificate_id=None, configuration_id=None, allow_module_overwrite=None, debug_mode=False, status_retention_days=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    For detailed descriptions of the parameters see:\n    https://msdn.microsoft.com/en-us/PowerShell/DSC/metaConfig\n\n    config_mode (str): How the LCM applies the configuration. Valid values\n        are:\n\n        - ApplyOnly\n        - ApplyAndMonitor\n        - ApplyAndAutoCorrect\n\n    config_mode_freq (int): How often, in minutes, the current configuration\n        is checked and applied. Ignored if config_mode is set to ApplyOnly.\n        Default is 15.\n\n    refresh_mode (str): How the LCM gets configurations. Valid values are:\n\n        - Disabled\n        - Push\n        - Pull\n\n    refresh_freq (int): How often, in minutes, the LCM checks for updated\n        configurations. (pull mode only) Default is 30.\n\n    reboot_if_needed (bool): Reboot the machine if needed after a\n        configuration is applied. Default is False.\n\n    action_after_reboot (str): Action to take after reboot. Valid values\n        are:\n\n        - ContinueConfiguration\n        - StopConfiguration\n\n    certificate_id (guid): A GUID that specifies a certificate used to\n        access the configuration: (pull mode)\n\n    configuration_id (guid): A GUID that identifies the config file to get\n        from a pull server. (pull mode)\n\n    allow_module_overwrite (bool): New configs are allowed to overwrite old\n        ones on the target node.\n\n    debug_mode (str): Sets the debug level. Valid values are:\n\n        - None\n        - ForceModuleImport\n        - All\n\n    status_retention_days (int): Number of days to keep status of the\n        current config.\n\n    .. note::\n        Either ``config_mode_freq`` or ``refresh_freq`` needs to be a\n        multiple of the other. See documentation on MSDN for more details.\n\n    Returns:\n        bool: True if successful, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dsc.set_lcm_config ApplyOnly\n    "
    temp_dir = os.getenv('TEMP', '{}\\temp'.format(os.getenv('WINDIR')))
    cmd = 'Configuration SaltConfig {'
    cmd += '    Node localhost {'
    cmd += '        LocalConfigurationManager {'
    if config_mode:
        if config_mode not in ('ApplyOnly', 'ApplyAndMonitor', 'ApplyAndAutoCorrect'):
            error = 'config_mode must be one of ApplyOnly, ApplyAndMonitor, or ApplyAndAutoCorrect. Passed {}'.format(config_mode)
            raise SaltInvocationError(error)
        cmd += '            ConfigurationMode = "{}";'.format(config_mode)
    if config_mode_freq:
        if not isinstance(config_mode_freq, int):
            error = 'config_mode_freq must be an integer. Passed {}'.format(config_mode_freq)
            raise SaltInvocationError(error)
        cmd += '            ConfigurationModeFrequencyMins = {};'.format(config_mode_freq)
    if refresh_mode:
        if refresh_mode not in ('Disabled', 'Push', 'Pull'):
            raise SaltInvocationError('refresh_mode must be one of Disabled, Push, or Pull')
        cmd += '            RefreshMode = "{}";'.format(refresh_mode)
    if refresh_freq:
        if not isinstance(refresh_freq, int):
            raise SaltInvocationError('refresh_freq must be an integer')
        cmd += '            RefreshFrequencyMins = {};'.format(refresh_freq)
    if reboot_if_needed is not None:
        if not isinstance(reboot_if_needed, bool):
            raise SaltInvocationError('reboot_if_needed must be a boolean value')
        if reboot_if_needed:
            reboot_if_needed = '$true'
        else:
            reboot_if_needed = '$false'
        cmd += '            RebootNodeIfNeeded = {};'.format(reboot_if_needed)
    if action_after_reboot:
        if action_after_reboot not in ('ContinueConfiguration', 'StopConfiguration'):
            raise SaltInvocationError('action_after_reboot must be one of ContinueConfiguration or StopConfiguration')
        cmd += '            ActionAfterReboot = "{}"'.format(action_after_reboot)
    if certificate_id is not None:
        if certificate_id == '':
            certificate_id = None
        cmd += '            CertificateID = "{}";'.format(certificate_id)
    if configuration_id is not None:
        if configuration_id == '':
            configuration_id = None
        cmd += '            ConfigurationID = "{}";'.format(configuration_id)
    if allow_module_overwrite is not None:
        if not isinstance(allow_module_overwrite, bool):
            raise SaltInvocationError('allow_module_overwrite must be a boolean value')
        if allow_module_overwrite:
            allow_module_overwrite = '$true'
        else:
            allow_module_overwrite = '$false'
        cmd += '            AllowModuleOverwrite = {};'.format(allow_module_overwrite)
    if debug_mode is not False:
        if debug_mode is None:
            debug_mode = 'None'
        if debug_mode not in ('None', 'ForceModuleImport', 'All'):
            raise SaltInvocationError('debug_mode must be one of None, ForceModuleImport, ResourceScriptBreakAll, or All')
        cmd += '            DebugMode = "{}";'.format(debug_mode)
    if status_retention_days:
        if not isinstance(status_retention_days, int):
            raise SaltInvocationError('status_retention_days must be an integer')
        cmd += '            StatusRetentionTimeInDays = {};'.format(status_retention_days)
    cmd += '        }}};'
    cmd += 'SaltConfig -OutputPath "{}\\SaltConfig"'.format(temp_dir)
    _pshell(cmd)
    cmd = 'Set-DscLocalConfigurationManager -Path "{}\\SaltConfig"'.format(temp_dir)
    ret = __salt__['cmd.run_all'](cmd, shell='powershell', python_shell=True)
    __salt__['file.remove']('{}\\SaltConfig'.format(temp_dir))
    if not ret['retcode']:
        log.info('DSC: LCM config applied successfully')
        return True
    else:
        log.error('DSC: Failed to apply LCM config. Error %s', ret)
        return False