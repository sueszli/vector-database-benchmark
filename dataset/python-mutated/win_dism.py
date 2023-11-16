"""
Install features/packages for Windows using DISM, which is useful for minions
not running server versions\xa0of Windows. Some functions are only available on
Windows 10.

"""
import logging
import os
import re
import salt.utils.platform
import salt.utils.versions
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'dism'
try:
    if not salt.utils.platform.is_windows():
        raise OSError
    if os.path.exists(os.path.join(os.environ.get('SystemRoot'), 'SysNative')):
        bin_path = os.path.join(os.environ.get('SystemRoot'), 'SysNative')
    else:
        bin_path = os.path.join(os.environ.get('SystemRoot'), 'System32')
    bin_dism = os.path.join(bin_path, 'dism.exe')
except OSError:
    log.trace('win_dism: Non-Windows system')
    bin_dism = 'dism.exe'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on Windows\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Only available on Windows systems')
    return __virtualname__

def _get_components(type_regex, plural_type, install_value, image=None):
    if False:
        print('Hello World!')
    cmd = [bin_dism, '/English', f'/Image:{image}' if image else '/Online', f'/Get-{plural_type}']
    out = __salt__['cmd.run'](cmd)
    if install_value:
        pattern = f'{type_regex} : (.*)\\r\\n.*State : {install_value}\\r\\n'
    else:
        pattern = f'{type_regex} : (.*)\\r\\n.*'
    capabilities = re.findall(pattern, out, re.MULTILINE)
    capabilities.sort()
    return capabilities

def add_capability(capability, source=None, limit_access=False, image=None, restart=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Install a capability\n\n    Args:\n        capability (str): The capability to install\n        source (Optional[str]): The optional source of the capability. Default\n            is set by group policy and can be Windows Update.\n        limit_access (Optional[bool]): Prevent DISM from contacting Windows\n            Update for the source package\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n        restart (Optional[bool]): Reboot the machine if required by the install\n\n    Raises:\n        NotImplementedError: For all versions of Windows that are not Windows 10\n        and later. Server editions of Windows use ServerManager instead.\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.add_capability Tools.Graphics.DirectX~~~~0.0.1.0\n    "
    if salt.utils.versions.version_cmp(__grains__['osversion'], '10') == -1:
        raise NotImplementedError(f"`install_capability` is not available on this version of Windows: {__grains__['osversion']}")
    cmd = [bin_dism, '/Quiet', f'/Image:{image}' if image else '/Online', '/Add-Capability', f'/CapabilityName:{capability}']
    if source:
        cmd.append(f'/Source:{source}')
    if limit_access:
        cmd.append('/LimitAccess')
    if not restart:
        cmd.append('/NoRestart')
    return __salt__['cmd.run_all'](cmd)

def remove_capability(capability, image=None, restart=False):
    if False:
        return 10
    "\n    Uninstall a capability\n\n    Args:\n        capability(str): The capability to be removed\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n        restart (Optional[bool]): Reboot the machine if required by the install\n\n    Raises:\n        NotImplementedError: For all versions of Windows that are not Windows 10\n        and later. Server editions of Windows use ServerManager instead.\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.remove_capability Tools.Graphics.DirectX~~~~0.0.1.0\n    "
    if salt.utils.versions.version_cmp(__grains__['osversion'], '10') == -1:
        raise NotImplementedError(f"`uninstall_capability` is not available on this version of Windows: {__grains__['osversion']}")
    cmd = [bin_dism, '/Quiet', f'/Image:{image}' if image else '/Online', '/Remove-Capability', f'/CapabilityName:{capability}']
    if not restart:
        cmd.append('/NoRestart')
    return __salt__['cmd.run_all'](cmd)

def get_capabilities(image=None):
    if False:
        return 10
    "\n    List all capabilities on the system\n\n    Args:\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Raises:\n        NotImplementedError: For all versions of Windows that are not Windows 10\n        and later. Server editions of Windows use ServerManager instead.\n\n    Returns:\n        list: A list of capabilities\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.get_capabilities\n    "
    if salt.utils.versions.version_cmp(__grains__['osversion'], '10') == -1:
        raise NotImplementedError(f"`installed_capabilities` is not available on this version of Windows: {__grains__['osversion']}")
    cmd = [bin_dism, '/English', f'/Image:{image}' if image else '/Online', '/Get-Capabilities']
    out = __salt__['cmd.run'](cmd)
    pattern = 'Capability Identity : (.*)\\r\\n'
    capabilities = re.findall(pattern, out, re.MULTILINE)
    capabilities.sort()
    return capabilities

def installed_capabilities(image=None):
    if False:
        return 10
    "\n    List the capabilities installed on the system\n\n    Args:\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Raises:\n        NotImplementedError: For all versions of Windows that are not Windows 10\n        and later. Server editions of Windows use ServerManager instead.\n\n    Returns:\n        list: A list of installed capabilities\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.installed_capabilities\n    "
    if salt.utils.versions.version_cmp(__grains__['osversion'], '10') == -1:
        raise NotImplementedError(f"`installed_capabilities` is not available on this version of Windows: {__grains__['osversion']}")
    return _get_components('Capability Identity', 'Capabilities', 'Installed')

def available_capabilities(image=None):
    if False:
        while True:
            i = 10
    "\n    List the capabilities available on the system\n\n    Args:\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Raises:\n        NotImplementedError: For all versions of Windows that are not Windows 10\n        and later. Server editions of Windows use ServerManager instead.\n\n    Returns:\n        list: A list of available capabilities\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.installed_capabilities\n    "
    if salt.utils.versions.version_cmp(__grains__['osversion'], '10') == -1:
        raise NotImplementedError(f"`installed_capabilities` is not available on this version of Windows: {__grains__['osversion']}")
    return _get_components('Capability Identity', 'Capabilities', 'Not Present')

def add_feature(feature, package=None, source=None, limit_access=False, enable_parent=False, image=None, restart=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Install a feature using DISM\n\n    Args:\n        feature (str): The feature to install\n        package (Optional[str]): The parent package for the feature. You do not\n            have to specify the package if it is the Windows Foundation Package.\n            Otherwise, use package to specify the parent package of the feature\n        source (Optional[str]): The optional source of the capability. Default\n            is set by group policy and can be Windows Update\n        limit_access (Optional[bool]): Prevent DISM from contacting Windows\n            Update for the source package\n        enable_parent (Optional[bool]): True will enable all parent features of\n            the specified feature\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n        restart (Optional[bool]): Reboot the machine if required by the install\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.add_feature NetFx3\n    "
    cmd = [bin_dism, '/Quiet', f'/Image:{image}' if image else '/Online', '/Enable-Feature', f'/FeatureName:{feature}']
    if package:
        cmd.append(f'/PackageName:{package}')
    if source:
        cmd.append(f'/Source:{source}')
    if limit_access:
        cmd.append('/LimitAccess')
    if enable_parent:
        cmd.append('/All')
    if not restart:
        cmd.append('/NoRestart')
    return __salt__['cmd.run_all'](cmd)

def remove_feature(feature, remove_payload=False, image=None, restart=False):
    if False:
        i = 10
        return i + 15
    "\n    Disables the feature.\n\n    Args:\n        feature (str): The feature to uninstall\n        remove_payload (Optional[bool]): Remove the feature's payload. Must\n            supply source when enabling in the future.\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n        restart (Optional[bool]): Reboot the machine if required by the install\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.remove_feature NetFx3\n    "
    cmd = [bin_dism, '/Quiet', f'/Image:{image}' if image else '/Online', '/Disable-Feature', f'/FeatureName:{feature}']
    if remove_payload:
        cmd.append('/Remove')
    if not restart:
        cmd.append('/NoRestart')
    return __salt__['cmd.run_all'](cmd)

def get_features(package=None, image=None):
    if False:
        print('Hello World!')
    "\n    List features on the system or in a package\n\n    Args:\n        package (Optional[str]): The full path to the package. Can be either a\n            .cab file or a folder. Should point to the original source of the\n            package, not to where the file is installed. You cannot use this\n            command to get package information for .msu files\n\n            This can also be the name of a package as listed in\n            ``dism.installed_packages``\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Returns:\n        list: A list of features\n\n    CLI Example:\n\n        .. code-block:: bash\n\n            # Return all features on the system\n            salt '*' dism.get_features\n\n            # Return all features in package.cab\n            salt '*' dism.get_features C:\\packages\\package.cab\n\n            # Return all features in the calc package\n            salt '*' dism.get_features Microsoft.Windows.Calc.Demo~6595b6144ccf1df~x86~en~1.0.0.0\n    "
    cmd = [bin_dism, '/English', f'/Image:{image}' if image else '/Online', '/Get-Features']
    if package:
        if '~' in package:
            cmd.append(f'/PackageName:{package}')
        else:
            cmd.append(f'/PackagePath:{package}')
    out = __salt__['cmd.run'](cmd)
    pattern = 'Feature Name : (.*)\\r\\n'
    features = re.findall(pattern, out, re.MULTILINE)
    features.sort()
    return features

def installed_features(image=None):
    if False:
        print('Hello World!')
    "\n    List the features installed on the system\n\n    Args:\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Returns:\n        list: A list of installed features\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.installed_features\n    "
    return _get_components('Feature Name', 'Features', 'Enabled')

def available_features(image=None):
    if False:
        while True:
            i = 10
    "\n    List the features available on the system\n\n    Args:\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Returns:\n        list: A list of available features\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.available_features\n    "
    return _get_components('Feature Name', 'Features', 'Disabled')

def add_package(package, ignore_check=False, prevent_pending=False, image=None, restart=False):
    if False:
        return 10
    "\n    Install a package using DISM\n\n    Args:\n        package (str):\n            The package to install. Can be a .cab file, a .msu file, or a folder\n\n            .. note::\n                An `.msu` package is supported only when the target image is\n                offline, either mounted or applied.\n\n        ignore_check (Optional[bool]):\n            Skip installation of the package if the applicability checks fail\n\n        prevent_pending (Optional[bool]):\n            Skip the installation of the package if there are pending online\n            actions\n\n        image (Optional[str]):\n            The path to the root directory of an offline Windows image. If\n            ``None`` is passed, the running operating system is targeted.\n            Default is None.\n\n        restart (Optional[bool]):\n            Reboot the machine if required by the install\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.add_package C:\\Packages\\package.cab\n    "
    cmd = [bin_dism, '/Quiet', f'/Image:{image}' if image else '/Online', '/Add-Package', f'/PackagePath:{package}']
    if ignore_check:
        cmd.append('/IgnoreCheck')
    if prevent_pending:
        cmd.append('/PreventPending')
    if not restart:
        cmd.append('/NoRestart')
    return __salt__['cmd.run_all'](cmd)

def add_provisioned_package(package, image=None, restart=False):
    if False:
        i = 10
        return i + 15
    "\n    Provision a package using DISM. A provisioned package will install for new\n    users on the system. It will also be reinstalled on each user if the system\n    is updated.\n\n    .. versionadded:: 3007.0\n\n    Args:\n\n        package (str):\n            The package to install. Can be one of the following:\n\n            - ``.appx`` or ``.appxbundle``\n            - ``.msix`` or ``.msixbundle``\n            - ``.ppkg``\n\n        image (Optional[str]):\n            The path to the root directory of an offline Windows image. If\n            ``None`` is passed, the running operating system is targeted.\n            Default is ``None``.\n\n        restart (Optional[bool]):\n            Reboot the machine if required by the installation. Default is\n            ``False``\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.add_provisioned_package C:\\Packages\\package.appx\n        salt '*' dism.add_provisioned_package C:\\Packages\\package.appxbundle\n        salt '*' dism.add_provisioned_package C:\\Packages\\package.msix\n        salt '*' dism.add_provisioned_package C:\\Packages\\package.msixbundle\n        salt '*' dism.add_provisioned_package C:\\Packages\\package.ppkg\n    "
    cmd = [bin_dism, '/Quiet', f'/Image:{image}' if image else '/Online', '/Add-ProvisionedAppxPackage', f'/PackagePath:{package}', '/SkipLicense']
    if not restart:
        cmd.append('/NoRestart')
    return __salt__['cmd.run_all'](cmd)

def remove_package(package, image=None, restart=False):
    if False:
        i = 10
        return i + 15
    "\n    Uninstall a package\n\n    Args:\n        package (str): The full path to the package. Can be either a .cab file\n            or a folder. Should point to the original source of the package, not\n            to where the file is installed. This can also be the name of a\n            package as listed in ``dism.installed_packages``\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n        restart (Optional[bool]): Reboot the machine if required by the\n            uninstall\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Remove the Calc Package\n        salt '*' dism.remove_package Microsoft.Windows.Calc.Demo~6595b6144ccf1df~x86~en~1.0.0.0\n\n        # Remove the package.cab (does not remove C:\\packages\\package.cab)\n        salt '*' dism.remove_package C:\\packages\\package.cab\n    "
    cmd = [bin_dism, '/Quiet', f'/Image:{image}' if image else '/Online', '/Remove-Package']
    if not restart:
        cmd.append('/NoRestart')
    if '~' in package:
        cmd.append(f'/PackageName:{package}')
    else:
        cmd.append(f'/PackagePath:{package}')
    return __salt__['cmd.run_all'](cmd)

def get_kb_package_name(kb, image=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the actual package name on the system based on the KB name\n\n    .. versionadded:: 3006.0\n\n    Args:\n        kb (str): The name of the KB to remove. Can also be just the KB number\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Returns:\n        str: The name of the package found on the system\n        None: If the package is not installed on the system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Get the package name for KB1231231\n        salt '*' dism.get_kb_package_name KB1231231\n\n        # Get the package name for KB1231231 using just the number\n        salt '*' dism.get_kb_package_name 1231231\n    "
    packages = installed_packages(image=image)
    search = kb.upper() if kb.lower().startswith('kb') else f'KB{kb}'
    for package in packages:
        if f'_{search}~' in package:
            return package
    return None

def remove_kb(kb, image=None, restart=False):
    if False:
        i = 10
        return i + 15
    "\n    Remove a package by passing a KB number. This searches the installed\n    packages to get the full package name of the KB. It then calls the\n    ``dism.remove_package`` function to remove the package.\n\n    .. versionadded:: 3006.0\n\n    Args:\n        kb (str): The name of the KB to remove. Can also be just the KB number\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n        restart (Optional[bool]): Reboot the machine if required by the\n            uninstall\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Remove the KB5007575 just passing the number\n        salt '*' dism.remove_kb 5007575\n\n        # Remove the KB5007575 just passing the full name\n        salt '*' dism.remove_kb KB5007575\n    "
    pkg_name = get_kb_package_name(kb=kb, image=image)
    if pkg_name is None:
        msg = f'{kb} not installed'
        raise CommandExecutionError(msg)
    log.debug('Found: %s', pkg_name)
    return remove_package(package=pkg_name, image=image, restart=restart)

def installed_packages(image=None):
    if False:
        print('Hello World!')
    "\n    List the packages installed on the system\n\n    Args:\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Returns:\n        list: A list of installed packages\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.installed_packages\n    "
    return _get_components(type_regex='Package Identity', plural_type='Packages', install_value='Installed', image=image)

def provisioned_packages(image=None):
    if False:
        return 10
    "\n    List the packages installed on the system\n\n    .. versionadded:: 3007.0\n\n    Args:\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Returns:\n        list: A list of installed packages\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.installed_packages\n    "
    return _get_components(type_regex='PackageName', plural_type='ProvisionedAppxPackages', install_value='', image=image)

def package_info(package, image=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display information about a package\n\n    Args:\n        package (str): The full path to the package. Can be either a .cab file\n            or a folder. Should point to the original source of the package, not\n            to where the file is installed. You cannot use this command to get\n            package information for .msu files\n        image (Optional[str]): The path to the root directory of an offline\n            Windows image. If `None` is passed, the running operating system is\n            targeted. Default is None.\n\n    Returns:\n        dict: A dictionary containing the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' dism.package_info C:\\packages\\package.cab\n    "
    cmd = [bin_dism, '/English', f'/Image:{image}' if image else '/Online', '/Get-PackageInfo']
    if '~' in package:
        cmd.append(f'/PackageName:{package}')
    else:
        cmd.append(f'/PackagePath:{package}')
    out = __salt__['cmd.run_all'](cmd)
    if out['retcode'] == 0:
        ret = dict()
        for line in str(out['stdout']).splitlines():
            if ' : ' in line:
                info = line.split(' : ')
                if len(info) < 2:
                    continue
                ret[info[0]] = info[1]
    else:
        ret = out
    return ret