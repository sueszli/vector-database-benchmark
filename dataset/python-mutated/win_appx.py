"""
Manage provisioned apps
=======================

.. versionadded:: 3007.0

Provisioned apps are part of the image and are installed for every user the
first time the user logs on. Provisioned apps are also updated and sometimes
reinstalled when the system is updated.

Apps removed with this module will remove the app for all users and deprovision
the app. Deprovisioned apps will neither be installed for new users nor will
they be upgraded.

An app removed with this module can only be re-provisioned on the machine, but
it can't be re-installed for all users. Also, once a package has been
deprovisioned, the only way to reinstall it is to download the package. This is
difficult. The steps are outlined below:

1. Obtain the Microsoft Store URL for the app:
    - Open the page for the app in the Microsoft Store
    - Click the share button and copy the URL

2. Look up the packages on https://store.rg-adguard.net/:
    - Ensure ``URL (link)`` is selected in the first dropdown
    - Paste the URL in the search field
    - Ensure Retail is selected in the 2nd dropdown
    - Click the checkmark button

This should return a list of URLs for the package and all dependencies for all
architectures. Download the package and all dependencies for your system
architecture. These will usually have one of the following file extensions:

- ``.appx``
- ``.appxbundle``
- ``.msix``
- ``.msixbundle``

Dependencies will need to be installed first.

Not all packages can be found this way, but it seems like most of them can.

Use the ``appx.install`` function to provision the new app.
"""
import fnmatch
import logging
import salt.utils.platform
import salt.utils.win_pwsh
import salt.utils.win_reg
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
CURRENT_VERSION_KEY = 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion'
DEPROVISIONED_KEY = f'{CURRENT_VERSION_KEY}\\Appx\\AppxAllUserStore\\Deprovisioned'
__virtualname__ = 'appx'
__func_alias__ = {'list_': 'list'}

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Load only on Windows\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'Appx module: Only available on Windows systems')
    pwsh_info = __salt__['cmd.shell_info'](shell='powershell', list_modules=False)
    if not pwsh_info['installed']:
        return (False, 'Appx module: PowerShell not available')
    return __virtualname__

def _pkg_list(raw, field='Name'):
    if False:
        for i in range(10):
            print('nop')
    result = []
    if raw:
        if isinstance(raw, list):
            for pkg in raw:
                result.append(pkg[field])
        else:
            result.append(raw[field])
    else:
        result = None
    return result

def list_(query=None, field='Name', include_store=False, frameworks=False, bundles=True):
    if False:
        while True:
            i = 10
    '\n    Get a list of Microsoft Store packages installed on the system.\n\n    Args:\n\n        query (str):\n            The query string to use to filter packages to be listed. The string\n            can match multiple packages. ``None`` will return all packages. Here\n            are some example strings:\n\n            - ``*teams*`` - Returns Microsoft Teams\n            - ``*zune*`` - Returns Windows Media Player and ZuneVideo\n            - ``*zuneMusic*`` - Only returns Windows Media Player\n            - ``*xbox*`` - Returns all xbox packages, there are 5 by default\n            - ``*`` - Returns everything but the Microsoft Store, unless\n              ``include_store=True``\n\n        field (str):\n            This function returns a list of packages on the system. It can\n            display a short name or a full name. If ``None`` is passed, a\n            dictionary will be returned with some common fields. The default is\n            ``Name``. Valid options are any fields returned by the powershell\n            command ``Get-AppxPackage``. Here are some useful fields:\n\n            - Name\n            - Version\n            - PackageFullName\n            - PackageFamilyName\n\n        include_store (bool):\n            Include the Microsoft Store in the results. Default is ``False``\n\n        frameworks (bool):\n            Include frameworks in the results. Default is ``False``\n\n        bundles (bool):\n            If ``True``, this will return application bundles only. If\n            ``False``, this will return individual packages only, even if they\n            are part of a bundle.\n\n    Returns:\n        list: A list of packages ordered by the string passed in field\n        list: A list of dictionaries of package information if field is ``None``\n\n    Raises:\n        CommandExecutionError: If an error is encountered retrieving packages\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # List installed apps that contain the word "candy"\n        salt \'*\' appx.list *candy*\n\n        # Return more information about the package\n        salt \'*\' appx.list *candy* field=None\n\n        # List all installed apps, including the Microsoft Store\n        salt \'*\' appx.list include_store=True\n\n        # List all installed apps, including frameworks\n        salt \'*\' appx.list frameworks=True\n\n        # List all installed apps that are bundles\n        salt \'*\' appx.list bundles=True\n    '
    cmd = []
    if bundles:
        cmd_str = 'Get-AppxPackage -AllUsers -PackageTypeFilter Bundle'
    else:
        cmd_str = 'Get-AppxPackage -AllUsers'
    if query:
        cmd.append(f'{cmd_str} -Name {query}')
    else:
        cmd.append(f'{cmd_str}')
    if not include_store:
        cmd.append('Where-Object {$_.name -notlike "Microsoft.WindowsStore*"}')
    if not frameworks:
        cmd.append('Where-Object -Property IsFramework -eq $false')
    cmd.append('Where-Object -Property NonRemovable -eq $false')
    if not field:
        cmd.append('Sort-Object Name')
        cmd.append('Select Name, Version, PackageFullName, PackageFamilyName, IsBundle, IsFramework')
        return salt.utils.win_pwsh.run_dict(' | '.join(cmd))
    else:
        cmd.append(f'Sort-Object {field}')
        return _pkg_list(salt.utils.win_pwsh.run_dict(' | '.join(cmd)), field)

def remove(query=None, include_store=False, frameworks=False, deprovision_only=False):
    if False:
        i = 10
        return i + 15
    '\n    Removes Microsoft Store packages from the system. If the package is part of\n    a bundle, the entire bundle will be removed.\n\n    This function removes the package for all users on the system. It also\n    deprovisions the package so that it isn\'t re-installed by later system\n    updates. To only deprovision a package and not remove it for all users, set\n    ``deprovision_only=True``.\n\n    Args:\n\n        query (str):\n            The query string to use to select the packages to be removed. If the\n            string matches multiple packages, they will all be removed. Here are\n            some example strings:\n\n            - ``*teams*`` - Remove Microsoft Teams\n            - ``*zune*`` - Remove Windows Media Player and ZuneVideo\n            - ``*zuneMusic*`` - Only remove Windows Media Player\n            - ``*xbox*`` - Remove all xbox packages, there are 5 by default\n            - ``*`` - Remove everything but the Microsoft Store, unless\n              ``include_store=True``\n\n            .. note::\n                Use the ``appx.list`` function to make sure your query is\n                returning what you expect. Then use the same query to remove\n                those packages\n\n        include_store (bool):\n            Include the Microsoft Store in the results of the query to be\n            removed. Use this with caution. It is difficult to reinstall the\n            Microsoft Store once it has been removed with this function. Default\n            is ``False``\n\n        frameworks (bool):\n            Include frameworks in the results of the query to be removed.\n            Default is ``False``\n\n        deprovision_only (bool):\n            Only deprovision the package. The package will be removed from the\n            current user and added to the list of deprovisioned packages. The\n            package will not be re-installed in future system updates. New users\n            of the system will not have the package installed. However, the\n            package will still be installed for existing users. Default is\n            ``False``\n\n    Returns:\n        bool: ``True`` if successful, ``None`` if no packages found\n\n    Raises:\n        CommandExecutionError: On errors encountered removing the package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt "*" appx.remove *candy*\n    '
    packages = list_(query=query, field=None, include_store=include_store, frameworks=frameworks, bundles=False)

    def remove_package(package):
        if False:
            i = 10
            return i + 15
        remove_name = package['PackageFullName']
        if not package['IsBundle']:
            bundle = list_(query=f"{package['Name']}*", field=None, include_store=include_store, frameworks=frameworks, bundles=True)
            if isinstance(bundle, list):
                for item in bundle:
                    remove_package(item)
            elif bundle and bundle['IsBundle']:
                log.debug(f"Found bundle: {bundle['PackageFullName']}")
                remove_name = bundle['PackageFullName']
        if deprovision_only:
            log.debug('Deprovisioning package: %s', remove_name)
            remove_cmd = f'Remove-AppxProvisionedPackage -Online -PackageName {remove_name}'
        else:
            log.debug('Removing package: %s', remove_name)
            remove_cmd = f'Remove-AppxPackage -AllUsers -Package {remove_name}'
        try:
            salt.utils.win_pwsh.run_dict(remove_cmd)
        except CommandExecutionError as exc:
            log.debug(f'There was an error removing package: {remove_name}')
            log.debug(exc)
    if isinstance(packages, list):
        log.debug('Removing %s packages', len(packages))
        for pkg in packages:
            remove_package(package=pkg)
    elif packages:
        log.debug('Removing a single package')
        remove_package(package=packages)
    else:
        log.debug('Package not found: %s', query)
        return None
    return True

def list_deprovisioned(query=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    When an app is deprovisioned, a registry key is created that will keep it\n    from being reinstalled during a major system update. This function returns a\n    list of keys for apps that have been deprovisioned.\n\n    Args:\n\n        query (str):\n            The query string to use to filter packages to be listed. The string\n            can match multiple packages. ``None`` will return all packages. Here\n            are some example strings:\n\n            - ``*teams*`` - Returns Microsoft Teams\n            - ``*zune*`` - Returns Windows Media Player and ZuneVideo\n            - ``*zuneMusic*`` - Only returns Windows Media Player\n            - ``*xbox*`` - Returns all xbox packages, there are 5 by default\n            - ``*`` - Returns everything but the Microsoft Store, unless\n              ``include_store=True``\n\n    Returns:\n        list: A list of packages matching the query criteria\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt "*" appx.list_deprovisioned *zune*\n    '
    ret = salt.utils.win_reg.list_keys(hive='HKLM', key=f'{DEPROVISIONED_KEY}')
    if query is None:
        return ret
    return fnmatch.filter(ret, query)

def install(package):
    if False:
        while True:
            i = 10
    '\n    This function uses ``dism`` to provision a package. This means that it will\n    be made a part of the online image and added to new users on the system. If\n    a package has dependencies, those must be installed first.\n\n    If a package installed using this function has been deprovisioned\n    previously, the registry entry marking it as deprovisioned will be removed.\n\n    .. NOTE::\n        There is no ``appx.present`` state. Instead, use the\n        ``dism.provisioned_package_installed`` state.\n\n    Args:\n\n        package (str):\n            The full path to the package to install. Can be one of the\n            following:\n\n            - ``.appx`` or ``.appxbundle``\n            - ``.msix`` or ``.msixbundle``\n            - ``.ppkg``\n\n    Returns:\n        bool: ``True`` if successful, otherwise ``False``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt "*" appx.install "C:\\Temp\\Microsoft.ZuneMusic.msixbundle"\n    '
    ret = __salt__['dism.add_provisioned_package'](package)
    return ret['retcode'] == 0