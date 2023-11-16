"""
A module to manage software on Windows

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.

The following functions require the existence of a :ref:`windows repository
<windows-package-manager>` metadata DB, typically created by running
:py:func:`pkg.refresh_db <salt.modules.win_pkg.refresh_db>`:

- :py:func:`pkg.get_repo_data <salt.modules.win_pkg.get_repo_data>`
- :py:func:`pkg.install <salt.modules.win_pkg.install>`
- :py:func:`pkg.latest_version <salt.modules.win_pkg.latest_version>`
- :py:func:`pkg.list_available <salt.modules.win_pkg.list_available>`
- :py:func:`pkg.list_pkgs <salt.modules.win_pkg.list_pkgs>`
- :py:func:`pkg.list_upgrades <salt.modules.win_pkg.list_upgrades>`
- :py:func:`pkg.remove <salt.modules.win_pkg.remove>`

If a metadata DB does not already exist and one of these functions is run, then
one will be created from the repo SLS files that are present.

As the creation of this metadata can take some time, the
:conf_minion:`winrepo_cache_expire_min` minion config option can be used to
suppress refreshes when the metadata is less than a given number of seconds
old.

.. note::
    Version numbers can be ``version number string``, ``latest`` and ``Not
    Found``, where ``Not Found`` means this module was not able to determine
    the version of the software installed, it can also be used as the version
    number in sls definitions file in these cases. Versions numbers are sorted
    in order of 0, ``Not Found``, ``order version numbers``, ..., ``latest``.

"""
import collections
import datetime
import errno
import logging
import os
import re
import sys
import time
import urllib.parse
from functools import cmp_to_key
import salt.payload
import salt.syspaths
import salt.utils.args
import salt.utils.data
import salt.utils.files
import salt.utils.hashutils
import salt.utils.path
import salt.utils.pkg
import salt.utils.platform
import salt.utils.versions
import salt.utils.win_functions
from salt.exceptions import CommandExecutionError, MinionError, SaltInvocationError, SaltRenderError
from salt.utils.versions import LooseVersion
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Set the virtual pkg module if the os is Windows\n    '
    if salt.utils.platform.is_windows():
        return __virtualname__
    return (False, 'Module win_pkg: module only works on Windows systems')

def latest_version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    .. note::\n        Since this is looking for the latest version available, a refresh_db\n        will be triggered by default. This can take some time. To avoid this set\n        ``refresh`` to ``False``.\n\n    Args:\n        names (str): A single or multiple names to lookup\n\n    Kwargs:\n        saltenv (str): Salt environment. Default ``base``\n        refresh (bool): Refresh package metadata. Default ``True``\n\n    Returns:\n        dict: A dictionary of packages with the latest version available\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n    "
    if not names:
        return ''
    ret = {}
    for name in names:
        ret[name] = ''
    saltenv = kwargs.get('saltenv', 'base')
    refresh = salt.utils.data.is_true(kwargs.get('refresh', True))
    installed_pkgs = list_pkgs(versions_as_list=True, saltenv=saltenv, refresh=refresh)
    log.trace('List of installed packages: %s', installed_pkgs)
    for name in names:
        latest_installed = '0'
        if name in installed_pkgs:
            log.trace('Determining latest installed version of %s', name)
            try:
                latest_installed = sorted(installed_pkgs[name], key=cmp_to_key(_reverse_cmp_pkg_versions)).pop()
            except IndexError:
                log.warning('%s was empty in pkg.list_pkgs return data, this is probably a bug in list_pkgs', name)
            else:
                log.debug('Latest installed version of %s is %s', name, latest_installed)
        pkg_info = _get_package_info(name, saltenv=saltenv)
        log.trace('Raw winrepo pkg_info for %s is %s', name, pkg_info)
        latest_available = _get_latest_pkg_version(pkg_info)
        if latest_available:
            log.debug('Latest available version of package %s is %s', name, latest_available)
            if compare_versions(ver1=str(latest_available), oper='>', ver2=str(latest_installed)):
                log.debug('Upgrade of %s from %s to %s is available', name, latest_installed, latest_available)
                ret[name] = latest_available
            else:
                log.debug('No newer version than %s of %s is available', latest_installed, name)
    if len(names) == 1:
        return ret[names[0]]
    return ret

def upgrade_available(name, **kwargs):
    if False:
        return 10
    "\n    Check whether or not an upgrade is available for a given package\n\n    Args:\n        name (str): The name of a single package\n\n    Kwargs:\n        refresh (bool): Refresh package metadata. Default ``True``\n        saltenv (str): The salt environment. Default ``base``\n\n    Returns:\n        bool: True if new version available, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    saltenv = kwargs.get('saltenv', 'base')
    refresh = salt.utils.data.is_true(kwargs.get('refresh', True))
    return latest_version(name, saltenv=saltenv, refresh=refresh) != ''

def list_upgrades(refresh=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all available package upgrades on this system\n\n    Args:\n        refresh (bool): Refresh package metadata. Default ``True``\n\n    Kwargs:\n        saltenv (str): Salt environment. Default ``base``\n\n    Returns:\n        dict: A dictionary of packages with available upgrades\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    saltenv = kwargs.get('saltenv', 'base')
    refresh = salt.utils.data.is_true(refresh)
    _refresh_db_conditional(saltenv, force=refresh)
    installed_pkgs = list_pkgs(refresh=False, saltenv=saltenv)
    available_pkgs = get_repo_data(saltenv).get('repo')
    pkgs = {}
    for pkg in installed_pkgs:
        if pkg in available_pkgs:
            latest_ver = latest_version(pkg, refresh=False, saltenv=saltenv)
            if latest_ver:
                pkgs[pkg] = latest_ver
    return pkgs

def list_available(*names, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return a list of available versions of the specified package.\n\n    Args:\n        names (str): One or more package names\n\n    Kwargs:\n\n        saltenv (str): The salt environment to use. Default ``base``.\n\n        refresh (bool): Refresh package metadata. Default ``False``.\n\n        return_dict_always (bool):\n            Default ``False`` dict when a single package name is queried.\n\n    Returns:\n        dict: The package name with its available versions\n\n    .. code-block:: cfg\n\n        {'<package name>': ['<version>', '<version>', ]}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_available <package name> return_dict_always=True\n        salt '*' pkg.list_available <package name01> <package name02>\n    "
    if not names:
        return ''
    saltenv = kwargs.get('saltenv', 'base')
    refresh = salt.utils.data.is_true(kwargs.get('refresh', False))
    _refresh_db_conditional(saltenv, force=refresh)
    return_dict_always = salt.utils.data.is_true(kwargs.get('return_dict_always', False))
    if len(names) == 1 and (not return_dict_always):
        pkginfo = _get_package_info(names[0], saltenv=saltenv)
        if not pkginfo:
            return ''
        versions = sorted(list(pkginfo.keys()), key=cmp_to_key(_reverse_cmp_pkg_versions))
    else:
        versions = {}
        for name in names:
            pkginfo = _get_package_info(name, saltenv=saltenv)
            if not pkginfo:
                continue
            verlist = sorted(list(pkginfo.keys()) if pkginfo else [], key=cmp_to_key(_reverse_cmp_pkg_versions))
            versions[name] = verlist
    return versions

def version(*names, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    Args:\n        name (str): One or more package names\n\n    Kwargs:\n        saltenv (str): The salt environment to use. Default ``base``.\n        refresh (bool): Refresh package metadata. Default ``False``.\n\n    Returns:\n        str: version string when a single package is specified.\n        dict: The package name(s) with the installed versions.\n\n    .. code-block:: cfg\n\n        {['<version>', '<version>', ]} OR\n        {'<package name>': ['<version>', '<version>', ]}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package name01> <package name02>\n\n    "
    saltenv = kwargs.get('saltenv', 'base')
    installed_pkgs = list_pkgs(saltenv=saltenv, refresh=kwargs.get('refresh', False))
    if len(names) == 1:
        return installed_pkgs.get(names[0], '')
    ret = {}
    for name in names:
        ret[name] = installed_pkgs.get(name, '')
    return ret

def list_pkgs(versions_as_list=False, include_components=True, include_updates=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    List the packages currently installed.\n\n    .. note::\n        To view installed software as displayed in the Add/Remove Programs, set\n        ``include_components`` and ``include_updates`` to False.\n\n    Args:\n\n        versions_as_list (bool):\n            Returns the versions as a list\n\n        include_components (bool):\n            Include sub components of installed software. Default is ``True``\n\n        include_updates (bool):\n            Include software updates and Windows updates. Default is ``True``\n\n    Kwargs:\n\n        saltenv (str):\n            The salt environment to use. Default ``base``\n\n        refresh (bool):\n            Refresh package metadata. Default ``False``\n\n    Returns:\n        dict: A dictionary of installed software with versions installed\n\n    .. code-block:: cfg\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n        salt '*' pkg.list_pkgs versions_as_list=True\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    saltenv = kwargs.get('saltenv', 'base')
    refresh = salt.utils.data.is_true(kwargs.get('refresh', False))
    _refresh_db_conditional(saltenv, force=refresh)
    ret = {}
    name_map = _get_name_map(saltenv)
    for (pkg_name, val_list) in _get_reg_software(include_components=include_components, include_updates=include_updates).items():
        if pkg_name in name_map:
            key = name_map[pkg_name]
            for val in val_list:
                if val == 'Not Found':
                    pkg_info = _get_package_info(key, saltenv=saltenv)
                    if not pkg_info:
                        continue
                    for pkg_ver in pkg_info.keys():
                        if pkg_info[pkg_ver]['full_name'] == pkg_name:
                            val = pkg_ver
                __salt__['pkg_resource.add_pkg'](ret, key, val)
        else:
            key = pkg_name
            for val in val_list:
                __salt__['pkg_resource.add_pkg'](ret, key, val)
    __salt__['pkg_resource.sort_pkglist'](ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def _get_reg_software(include_components=True, include_updates=True):
    if False:
        i = 10
        return i + 15
    "\n    This searches the uninstall keys in the registry to find a match in the sub\n    keys, it will return a dict with the display name as the key and the\n    version as the value\n\n    Args:\n\n        include_components (bool):\n            Include sub components of installed software. Default is ``True``\n\n        include_updates (bool):\n            Include software updates and Windows updates. Default is ``True``\n\n    Returns:\n        dict: A dictionary of installed software with versions installed\n\n    .. code-block:: cfg\n\n        {'<package_name>': '<version>'}\n    "
    reg_software = {}

    def skip_component(hive, key, sub_key, use_32bit_registry):
        if False:
            i = 10
            return i + 15
        "\n        'SystemComponent' must be either absent or present with a value of 0,\n        because this value is usually set on programs that have been installed\n        via a Windows Installer Package (MSI).\n\n        Returns:\n            bool: True if the package needs to be skipped, otherwise False\n        "
        if include_components:
            return False
        if __utils__['reg.value_exists'](hive=hive, key=f'{key}\\{sub_key}', vname='SystemComponent', use_32bit_registry=use_32bit_registry):
            if __utils__['reg.read_value'](hive=hive, key=f'{key}\\{sub_key}', vname='SystemComponent', use_32bit_registry=use_32bit_registry)['vdata'] > 0:
                return True
        return False

    def skip_win_installer(hive, key, sub_key, use_32bit_registry):
        if False:
            for i in range(10):
                print('nop')
        "\n        'WindowsInstaller' must be either absent or present with a value of 0.\n        If the value is set to 1, then the application is included in the list\n        if and only if the corresponding compressed guid is also present in\n        HKLM:\\Software\\Classes\\Installer\\Products\n\n        Returns:\n            bool: True if the package needs to be skipped, otherwise False\n        "
        products_key = 'Software\\Classes\\Installer\\Products\\{0}'
        if __utils__['reg.value_exists'](hive=hive, key=f'{key}\\{sub_key}', vname='WindowsInstaller', use_32bit_registry=use_32bit_registry):
            if __utils__['reg.read_value'](hive=hive, key=f'{key}\\{sub_key}', vname='WindowsInstaller', use_32bit_registry=use_32bit_registry)['vdata'] > 0:
                squid = salt.utils.win_functions.guid_to_squid(sub_key)
                if not __utils__['reg.key_exists'](hive='HKLM', key=products_key.format(squid), use_32bit_registry=use_32bit_registry):
                    return True
        return False

    def skip_uninstall_string(hive, key, sub_key, use_32bit_registry):
        if False:
            i = 10
            return i + 15
        "\n        `UninstallString` must be present, because it stores the command line\n        that gets executed by Add/Remove programs, when the user tries to\n        uninstall a program. Skip those, unless `NoRemove` contains a non-zero\n        value in which case there is no `UninstallString` value.\n\n        We want to display these in case we're trying to install software that\n        will set the `NoRemove` option.\n\n        Returns:\n            bool: True if the package needs to be skipped, otherwise False\n        "
        if __utils__['reg.value_exists'](hive=hive, key=f'{key}\\{sub_key}', vname='NoRemove', use_32bit_registry=use_32bit_registry):
            if __utils__['reg.read_value'](hive=hive, key=f'{key}\\{sub_key}', vname='NoRemove', use_32bit_registry=use_32bit_registry)['vdata'] > 0:
                return False
        if not __utils__['reg.value_exists'](hive=hive, key=f'{key}\\{sub_key}', vname='UninstallString', use_32bit_registry=use_32bit_registry):
            return True
        return False

    def skip_release_type(hive, key, sub_key, use_32bit_registry):
        if False:
            i = 10
            return i + 15
        "\n        'ReleaseType' must either be absent or if present must not have a\n        value set to 'Security Update', 'Update Rollup', or 'Hotfix', because\n        that indicates it's an update to an existing program.\n\n        Returns:\n            bool: True if the package needs to be skipped, otherwise False\n        "
        if include_updates:
            return False
        skip_types = ['Hotfix', 'Security Update', 'Update Rollup']
        if __utils__['reg.value_exists'](hive=hive, key=f'{key}\\{sub_key}', vname='ReleaseType', use_32bit_registry=use_32bit_registry):
            if __utils__['reg.read_value'](hive=hive, key=f'{key}\\{sub_key}', vname='ReleaseType', use_32bit_registry=use_32bit_registry)['vdata'] in skip_types:
                return True
        return False

    def skip_parent_key(hive, key, sub_key, use_32bit_registry):
        if False:
            i = 10
            return i + 15
        "\n        'ParentKeyName' must NOT be present, because that indicates it's an\n        update to the parent program.\n\n        Returns:\n            bool: True if the package needs to be skipped, otherwise False\n        "
        if __utils__['reg.value_exists'](hive=hive, key=f'{key}\\{sub_key}', vname='ParentKeyName', use_32bit_registry=use_32bit_registry):
            return True
        return False

    def add_software(hive, key, sub_key, use_32bit_registry):
        if False:
            while True:
                i = 10
        "\n        'DisplayName' must be present with a valid value, as this is reflected\n        as the software name returned by pkg.list_pkgs. Also, its value must\n        not start with 'KB' followed by 6 numbers - as that indicates a\n        Windows update.\n        "
        d_name_regdata = __utils__['reg.read_value'](hive=hive, key=f'{key}\\{sub_key}', vname='DisplayName', use_32bit_registry=use_32bit_registry)
        if not d_name_regdata['success'] or d_name_regdata['vtype'] not in ['REG_SZ', 'REG_EXPAND_SZ'] or d_name_regdata['vdata'] in ['(value not set)', None, False]:
            return
        d_name = d_name_regdata['vdata']
        if not include_updates:
            if re.match('^KB[0-9]{6}', d_name):
                return
        d_vers_regdata = __utils__['reg.read_value'](hive=hive, key=f'{key}\\{sub_key}', vname='DisplayVersion', use_32bit_registry=use_32bit_registry)
        d_vers = 'Not Found'
        if d_vers_regdata['success'] and d_vers_regdata['vtype'] in ['REG_SZ', 'REG_EXPAND_SZ', 'REG_DWORD']:
            if isinstance(d_vers_regdata['vdata'], int):
                d_vers = str(d_vers_regdata['vdata'])
            elif d_vers_regdata['vdata'] and d_vers_regdata['vdata'] != '(value not set)':
                d_vers = d_vers_regdata['vdata']
        reg_software.setdefault(d_name, []).append(d_vers)
    kwargs = {'hive': 'HKLM', 'key': 'Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall', 'use_32bit_registry': False}
    for sub_key in __utils__['reg.list_keys'](**kwargs):
        kwargs['sub_key'] = sub_key
        if skip_component(**kwargs):
            continue
        if skip_win_installer(**kwargs):
            continue
        if skip_uninstall_string(**kwargs):
            continue
        if skip_release_type(**kwargs):
            continue
        if skip_parent_key(**kwargs):
            continue
        add_software(**kwargs)
    kwargs['use_32bit_registry'] = True
    kwargs.pop('sub_key', False)
    for sub_key in __utils__['reg.list_keys'](**kwargs):
        kwargs['sub_key'] = sub_key
        if skip_component(**kwargs):
            continue
        if skip_win_installer(**kwargs):
            continue
        if skip_uninstall_string(**kwargs):
            continue
        if skip_release_type(**kwargs):
            continue
        if skip_parent_key(**kwargs):
            continue
        add_software(**kwargs)
    kwargs = {'hive': 'HKLM', 'key': 'Software\\Classes\\Installer\\Products', 'use_32bit_registry': False}
    userdata_key = 'Software\\Microsoft\\Windows\\CurrentVersion\\Installer\\UserData\\S-1-5-18\\Products'
    for sub_key in __utils__['reg.list_keys'](**kwargs):
        if not __utils__['reg.key_exists'](hive=kwargs['hive'], key=f'{userdata_key}\\{sub_key}'):
            continue
        kwargs['sub_key'] = sub_key
        if skip_component(**kwargs):
            continue
        if skip_win_installer(**kwargs):
            continue
        add_software(**kwargs)
    hive_hku = 'HKU'
    uninstall_key = '{0}\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall'
    product_key = '{0}\\Software\\Microsoft\\Installer\\Products'
    user_data_key = 'Software\\Microsoft\\Windows\\CurrentVersion\\Installer\\UserData\\{0}\\Products\\{1}'
    for user_guid in __utils__['reg.list_keys'](hive=hive_hku):
        kwargs = {'hive': hive_hku, 'key': uninstall_key.format(user_guid), 'use_32bit_registry': False}
        if __utils__['reg.key_exists'](**kwargs):
            for sub_key in __utils__['reg.list_keys'](**kwargs):
                kwargs['sub_key'] = sub_key
                if skip_component(**kwargs):
                    continue
                if skip_win_installer(**kwargs):
                    continue
                if skip_uninstall_string(**kwargs):
                    continue
                if skip_release_type(**kwargs):
                    continue
                if skip_parent_key(**kwargs):
                    continue
                add_software(**kwargs)
        kwargs = {'hive': hive_hku, 'key': product_key.format(user_guid), 'use_32bit_registry': False}
        if __utils__['reg.key_exists'](**kwargs):
            for sub_key in __utils__['reg.list_keys'](**kwargs):
                kwargs = {'hive': 'HKLM', 'key': user_data_key.format(user_guid, sub_key), 'use_32bit_registry': False}
                if __utils__['reg.key_exists'](**kwargs):
                    kwargs['sub_key'] = 'InstallProperties'
                    if skip_component(**kwargs):
                        continue
                    add_software(**kwargs)
    for user_guid in __utils__['reg.list_keys'](hive=hive_hku, use_32bit_registry=True):
        kwargs = {'hive': hive_hku, 'key': uninstall_key.format(user_guid), 'use_32bit_registry': True}
        if __utils__['reg.key_exists'](**kwargs):
            for sub_key in __utils__['reg.list_keys'](**kwargs):
                kwargs['sub_key'] = sub_key
                if skip_component(**kwargs):
                    continue
                if skip_win_installer(**kwargs):
                    continue
                if skip_uninstall_string(**kwargs):
                    continue
                if skip_release_type(**kwargs):
                    continue
                if skip_parent_key(**kwargs):
                    continue
                add_software(**kwargs)
        kwargs = {'hive': hive_hku, 'key': product_key.format(user_guid), 'use_32bit_registry': True}
        if __utils__['reg.key_exists'](**kwargs):
            for sub_key_2 in __utils__['reg.list_keys'](**kwargs):
                kwargs = {'hive': 'HKLM', 'key': user_data_key.format(user_guid, sub_key_2), 'use_32bit_registry': True}
                if __utils__['reg.key_exists'](**kwargs):
                    kwargs['sub_key'] = 'InstallProperties'
                    if skip_component(**kwargs):
                        continue
                    add_software(**kwargs)
    return reg_software

def _refresh_db_conditional(saltenv, **kwargs):
    if False:
        print('Hello World!')
    '\n    Internal use only in this module, has a different set of defaults and\n    returns True or False. And supports checking the age of the existing\n    generated metadata db, as well as ensure metadata db exists to begin with\n\n    Args:\n        saltenv (str): Salt environment\n\n    Kwargs:\n\n        force (bool):\n            Force a refresh if the minimum age has been reached. Default is\n            False.\n\n        failhard (bool):\n            If ``True``, an error will be raised if any repo SLS files failed to\n            process.\n\n    Returns:\n        bool: True Fetched or Cache uptodate, False to indicate an issue\n\n    :codeauthor: Damon Atkins <https://github.com/damon-atkins>\n    '
    force = salt.utils.data.is_true(kwargs.pop('force', False))
    failhard = salt.utils.data.is_true(kwargs.pop('failhard', False))
    expired_max = __opts__['winrepo_cache_expire_max']
    expired_min = __opts__['winrepo_cache_expire_min']
    repo_details = _get_repo_details(saltenv)
    if force and expired_min > 0 and (repo_details.winrepo_age < expired_min):
        log.info('Refresh skipped, age of winrepo metadata in seconds (%s) is less than winrepo_cache_expire_min (%s)', repo_details.winrepo_age, expired_min)
        force = False
    refresh = True if force or repo_details.winrepo_age == -1 or repo_details.winrepo_age > expired_max else False
    if not refresh:
        log.debug("Using existing pkg metadata db for saltenv '%s' (age is %s)", saltenv, datetime.timedelta(seconds=repo_details.winrepo_age))
        return True
    if repo_details.winrepo_age == -1:
        log.debug("No winrepo.p cache file for saltenv '%s', creating one now", saltenv)
    results = refresh_db(saltenv=saltenv, verbose=False, failhard=failhard)
    try:
        return not bool(results.get('failed', 0))
    except AttributeError:
        return False

def refresh_db(**kwargs):
    if False:
        return 10
    "\n    Generates the local software metadata database (`winrepo.p`) on the minion.\n    The database is stored in a serialized format located by default at the\n    following location:\n\n    ``C:\\salt\\var\\cache\\salt\\minion\\files\\base\\win\\repo-ng\\winrepo.p``\n\n    This module performs the following steps to generate the software metadata\n    database:\n\n    - Fetch the package definition files (.sls) from `winrepo_source_dir`\n      (default `salt://win/repo-ng`) and cache them in\n      `<cachedir>\\files\\<saltenv>\\<winrepo_source_dir>`\n      (default: ``C:\\salt\\var\\cache\\salt\\minion\\files\\base\\win\\repo-ng``)\n    - Call :py:func:`pkg.genrepo <salt.modules.win_pkg.genrepo>` to parse the\n      package definition files and generate the repository metadata database\n      file (`winrepo.p`)\n    - Return the report received from\n      :py:func:`pkg.genrepo <salt.modules.win_pkg.genrepo>`\n\n    The default winrepo directory on the master is `/srv/salt/win/repo-ng`. All\n    files that end with `.sls` in this and all subdirectories will be used to\n    generate the repository metadata database (`winrepo.p`).\n\n    .. note::\n        - Hidden directories (directories beginning with '`.`', such as\n          '`.git`') will be ignored.\n\n    .. note::\n        There is no need to call `pkg.refresh_db` every time you work with the\n        pkg module. Automatic refresh will occur based on the following minion\n        configuration settings:\n\n        - `winrepo_cache_expire_min`\n        - `winrepo_cache_expire_max`\n\n        However, if the package definition files have changed, as would be the\n        case if you are developing a new package definition, this function\n        should be called to ensure the minion has the latest information about\n        packages available to it.\n\n    .. warning::\n        Directories and files fetched from <winrepo_source_dir>\n        (`/srv/salt/win/repo-ng`) will be processed in alphabetical order. If\n        two or more software definition files contain the same name, the last\n        one processed replaces all data from the files processed before it.\n\n    For more information see\n    :ref:`Windows Software Repository <windows-package-manager>`\n\n    Arguments:\n\n    saltenv (str): Salt environment. Default: ``base``\n\n    verbose (bool):\n        Return a verbose data structure which includes 'success_list', a\n        list of all sls files and the package names contained within.\n        Default is 'False'\n\n    failhard (bool):\n        If ``True``, an error will be raised if any repo SLS files fails to\n        process. If ``False``, no error will be raised, and a dictionary\n        containing the full results will be returned.\n\n    Returns:\n        dict: A dictionary containing the results of the database refresh.\n\n    .. note::\n        A result with a `total: 0` generally means that the files are in the\n        wrong location on the master. Try running the following command on the\n        minion: `salt-call -l debug pkg.refresh saltenv=base`\n\n    .. warning::\n        When calling this command from a state using `module.run` be sure to\n        pass `failhard: False`. Otherwise the state will report failure if it\n        encounters a bad software definition file.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n        salt '*' pkg.refresh_db saltenv=base\n    "
    salt.utils.pkg.clear_rtag(__opts__)
    saltenv = kwargs.pop('saltenv', 'base')
    verbose = salt.utils.data.is_true(kwargs.pop('verbose', False))
    failhard = salt.utils.data.is_true(kwargs.pop('failhard', True))
    __context__.pop('winrepo.data', None)
    repo_details = _get_repo_details(saltenv)
    log.debug("Refreshing pkg metadata db for saltenv '%s' (age of existing metadata is %s)", saltenv, datetime.timedelta(seconds=repo_details.winrepo_age))
    log.info("Removing all *.sls files under '%s'", repo_details.local_dest)
    failed = []
    for (root, _, files) in salt.utils.path.os_walk(repo_details.local_dest, followlinks=False):
        for name in files:
            if name.endswith('.sls'):
                full_filename = os.path.join(root, name)
                try:
                    os.remove(full_filename)
                except OSError as exc:
                    if exc.errno != errno.ENOENT:
                        log.error('Failed to remove %s: %s', full_filename, exc)
                        failed.append(full_filename)
    if failed:
        raise CommandExecutionError('Failed to clear one or more winrepo cache files', info={'failed': failed})
    log.info('Fetching *.sls files from %s', repo_details.winrepo_source_dir)
    try:
        __salt__['cp.cache_dir'](path=repo_details.winrepo_source_dir, saltenv=saltenv, include_pat='*.sls', exclude_pat='E@\\/\\..*?\\/')
    except MinionError as exc:
        log.exception('Failed to cache %s', repo_details.winrepo_source_dir, exc_info=exc)
    return genrepo(saltenv=saltenv, verbose=verbose, failhard=failhard)

def _get_repo_details(saltenv):
    if False:
        while True:
            i = 10
    '\n    Return repo details for the specified saltenv as a namedtuple\n    '
    contextkey = f'winrepo._get_repo_details.{saltenv}'
    if contextkey in __context__:
        (winrepo_source_dir, local_dest, winrepo_file) = __context__[contextkey]
    else:
        winrepo_source_dir = __opts__['winrepo_source_dir']
        dirs = [__opts__['cachedir'], 'files', saltenv]
        url_parts = urllib.parse.urlparse(winrepo_source_dir)
        dirs.append(url_parts.netloc)
        dirs.extend(url_parts.path.strip('/').split('/'))
        local_dest = os.sep.join(dirs)
        winrepo_file = os.path.join(local_dest, 'winrepo.p')
        if not re.search('[\\/:*?"<>|]', __opts__['winrepo_cachefile'], flags=re.IGNORECASE):
            winrepo_file = os.path.join(local_dest, __opts__['winrepo_cachefile'])
        else:
            log.error("minion configuration option 'winrepo_cachefile' has been ignored as its value (%s) is invalid. Please ensure this option is set to a valid filename.", __opts__['winrepo_cachefile'])
        system_root = os.environ.get('SystemRoot', 'C:\\Windows')
        if not salt.utils.path.safe_path(path=local_dest, allow_path='\\'.join([system_root, 'TEMP'])):
            raise CommandExecutionError('Attempting to delete files from a possibly unsafe location: {}'.format(local_dest))
        __context__[contextkey] = (winrepo_source_dir, local_dest, winrepo_file)
    try:
        os.makedirs(local_dest)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise CommandExecutionError(f'Failed to create {local_dest}: {exc}')
    winrepo_age = -1
    try:
        stat_result = os.stat(winrepo_file)
        mtime = stat_result.st_mtime
        winrepo_age = time.time() - mtime
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise CommandExecutionError(f'Failed to get age of {winrepo_file}: {exc}')
    except AttributeError:
        log.warning('st_mtime missing from stat result %s', stat_result)
    except TypeError:
        log.warning('mtime of %s (%s) is an invalid type', winrepo_file, mtime)
    repo_details = collections.namedtuple('RepoDetails', ('winrepo_source_dir', 'local_dest', 'winrepo_file', 'winrepo_age'))
    return repo_details(winrepo_source_dir, local_dest, winrepo_file, winrepo_age)

def genrepo(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Generate package metadata db based on files within the winrepo_source_dir\n\n    Kwargs:\n\n        saltenv (str): Salt environment. Default: ``base``\n\n        verbose (bool):\n            Return verbose data structure which includes 'success_list', a list\n            of all sls files and the package names contained within.\n            Default ``False``.\n\n        failhard (bool):\n            If ``True``, an error will be raised if any repo SLS files failed\n            to process. If ``False``, no error will be raised, and a dictionary\n            containing the full results will be returned.\n\n    .. note::\n        - Hidden directories (directories beginning with '`.`', such as\n          '`.git`') will be ignored.\n\n    Returns:\n        dict: A dictionary of the results of the command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt-run pkg.genrepo\n        salt -G 'os:windows' pkg.genrepo verbose=true failhard=false\n        salt -G 'os:windows' pkg.genrepo saltenv=base\n    "
    saltenv = kwargs.pop('saltenv', 'base')
    verbose = salt.utils.data.is_true(kwargs.pop('verbose', False))
    failhard = salt.utils.data.is_true(kwargs.pop('failhard', True))
    ret = {}
    successful_verbose = {}
    total_files_processed = 0
    ret['repo'] = {}
    ret['errors'] = {}
    repo_details = _get_repo_details(saltenv)
    for (root, _, files) in salt.utils.path.os_walk(repo_details.local_dest, followlinks=False):
        if re.search('[\\\\/]\\..*', root):
            log.debug('Skipping files in directory: %s', root)
            continue
        short_path = os.path.relpath(root, repo_details.local_dest)
        if short_path == '.':
            short_path = ''
        for name in files:
            if name.endswith('.sls'):
                total_files_processed += 1
                _repo_process_pkg_sls(os.path.join(root, name), os.path.join(short_path, name), ret, successful_verbose)
    with salt.utils.files.fopen(repo_details.winrepo_file, 'wb') as repo_cache:
        repo_cache.write(salt.payload.dumps(ret))
    successful_count = len(successful_verbose)
    error_count = len(ret['errors'])
    if verbose:
        results = {'total': total_files_processed, 'success': successful_count, 'failed': error_count, 'success_list': successful_verbose, 'failed_list': ret['errors']}
    elif error_count > 0:
        results = {'total': total_files_processed, 'success': successful_count, 'failed': error_count, 'failed_list': ret['errors']}
    else:
        results = {'total': total_files_processed, 'success': successful_count, 'failed': error_count}
    if error_count > 0 and failhard:
        raise CommandExecutionError('Error occurred while generating repo db', info=results)
    else:
        return results

def _repo_process_pkg_sls(filename, short_path_name, ret, successful_verbose):
    if False:
        return 10
    renderers = salt.loader.render(__opts__, __salt__)

    def _failed_compile(prefix_msg, error_msg):
        if False:
            return 10
        log.error("%s '%s': %s", prefix_msg, short_path_name, error_msg)
        ret.setdefault('errors', {})[short_path_name] = [f'{prefix_msg}, {error_msg} ']
        return False
    try:
        config = salt.template.compile_template(filename, renderers, __opts__['renderer'], __opts__.get('renderer_blacklist', ''), __opts__.get('renderer_whitelist', ''))
    except SaltRenderError as exc:
        return _failed_compile('Failed to compile', exc)
    except Exception as exc:
        return _failed_compile('Failed to read', exc)
    if config and isinstance(config, dict):
        revmap = {}
        errors = []
        for (pkgname, version_list) in config.items():
            if pkgname in ret['repo']:
                log.error("package '%s' within '%s' already defined, skipping", pkgname, short_path_name)
                errors.append(f"package '{pkgname}' already defined")
                break
            for (version_str, repodata) in version_list.items():
                if not isinstance(version_str, str):
                    log.error("package '%s' within '%s', version number %s' is not a string", pkgname, short_path_name, version_str)
                    errors.append("package '{}', version number {} is not a string".format(pkgname, version_str))
                    continue
                if not isinstance(repodata, dict):
                    log.error("package '%s' within '%s', repo data for version number %s is not defined as a dictionary", pkgname, short_path_name, version_str)
                    errors.append("package '{}', repo data for version number {} is not defined as a dictionary".format(pkgname, version_str))
                    continue
                revmap[repodata['full_name']] = pkgname
        if errors:
            ret.setdefault('errors', {})[short_path_name] = errors
        else:
            ret.setdefault('repo', {}).update(config)
            ret.setdefault('name_map', {}).update(revmap)
            successful_verbose[short_path_name] = list(config.keys())
    elif config:
        return _failed_compile('Compiled contents', 'not a dictionary/hash')
    else:
        log.debug("No data within '%s' after processing", short_path_name)
        successful_verbose[short_path_name] = []

def _get_source_sum(source_hash, file_path, saltenv, verify_ssl=True):
    if False:
        while True:
            i = 10
    '\n    Extract the hash sum, whether it is in a remote hash file, or just a string.\n    '
    ret = dict()
    schemes = ('salt', 'http', 'https', 'ftp', 'swift', 's3', 'file')
    invalid_hash_msg = "Source hash '{}' format is invalid. It must be in the format <hash type>=<hash>".format(source_hash)
    source_hash = str(source_hash)
    source_hash_scheme = urllib.parse.urlparse(source_hash).scheme
    if source_hash_scheme in schemes:
        try:
            cached_hash_file = __salt__['cp.cache_file'](source_hash, saltenv=saltenv, verify_ssl=verify_ssl, use_etag=True)
        except MinionError as exc:
            log.exception('Failed to cache %s', source_hash, exc_info=exc)
            raise
        if not cached_hash_file:
            raise CommandExecutionError(f'Source hash file {source_hash} not found')
        ret = __salt__['file.extract_hash'](cached_hash_file, '', file_path)
        if ret is None:
            raise SaltInvocationError(invalid_hash_msg)
    else:
        items = source_hash.split('=', 1)
        if len(items) != 2:
            invalid_hash_msg = '{}, or it must be a supported protocol: {}'.format(invalid_hash_msg, ', '.join(schemes))
            raise SaltInvocationError(invalid_hash_msg)
        (ret['hash_type'], ret['hsum']) = (item.strip().lower() for item in items)
    return ret

def _get_msiexec(use_msiexec):
    if False:
        while True:
            i = 10
    '\n    Return if msiexec.exe will be used and the command to invoke it.\n    '
    if use_msiexec is False:
        return (False, '')
    if isinstance(use_msiexec, str):
        if os.path.isfile(use_msiexec):
            return (True, use_msiexec)
        else:
            log.warning("msiexec path '%s' not found. Using system registered msiexec instead", use_msiexec)
            use_msiexec = True
    if use_msiexec is True:
        return (True, 'msiexec')

def normalize_name(name):
    if False:
        print('Hello World!')
    "\n    Nothing to do on Windows. We need this function so that Salt doesn't go\n    through every module looking for ``pkg.normalize_name``.\n\n    .. versionadded:: 3006.0\n\n    Args:\n        name (str): The name of the package\n\n    Returns:\n        str: The name of the package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.normalize_name git\n    "
    return name

def install(name=None, refresh=False, pkgs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Install the passed package(s) on the system using winrepo\n\n    Args:\n\n        name (str):\n            The name of a single package, or a comma-separated list of packages\n            to install. (no spaces after the commas)\n\n        refresh (bool):\n            Boolean value representing whether or not to refresh the winrepo db.\n            Default ``False``.\n\n        pkgs (list):\n            A list of packages to install from a software repository. All\n            packages listed under ``pkgs`` will be installed via a single\n            command.\n\n            You can specify a version by passing the item as a dict:\n\n            CLI Example:\n\n            .. code-block:: bash\n\n                # will install the latest version of foo and bar\n                salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n\n                # will install the latest version of foo and version 1.2.3 of bar\n                salt \'*\' pkg.install pkgs=\'["foo", {"bar": "1.2.3"}]\'\n\n    Kwargs:\n\n        version (str):\n            The specific version to install. If omitted, the latest version will\n            be installed. Recommend for use when installing a single package.\n\n            If passed with a list of packages in the ``pkgs`` parameter, the\n            version will be ignored.\n\n            CLI Example:\n\n             .. code-block:: bash\n\n                # Version is ignored\n                salt \'*\' pkg.install pkgs="[\'foo\', \'bar\']" version=1.2.3\n\n            If passed with a comma separated list in the ``name`` parameter, the\n            version will apply to all packages in the list.\n\n            CLI Example:\n\n             .. code-block:: bash\n\n                # Version 1.2.3 will apply to packages foo and bar\n                salt \'*\' pkg.install foo,bar version=1.2.3\n\n        extra_install_flags (str):\n            Additional install flags that will be appended to the\n            ``install_flags`` defined in the software definition file. Only\n            applies when single package is passed.\n\n        saltenv (str):\n            Salt environment. Default \'base\'\n\n        report_reboot_exit_codes (bool):\n            If the installer exits with a recognized exit code indicating that\n            a reboot is required, the module function\n\n               *win_system.set_reboot_required_witnessed*\n\n            will be called, preserving the knowledge of this event for the\n            remainder of the current boot session. For the time being, 3010 is\n            the only recognized exit code. The value of this param defaults to\n            True.\n\n            .. versionadded:: 2016.11.0\n\n    Returns:\n        dict: Return a dict containing the new package names and versions. If\n        the package is already installed, an empty dict is returned.\n\n        If the package is installed by ``pkg.install``:\n\n        .. code-block:: cfg\n\n            {\'<package>\': {\'old\': \'<old-version>\',\n                           \'new\': \'<new-version>\'}}\n\n    The following example will refresh the winrepo and install a single\n    package, 7zip.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install 7zip refresh=True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install 7zip\n        salt \'*\' pkg.install 7zip,filezilla\n        salt \'*\' pkg.install pkgs=\'["7zip","filezilla"]\'\n\n    WinRepo Definition File Examples:\n\n    The following example demonstrates the use of ``cache_file``. This would be\n    used if you have multiple installers in the same directory that use the\n    same ``install.ini`` file and you don\'t want to download the additional\n    installers.\n\n    .. code-block:: bash\n\n        ntp:\n          4.2.8:\n            installer: \'salt://win/repo/ntp/ntp-4.2.8-win32-setup.exe\'\n            full_name: Meinberg NTP Windows Client\n            locale: en_US\n            reboot: False\n            cache_file: \'salt://win/repo/ntp/install.ini\'\n            install_flags: \'/USEFILE=C:\\salt\\var\\cache\\salt\\minion\\files\\base\\win\\repo\\ntp\\install.ini\'\n            uninstaller: \'NTP/uninst.exe\'\n\n    The following example demonstrates the use of ``cache_dir``. It assumes a\n    file named ``install.ini`` resides in the same directory as the installer.\n\n    .. code-block:: bash\n\n        ntp:\n          4.2.8:\n            installer: \'salt://win/repo/ntp/ntp-4.2.8-win32-setup.exe\'\n            full_name: Meinberg NTP Windows Client\n            locale: en_US\n            reboot: False\n            cache_dir: True\n            install_flags: \'/USEFILE=C:\\salt\\var\\cache\\salt\\minion\\files\\base\\win\\repo\\ntp\\install.ini\'\n            uninstaller: \'NTP/uninst.exe\'\n    '
    ret = {}
    saltenv = kwargs.pop('saltenv', 'base')
    refresh = salt.utils.data.is_true(refresh)
    if not name and (not pkgs):
        return 'Must pass a single package or a list of packages'
    pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs, **kwargs)[0]
    if len(pkg_params) > 1:
        if kwargs.get('extra_install_flags') is not None:
            log.warning("'extra_install_flags' argument will be ignored for multiple package targets")
    for pkg in pkg_params:
        pkg_params[pkg] = {'version': pkg_params[pkg]}
    if not pkg_params:
        log.error('No package definition found')
        return {}
    if not pkgs and len(pkg_params) == 1:
        pkg_params = {name: {'version': kwargs.get('version'), 'extra_install_flags': kwargs.get('extra_install_flags')}}
    elif len(pkg_params) == 1:
        pkg = next(iter(pkg_params))
        pkg_params[pkg]['extra_install_flags'] = kwargs.get('extra_install_flags')
    old = list_pkgs(saltenv=saltenv, refresh=refresh, versions_as_list=True)
    changed = []
    for (pkg_name, options) in pkg_params.items():
        pkginfo = _get_package_info(pkg_name, saltenv=saltenv)
        if not pkginfo:
            log.error('Unable to locate package %s', pkg_name)
            ret[pkg_name] = f'Unable to locate package {pkg_name}'
            continue
        version_num = options.get('version')
        if not isinstance(version_num, str) and version_num is not None:
            version_num = str(version_num)
        if not version_num:
            if pkg_name in old:
                log.debug("pkg.install: '%s' version '%s' is already installed", pkg_name, old[pkg_name][0])
                continue
            version_num = _get_latest_pkg_version(pkginfo)
        if version_num == 'latest' and 'latest' not in pkginfo:
            version_num = _get_latest_pkg_version(pkginfo)
        if version_num in old.get(pkg_name, []):
            log.debug("pkg.install: '%s' version '%s' is already installed", pkg_name, version_num)
            continue
        elif version_num != 'latest' and version_num not in pkginfo:
            log.error('Version %s not found for package %s', version_num, pkg_name)
            ret[pkg_name] = {'not found': version_num}
            continue
        installer = pkginfo[version_num].get('installer', '')
        cache_dir = pkginfo[version_num].get('cache_dir', False)
        cache_file = pkginfo[version_num].get('cache_file', '')
        if not installer:
            log.error('No installer configured for version %s of package %s', version_num, pkg_name)
            ret[pkg_name] = {'no installer': version_num}
            continue
        installer_hash = __salt__['cp.hash_file'](installer, saltenv)
        if isinstance(installer_hash, dict):
            installer_hash = installer_hash['hsum']
        else:
            installer_hash = None
        if __salt__['config.valid_fileproto'](installer):
            if cache_dir and installer.startswith('salt:'):
                (path, _) = os.path.split(installer)
                log.debug(f'PKG: Caching directory: {path}')
                try:
                    __salt__['cp.cache_dir'](path=path, saltenv=saltenv, include_empty=False, include_pat=None, exclude_pat='E@init.sls$')
                except MinionError as exc:
                    msg = f'Failed to cache {path}'
                    log.exception(msg, exc_info=exc)
                    return f'{msg}\n{exc}'
            if cache_file and cache_file.startswith('salt:'):
                cache_file_hash = __salt__['cp.hash_file'](cache_file, saltenv)
                log.debug(f'PKG: Caching file: {cache_file}')
                try:
                    cached_file = __salt__['cp.cache_file'](cache_file, saltenv=saltenv, source_hash=cache_file_hash, verify_ssl=kwargs.get('verify_ssl', True))
                except MinionError as exc:
                    msg = f'Failed to cache {cache_file}'
                    log.exception(msg, exc_info=exc)
                    return f'{msg}\n{exc}'
                if not cached_file:
                    log.error('Unable to cache %s', cache_file)
                    ret[pkg_name] = {'failed to cache cache_file': cache_file}
                    continue
            cached_pkg = False
            if version_num != 'latest' and (not installer.startswith('salt:')):
                cached_pkg = __salt__['cp.is_cached'](installer, saltenv)
            if not cached_pkg:
                log.debug(f'PKG: Caching file: {installer}')
                try:
                    cached_pkg = __salt__['cp.cache_file'](installer, saltenv=saltenv, source_hash=installer_hash, verify_ssl=kwargs.get('verify_ssl', True), use_etag=True)
                except MinionError as exc:
                    msg = f'Failed to cache {installer}'
                    log.exception(msg, exc_info=exc)
                    return f'{msg}\n{exc}'
                if not cached_pkg:
                    log.error('Unable to cache file %s from saltenv: %s', installer, saltenv)
                    ret[pkg_name] = {'unable to cache': installer}
                    continue
        else:
            cached_pkg = installer
        cached_pkg = cached_pkg.replace('/', '\\')
        cache_path = os.path.dirname(cached_pkg)
        source_hash = pkginfo[version_num].get('source_hash', False)
        if source_hash:
            source_sum = _get_source_sum(source_hash, cached_pkg, saltenv=saltenv, verify_ssl=kwargs.get('verify_ssl', True))
            log.debug('pkg.install: Source %s hash: %s', source_sum['hash_type'], source_sum['hsum'])
            cached_pkg_sum = salt.utils.hashutils.get_hash(cached_pkg, source_sum['hash_type'])
            log.debug('pkg.install: Package %s hash: %s', source_sum['hash_type'], cached_pkg_sum)
            if source_sum['hsum'] != cached_pkg_sum:
                raise SaltInvocationError("Source hash '{}' does not match package hash '{}'".format(source_sum['hsum'], cached_pkg_sum))
            log.debug('pkg.install: Source hash matches package hash.')
        install_flags = pkginfo[version_num].get('install_flags', '')
        if options and options.get('extra_install_flags'):
            install_flags = '{} {}'.format(install_flags, options.get('extra_install_flags', ''))
        (use_msiexec, msiexec) = _get_msiexec(pkginfo[version_num].get('msiexec', False))
        cmd_shell = os.getenv('ComSpec', '{}\\system32\\cmd.exe'.format(os.getenv('WINDIR')))
        if use_msiexec:
            arguments = f'"{msiexec}" /I "{cached_pkg}"'
            if pkginfo[version_num].get('allusers', True):
                arguments = f'{arguments} ALLUSERS=1'
        else:
            arguments = f'"{cached_pkg}"'
        if install_flags:
            arguments = f'{arguments} {install_flags}'
        log.debug('PKG : cmd: %s /c %s', cmd_shell, arguments)
        log.debug('PKG : pwd: %s', cache_path)
        if pkginfo[version_num].get('use_scheduler', False):
            __salt__['task.create_task'](name='update-salt-software', user_name='System', force=True, action_type='Execute', cmd=cmd_shell, arguments=f'/c "{arguments}"', start_in=cache_path, trigger_type='Once', start_date='1975-01-01', start_time='01:00', ac_only=False, stop_if_on_batteries=False)
            if re.search('salt[\\s_.-]*minion', pkg_name, flags=re.IGNORECASE + re.UNICODE) is not None:
                ret[pkg_name] = {'install status': 'task started'}
                if not __salt__['task.run'](name='update-salt-software'):
                    log.error('Scheduled Task failed to run. Failed to install %s', pkg_name)
                    ret[pkg_name] = {'install status': 'failed'}
                else:
                    t_end = time.time() + 5
                    while time.time() < t_end:
                        time.sleep(0.25)
                        task_running = __salt__['task.status']('update-salt-software') == 'Running'
                        if task_running:
                            break
                    if not task_running:
                        log.error('Scheduled Task failed to run. Failed to install %s', pkg_name)
                        ret[pkg_name] = {'install status': 'failed'}
            elif not __salt__['task.run_wait'](name='update-salt-software'):
                log.error('Scheduled Task failed to run. Failed to install %s', pkg_name)
                ret[pkg_name] = {'install status': 'failed'}
        else:
            result = __salt__['cmd.run_all'](f'"{cmd_shell}" /c "{arguments}"', cache_path, output_loglevel='trace', python_shell=False, redirect_stderr=True)
            log.debug('PKG : retcode: %s', result['retcode'])
            if not result['retcode']:
                ret[pkg_name] = {'install status': 'success'}
                changed.append(pkg_name)
            elif result['retcode'] == 3010:
                report_reboot_exit_codes = kwargs.pop('report_reboot_exit_codes', True)
                if report_reboot_exit_codes:
                    __salt__['system.set_reboot_required_witnessed']()
                ret[pkg_name] = {'install status': 'success, reboot required'}
                changed.append(pkg_name)
            elif result['retcode'] == 1641:
                ret[pkg_name] = {'install status': 'success, reboot initiated'}
                changed.append(pkg_name)
            else:
                log.error('Failed to install %s; retcode: %s; installer output: %s', pkg_name, result['retcode'], result['stdout'])
                ret[pkg_name] = {'install status': 'failed'}
    new = list_pkgs(saltenv=saltenv, refresh=False)
    __salt__['pkg_resource.stringify'](old)
    difference = salt.utils.data.compare_dicts(old, new)
    ret.update(difference)
    return ret

def upgrade(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Upgrade all software. Currently not implemented\n\n    Kwargs:\n        saltenv (str): The salt environment to use. Default ``base``.\n        refresh (bool): Refresh package metadata. Default ``True``.\n\n    .. note::\n        This feature is not yet implemented for Windows.\n\n    Returns:\n        dict: Empty dict, until implemented\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n    "
    log.warning('pkg.upgrade not implemented on Windows yet')
    refresh = salt.utils.data.is_true(kwargs.get('refresh', True))
    saltenv = kwargs.get('saltenv', 'base')
    log.warning('pkg.upgrade not implemented on Windows yet refresh:%s saltenv:%s', refresh, saltenv)
    return {}

def remove(name=None, pkgs=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Remove the passed package(s) from the system using winrepo\n\n    .. versionadded:: 0.16.0\n\n    Args:\n        name (str):\n            The name(s) of the package(s) to be uninstalled. Can be a\n            single package or a comma delimited list of packages, no spaces.\n\n        pkgs (list):\n            A list of packages to delete. Must be passed as a python list. The\n            ``name`` parameter will be ignored if this option is passed.\n\n    Kwargs:\n\n        version (str):\n            The version of the package to be uninstalled. If this option is\n            used to to uninstall multiple packages, then this version will be\n            applied to all targeted packages. Recommended using only when\n            uninstalling a single package. If this parameter is omitted, the\n            latest version will be uninstalled.\n\n        saltenv (str): Salt environment. Default ``base``\n        refresh (bool): Refresh package metadata. Default ``False``\n\n    Returns:\n        dict: Returns a dict containing the changes.\n\n        If the package is removed by ``pkg.remove``:\n\n            {\'<package>\': {\'old\': \'<old-version>\',\n                           \'new\': \'<new-version>\'}}\n\n        If the package is already uninstalled:\n\n            {\'<package>\': {\'current\': \'not installed\'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    saltenv = kwargs.get('saltenv', 'base')
    refresh = salt.utils.data.is_true(kwargs.get('refresh', False))
    ret = {}
    if not name and (not pkgs):
        return 'Must pass a single package or a list of packages'
    pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs, **kwargs)[0]
    old = list_pkgs(saltenv=saltenv, refresh=refresh, versions_as_list=True)
    changed = []
    for (pkgname, version_num) in pkg_params.items():
        pkginfo = _get_package_info(pkgname, saltenv=saltenv)
        if not pkginfo:
            msg = f'Unable to locate package {pkgname}'
            log.error(msg)
            ret[pkgname] = msg
            continue
        if pkgname not in old:
            log.debug('%s %s not installed', pkgname, version_num if version_num else '')
            ret[pkgname] = {'current': 'not installed'}
            continue
        removal_targets = []
        if version_num is not None:
            version_num = str(version_num)
        if version_num is None:
            for ver_install in old[pkgname]:
                if ver_install not in pkginfo and 'latest' in pkginfo:
                    log.debug('%s %s using package latest entry to to remove', pkgname, version_num)
                    removal_targets.append('latest')
                else:
                    removal_targets.append(ver_install)
        elif version_num in pkginfo:
            if version_num in old[pkgname]:
                removal_targets.append(version_num)
            else:
                log.debug('%s %s not installed', pkgname, version_num)
                ret[pkgname] = {'current': f'{version_num} not installed'}
                continue
        elif 'latest' in pkginfo:
            log.debug('%s %s using package latest entry to to remove', pkgname, version_num)
            removal_targets.append('latest')
        if not removal_targets:
            log.error('%s %s no definition to remove this version', pkgname, version_num)
            ret[pkgname] = {'current': f'{version_num} no definition, cannot removed'}
            continue
        for target in removal_targets:
            uninstaller = pkginfo[target].get('uninstaller', '')
            cache_dir = pkginfo[target].get('cache_dir', False)
            uninstall_flags = pkginfo[target].get('uninstall_flags', '')
            if not uninstaller and uninstall_flags:
                uninstaller = pkginfo[target].get('installer', '')
            if not uninstaller:
                log.error('No installer or uninstaller configured for package %s', pkgname)
                ret[pkgname] = {'no uninstaller defined': target}
                continue
            uninstaller_hash = __salt__['cp.hash_file'](uninstaller, saltenv)
            if isinstance(uninstaller_hash, dict):
                uninstaller_hash = uninstaller_hash['hsum']
            else:
                uninstaller_hash = None
            if __salt__['config.valid_fileproto'](uninstaller):
                if cache_dir and uninstaller.startswith('salt:'):
                    (path, _) = os.path.split(uninstaller)
                    log.debug(f'PKG: Caching dir: {path}')
                    try:
                        __salt__['cp.cache_dir'](path=path, saltenv=saltenv, include_empty=False, include_pat=None, exclude_pat='E@init.sls$')
                    except MinionError as exc:
                        msg = f'Failed to cache {path}'
                        log.exception(msg, exc_info=exc)
                        return f'{msg}\n{exc}'
                cached_pkg = __salt__['cp.is_cached'](uninstaller, saltenv)
                if not cached_pkg:
                    log.debug(f'PKG: Caching file: {uninstaller}')
                    try:
                        cached_pkg = __salt__['cp.cache_file'](uninstaller, saltenv=saltenv, source_hash=uninstaller_hash, verify_ssl=kwargs.get('verify_ssl', True), use_etag=True)
                    except MinionError as exc:
                        msg = f'Failed to cache {uninstaller}'
                        log.exception(msg, exc_info=exc)
                        return f'{msg}\n{exc}'
                    if not cached_pkg:
                        log.error('Unable to cache %s', uninstaller)
                        ret[pkgname] = {'unable to cache': uninstaller}
                        continue
            else:
                cached_pkg = os.path.expandvars(uninstaller)
            cached_pkg = cached_pkg.replace('/', '\\')
            (cache_path, _) = os.path.split(cached_pkg)
            if kwargs.get('extra_uninstall_flags'):
                uninstall_flags = '{} {}'.format(uninstall_flags, kwargs.get('extra_uninstall_flags', ''))
            (use_msiexec, msiexec) = _get_msiexec(pkginfo[target].get('msiexec', False))
            cmd_shell = os.getenv('ComSpec', '{}\\system32\\cmd.exe'.format(os.getenv('WINDIR')))
            if use_msiexec:
                arguments = f'"{msiexec}" /X "{cached_pkg}"'
            else:
                arguments = f'"{cached_pkg}"'
            if uninstall_flags:
                arguments = f'{arguments} {uninstall_flags}'
            changed.append(pkgname)
            log.debug('PKG : cmd: %s /c %s', cmd_shell, arguments)
            log.debug('PKG : pwd: %s', cache_path)
            if pkginfo[target].get('use_scheduler', False):
                __salt__['task.create_task'](name='update-salt-software', user_name='System', force=True, action_type='Execute', cmd=cmd_shell, arguments=f'/c "{arguments}"', start_in=cache_path, trigger_type='Once', start_date='1975-01-01', start_time='01:00', ac_only=False, stop_if_on_batteries=False)
                if not __salt__['task.run_wait'](name='update-salt-software'):
                    log.error('Scheduled Task failed to run. Failed to remove %s', pkgname)
                    ret[pkgname] = {'uninstall status': 'failed'}
            else:
                result = __salt__['cmd.run_all'](f'"{cmd_shell}" /c "{arguments}"', output_loglevel='trace', python_shell=False, redirect_stderr=True)
                log.debug('PKG : retcode: %s', result['retcode'])
                if not result['retcode']:
                    ret[pkgname] = {'uninstall status': 'success'}
                    changed.append(pkgname)
                elif result['retcode'] == 3010:
                    report_reboot_exit_codes = kwargs.pop('report_reboot_exit_codes', True)
                    if report_reboot_exit_codes:
                        __salt__['system.set_reboot_required_witnessed']()
                    ret[pkgname] = {'uninstall status': 'success, reboot required'}
                    changed.append(pkgname)
                elif result['retcode'] == 1641:
                    ret[pkgname] = {'uninstall status': 'success, reboot initiated'}
                    changed.append(pkgname)
                else:
                    log.error('Failed to remove %s; retcode: %s; uninstaller output: %s', pkgname, result['retcode'], result['stdout'])
                    ret[pkgname] = {'uninstall status': 'failed'}
    new = list_pkgs(saltenv=saltenv, refresh=False)
    __salt__['pkg_resource.stringify'](old)
    difference = salt.utils.data.compare_dicts(old, new)
    found_chgs = all((name in difference for name in changed))
    end_t = time.time() + 3
    while not found_chgs and time.time() < end_t:
        time.sleep(0.5)
        new = list_pkgs(saltenv=saltenv, refresh=False)
        difference = salt.utils.data.compare_dicts(old, new)
        found_chgs = all((name in difference for name in changed))
    if not found_chgs:
        log.warning('Expected changes for package removal may not have occurred')
    ret.update(difference)
    return ret

def purge(name=None, pkgs=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Package purges are not supported on Windows, this function is identical to\n    ``remove()``.\n\n    .. note::\n        At some point in the future, ``pkg.purge`` may direct the installer to\n        remove all configs and settings for software packages that support that\n        option.\n\n    .. versionadded:: 0.16.0\n\n    Args:\n\n        name (str): The name of the package to be deleted.\n\n        version (str):\n            The version of the package to be deleted. If this option is\n            used in combination with the ``pkgs`` option below, then this\n            version will be applied to all targeted packages.\n\n        pkgs (list):\n            A list of packages to delete. Must be passed as a python\n            list. The ``name`` parameter will be ignored if this option is\n            passed.\n\n    Kwargs:\n        saltenv (str): Salt environment. Default ``base``\n        refresh (bool): Refresh package metadata. Default ``False``\n\n    Returns:\n        dict: A dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    return remove(name=name, pkgs=pkgs, **kwargs)

def get_repo_data(saltenv='base'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the existing package metadata db. Will create it, if it does not\n    exist, however will not refresh it.\n\n    Args:\n        saltenv (str): Salt environment. Default ``base``\n\n    Returns:\n        dict: A dict containing contents of metadata db.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.get_repo_data\n    "
    repo_details = _get_repo_details(saltenv)
    if repo_details.winrepo_age == -1:
        log.debug('No winrepo.p cache file. Refresh pkg db now.')
        refresh_db(saltenv=saltenv)
    if 'winrepo.data' in __context__:
        log.trace('get_repo_data returning results from __context__')
        return __context__['winrepo.data']
    else:
        log.trace('get_repo_data called reading from disk')
    try:
        with salt.utils.files.fopen(repo_details.winrepo_file, 'rb') as repofile:
            try:
                repodata = salt.utils.data.decode(salt.payload.loads(repofile.read()) or {})
                __context__['winrepo.data'] = repodata
                return repodata
            except Exception as exc:
                log.exception(exc)
                return {}
    except OSError as exc:
        log.exception('Not able to read repo file: %s', exc)
        return {}

def _get_name_map(saltenv='base'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a reverse map of full pkg names to the names recognized by winrepo.\n    '
    u_name_map = {}
    name_map = get_repo_data(saltenv).get('name_map', {})
    return name_map

def get_package_info(name, saltenv='base'):
    if False:
        print('Hello World!')
    '\n    Return package info. Returns empty map if package not available.\n    '
    return _get_package_info(name=name, saltenv=saltenv)

def _get_package_info(name, saltenv='base'):
    if False:
        print('Hello World!')
    '\n    Return package info. Returns empty map if package not available\n    TODO: Add option for version\n    '
    return get_repo_data(saltenv).get('repo', {}).get(name, {})

def _reverse_cmp_pkg_versions(pkg1, pkg2):
    if False:
        return 10
    '\n    Compare software package versions\n    '
    return 1 if LooseVersion(pkg1) > LooseVersion(pkg2) else -1

def _get_latest_pkg_version(pkginfo):
    if False:
        print('Hello World!')
    "\n    Returns the latest version of the package.\n    Will return 'latest' or version number string, and\n    'Not Found' if 'Not Found' is the only entry.\n    "
    if len(pkginfo) == 1:
        return next(iter(pkginfo.keys()))
    try:
        return sorted(pkginfo, key=cmp_to_key(_reverse_cmp_pkg_versions)).pop()
    except IndexError:
        return ''

def compare_versions(ver1='', oper='==', ver2=''):
    if False:
        i = 10
        return i + 15
    "\n    Compare software package versions. Made public for use with Jinja\n\n    Args:\n        ver1 (str): A software version to compare\n        oper (str): The operand to use to compare\n        ver2 (str): A software version to compare\n\n    Returns:\n        bool: True if the comparison is valid, otherwise False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.compare_versions 1.2 >= 1.3\n    "
    if not ver1:
        raise SaltInvocationError('compare_version, ver1 is blank')
    if not ver2:
        raise SaltInvocationError('compare_version, ver2 is blank')
    if ver1 == 'latest':
        ver1 = str(sys.maxsize)
    if ver2 == 'latest':
        ver2 = str(sys.maxsize)
    if ver1 == 'Not Found':
        ver1 = '0.0.0.0.0'
    if ver2 == 'Not Found':
        ver2 = '0.0.0.0.0'
    return salt.utils.versions.compare(ver1, oper, ver2, ignore_epoch=True)