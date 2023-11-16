"""
Support for MacPorts under macOS.

This module has some caveats.

1. Updating the database of available ports is quite resource-intensive.
However, `refresh=True` is the default for all operations that need an
up-to-date copy of available ports.  Consider `refresh=False` when you are
sure no db update is needed.

2. In some cases MacPorts doesn't always realize when another copy of itself
is running and will gleefully tromp all over the available ports database.
This makes MacPorts behave in undefined ways until a fresh complete
copy is retrieved.

Because of 1 and 2 it is possible to get the salt-minion into a state where
`salt mac-machine pkg./something/` won't want to return.  Use

`salt-run jobs.active`

on the master to check for potentially long-running calls to `port`.

Finally, ports database updates are always handled with `port selfupdate`
as opposed to `port sync`.  This makes sense in the MacPorts user community
but may confuse experienced Linux admins as Linux package managers
don't upgrade the packaging software when doing a package database update.
In other words `salt mac-machine pkg.refresh_db` is more like
`apt-get update; apt-get upgrade dpkg apt-get` than simply `apt-get update`.

"""
import copy
import logging
import re
import salt.utils.data
import salt.utils.functools
import salt.utils.mac_utils
import salt.utils.path
import salt.utils.pkg
import salt.utils.platform
import salt.utils.versions
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
LIST_ACTIVE_ONLY = True
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Confine this module to Mac OS with MacPorts.\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'mac_ports only available on MacOS')
    if not salt.utils.path.which('port'):
        return (False, 'mac_ports requires the "port" binary')
    return __virtualname__

def _list(query=''):
    if False:
        while True:
            i = 10
    cmd = 'port list {}'.format(query)
    out = salt.utils.mac_utils.execute_return_result(cmd)
    ret = {}
    for line in out.splitlines():
        try:
            (name, version_num, category) = re.split('\\s+', line.lstrip())[0:3]
            version_num = version_num[1:]
        except ValueError:
            continue
        ret[name] = version_num
    return ret

def _list_pkgs_from_context(versions_as_list):
    if False:
        i = 10
        return i + 15
    '\n    Use pkg list from __context__\n    '
    if versions_as_list:
        return __context__['pkg.list_pkgs']
    else:
        ret = copy.deepcopy(__context__['pkg.list_pkgs'])
        __salt__['pkg_resource.stringify'](ret)
        return ret

def list_pkgs(versions_as_list=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List the packages currently installed in a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    ret = {}
    cmd = ['port', 'installed']
    out = salt.utils.mac_utils.execute_return_result(cmd)
    for line in out.splitlines():
        try:
            (name, version_num, active) = re.split('\\s+', line.lstrip())[0:3]
            version_num = version_num[1:]
        except ValueError:
            continue
        if not LIST_ACTIVE_ONLY or active == '(active)':
            __salt__['pkg_resource.add_pkg'](ret, name, version_num)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def version(*names, **kwargs):
    if False:
        return 10
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3>\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def latest_version(*names, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the latest version of the named package available for upgrade or\n    installation\n\n    Options:\n\n    refresh\n        Update ports with ``port selfupdate``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3>\n    "
    if salt.utils.data.is_true(kwargs.get('refresh', True)):
        refresh_db()
    available = _list(' '.join(names)) or {}
    installed = __salt__['pkg.list_pkgs']() or {}
    ret = {}
    for (key, val) in available.items():
        if key not in installed or salt.utils.versions.compare(ver1=installed[key], oper='<', ver2=val):
            ret[key] = val
        else:
            ret[key] = '{} (installed)'.format(version(key))
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def remove(name=None, pkgs=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Removes packages with ``port uninstall``.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs, **kwargs)[0]
    old = list_pkgs()
    targets = [x for x in pkg_params if x in old]
    if not targets:
        return {}
    cmd = ['port', 'uninstall']
    cmd.extend(targets)
    err_message = ''
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        err_message = exc.strerror
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if err_message:
        raise CommandExecutionError('Problem encountered removing package(s)', info={'errors': err_message, 'changes': ret})
    return ret

def install(name=None, refresh=False, pkgs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Install the passed package(s) with ``port install``\n\n    name\n        The name of the formula to be installed. Note that this parameter is\n        ignored if "pkgs" is passed.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n\n    version\n        Specify a version to pkg to install. Ignored if pkgs is specified.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n            salt \'*\' pkg.install git-core version=\'1.8.5.5\'\n\n    variant\n        Specify a variant to pkg to install. Ignored if pkgs is specified.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n            salt \'*\' pkg.install git-core version=\'1.8.5.5\' variant=\'+credential_osxkeychain+doc+pcre\'\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of formulas to install. Must be passed as a python list.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo","bar"]\'\n            salt \'*\' pkg.install pkgs=\'["foo@1.2","bar"]\'\n            salt \'*\' pkg.install pkgs=\'["foo@1.2+ssl","bar@2.3"]\'\n\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install \'package package package\'\n    '
    (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, {})
    if salt.utils.data.is_true(refresh):
        refresh_db()
    if pkgs is None:
        version_num = kwargs.get('version')
        variant_spec = kwargs.get('variant')
        spec = {}
        if version_num:
            spec['version'] = version_num
        if variant_spec:
            spec['variant'] = variant_spec
        pkg_params = {name: spec}
    if not pkg_params:
        return {}
    formulas_array = []
    for (pname, pparams) in pkg_params.items():
        formulas_array.append(pname)
        if pparams:
            if 'version' in pparams:
                formulas_array.append('@' + pparams['version'])
            if 'variant' in pparams:
                formulas_array.append(pparams['variant'])
    old = list_pkgs()
    cmd = ['port', 'install']
    cmd.extend(formulas_array)
    err_message = ''
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        err_message = exc.strerror
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if err_message:
        raise CommandExecutionError('Problem encountered installing package(s)', info={'errors': err_message, 'changes': ret})
    return ret

def list_upgrades(refresh=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Check whether or not an upgrade is available for all packages\n\n    Options:\n\n    refresh\n        Update ports with ``port selfupdate``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    if refresh:
        refresh_db()
    return _list('outdated')

def upgrade_available(pkg, refresh=True, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Check whether or not an upgrade is available for a given package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return pkg in list_upgrades(refresh=refresh)

def refresh_db(**kwargs):
    if False:
        print('Hello World!')
    '\n    Update ports with ``port selfupdate``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt mac pkg.refresh_db\n    '
    salt.utils.pkg.clear_rtag(__opts__)
    cmd = ['port', 'selfupdate']
    return salt.utils.mac_utils.execute_return_success(cmd)

def upgrade(refresh=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run a full upgrade using MacPorts 'port upgrade outdated'\n\n    Options:\n\n    refresh\n        Update ports with ``port selfupdate``\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {'<package>':  {'old': '<old-version>',\n                        'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n    "
    if refresh:
        refresh_db()
    old = list_pkgs()
    cmd = ['port', 'upgrade', 'outdated']
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    return ret