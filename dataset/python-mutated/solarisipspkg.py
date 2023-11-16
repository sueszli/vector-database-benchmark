"""
IPS pkg support for Solaris

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.

This module provides support for Solaris 11 new package management - IPS (Image Packaging System).
This is the default pkg module for Solaris 11 (and later).

If you want to use also other packaging module (e.g. pkgutil) together with IPS, you need to override the ``pkg`` provider
in sls for each package:

.. code-block:: yaml

    mypackage:
      pkg.installed:
        - provider: pkgutil

Or you can override it globally by setting the :conf_minion:`providers` parameter in your Minion config file like this:

.. code-block:: yaml

    providers:
      pkg: pkgutil

Or you can override it globally by setting the :conf_minion:`providers` parameter in your Minion config file like this:

.. code-block:: yaml

    providers:
      pkg: pkgutil

"""
import copy
import logging
import salt.utils.data
import salt.utils.functools
import salt.utils.path
import salt.utils.pkg
from salt.exceptions import CommandExecutionError
__virtualname__ = 'pkg'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the virtual pkg module if the os is Solaris 11\n    '
    if __grains__['os_family'] == 'Solaris' and float(__grains__['kernelrelease']) > 5.1 and salt.utils.path.which('pkg'):
        return __virtualname__
    return (False, 'The solarisips execution module failed to load: only available on Solaris >= 11.')
ips_pkg_return_values = {0: 'Command succeeded.', 1: 'An error occurred.', 2: 'Invalid command line options were specified.', 3: 'Multiple operations were requested, but only some of them succeeded.', 4: 'No changes were made - nothing to do.', 5: 'The requested operation cannot be performed on a  live image.', 6: 'The requested operation cannot be completed because the licenses for the packages being installed or updated have not been accepted.', 7: 'The image is currently in use by another process and cannot be modified.'}

def _ips_get_pkgname(line):
    if False:
        print('Hello World!')
    '\n    Extracts package name from "pkg list -v" output.\n    Input: one line of the command output\n    Output: pkg name (e.g.: "pkg://solaris/x11/library/toolkit/libxt")\n    Example use:\n    line = "pkg://solaris/x11/library/toolkit/libxt@1.1.3,5.11-0.175.1.0.0.24.1317:20120904T180030Z i--"\n    name = _ips_get_pkgname(line)\n    '
    return line.split()[0].split('@')[0].strip()

def _ips_get_pkgversion(line):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extracts package version from "pkg list -v" output.\n    Input: one line of the command output\n    Output: package version (e.g.: "1.1.3,5.11-0.175.1.0.0.24.1317:20120904T180030Z")\n    Example use:\n    line = "pkg://solaris/x11/library/toolkit/libxt@1.1.3,5.11-0.175.1.0.0.24.1317:20120904T180030Z i--"\n    name = _ips_get_pkgversion(line)\n    '
    return line.split()[0].split('@')[1].strip()

def refresh_db(full=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Updates the remote repos database.\n\n    full : False\n\n        Set to ``True`` to force a refresh of the pkg DB from all publishers,\n        regardless of the last refresh time.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n        salt '*' pkg.refresh_db full=True\n    "
    salt.utils.pkg.clear_rtag(__opts__)
    if full:
        return __salt__['cmd.retcode']('/bin/pkg refresh --full') == 0
    else:
        return __salt__['cmd.retcode']('/bin/pkg refresh') == 0

def upgrade_available(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Check if there is an upgrade available for a certain package\n    Accepts full or partial FMRI. Returns all matches found.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available apache-22\n    "
    version = None
    cmd = ['pkg', 'list', '-Huv', name]
    lines = __salt__['cmd.run_stdout'](cmd).splitlines()
    if not lines:
        return {}
    ret = {}
    for line in lines:
        ret[_ips_get_pkgname(line)] = _ips_get_pkgversion(line)
    return ret

def list_upgrades(refresh=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Lists all packages available for update.\n\n    When run in global zone, it reports only upgradable packages for the global\n    zone.\n\n    When run in non-global zone, it can report more upgradable packages than\n    ``pkg update -vn``, because ``pkg update`` hides packages that require\n    newer version of ``pkg://solaris/entire`` (which means that they can be\n    upgraded only from the global zone). If ``pkg://solaris/entire`` is found\n    in the list of upgrades, then the global zone should be updated to get all\n    possible updates. Use ``refresh=True`` to refresh the package database.\n\n    refresh : True\n        Runs a full package database refresh before listing. Set to ``False`` to\n        disable running the refresh.\n\n        .. versionchanged:: 2017.7.0\n\n        In previous versions of Salt, ``refresh`` defaulted to ``False``. This was\n        changed to default to ``True`` in the 2017.7.0 release to make the behavior\n        more consistent with the other package modules, which all default to ``True``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n        salt '*' pkg.list_upgrades refresh=False\n    "
    if salt.utils.data.is_true(refresh):
        refresh_db(full=True)
    upgrades = {}
    lines = __salt__['cmd.run_stdout']('/bin/pkg list -Huv').splitlines()
    for line in lines:
        upgrades[_ips_get_pkgname(line)] = _ips_get_pkgversion(line)
    return upgrades

def upgrade(refresh=False, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Upgrade all packages to the latest possible version.\n    When run in global zone, it updates also all non-global zones.\n    In non-global zones upgrade is limited by dependency constrains linked to\n    the version of pkg://solaris/entire.\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {'<package>':  {'old': '<old-version>',\n                        'new': '<new-version>'}}\n\n    When there is a failure, an explanation is also included in the error\n    message, based on the return code of the ``pkg update`` command.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n    "
    if salt.utils.data.is_true(refresh):
        refresh_db()
    old = list_pkgs()
    cmd = ['pkg', 'update', '-v', '--accept']
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'retcode': ips_pkg_return_values[result['retcode']], 'result': result})
    return ret

def _list_pkgs_from_context(versions_as_list):
    if False:
        for i in range(10):
            print('nop')
    '\n    Use pkg list from __context__\n    '
    if versions_as_list:
        return __context__['pkg.list_pkgs']
    else:
        ret = copy.deepcopy(__context__['pkg.list_pkgs'])
        __salt__['pkg_resource.stringify'](ret)
        return ret

def list_pkgs(versions_as_list=False, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List the currently installed packages as a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    ret = {}
    cmd = '/bin/pkg list -Hv'
    lines = __salt__['cmd.run_stdout'](cmd).splitlines()
    for line in lines:
        name = _ips_get_pkgname(line)
        version = _ips_get_pkgversion(line)
        __salt__['pkg_resource.add_pkg'](ret, name, version)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def version(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    Common interface for obtaining the version of installed packages.\n    Accepts full or partial FMRI. If called using pkg_resource, full FMRI is required.\n    Partial FMRI is returned if the package is not installed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version vim\n        salt '*' pkg.version foo bar baz\n        salt '*' pkg_resource.version pkg://solaris/entire\n\n    "
    if not names:
        return ''
    cmd = ['/bin/pkg', 'list', '-Hv']
    cmd.extend(names)
    lines = __salt__['cmd.run_stdout'](cmd, ignore_retcode=True).splitlines()
    ret = {}
    for line in lines:
        ret[_ips_get_pkgname(line)] = _ips_get_pkgversion(line)
    for name in names:
        if name not in ret:
            ret[name] = ''
    if len(names) == 1:
        try:
            return next(iter(ret.values()))
        except StopIteration:
            return ''
    return ret

def latest_version(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    The available version of packages in the repository.\n    Accepts full or partial FMRI. Partial FMRI is returned if the full FMRI\n    could not be resolved.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    Please use pkg.latest_version as pkg.available_version is being deprecated.\n\n    .. versionchanged:: 2019.2.0\n        Support for multiple package names added.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version bash\n        salt '*' pkg.latest_version pkg://solaris/entire\n        salt '*' pkg.latest_version postfix sendmail\n    "
    if not names:
        return ''
    cmd = ['/bin/pkg', 'list', '-Hnv']
    cmd.extend(names)
    lines = __salt__['cmd.run_stdout'](cmd, ignore_retcode=True).splitlines()
    ret = {}
    for line in lines:
        ret[_ips_get_pkgname(line)] = _ips_get_pkgversion(line)
    installed = version(*names)
    if len(names) == 1:
        installed = {list(ret)[0] if ret else names[0]: installed}
    for name in ret:
        if name not in installed:
            continue
        if ret[name] == installed[name]:
            ret[name] = ''
    for name in names:
        if name not in ret:
            ret[name] = ''
    if len(names) == 1:
        try:
            return next(iter(ret.values()))
        except StopIteration:
            return ''
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def get_fmri(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns FMRI from partial name. Returns empty string ('') if not found.\n    In case of multiple match, the function returns list of all matched packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.get_fmri bash\n    "
    if name.startswith('pkg://'):
        return name
    cmd = ['/bin/pkg', 'list', '-aHv', name]
    lines = __salt__['cmd.run_stdout'](cmd).splitlines()
    if not lines:
        return ''
    ret = []
    for line in lines:
        ret.append(_ips_get_pkgname(line))
    return ret

def normalize_name(name, **kwargs):
    if False:
        return 10
    "\n    Internal function. Normalizes pkg name to full FMRI before running\n    pkg.install. In case of multiple matches or no match, it returns the name\n    without modifications.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.normalize_name vim\n    "
    if name.startswith('pkg://'):
        return name
    cmd = ['/bin/pkg', 'list', '-aHv', name]
    lines = __salt__['cmd.run_stdout'](cmd).splitlines()
    if len(lines) != 1:
        return name
    return _ips_get_pkgname(lines[0])

def is_installed(name, **kwargs):
    if False:
        return 10
    "\n    Returns True if the package is installed. Otherwise returns False.\n    Name can be full or partial FMRI.\n    In case of multiple match from partial FMRI name, it returns True.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.is_installed bash\n    "
    cmd = ['/bin/pkg', 'list', '-Hv', name]
    return __salt__['cmd.retcode'](cmd) == 0

def search(name, versions_as_list=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Searches the repository for given pkg name.\n    The name can be full or partial FMRI. All matches are printed. Globs are\n    also supported.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.search bash\n    "
    ret = {}
    cmd = ['/bin/pkg', 'list', '-aHv', name]
    out = __salt__['cmd.run_all'](cmd, ignore_retcode=True)
    if out['retcode'] != 0:
        return {}
    for line in out['stdout'].splitlines():
        name = _ips_get_pkgname(line)
        version = _ips_get_pkgversion(line)
        __salt__['pkg_resource.add_pkg'](ret, name, version)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def install(name=None, refresh=False, pkgs=None, version=None, test=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Install the named package using the IPS pkg command.\n    Accepts full or partial FMRI.\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install. Must be passed as a python list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install vim\n        salt \'*\' pkg.install pkg://solaris/editor/vim\n        salt \'*\' pkg.install pkg://solaris/editor/vim refresh=True\n        salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n    '
    if not pkgs:
        if is_installed(name):
            return {}
    if refresh:
        refresh_db(full=True)
    pkg2inst = ''
    if pkgs:
        pkg2inst = []
        for pkg in pkgs:
            if getattr(pkg, 'items', False):
                if list(pkg.items())[0][1]:
                    pkg2inst.append('{}@{}'.format(list(pkg.items())[0][0], list(pkg.items())[0][1]))
                else:
                    pkg2inst.append(list(pkg.items())[0][0])
            else:
                pkg2inst.append('{}'.format(pkg))
        log.debug('Installing these packages instead of %s: %s', name, pkg2inst)
    elif version:
        pkg2inst = '{}@{}'.format(name, version)
    else:
        pkg2inst = '{}'.format(name)
    cmd = ['pkg', 'install', '-v', '--accept']
    if test:
        cmd.append('-n')
    old = list_pkgs()
    if isinstance(pkg2inst, str):
        cmd.append(pkg2inst)
    elif isinstance(pkg2inst, list):
        cmd = cmd + pkg2inst
    out = __salt__['cmd.run_all'](cmd, output_loglevel='trace')
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if out['retcode'] != 0:
        raise CommandExecutionError('Error occurred installing package(s)', info={'changes': ret, 'retcode': ips_pkg_return_values[out['retcode']], 'errors': [out['stderr']]})
    if test:
        return 'Test succeeded.'
    return ret

def remove(name=None, pkgs=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Remove specified package. Accepts full or partial FMRI.\n    In case of multiple match, the command fails and won\'t modify the OS.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n\n    Returns a list containing the removed packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove tcsh\n        salt \'*\' pkg.remove pkg://solaris/shell/tcsh\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    targets = salt.utils.args.split_input(pkgs) if pkgs else [name]
    if not targets:
        return {}
    if pkgs:
        log.debug('Removing these packages instead of %s: %s', name, targets)
    old = list_pkgs()
    cmd = ['/bin/pkg', 'uninstall', '-v'] + targets
    out = __salt__['cmd.run_all'](cmd, output_loglevel='trace')
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if out['retcode'] != 0:
        raise CommandExecutionError('Error occurred removing package(s)', info={'changes': ret, 'retcode': ips_pkg_return_values[out['retcode']], 'errors': [out['stderr']]})
    return ret

def purge(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove specified package. Accepts full or partial FMRI.\n\n    Returns a list containing the removed packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.purge <package name>\n    "
    return remove(name, **kwargs)