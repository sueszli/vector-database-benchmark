"""
Package support for pkgin based systems, inspired from freebsdpkg module

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import copy
import logging
import os
import re
import salt.utils.data
import salt.utils.decorators as decorators
import salt.utils.functools
import salt.utils.path
import salt.utils.pkg
from salt.exceptions import CommandExecutionError, MinionError
VERSION_MATCH = re.compile('pkgin(?:[\\s]+)([\\d.]+)(?:[\\s]+)(?:.*)')
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

@decorators.memoize
def _check_pkgin():
    if False:
        print('Hello World!')
    '\n    Looks to see if pkgin is present on the system, return full path\n    '
    ppath = salt.utils.path.which('pkgin')
    if ppath is None:
        try:
            localbase = __salt__['cmd.run']('pkg_info -Q LOCALBASE pkgin', output_loglevel='trace')
            if localbase is not None:
                ppath = f'{localbase}/bin/pkgin'
                if not os.path.exists(ppath):
                    return None
        except CommandExecutionError:
            return None
    return ppath

@decorators.memoize
def _get_version():
    if False:
        while True:
            i = 10
    '\n    Get the pkgin version\n    '
    version_string = __salt__['cmd.run']([_check_pkgin(), '-v'], output_loglevel='trace')
    if version_string is None:
        return False
    version_match = VERSION_MATCH.search(version_string)
    if not version_match:
        return False
    return version_match.group(1).split('.')

@decorators.memoize
def _supports_regex():
    if False:
        return 10
    '\n    Check support of regexp\n    '
    return tuple((int(i) for i in _get_version())) > (0, 5)

@decorators.memoize
def _supports_parsing():
    if False:
        return 10
    '\n    Check support of parsing\n    '
    return tuple((int(i) for i in _get_version())) > (0, 6)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set the virtual pkg module if the os is supported by pkgin\n    '
    supported = ['NetBSD', 'SunOS', 'DragonFly', 'Minix', 'Darwin', 'SmartOS']
    if __grains__['os'] in supported and _check_pkgin():
        return __virtualname__
    return (False, 'The pkgin execution module cannot be loaded: only available on {} systems.'.format(', '.join(supported)))

def _splitpkg(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Split package name from versioned string\n    '
    if name[0].isalnum() and name != 'No':
        return name.split(';', 1)[0].rsplit('-', 1)

def search(pkg_name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Searches for an exact match using pkgin ^package$\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.search 'mysql-server'\n    "
    pkglist = {}
    pkgin = _check_pkgin()
    if not pkgin:
        return pkglist
    if _supports_regex():
        pkg_name = f'^{pkg_name}$'
    out = __salt__['cmd.run']([pkgin, 'se', pkg_name], output_loglevel='trace')
    for line in out.splitlines():
        if line:
            match = _splitpkg(line.split()[0])
            if match:
                pkglist[match[0]] = match[1]
    return pkglist

def latest_version(*names, **kwargs):
    if False:
        return 10
    "\n    .. versionchanged:: 2016.3.0\n\n    Return the latest version of the named package available for upgrade or\n    installation.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> ...\n    "
    refresh = salt.utils.data.is_true(kwargs.pop('refresh', True))
    pkglist = {}
    pkgin = _check_pkgin()
    if not pkgin:
        return pkglist
    if refresh:
        refresh_db()
    cmd_prefix = [pkgin, 'se']
    if _supports_parsing():
        cmd_prefix.insert(1, '-p')
    for name in names:
        cmd = copy.deepcopy(cmd_prefix)
        cmd.append(f'^{name}$' if _supports_regex() else name)
        out = __salt__['cmd.run'](cmd, output_loglevel='trace')
        for line in out.splitlines():
            if line.startswith('No results found for'):
                return pkglist
            p = line.split(';' if _supports_parsing() else None)
            if p and p[0] in ('=:', '<:', '>:', ''):
                continue
            elif p:
                s = _splitpkg(p[0])
                if s:
                    if not s[0] in pkglist:
                        if len(p) > 1 and p[1] in ('<', '', '='):
                            pkglist[s[0]] = s[1]
                        else:
                            pkglist[s[0]] = ''
    if pkglist and len(names) == 1:
        if names[0] in pkglist:
            return pkglist[names[0]]
    else:
        return pkglist
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def version(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def refresh_db(force=False, **kwargs):
    if False:
        return 10
    "\n    Use pkg update to get latest pkg_summary\n\n    force\n        Pass -f so that the cache is always refreshed.\n\n        .. versionadded:: 2018.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n    "
    salt.utils.pkg.clear_rtag(__opts__)
    pkgin = _check_pkgin()
    if pkgin:
        cmd = [pkgin, 'up']
        if force:
            cmd.insert(1, '-f')
        call = __salt__['cmd.run_all'](cmd, output_loglevel='trace')
        if call['retcode'] != 0:
            comment = ''
            if 'stderr' in call:
                comment += call['stderr']
            raise CommandExecutionError(comment)
    return True

def _list_pkgs_from_context(versions_as_list):
    if False:
        while True:
            i = 10
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
    "\n    .. versionchanged:: 2016.3.0\n\n    List the packages currently installed as a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    pkgin = _check_pkgin()
    ret = {}
    out = __salt__['cmd.run']([pkgin, 'ls'] if pkgin else ['pkg_info'], output_loglevel='trace')
    for line in out.splitlines():
        try:
            (pkg, ver) = re.split('[; ]', line, 1)[0].rsplit('-', 1)
        except ValueError:
            continue
        __salt__['pkg_resource.add_pkg'](ret, pkg, ver)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def list_upgrades(refresh=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    List all available package upgrades.\n\n    .. versionadded:: 2018.3.0\n\n    refresh\n        Whether or not to refresh the package database before installing.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    pkgs = {}
    for pkg in sorted(list_pkgs(refresh=refresh).keys()):
        pkg_upgrade = latest_version(pkg, refresh=False)
        if pkg_upgrade:
            pkgs[pkg] = pkg_upgrade
    return pkgs

def install(name=None, refresh=False, fromrepo=None, pkgs=None, sources=None, **kwargs):
    if False:
        return 10
    '\n    Install the passed package\n\n    name\n        The name of the package to be installed.\n\n    refresh\n        Whether or not to refresh the package database before installing.\n\n    fromrepo\n        Specify a package repository to install from.\n\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install from a software repository. Must be\n        passed as a python list.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo","bar"]\'\n\n    sources\n        A list of packages to install. Must be passed as a list of dicts,\n        with the keys being package names, and the values being the source URI\n        or local path to the package.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install sources=\'[{"foo": "salt://foo.deb"},{"bar": "salt://bar.deb"}]\'\n\n    Return a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install <package name>\n    '
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, sources, **kwargs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    repo = kwargs.get('repo', '')
    if not fromrepo and repo:
        fromrepo = repo
    if not pkg_params:
        return {}
    env = []
    args = []
    pkgin = _check_pkgin()
    if pkgin:
        cmd = pkgin
        if fromrepo:
            log.info('Setting PKG_REPOS=%s', fromrepo)
            env.append(('PKG_REPOS', fromrepo))
    else:
        cmd = 'pkg_add'
        if fromrepo:
            log.info('Setting PKG_PATH=%s', fromrepo)
            env.append(('PKG_PATH', fromrepo))
    if pkg_type == 'file':
        cmd = 'pkg_add'
    elif pkg_type == 'repository':
        if pkgin:
            if refresh:
                args.append('-f')
            args.extend(('-y', 'in'))
    args.insert(0, cmd)
    args.extend(pkg_params)
    old = list_pkgs()
    out = __salt__['cmd.run_all'](args, env=env, output_loglevel='trace')
    if out['retcode'] != 0 and out['stderr']:
        errors = [out['stderr']]
    else:
        errors = []
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        raise CommandExecutionError('Problem encountered installing package(s)', info={'errors': errors, 'changes': ret})
    _rehash()
    return ret

def upgrade(refresh=True, pkgs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run pkg upgrade, if pkgin used. Otherwise do nothing\n\n    refresh\n        Whether or not to refresh the package database before installing.\n\n    Multiple Package Upgrade Options:\n\n    pkgs\n        A list of packages to upgrade from a software repository. Must be\n        passed as a python list.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.upgrade pkgs=\'["foo","bar"]\'\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {\'<package>\':  {\'old\': \'<old-version>\',\n                        \'new\': \'<new-version>\'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.upgrade\n    '
    pkgin = _check_pkgin()
    if not pkgin:
        return {}
    if salt.utils.data.is_true(refresh):
        refresh_db()
    old = list_pkgs()
    cmds = []
    if not pkgs:
        cmds.append([pkgin, '-y', 'full-upgrade'])
    elif salt.utils.data.is_list(pkgs):
        for pkg in pkgs:
            cmds.append([pkgin, '-y', 'install', pkg])
    else:
        result = {'retcode': 1, 'reason': 'Ignoring the parameter `pkgs` because it is not a list!'}
        log.error(result['reason'])
    for cmd in cmds:
        result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
        if result['retcode'] != 0:
            break
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    return ret

def remove(name=None, pkgs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a list containing the removed packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    old = list_pkgs()
    args = []
    for param in pkg_params:
        ver = old.get(param, [])
        if not ver:
            continue
        if isinstance(ver, list):
            args.extend([f'{param}-{v}' for v in ver])
        else:
            args.append(f'{param}-{ver}')
    if not args:
        return {}
    pkgin = _check_pkgin()
    cmd = [pkgin, '-y', 'remove'] if pkgin else ['pkg_remove']
    cmd.extend(args)
    out = __salt__['cmd.run_all'](cmd, output_loglevel='trace')
    if out['retcode'] != 0 and out['stderr']:
        errors = [out['stderr']]
    else:
        errors = []
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        raise CommandExecutionError('Problem encountered removing package(s)', info={'errors': errors, 'changes': ret})
    return ret

def purge(name=None, pkgs=None, **kwargs):
    if False:
        return 10
    '\n    Package purges are not supported, this function is identical to\n    ``remove()``.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    return remove(name=name, pkgs=pkgs)

def _rehash():
    if False:
        while True:
            i = 10
    '\n    Recomputes internal hash table for the PATH variable.\n    Use whenever a new command is created during the current\n    session.\n    '
    shell = __salt__['environ.get']('SHELL')
    if shell.split('/')[-1] in ('csh', 'tcsh'):
        __salt__['cmd.run']('rehash', output_loglevel='trace')

def file_list(package, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List the files that belong to a package.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list nginx\n    "
    ret = file_dict(package)
    files = []
    for pkg_files in ret['files'].values():
        files.extend(pkg_files)
    ret['files'] = files
    return ret

def file_dict(*packages, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionchanged:: 2016.3.0\n\n    List the files that belong to a package.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_dict nginx\n        salt '*' pkg.file_dict nginx varnish\n    "
    errors = []
    files = {}
    for package in packages:
        cmd = ['pkg_info', '-qL', package]
        ret = __salt__['cmd.run_all'](cmd, output_loglevel='trace')
        files[package] = []
        for line in ret['stderr'].splitlines():
            errors.append(line)
        for line in ret['stdout'].splitlines():
            if line.startswith('/'):
                files[package].append(line)
            else:
                continue
    ret = {'errors': errors, 'files': files}
    for field in list(ret):
        if not ret[field] or ret[field] == '':
            del ret[field]
    return ret

def normalize_name(pkgs, **kwargs):
    if False:
        print('Hello World!')
    '\n    Normalize package names\n\n    .. note::\n        Nothing special to do to normalize, just return\n        the original. (We do need it to be compatible\n        with the pkg_resource provider.)\n    '
    return pkgs