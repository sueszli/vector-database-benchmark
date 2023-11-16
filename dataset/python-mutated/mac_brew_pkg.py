"""
Homebrew for macOS

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import copy
import logging
import salt.utils.data
import salt.utils.functools
import salt.utils.json
import salt.utils.path
import salt.utils.pkg
import salt.utils.versions
from salt.exceptions import CommandExecutionError, MinionError, SaltInvocationError
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Confine this module to Mac OS with Homebrew.\n    '
    if __grains__['os'] != 'MacOS':
        return (False, 'brew module is macos specific')
    if not _homebrew_os_bin():
        return (False, "The 'brew' binary was not found")
    return __virtualname__

def _list_taps():
    if False:
        i = 10
        return i + 15
    '\n    List currently installed brew taps\n    '
    return _call_brew('tap')['stdout'].splitlines()

def _list_pinned():
    if False:
        while True:
            i = 10
    '\n    List currently pinned formulas\n    '
    return _call_brew('list', '--pinned')['stdout'].splitlines()

def _pin(pkg, runas=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Pin pkg\n    '
    try:
        _call_brew('pin', pkg)
    except CommandExecutionError:
        log.error('Failed to pin "%s"', pkg)
        return False
    return True

def _unpin(pkg, runas=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Pin pkg\n    '
    try:
        _call_brew('unpin', pkg)
    except CommandExecutionError:
        log.error('Failed to unpin "%s"', pkg)
        return False
    return True

def _tap(tap, runas=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add unofficial GitHub repos to the list of formulas that brew tracks,\n    updates, and installs from.\n    '
    if tap in _list_taps():
        return True
    try:
        _call_brew('tap', tap)
    except CommandExecutionError:
        log.error('Failed to tap "%s"', tap)
        return False
    return True

def _homebrew_os_bin():
    if False:
        for i in range(10):
            print('nop')
    '\n    Fetch PATH binary brew full path eg: /usr/local/bin/brew (symbolic link)\n    '
    return salt.utils.path.which('brew')

def _homebrew_bin():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the full path to the homebrew binary in the homebrew installation folder\n    '
    brew = _homebrew_os_bin()
    if brew:
        brew = __salt__['cmd.run'](f'{brew} --prefix', output_loglevel='trace')
        brew += '/bin/brew'
    return brew

def _call_brew(*cmd, failhard=True):
    if False:
        while True:
            i = 10
    '\n    Calls the brew command with the user account of brew\n    '
    user = __salt__['file.get_user'](_homebrew_bin())
    runas = user if user != __opts__['user'] else None
    _cmd = []
    if runas:
        _cmd = [f'sudo -i -n -H -u {runas} -- ']
    _cmd = _cmd + [_homebrew_bin()] + list(cmd)
    _cmd = ' '.join(_cmd)
    runas = None
    result = __salt__['cmd.run_all'](cmd=_cmd, runas=runas, output_loglevel='trace', python_shell=False)
    if failhard and result['retcode'] != 0:
        raise CommandExecutionError('Brew command failed', info={'result': result})
    return result

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
    package_info = salt.utils.json.loads(_call_brew('info', '--json=v2', '--installed')['stdout'])
    for package in package_info['formulae']:
        pkg_versions = [v['version'] for v in package['installed']]
        pkg_names = package['aliases'] + [package['name'], package['full_name']]
        combinations = [(n, v) for n in pkg_names for v in pkg_versions]
        for (pkg_name, pkg_version) in combinations:
            __salt__['pkg_resource.add_pkg'](ret, pkg_name, pkg_version)
    for package in package_info['casks']:
        pkg_version = package['installed']
        pkg_names = {package['full_token'], package['token']}
        pkg_tap = package.get('tap', None)
        if not pkg_tap:
            pkg_tap = 'homebrew/cask'
        pkg_names.add('/'.join([pkg_tap, package['token']]))
        for pkg_name in pkg_names:
            __salt__['pkg_resource.add_pkg'](ret, pkg_name, pkg_version)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def version(*names, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3>\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def latest_version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the latest version of the named package available for upgrade or\n    installation\n\n    Currently chooses stable versions, falling back to devel if that does not\n    exist.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3>\n    "
    refresh = salt.utils.data.is_true(kwargs.pop('refresh', True))
    if refresh:
        refresh_db()

    def get_version(pkg_info):
        if False:
            i = 10
            return i + 15
        version = pkg_info['versions']['stable'] or pkg_info['versions']['devel']
        if pkg_info['versions']['bottle'] and pkg_info['revision'] >= 1:
            version = '{}_{}'.format(version, pkg_info['revision'])
        return version
    versions_dict = {key: get_version(val) for (key, val) in _info(*names).items()}
    if len(names) == 1:
        return next(iter(versions_dict.values()))
    else:
        return versions_dict
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def remove(name=None, pkgs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Removes packages with ``brew uninstall``.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs, **kwargs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    old = list_pkgs()
    targets = [x for x in pkg_params if x in old]
    if not targets:
        return {}
    out = _call_brew('uninstall', *targets)
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

def refresh_db(**kwargs):
    if False:
        return 10
    "\n    Update the homebrew package repository.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n    "
    salt.utils.pkg.clear_rtag(__opts__)
    if _call_brew('update')['retcode']:
        log.error('Failed to update')
        return False
    return True

def _info(*pkgs):
    if False:
        i = 10
        return i + 15
    "\n    Get all info brew can provide about a list of packages.\n\n    Does not do any kind of processing, so the format depends entirely on\n    the output brew gives. This may change if a new version of the format is\n    requested.\n\n    On failure, returns an empty dict and logs failure.\n    On success, returns a dict mapping each item in pkgs to its corresponding\n    object in the output of 'brew info'.\n\n    Caveat: If one of the packages does not exist, no packages will be\n            included in the output.\n    "
    brew_result = _call_brew('info', '--json=v2', *pkgs)
    if brew_result['retcode']:
        log.error('Failed to get info about packages: %s', ' '.join(pkgs))
        return {}
    output = salt.utils.json.loads(brew_result['stdout'])
    meta_info = {'formulae': ['name', 'full_name'], 'casks': ['token', 'full_token']}
    pkgs_info = dict()
    for (tap, keys) in meta_info.items():
        data = output[tap]
        if len(data) == 0:
            continue
        for _pkg in data:
            for key in keys:
                if _pkg[key] in pkgs:
                    pkgs_info[_pkg[key]] = _pkg
    return pkgs_info

def install(name=None, pkgs=None, taps=None, options=None, **kwargs):
    if False:
        return 10
    '\n    Install the passed package(s) with ``brew install``\n\n    name\n        The name of the formula to be installed. Note that this parameter is\n        ignored if "pkgs" is passed.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n\n    taps\n        Unofficial GitHub repos to use when updating and installing formulas.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name> tap=\'<tap>\'\n            salt \'*\' pkg.install zlib taps=\'homebrew/dupes\'\n            salt \'*\' pkg.install php54 taps=\'["josegonzalez/php", "homebrew/dupes"]\'\n\n    options\n        Options to pass to brew. Only applies to initial install. Due to how brew\n        works, modifying chosen options requires a full uninstall followed by a\n        fresh install. Note that if "pkgs" is used, all options will be passed\n        to all packages. Unrecognized options for a package will be silently\n        ignored by brew.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name> tap=\'<tap>\'\n            salt \'*\' pkg.install php54 taps=\'["josegonzalez/php", "homebrew/dupes"]\' options=\'["--with-fpm"]\'\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of formulas to install. Must be passed as a python list.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo","bar"]\'\n\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install \'package package package\'\n    '
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, kwargs.get('sources', {}))
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    cmd = ['install']
    cmd.extend(list(pkg_params))
    old = list_pkgs()
    if taps:
        if not isinstance(taps, list):
            taps = [taps]
        for tap in taps:
            _tap(tap)
    if options:
        cmd.extend(options)
    out = _call_brew(*cmd)
    if out['retcode'] != 0 and out['stderr']:
        errors = [out['stderr']]
    else:
        errors = []
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        raise CommandExecutionError('Problem encountered installing package(s)', info={'errors': errors, 'changes': ret})
    return ret

def list_upgrades(refresh=True, include_casks=False, **kwargs):
    if False:
        print('Hello World!')
    "\n    Check whether or not an upgrade is available for all packages\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    if refresh:
        refresh_db()
    res = _call_brew('outdated', '--json=v2')
    ret = {}
    try:
        data = salt.utils.json.loads(res['stdout'])
    except ValueError as err:
        msg = f'unable to interpret output from "brew outdated": {err}'
        log.error(msg)
        raise CommandExecutionError(msg)
    for pkg in data['formulae']:
        ret[pkg['name']] = pkg['current_version']
    if include_casks:
        for pkg in data['casks']:
            ret[pkg['name']] = pkg['current_version']
    return ret

def upgrade_available(pkg, **kwargs):
    if False:
        print('Hello World!')
    "\n    Check whether or not an upgrade is available for a given package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return pkg in list_upgrades(**kwargs)

def upgrade(refresh=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Upgrade outdated, unpinned brews.\n\n    refresh\n        Fetch the newest version of Homebrew and all formulae from GitHub before installing.\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {'<package>':  {'old': '<old-version>',\n                        'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n    "
    ret = {'changes': {}, 'result': True, 'comment': ''}
    old = list_pkgs()
    if salt.utils.data.is_true(refresh):
        refresh_db()
    result = _call_brew('upgrade', failhard=False)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    return ret

def info_installed(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the information of the named package(s) installed on the system.\n\n    .. versionadded:: 2016.3.1\n\n    names\n        The names of the packages for which to return information.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.info_installed <package1>\n        salt '*' pkg.info_installed <package1> <package2> <package3> ...\n    "
    return _info(*names)

def hold(name=None, pkgs=None, sources=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Set package in \'hold\' state, meaning it will not be upgraded.\n\n    .. versionadded:: 3001\n\n    name\n        The name of the package, e.g., \'tmux\'\n\n    CLI Example:\n\n     .. code-block:: bash\n\n        salt \'*\' pkg.hold <package name>\n\n    pkgs\n        A list of packages to hold. Must be passed as a python list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.hold pkgs=\'["foo", "bar"]\'\n    '
    if not name and (not pkgs) and (not sources):
        raise SaltInvocationError('One of name, pkgs, or sources must be specified.')
    if pkgs and sources:
        raise SaltInvocationError('Only one of pkgs or sources can be specified.')
    targets = []
    if pkgs:
        targets.extend(pkgs)
    elif sources:
        for source in sources:
            targets.append(next(iter(source)))
    else:
        targets.append(name)
    ret = {}
    pinned = _list_pinned()
    installed = list_pkgs()
    for target in targets:
        if isinstance(target, dict):
            target = next(iter(target))
        ret[target] = {'name': target, 'changes': {}, 'result': False, 'comment': ''}
        if target not in installed:
            ret[target]['comment'] = f'Package {target} does not have a state.'
        elif target not in pinned:
            if 'test' in __opts__ and __opts__['test']:
                ret[target].update(result=None)
                ret[target]['comment'] = f'Package {target} is set to be held.'
            else:
                result = _pin(target)
                if result:
                    changes = {'old': 'install', 'new': 'hold'}
                    ret[target].update(changes=changes, result=True)
                    ret[target]['comment'] = 'Package {} is now being held.'.format(target)
                else:
                    ret[target].update(result=False)
                    ret[target]['comment'] = f'Unable to hold package {target}.'
        else:
            ret[target].update(result=True)
            ret[target]['comment'] = 'Package {} is already set to be held.'.format(target)
    return ret
pin = hold

def unhold(name=None, pkgs=None, sources=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Set package current in \'hold\' state to install state,\n    meaning it will be upgraded.\n\n    .. versionadded:: 3001\n\n    name\n        The name of the package, e.g., \'tmux\'\n\n     CLI Example:\n\n     .. code-block:: bash\n\n        salt \'*\' pkg.unhold <package name>\n\n    pkgs\n        A list of packages to unhold. Must be passed as a python list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.unhold pkgs=\'["foo", "bar"]\'\n    '
    if not name and (not pkgs) and (not sources):
        raise SaltInvocationError('One of name, pkgs, or sources must be specified.')
    if pkgs and sources:
        raise SaltInvocationError('Only one of pkgs or sources can be specified.')
    targets = []
    if pkgs:
        targets.extend(pkgs)
    elif sources:
        for source in sources:
            targets.append(next(iter(source)))
    else:
        targets.append(name)
    ret = {}
    pinned = _list_pinned()
    installed = list_pkgs()
    for target in targets:
        if isinstance(target, dict):
            target = next(iter(target))
        ret[target] = {'name': target, 'changes': {}, 'result': False, 'comment': ''}
        if target not in installed:
            ret[target]['comment'] = f'Package {target} does not have a state.'
        elif target in pinned:
            if 'test' in __opts__ and __opts__['test']:
                ret[target].update(result=None)
                ret[target]['comment'] = 'Package {} is set to be unheld.'.format(target)
            else:
                result = _unpin(target)
                if result:
                    changes = {'old': 'hold', 'new': 'install'}
                    ret[target].update(changes=changes, result=True)
                    ret[target]['comment'] = f'Package {target} is no longer being held.'
                else:
                    ret[target].update(result=False)
                    ret[target]['comment'] = 'Unable to unhold package {}.'.format(target)
        else:
            ret[target].update(result=True)
            ret[target]['comment'] = 'Package {} is already set not to be held.'.format(target)
    return ret
unpin = unhold