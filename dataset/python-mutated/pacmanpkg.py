"""
A module to wrap pacman calls, since Arch is the best
(https://wiki.archlinux.org/index.php/Arch_is_the_best)

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import copy
import fnmatch
import logging
import os.path
import salt.utils.args
import salt.utils.data
import salt.utils.functools
import salt.utils.itertools
import salt.utils.pkg
import salt.utils.systemd
from salt.exceptions import CommandExecutionError, MinionError
from salt.utils.versions import LooseVersion
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set the virtual pkg module if the os is Arch\n    '
    if __grains__['os_family'] == 'Arch':
        return __virtualname__
    return (False, 'The pacman module could not be loaded: unsupported OS family.')

def _list_removed(old, new):
    if False:
        return 10
    '\n    List the packages which have been removed between the two package objects\n    '
    return [x for x in old if x not in new]

def latest_version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n    "
    refresh = salt.utils.data.is_true(kwargs.pop('refresh', False))
    if not names:
        return ''
    if refresh:
        refresh_db()
    ret = {}
    for name in names:
        ret[name] = ''
    cmd = ['pacman', '-Sp', '--needed', '--print-format', '%n %v']
    cmd.extend(names)
    if 'root' in kwargs:
        cmd.extend(('-r', kwargs['root']))
    out = __salt__['cmd.run_stdout'](cmd, output_loglevel='trace', python_shell=False)
    for line in salt.utils.itertools.split(out, '\n'):
        try:
            (name, version_num) = line.split()
            if name in names:
                ret[name] = version_num
        except (ValueError, IndexError):
            pass
    if len(names) == 1:
        return ret[names[0]]
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def upgrade_available(name, **kwargs):
    if False:
        return 10
    "\n    Check whether or not an upgrade is available for a given package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return latest_version(name) != ''

def list_upgrades(refresh=False, root=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List all available package upgrades on this system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    upgrades = {}
    cmd = ['pacman', '-S', '-p', '-u', '--print-format', '%n %v']
    if root is not None:
        cmd.extend(('-r', root))
    if refresh:
        cmd.append('-y')
    call = __salt__['cmd.run_all'](cmd, python_shell=False, output_loglevel='trace')
    if call['retcode'] != 0:
        comment = ''
        if 'stderr' in call:
            comment += call['stderr']
        if 'stdout' in call:
            comment += call['stdout']
        if comment:
            comment = ': ' + comment
        raise CommandExecutionError('Error listing upgrades' + comment)
    else:
        out = call['stdout']
    for line in salt.utils.itertools.split(out, '\n'):
        try:
            (pkgname, pkgver) = line.split()
        except ValueError:
            continue
        if pkgname.lower() == 'downloading' and '.db' in pkgver.lower():
            continue
        upgrades[pkgname] = pkgver
    return upgrades

def version(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

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
        return 10
    "\n    List the packages currently installed as a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    cmd = ['pacman', '-Q']
    if 'root' in kwargs:
        cmd.extend(('-r', kwargs['root']))
    ret = {}
    out = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    for line in salt.utils.itertools.split(out, '\n'):
        if not line:
            continue
        try:
            (name, version_num) = line.split()[0:2]
        except ValueError:
            log.error("Problem parsing pacman -Q: Unexpected formatting in line: '%s'", line)
        else:
            __salt__['pkg_resource.add_pkg'](ret, name, version_num)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def group_list():
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2016.11.0\n\n    Lists all groups known by pacman on this system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.group_list\n    "
    ret = {'installed': [], 'partially_installed': [], 'available': []}
    cmd = ['pacman', '-Sgg']
    out = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    available = {}
    for line in salt.utils.itertools.split(out, '\n'):
        if not line:
            continue
        try:
            (group, pkg) = line.split()[0:2]
        except ValueError:
            log.error("Problem parsing pacman -Sgg: Unexpected formatting in line: '%s'", line)
        else:
            available.setdefault(group, []).append(pkg)
    cmd = ['pacman', '-Qg']
    out = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    installed = {}
    for line in salt.utils.itertools.split(out, '\n'):
        if not line:
            continue
        try:
            (group, pkg) = line.split()[0:2]
        except ValueError:
            log.error("Problem parsing pacman -Qg: Unexpected formatting in line: '%s'", line)
        else:
            installed.setdefault(group, []).append(pkg)
    for group in installed:
        if group not in available:
            log.error('Pacman reports group %s installed, but it is not in the available list (%s)!', group, available)
            continue
        if len(installed[group]) == len(available[group]):
            ret['installed'].append(group)
        else:
            ret['partially_installed'].append(group)
        available.pop(group)
    ret['installed'].sort()
    ret['partially_installed'].sort()
    ret['available'] = sorted(available.keys())
    return ret

def group_info(name):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.11.0\n\n    Lists all packages in the specified group\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.group_info 'xorg'\n    "
    pkgtypes = ('mandatory', 'optional', 'default', 'conditional')
    ret = {}
    for pkgtype in pkgtypes:
        ret[pkgtype] = set()
    cmd = ['pacman', '-Sgg', name]
    out = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    for line in salt.utils.itertools.split(out, '\n'):
        if not line:
            continue
        try:
            pkg = line.split()[1]
        except ValueError:
            log.error("Problem parsing pacman -Sgg: Unexpected formatting in line: '%s'", line)
        else:
            ret['default'].add(pkg)
    for pkgtype in pkgtypes:
        ret[pkgtype] = sorted(ret[pkgtype])
    return ret

def group_diff(name):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.11.0\n\n    Lists which of a group's packages are installed and which are not\n    installed\n\n    Compatible with yumpkg.group_diff for easy support of state.pkg.group_installed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.group_diff 'xorg'\n    "
    pkgtypes = ('mandatory', 'optional', 'default', 'conditional')
    ret = {}
    for pkgtype in pkgtypes:
        ret[pkgtype] = {'installed': [], 'not installed': []}
    pkgs = __salt__['pkg.list_pkgs']()
    group_pkgs = __salt__['pkg.group_info'](name)
    for pkgtype in pkgtypes:
        for member in group_pkgs.get(pkgtype, []):
            if member in pkgs:
                ret[pkgtype]['installed'].append(member)
            else:
                ret[pkgtype]['not installed'].append(member)
    return ret

def refresh_db(root=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Just run a ``pacman -Sy``, return a dict::\n\n        {'<database name>': Bool}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n    "
    salt.utils.pkg.clear_rtag(__opts__)
    cmd = ['pacman', '-Sy']
    if root is not None:
        cmd.extend(('-r', root))
    ret = {}
    call = __salt__['cmd.run_all'](cmd, output_loglevel='trace', env={'LANG': 'C'}, python_shell=False)
    if call['retcode'] != 0:
        comment = ''
        if 'stderr' in call:
            comment += ': ' + call['stderr']
        raise CommandExecutionError('Error refreshing package database' + comment)
    else:
        out = call['stdout']
    for line in salt.utils.itertools.split(out, '\n'):
        if line.strip().startswith('::'):
            continue
        if not line:
            continue
        key = line.strip().split()[0]
        if 'is up to date' in line:
            ret[key] = False
        elif 'downloading' in line:
            key = line.strip().split()[1].split('.')[0]
            ret[key] = True
    return ret

def install(name=None, refresh=False, sysupgrade=None, pkgs=None, sources=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any pacman commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Install (``pacman -S``) the specified packag(s). Add ``refresh=True`` to\n    install with ``-y``, add ``sysupgrade=True`` to install with ``-u``.\n\n    name\n        The name of the package to be installed. Note that this parameter is\n        ignored if either ``pkgs`` or ``sources`` is passed. Additionally,\n        please note that this option can only be used to install packages from\n        a software repository. To install a package file manually, use the\n        ``sources`` option.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n\n    refresh\n        Whether or not to refresh the package database before installing.\n\n    sysupgrade\n        Whether or not to upgrade the system packages before installing.\n        If refresh is set to ``True`` but sysupgrade is not specified, ``-u`` will be\n        applied\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install from a software repository. Must be\n        passed as a python list. A specific version number can be specified\n        by using a single-element dict representing the package and its\n        version. As with the ``version`` parameter above, comparison operators\n        can be used to target a specific version of a package.\n\n        CLI Examples:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n            salt \'*\' pkg.install pkgs=\'["foo", {"bar": "1.2.3-4"}]\'\n            salt \'*\' pkg.install pkgs=\'["foo", {"bar": "<1.2.3-4"}]\'\n\n    sources\n        A list of packages to install. Must be passed as a list of dicts,\n        with the keys being package names, and the values being the source URI\n        or local path to the package.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install                 sources=\'[{"foo": "salt://foo.pkg.tar.xz"},                 {"bar": "salt://bar.pkg.tar.xz"}]\'\n\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n    '
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, sources, **kwargs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    if 'root' in kwargs:
        pkg_params['-r'] = kwargs['root']
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.append('pacman')
    errors = []
    targets = []
    if pkg_type == 'file':
        cmd.extend(['-U', '--noprogressbar', '--noconfirm'])
        cmd.extend(pkg_params)
    elif pkg_type == 'repository':
        cmd.append('-S')
        if refresh is True:
            cmd.append('-y')
        if sysupgrade is True or (sysupgrade is None and refresh is True):
            cmd.append('-u')
        cmd.extend(['--noprogressbar', '--noconfirm', '--needed'])
        wildcards = []
        for (param, version_num) in pkg_params.items():
            if version_num is None:
                targets.append(param)
            else:
                (prefix, verstr) = salt.utils.pkg.split_comparison(version_num)
                if not prefix:
                    prefix = '='
                if '*' in verstr:
                    if prefix == '=':
                        wildcards.append((param, verstr))
                    else:
                        errors.append(f'Invalid wildcard for {param}{prefix}{verstr}')
                    continue
                targets.append(f'{param}{prefix}{verstr}')
        if wildcards:
            _available = list_repo_pkgs(*[x[0] for x in wildcards], refresh=refresh)
            for (pkgname, verstr) in wildcards:
                candidates = _available.get(pkgname, [])
                match = salt.utils.itertools.fnmatch_multiple(candidates, verstr)
                if match is not None:
                    targets.append('='.join((pkgname, match)))
                else:
                    errors.append("No version matching '{}' found for package '{}' (available: {})".format(verstr, pkgname, ', '.join(candidates) if candidates else 'none'))
            if refresh:
                try:
                    cmd.remove('-y')
                except ValueError:
                    pass
    if not errors:
        cmd.extend(targets)
        old = list_pkgs()
        out = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
        if out['retcode'] != 0 and out['stderr']:
            errors = [out['stderr']]
        else:
            errors = []
        __context__.pop('pkg.list_pkgs', None)
        new = list_pkgs()
        ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        try:
            changes = ret
        except UnboundLocalError:
            changes = {}
        raise CommandExecutionError('Problem encountered installing package(s)', info={'errors': errors, 'changes': changes})
    return ret

def upgrade(refresh=False, root=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon's control group. This is done to keep systemd\n        from killing any pacman commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Run a full system upgrade, a pacman -Syu\n\n    refresh\n        Whether or not to refresh the package database before installing.\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {'<package>':  {'old': '<old-version>',\n                        'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n    "
    ret = {'changes': {}, 'result': True, 'comment': ''}
    old = list_pkgs()
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.extend(['pacman', '-Su', '--noprogressbar', '--noconfirm'])
    if salt.utils.data.is_true(refresh):
        cmd.append('-y')
    if root is not None:
        cmd.extend(('-r', root))
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    return ret

def _uninstall(action='remove', name=None, pkgs=None, **kwargs):
    if False:
        return 10
    '\n    remove and purge do identical things but with different pacman commands,\n    this function performs the common logic.\n    '
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    old = list_pkgs()
    targets = [x for x in pkg_params if x in old]
    if not targets:
        return {}
    remove_arg = '-Rsc' if action == 'purge' else '-R'
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.extend(['pacman', remove_arg, '--noprogressbar', '--noconfirm'])
    cmd.extend(targets)
    if 'root' in kwargs:
        cmd.extend(('-r', kwargs['root']))
    out = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
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

def remove(name=None, pkgs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any pacman commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Remove packages with ``pacman -R``.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    return _uninstall(action='remove', name=name, pkgs=pkgs)

def purge(name=None, pkgs=None, **kwargs):
    if False:
        return 10
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any pacman commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Recursively remove a package and all dependencies which were installed\n    with it, this will call a ``pacman -Rsc``\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    return _uninstall(action='purge', name=name, pkgs=pkgs)

def file_list(*packages, **kwargs):
    if False:
        print('Hello World!')
    "\n    List the files that belong to a package. Not specifying any packages will\n    return a list of _every_ file on the system's package database (not\n    generally recommended).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list httpd\n        salt '*' pkg.file_list httpd postfix\n        salt '*' pkg.file_list\n    "
    errors = []
    ret = []
    cmd = ['pacman', '-Ql']
    if packages and os.path.exists(packages[0]):
        packages = list(packages)
        cmd.extend(('-r', packages.pop(0)))
    cmd.extend(packages)
    out = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    for line in salt.utils.itertools.split(out, '\n'):
        if line.startswith('error'):
            errors.append(line)
        else:
            comps = line.split()
            ret.append(' '.join(comps[1:]))
    return {'errors': errors, 'files': ret}

def file_dict(*packages, **kwargs):
    if False:
        return 10
    "\n    List the files that belong to a package, grouped by package. Not\n    specifying any packages will return a list of _every_ file on the system's\n    package database (not generally recommended).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list httpd\n        salt '*' pkg.file_list httpd postfix\n        salt '*' pkg.file_list\n    "
    errors = []
    ret = {}
    cmd = ['pacman', '-Ql']
    if packages and os.path.exists(packages[0]):
        packages = list(packages)
        cmd.extend(('-r', packages.pop(0)))
    cmd.extend(packages)
    out = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    for line in salt.utils.itertools.split(out, '\n'):
        if line.startswith('error'):
            errors.append(line)
        else:
            comps = line.split()
            if not comps[0] in ret:
                ret[comps[0]] = []
            ret[comps[0]].append(' '.join(comps[1:]))
    return {'errors': errors, 'packages': ret}

def owner(*paths, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2014.7.0\n\n    Return the name of the package that owns the file. Multiple file paths can\n    be passed. Like :mod:`pkg.version <salt.modules.yumpkg.version>`, if a\n    single path is passed, a string will be returned, and if multiple paths are\n    passed, a dictionary of file/package name pairs will be returned.\n\n    If the file is not owned by a package, or is not present on the minion,\n    then an empty string will be returned for that path.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.owner /usr/bin/apachectl\n        salt '*' pkg.owner /usr/bin/apachectl /usr/bin/zsh\n    "
    if not paths:
        return ''
    ret = {}
    cmd_prefix = ['pacman', '-Qqo']
    for path in paths:
        ret[path] = __salt__['cmd.run_stdout'](cmd_prefix + [path], python_shell=False)
    if len(ret) == 1:
        return next(iter(ret.values()))
    return ret

def list_repo_pkgs(*args, **kwargs):
    if False:
        print('Hello World!')
    "\n    Returns all available packages. Optionally, package names (and name globs)\n    can be passed and the results will be filtered to packages matching those\n    names.\n\n    This function can be helpful in discovering the version or repo to specify\n    in a :mod:`pkg.installed <salt.states.pkg.installed>` state.\n\n    The return data will be a dictionary mapping package names to a list of\n    version numbers, ordered from newest to oldest. If ``byrepo`` is set to\n    ``True``, then the return dictionary will contain repository names at the\n    top level, and each repository will map packages to lists of version\n    numbers. For example:\n\n    .. code-block:: python\n\n        # With byrepo=False (default)\n        {\n            'bash': ['4.4.005-2'],\n            'nginx': ['1.10.2-2']\n        }\n        # With byrepo=True\n        {\n            'core': {\n                'bash': ['4.4.005-2']\n            },\n            'extra': {\n                'nginx': ['1.10.2-2']\n            }\n        }\n\n    fromrepo : None\n        Only include results from the specified repo(s). Multiple repos can be\n        specified, comma-separated.\n\n    byrepo : False\n        When ``True``, the return data for each package will be organized by\n        repository.\n\n    refresh : False\n        When ``True``, the package database will be refreshed (i.e. ``pacman\n        -Sy``) before checking for available versions.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_repo_pkgs\n        salt '*' pkg.list_repo_pkgs foo bar baz\n        salt '*' pkg.list_repo_pkgs 'samba4*' fromrepo=base,updates\n        salt '*' pkg.list_repo_pkgs 'python2-*' byrepo=True\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    fromrepo = kwargs.pop('fromrepo', '') or ''
    byrepo = kwargs.pop('byrepo', False)
    refresh = kwargs.pop('refresh', False)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    if fromrepo:
        try:
            repos = [x.strip() for x in fromrepo.split(',')]
        except AttributeError:
            repos = [x.strip() for x in str(fromrepo).split(',')]
    else:
        repos = []
    if refresh:
        refresh_db()
    out = __salt__['cmd.run_all'](['pacman', '-Sl'], output_loglevel='trace', ignore_retcode=True, python_shell=False)
    ret = {}
    for line in salt.utils.itertools.split(out['stdout'], '\n'):
        try:
            (repo, pkg_name, pkg_ver) = line.strip().split()[:3]
        except ValueError:
            continue
        if repos and repo not in repos:
            continue
        if args:
            for arg in args:
                if fnmatch.fnmatch(pkg_name, arg):
                    skip_pkg = False
                    break
            else:
                continue
        ret.setdefault(repo, {}).setdefault(pkg_name, []).append(pkg_ver)
    if byrepo:
        for reponame in ret:
            for pkgname in ret[reponame]:
                sorted_versions = sorted((LooseVersion(x) for x in ret[reponame][pkgname]), reverse=True)
                ret[reponame][pkgname] = [x.vstring for x in sorted_versions]
        return ret
    else:
        byrepo_ret = {}
        for reponame in ret:
            for pkgname in ret[reponame]:
                byrepo_ret.setdefault(pkgname, []).extend(ret[reponame][pkgname])
        for pkgname in byrepo_ret:
            sorted_versions = sorted((LooseVersion(x) for x in byrepo_ret[pkgname]), reverse=True)
            byrepo_ret[pkgname] = [x.vstring for x in sorted_versions]
        return byrepo_ret