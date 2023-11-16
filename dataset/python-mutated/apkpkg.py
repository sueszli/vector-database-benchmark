"""
Support for apk

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.

.. versionadded:: 2017.7.0

"""
import copy
import logging
import salt.utils.data
import salt.utils.itertools
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Confirm this module is running on an Alpine Linux distribution\n    '
    if __grains__.get('os_family', False) == 'Alpine':
        return __virtualname__
    return (False, 'Module apk only works on Alpine Linux based systems')

def version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def refresh_db(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Updates the package list\n\n    - ``True``: Database updated successfully\n    - ``False``: Problem updating database\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n    "
    ret = {}
    cmd = ['apk', 'update']
    call = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if call['retcode'] == 0:
        errors = []
        ret = True
    else:
        errors = [call['stdout']]
        ret = False
    if errors:
        raise CommandExecutionError('Problem encountered installing package(s)', info={'errors': errors, 'changes': ret})
    return ret

def _list_pkgs_from_context(versions_as_list):
    if False:
        return 10
    '\n    Use pkg list from __context__\n    '
    if versions_as_list:
        return __context__['pkg.list_pkgs']
    else:
        ret = copy.deepcopy(__context__['pkg.list_pkgs'])
        __salt__['pkg_resource.stringify'](ret)
        return ret

def list_pkgs(versions_as_list=False, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List the packages currently installed in a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n        salt '*' pkg.list_pkgs versions_as_list=True\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    cmd = ['apk', 'info', '-v']
    ret = {}
    out = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    for line in salt.utils.itertools.split(out, '\n'):
        pkg_version = '-'.join(line.split('-')[-2:])
        pkg_name = '-'.join(line.split('-')[:-2])
        __salt__['pkg_resource.add_pkg'](ret, pkg_name, pkg_version)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def latest_version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n    "
    refresh = salt.utils.data.is_true(kwargs.pop('refresh', True))
    if not names:
        return ''
    ret = {}
    for name in names:
        ret[name] = ''
    pkgs = list_pkgs()
    if refresh:
        refresh_db()
    cmd = ['apk', 'upgrade', '-s']
    out = __salt__['cmd.run_stdout'](cmd, output_loglevel='trace', python_shell=False)
    for line in salt.utils.itertools.split(out, '\n'):
        try:
            name = line.split(' ')[2]
            _oldversion = line.split(' ')[3].strip('(')
            newversion = line.split(' ')[5].strip(')')
            if name in names:
                ret[name] = newversion
        except (ValueError, IndexError):
            pass
    for pkg in ret:
        if not ret[pkg]:
            installed = pkgs.get(pkg)
            cmd = ['apk', 'search', pkg]
            out = __salt__['cmd.run_stdout'](cmd, output_loglevel='trace', python_shell=False)
            for line in salt.utils.itertools.split(out, '\n'):
                try:
                    pkg_version = '-'.join(line.split('-')[-2:])
                    pkg_name = '-'.join(line.split('-')[:-2])
                    if pkg == pkg_name:
                        if installed == pkg_version:
                            ret[pkg] = ''
                        else:
                            ret[pkg] = pkg_version
                except ValueError:
                    pass
    if len(names) == 1:
        return ret[names[0]]
    return ret

def install(name=None, refresh=False, pkgs=None, sources=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Install the passed package, add refresh=True to update the apk database.\n\n    name\n        The name of the package to be installed. Note that this parameter is\n        ignored if either "pkgs" or "sources" is passed. Additionally, please\n        note that this option can only be used to install packages from a\n        software repository. To install a package file manually, use the\n        "sources" option.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n\n    refresh\n        Whether or not to refresh the package database before installing.\n\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install from a software repository. Must be\n        passed as a python list.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n\n    sources\n        A list of IPK packages to install. Must be passed as a list of dicts,\n        with the keys being package names, and the values being the source URI\n        or local path to the package.  Dependencies are automatically resolved\n        and marked as auto-installed.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install sources=\'[{"foo": "salt://foo.deb"},{"bar": "salt://bar.deb"}]\'\n\n    install_recommends\n        Whether to install the packages marked as recommended. Default is True.\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n    '
    refreshdb = salt.utils.data.is_true(refresh)
    pkg_to_install = []
    old = list_pkgs()
    if name and (not (pkgs or sources)):
        if ',' in name:
            pkg_to_install = name.split(',')
        else:
            pkg_to_install = [name]
    if pkgs:
        pkgs = [next(iter(p)) for p in pkgs if isinstance(p, dict)]
        pkg_to_install.extend(pkgs)
    if not pkg_to_install:
        return {}
    if refreshdb:
        refresh_db()
    cmd = ['apk', 'add']
    for _pkg in pkg_to_install:
        if old.get(_pkg):
            cmd.append('-u')
            break
    cmd.extend(pkg_to_install)
    out = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
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

def purge(name=None, pkgs=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Alias to remove\n    '
    return remove(name=name, pkgs=pkgs, purge=True)

def remove(name=None, pkgs=None, purge=False, **kwargs):
    if False:
        print('Hello World!')
    '\n    Remove packages using ``apk del``.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    old = list_pkgs()
    pkg_to_remove = []
    if name:
        if ',' in name:
            pkg_to_remove = name.split(',')
        else:
            pkg_to_remove = [name]
    if pkgs:
        pkg_to_remove.extend(pkgs)
    if not pkg_to_remove:
        return {}
    if purge:
        cmd = ['apk', 'del', '--purge']
    else:
        cmd = ['apk', 'del']
    cmd.extend(pkg_to_remove)
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

def upgrade(name=None, pkgs=None, refresh=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Upgrades all packages via ``apk upgrade`` or a specific package if name or\n    pkgs is specified. Name is ignored if pkgs is specified\n\n    Returns a dict containing the changes.\n\n        {'<package>':  {'old': '<old-version>',\n                        'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n    "
    ret = {'changes': {}, 'result': True, 'comment': ''}
    if salt.utils.data.is_true(refresh):
        refresh_db()
    old = list_pkgs()
    pkg_to_upgrade = []
    if name and (not pkgs):
        if ',' in name:
            pkg_to_upgrade = name.split(',')
        else:
            pkg_to_upgrade = [name]
    if pkgs:
        pkg_to_upgrade.extend(pkgs)
    if pkg_to_upgrade:
        cmd = ['apk', 'add', '-u']
        cmd.extend(pkg_to_upgrade)
    else:
        cmd = ['apk', 'upgrade']
    call = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False, redirect_stderr=True)
    if call['retcode'] != 0:
        ret['result'] = False
        if call['stdout']:
            ret['comment'] = call['stdout']
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret['changes'] = salt.utils.data.compare_dicts(old, new)
    return ret

def list_upgrades(refresh=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all available package upgrades.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    ret = {}
    if salt.utils.data.is_true(refresh):
        refresh_db()
    cmd = ['apk', 'upgrade', '-s']
    call = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if call['retcode'] != 0:
        comment = ''
        if 'stderr' in call:
            comment += call['stderr']
        if 'stdout' in call:
            comment += call['stdout']
        raise CommandExecutionError(comment)
    else:
        out = call['stdout']
    for line in out.splitlines():
        if 'Upgrading' in line:
            name = line.split(' ')[2]
            _oldversion = line.split(' ')[3].strip('(')
            newversion = line.split(' ')[5].strip(')')
            ret[name] = newversion
    return ret

def file_list(*packages, **kwargs):
    if False:
        return 10
    "\n    List the files that belong to a package. Not specifying any packages will\n    return a list of _every_ file on the system's package database (not\n    generally recommended).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list httpd\n        salt '*' pkg.file_list httpd postfix\n        salt '*' pkg.file_list\n    "
    return file_dict(*packages)

def file_dict(*packages, **kwargs):
    if False:
        print('Hello World!')
    "\n    List the files that belong to a package, grouped by package. Not\n    specifying any packages will return a list of _every_ file on the system's\n    package database (not generally recommended).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list httpd\n        salt '*' pkg.file_list httpd postfix\n        salt '*' pkg.file_list\n    "
    errors = []
    ret = {}
    cmd_files = ['apk', 'info', '-L']
    if not packages:
        return 'Package name should be provided'
    for package in packages:
        files = []
        cmd = cmd_files[:]
        cmd.append(package)
        out = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
        for line in out['stdout'].splitlines():
            if line.endswith('contains:'):
                continue
            else:
                files.append(line)
        if files:
            ret[package] = files
    return {'errors': errors, 'packages': ret}

def owner(*paths, **kwargs):
    if False:
        return 10
    "\n    Return the name of the package that owns the file. Multiple file paths can\n    be passed. Like :mod:`pkg.version <salt.modules.apk.version`, if a single\n    path is passed, a string will be returned, and if multiple paths are passed,\n    a dictionary of file/package name pairs will be returned.\n\n    If the file is not owned by a package, or is not present on the minion,\n    then an empty string will be returned for that path.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.owns /usr/bin/apachectl\n        salt '*' pkg.owns /usr/bin/apachectl /usr/bin/basename\n    "
    if not paths:
        return 'You must provide a path'
    ret = {}
    cmd_search = ['apk', 'info', '-W']
    for path in paths:
        cmd = cmd_search[:]
        cmd.append(path)
        output = __salt__['cmd.run_stdout'](cmd, output_loglevel='trace', python_shell=False)
        if output:
            if 'ERROR:' in output:
                ret[path] = 'Could not find owner package'
            else:
                ret[path] = output.split('by ')[1].strip()
        else:
            ret[path] = 'Error running {}'.format(cmd)
    return ret