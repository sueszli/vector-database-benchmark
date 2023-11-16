"""
Package support for OpenBSD

.. note::

    The package repository is configured on each host using ``/etc/installurl``
    from OpenBSD 6.1 onwards. Earlier releases relied on ``/etc/pkg.conf``.

.. versionchanged:: 2016.3.5

    Package versions on OpenBSD are not normally specified explicitly; instead
    packages may be available in multiple *flavors*, and *branches* which are
    specified by the format of the package name. This module allows you to use
    the same formatting as ``pkg_add(1)``, and will select the empty flavor and
    default branch by default. Examples:

    .. code-block:: yaml

      - rsync
      - vim--no_x11
      - ruby%2.3

"""
import copy
import logging
import re
import salt.utils.data
import salt.utils.versions
from salt.exceptions import CommandExecutionError, MinionError
log = logging.getLogger(__name__)
__PKG_RE = re.compile('^((?:[^-]+|-(?![0-9]))+)-([0-9][^-]*)(?:-(.*))?$')
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Set the virtual pkg module if the os is OpenBSD\n    '
    if __grains__['os'] == 'OpenBSD':
        return __virtualname__
    return (False, 'The openbsdpkg execution module cannot be loaded: only available on OpenBSD systems.')

def _list_pkgs_from_context(versions_as_list):
    if False:
        print('Hello World!')
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
    "\n    List the packages currently installed as a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    ret = {}
    cmd = 'pkg_info -q -a'
    out = __salt__['cmd.run_stdout'](cmd, output_loglevel='trace')
    for line in out.splitlines():
        try:
            (pkgname, pkgver, flavor) = __PKG_RE.match(line).groups()
        except AttributeError:
            continue
        pkgname += '--{}'.format(flavor) if flavor else ''
        __salt__['pkg_resource.add_pkg'](ret, pkgname, pkgver)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def latest_version(*names, **kwargs):
    if False:
        return 10
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n    "
    kwargs.pop('refresh', True)
    pkgs = list_pkgs()
    ret = {}
    for name in names:
        ret[name] = ''
        cmd = 'pkg_info -Q {}'.format(name)
        out = __salt__['cmd.run_stdout'](cmd, python_shell=False, output_loglevel='trace')
        for line in out.splitlines():
            try:
                (pkgname, pkgver, flavor) = __PKG_RE.match(line).groups()
            except AttributeError:
                continue
            match = re.match('.*\\(installed\\)$', pkgver)
            if match:
                break
            if '{}--{}'.format(pkgname, flavor) == name:
                pkgname += '--{}'.format(flavor)
            elif pkgname == name:
                pass
            else:
                continue
            cur = pkgs.get(pkgname, '')
            if not cur or salt.utils.versions.compare(ver1=cur, oper='<', ver2=pkgver):
                ret[pkgname] = pkgver
    if len(names) == 1:
        return ret[names[0]]
    return ret

def version(*names, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def install(name=None, pkgs=None, sources=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Install the passed package\n\n    Return a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    CLI Example, Install one package:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install <package name>\n\n    CLI Example, Install more than one package:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install pkgs=\'["<package name>", "<package name>"]\'\n\n    CLI Example, Install more than one package from a alternate source (e.g.\n    salt file-server, HTTP, FTP, local filesystem):\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install sources=\'[{"<pkg name>": "salt://pkgs/<pkg filename>"}]\'\n    '
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, sources, **kwargs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    old = list_pkgs()
    errors = []
    for pkg in pkg_params:
        if pkg_type == 'repository':
            (stem, branch) = (pkg.split('%') + [''])[:2]
            (base, flavor) = (stem.split('--') + [''])[:2]
            pkg = '{}--{}%{}'.format(base, flavor, branch)
        cmd = 'pkg_add -x -I {}'.format(pkg)
        out = __salt__['cmd.run_all'](cmd, python_shell=False, output_loglevel='trace')
        if out['retcode'] != 0 and out['stderr']:
            errors.append(out['stderr'])
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        raise CommandExecutionError('Problem encountered installing package(s)', info={'errors': errors, 'changes': ret})
    return ret

def remove(name=None, pkgs=None, purge=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Remove a single package with pkg_delete\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    try:
        pkg_params = [x.split('--')[0] for x in __salt__['pkg_resource.parse_targets'](name, pkgs)[0]]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    old = list_pkgs()
    targets = [x for x in pkg_params if x in old]
    if not targets:
        return {}
    cmd = ['pkg_delete', '-Ix', '-Ddependencies']
    if purge:
        cmd.append('-cqq')
    cmd.extend(targets)
    out = __salt__['cmd.run_all'](cmd, python_shell=False, output_loglevel='trace')
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
    '\n    Remove a package and extra configuration files.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    return remove(name=name, pkgs=pkgs, purge=True)

def upgrade_available(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Check whether or not an upgrade is available for a given package\n\n    .. versionadded:: 2019.2.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return latest_version(name) != ''

def upgrade(name=None, pkgs=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Run a full package upgrade (``pkg_add -u``), or upgrade a specific package\n    if ``name`` or ``pkgs`` is provided.\n    ``name`` is ignored when ``pkgs`` is specified.\n\n    Returns a dictionary containing the changes:\n\n    .. versionadded:: 2019.2.0\n\n    .. code-block:: python\n\n        {'<package>': {'old': '<old-version>',\n                       'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n        salt '*' pkg.upgrade python%2.7\n    "
    old = list_pkgs()
    cmd = ['pkg_add', '-Ix', '-u']
    if kwargs.get('noop', False):
        cmd.append('-n')
    if pkgs:
        cmd.extend(pkgs)
    elif name:
        cmd.append(name)
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    return ret