"""
Pkgutil support for Solaris

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import copy
import salt.utils.data
import salt.utils.functools
import salt.utils.pkg
import salt.utils.versions
from salt.exceptions import CommandExecutionError, MinionError
__virtualname__ = 'pkgutil'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set the virtual pkg module if the os is Solaris\n    '
    if __grains__['os_family'] == 'Solaris':
        return __virtualname__
    return (False, 'The pkgutil execution module cannot be loaded: only available on Solaris systems.')

def refresh_db():
    if False:
        print('Hello World!')
    "\n    Updates the pkgutil repo database (pkgutil -U)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.refresh_db\n    "
    salt.utils.pkg.clear_rtag(__opts__)
    return __salt__['cmd.retcode']('/opt/csw/bin/pkgutil -U') == 0

def upgrade_available(name):
    if False:
        return 10
    "\n    Check if there is an upgrade available for a certain package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.upgrade_available CSWpython\n    "
    version_num = None
    cmd = '/opt/csw/bin/pkgutil -c --parse --single {}'.format(name)
    out = __salt__['cmd.run_stdout'](cmd)
    if out:
        version_num = out.split()[2].strip()
    if version_num:
        if version_num == 'SAME':
            return ''
        else:
            return version_num
    return ''

def list_upgrades(refresh=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all available package upgrades on this system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.list_upgrades\n    "
    if salt.utils.data.is_true(refresh):
        refresh_db()
    upgrades = {}
    lines = __salt__['cmd.run_stdout']('/opt/csw/bin/pkgutil -A --parse').splitlines()
    for line in lines:
        comps = line.split('\t')
        if comps[2] == 'SAME':
            continue
        if comps[2] == 'not installed':
            continue
        upgrades[comps[0]] = comps[1]
    return upgrades

def upgrade(refresh=True):
    if False:
        while True:
            i = 10
    "\n    Upgrade all of the packages to the latest available version.\n\n    Returns a dict containing the changes::\n\n        {'<package>': {'old': '<old-version>',\n                       'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.upgrade\n    "
    if salt.utils.data.is_true(refresh):
        refresh_db()
    old = list_pkgs()
    cmd = '/opt/csw/bin/pkgutil -yu'
    __salt__['cmd.run_all'](cmd)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    return salt.utils.data.compare_dicts(old, new)

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
        return 10
    "\n    List the packages currently installed as a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n        salt '*' pkg.list_pkgs versions_as_list=True\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if salt.utils.data.is_true(kwargs.get('removed')):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    ret = {}
    cmd = '/usr/bin/pkginfo -x'
    lines = __salt__['cmd.run'](cmd).splitlines()
    for (index, line) in enumerate(lines):
        if index % 2 == 0:
            name = line.split()[0].strip()
        if index % 2 == 1:
            version_num = line.split()[1].strip()
            __salt__['pkg_resource.add_pkg'](ret, name, version_num)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a version if the package is installed, else returns an empty string\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.version CSWpython\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def latest_version(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkgutil.latest_version CSWpython\n        salt '*' pkgutil.latest_version <package1> <package2> <package3> ...\n    "
    refresh = salt.utils.data.is_true(kwargs.pop('refresh', True))
    if not names:
        return ''
    ret = {}
    for name in names:
        ret[name] = ''
    if refresh:
        refresh_db()
    pkgs = list_pkgs()
    cmd = '/opt/csw/bin/pkgutil -a --parse {}'.format(' '.join(names))
    output = __salt__['cmd.run_all'](cmd).get('stdout', '').splitlines()
    for line in output:
        try:
            (name, version_rev) = line.split()[1:3]
        except ValueError:
            continue
        if name in names:
            cver = pkgs.get(name, '')
            nver = version_rev.split(',')[0]
            if not cver or salt.utils.versions.compare(ver1=cver, oper='<', ver2=nver):
                ret[name] = version_rev
    if len(names) == 1:
        return ret[names[0]]
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def install(name=None, refresh=False, version=None, pkgs=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Install packages using the pkgutil tool.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install <package_name>\n        salt \'*\' pkg.install SMClgcc346\n\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install from OpenCSW. Must be passed as a python\n        list.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n            salt \'*\' pkg.install pkgs=\'["foo", {"bar": "1.2.3"}]\'\n\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n    '
    if refresh:
        refresh_db()
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs, **kwargs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    if pkgs is None and version and (len(pkg_params) == 1):
        pkg_params = {name: version}
    targets = []
    for (param, pkgver) in pkg_params.items():
        if pkgver is None:
            targets.append(param)
        else:
            targets.append('{}-{}'.format(param, pkgver))
    cmd = '/opt/csw/bin/pkgutil -yu {}'.format(' '.join(targets))
    old = list_pkgs()
    __salt__['cmd.run_all'](cmd)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    return salt.utils.data.compare_dicts(old, new)

def remove(name=None, pkgs=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Remove a package and all its dependencies which are not in use by other\n    packages.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    old = list_pkgs()
    targets = [x for x in pkg_params if x in old]
    if not targets:
        return {}
    cmd = '/opt/csw/bin/pkgutil -yr {}'.format(' '.join(targets))
    __salt__['cmd.run_all'](cmd)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    return salt.utils.data.compare_dicts(old, new)

def purge(name=None, pkgs=None, **kwargs):
    if False:
        return 10
    '\n    Package purges are not supported, this function is identical to\n    ``remove()``.\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    return remove(name=name, pkgs=pkgs)