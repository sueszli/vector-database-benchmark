"""
Package support for Solaris

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import copy
import logging
import os
import salt.utils.data
import salt.utils.files
import salt.utils.functools
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError, MinionError
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        return 10
    '\n    Set the virtual pkg module if the os is Solaris\n    '
    if __grains__['os_family'] == 'Solaris' and float(__grains__['kernelrelease']) <= 5.1:
        return __virtualname__
    return (False, 'The solarispkg execution module failed to load: only available on Solaris <= 10.')

def _write_adminfile(kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Create a temporary adminfile based on the keyword arguments passed to\n    pkg.install.\n    '
    email = kwargs.get('email', '')
    instance = kwargs.get('instance', 'quit')
    partial = kwargs.get('partial', 'nocheck')
    runlevel = kwargs.get('runlevel', 'nocheck')
    idepend = kwargs.get('idepend', 'nocheck')
    rdepend = kwargs.get('rdepend', 'nocheck')
    space = kwargs.get('space', 'nocheck')
    setuid = kwargs.get('setuid', 'nocheck')
    conflict = kwargs.get('conflict', 'nocheck')
    action = kwargs.get('action', 'nocheck')
    basedir = kwargs.get('basedir', 'default')
    adminfile = salt.utils.files.mkstemp(prefix='salt-')

    def _write_line(fp_, line):
        if False:
            return 10
        fp_.write(salt.utils.stringutils.to_str(line))
    with salt.utils.files.fopen(adminfile, 'w') as fp_:
        _write_line(fp_, 'email={}\n'.format(email))
        _write_line(fp_, 'instance={}\n'.format(instance))
        _write_line(fp_, 'partial={}\n'.format(partial))
        _write_line(fp_, 'runlevel={}\n'.format(runlevel))
        _write_line(fp_, 'idepend={}\n'.format(idepend))
        _write_line(fp_, 'rdepend={}\n'.format(rdepend))
        _write_line(fp_, 'space={}\n'.format(space))
        _write_line(fp_, 'setuid={}\n'.format(setuid))
        _write_line(fp_, 'conflict={}\n'.format(conflict))
        _write_line(fp_, 'action={}\n'.format(action))
        _write_line(fp_, 'basedir={}\n'.format(basedir))
    return adminfile

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
    "\n    List the packages currently installed as a dict:\n\n    .. code-block:: python\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    ret = {}
    cmd = '/usr/bin/pkginfo -x'
    lines = __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False).splitlines()
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

def latest_version(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n\n    NOTE: As package repositories are not presently supported for Solaris\n    pkgadd, this function will always return an empty string for a given\n    package.\n    "
    kwargs.pop('refresh', True)
    ret = {}
    if not names:
        return ''
    for name in names:
        ret[name] = ''
    if len(names) == 1:
        return ret[names[0]]
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def upgrade_available(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check whether or not an upgrade is available for a given package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return latest_version(name) != ''

def version(*names, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def install(name=None, sources=None, saltenv='base', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Install the passed package. Can install packages from the following\n    sources:\n\n    * Locally (package already exists on the minion\n    * HTTP/HTTPS server\n    * FTP server\n    * Salt master\n\n    Returns a dict containing the new package names and versions:\n\n    .. code-block:: python\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        # Installing a data stream pkg that already exists on the minion\n\n        salt \'*\' pkg.install sources=\'[{"<pkg name>": "/dir/on/minion/<pkg filename>"}]\'\n        salt \'*\' pkg.install sources=\'[{"SMClgcc346": "/var/spool/pkg/gcc-3.4.6-sol10-sparc-local.pkg"}]\'\n\n        # Installing a data stream pkg that exists on the salt master\n\n        salt \'*\' pkg.install sources=\'[{"<pkg name>": "salt://pkgs/<pkg filename>"}]\'\n        salt \'*\' pkg.install sources=\'[{"SMClgcc346": "salt://pkgs/gcc-3.4.6-sol10-sparc-local.pkg"}]\'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Installing a data stream pkg that exists on a HTTP server\n        salt \'*\' pkg.install sources=\'[{"<pkg name>": "http://packages.server.com/<pkg filename>"}]\'\n        salt \'*\' pkg.install sources=\'[{"SMClgcc346": "http://packages.server.com/gcc-3.4.6-sol10-sparc-local.pkg"}]\'\n\n    If working with solaris zones and you want to install a package only in the\n    global zone you can pass \'current_zone_only=True\' to salt to have the\n    package only installed in the global zone. (Behind the scenes this is\n    passing \'-G\' to the pkgadd command.) Solaris default when installing a\n    package in the global zone is to install it in all zones. This overrides\n    that and installs the package only in the global.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Installing a data stream package only in the global zone:\n        salt \'global_zone\' pkg.install sources=\'[{"SMClgcc346": "/var/spool/pkg/gcc-3.4.6-sol10-sparc-local.pkg"}]\' current_zone_only=True\n\n    By default salt automatically provides an adminfile, to automate package\n    installation, with these options set::\n\n        email=\n        instance=quit\n        partial=nocheck\n        runlevel=nocheck\n        idepend=nocheck\n        rdepend=nocheck\n        space=nocheck\n        setuid=nocheck\n        conflict=nocheck\n        action=nocheck\n        basedir=default\n\n    You can override any of these options in two ways. First you can optionally\n    pass any of the options as a kwarg to the module/state to override the\n    default value or you can optionally pass the \'admin_source\' option\n    providing your own adminfile to the minions.\n\n    Note: You can find all of the possible options to provide to the adminfile\n    by reading the admin man page:\n\n    .. code-block:: bash\n\n        man -s 4 admin\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Overriding the \'instance\' adminfile option when calling the module directly\n        salt \'*\' pkg.install sources=\'[{"<pkg name>": "salt://pkgs/<pkg filename>"}]\' instance="overwrite"\n\n    SLS Example:\n\n    .. code-block:: yaml\n\n        # Overriding the \'instance\' adminfile option when used in a state\n\n        SMClgcc346:\n          pkg.installed:\n            - sources:\n              - SMClgcc346: salt://srv/salt/pkgs/gcc-3.4.6-sol10-sparc-local.pkg\n            - instance: overwrite\n\n    .. note::\n        The ID declaration is ignored, as the package name is read from the\n        ``sources`` parameter.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Providing your own adminfile when calling the module directly\n\n        salt \'*\' pkg.install sources=\'[{"<pkg name>": "salt://pkgs/<pkg filename>"}]\' admin_source=\'salt://pkgs/<adminfile filename>\'\n\n        # Providing your own adminfile when using states\n\n        <pkg name>:\n          pkg.installed:\n            - sources:\n              - <pkg name>: salt://pkgs/<pkg filename>\n            - admin_source: salt://pkgs/<adminfile filename>\n\n    .. note::\n        The ID declaration is ignored, as the package name is read from the\n        ``sources`` parameter.\n    '
    if salt.utils.data.is_true(kwargs.get('refresh')):
        log.warning("'refresh' argument not implemented for solarispkg module")
    pkgs = kwargs.pop('pkgs', None)
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, sources, **kwargs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    if not sources:
        log.error('"sources" param required for solaris pkg_add installs')
        return {}
    try:
        if 'admin_source' in kwargs:
            adminfile = __salt__['cp.cache_file'](kwargs['admin_source'], saltenv)
        else:
            adminfile = _write_adminfile(kwargs)
        old = list_pkgs()
        cmd_prefix = ['/usr/sbin/pkgadd', '-n', '-a', adminfile]
        if kwargs.get('current_zone_only') in (True, 'True'):
            cmd_prefix.append('-G ')
        errors = []
        for pkg in pkg_params:
            cmd = cmd_prefix + ['-d', pkg, 'all']
            out = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
            if out['retcode'] != 0 and out['stderr']:
                errors.append(out['stderr'])
        __context__.pop('pkg.list_pkgs', None)
        new = list_pkgs()
        ret = salt.utils.data.compare_dicts(old, new)
        if errors:
            raise CommandExecutionError('Problem encountered installing package(s)', info={'errors': errors, 'changes': ret})
    finally:
        if 'admin_source' not in kwargs:
            try:
                os.remove(adminfile)
            except (NameError, OSError):
                pass
    return ret

def remove(name=None, pkgs=None, saltenv='base', **kwargs):
    if False:
        while True:
            i = 10
    '\n    Remove packages with pkgrm\n\n    name\n        The name of the package to be deleted\n\n    By default salt automatically provides an adminfile, to automate package\n    removal, with these options set::\n\n        email=\n        instance=quit\n        partial=nocheck\n        runlevel=nocheck\n        idepend=nocheck\n        rdepend=nocheck\n        space=nocheck\n        setuid=nocheck\n        conflict=nocheck\n        action=nocheck\n        basedir=default\n\n    You can override any of these options in two ways. First you can optionally\n    pass any of the options as a kwarg to the module/state to override the\n    default value or you can optionally pass the \'admin_source\' option\n    providing your own adminfile to the minions.\n\n    Note: You can find all of the possible options to provide to the adminfile\n    by reading the admin man page:\n\n    .. code-block:: bash\n\n        man -s 4 admin\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove SUNWgit\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    old = list_pkgs()
    targets = [x for x in pkg_params if x in old]
    if not targets:
        return {}
    try:
        if 'admin_source' in kwargs:
            adminfile = __salt__['cp.cache_file'](kwargs['admin_source'], saltenv)
        else:
            adminfile = _write_adminfile(kwargs)
        cmd = ['/usr/sbin/pkgrm', '-n', '-a', adminfile] + targets
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
    finally:
        if 'admin_source' not in kwargs:
            try:
                os.remove(adminfile)
            except (NameError, OSError):
                pass
    return ret

def purge(name=None, pkgs=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Package purges are not supported, this function is identical to\n    ``remove()``.\n\n    name\n        The name of the package to be deleted\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    return remove(name=name, pkgs=pkgs, **kwargs)