"""
Support for YUM/DNF

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.

.. note::
    DNF is fully supported as of version 2015.5.10 and 2015.8.4 (partial
    support for DNF was initially added in 2015.8.0), and DNF is used
    automatically in place of YUM in Fedora 22 and newer.

.. versionadded:: 3003
    Support for ``tdnf`` on Photon OS.
.. versionadded:: 3007.0
    Support for ``dnf5``` on Fedora 39
"""
import configparser
import contextlib
import datetime
import fnmatch
import itertools
import logging
import os
import re
import string
import salt.utils.args
import salt.utils.data
import salt.utils.decorators.path
import salt.utils.environment
import salt.utils.files
import salt.utils.functools
import salt.utils.itertools
import salt.utils.lazy
import salt.utils.path
import salt.utils.pkg
import salt.utils.pkg.rpm
import salt.utils.systemd
import salt.utils.versions
from salt.exceptions import CommandExecutionError, MinionError, SaltInvocationError
from salt.utils.versions import LooseVersion
try:
    import yum
    HAS_YUM = True
except ImportError:
    HAS_YUM = False
log = logging.getLogger(__name__)
__HOLD_PATTERN = '[\\w+]+(?:[.-][^-]+)*'
PKG_ARCH_SEPARATOR = '.'
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Confine this module to yum based systems\n    '
    if __opts__.get('yum_provider') == 'yumpkg_api':
        return (False, 'Module yumpkg: yumpkg_api provider not available')
    try:
        os_grain = __grains__['os'].lower()
        os_family = __grains__['os_family'].lower()
    except Exception:
        return (False, 'Module yumpkg: no yum based system detected')
    enabled = ('amazon', 'xcp', 'xenserver', 'virtuozzolinux', 'virtuozzo', 'issabel pbx', 'openeuler')
    if os_family == 'redhat' or os_grain in enabled:
        if _yum() is None:
            return (False, 'DNF nor YUM found')
        return __virtualname__
    return (False, 'Module yumpkg: no yum based system detected')

def _strip_headers(output, *args):
    if False:
        print('Hello World!')
    if not args:
        args_lc = ('installed packages', 'available packages', 'available upgrades', 'updated packages', 'upgraded packages')
    else:
        args_lc = [x.lower() for x in args]
    ret = ''
    for line in salt.utils.itertools.split(output, '\n'):
        if line.lower() not in args_lc:
            ret += line + '\n'
    return ret

def _get_copr_repo(copr):
    if False:
        while True:
            i = 10
    copr = copr.split(':', 1)[1]
    copr = copr.split('/', 1)
    return f'copr:copr.fedorainfracloud.org:{copr[0]}:{copr[1]}'

def _get_hold(line, pattern=__HOLD_PATTERN, full=True):
    if False:
        return 10
    '\n    Resolve a package name from a line containing the hold expression. If the\n    regex is not matched, None is returned.\n\n    yum ==> 2:vim-enhanced-7.4.629-5.el6.*\n    dnf ==> vim-enhanced-2:7.4.827-1.fc22.*\n    '
    if full:
        if _yum() in ('dnf', 'dnf5'):
            lock_re = f'({pattern}-\\S+)'
        else:
            lock_re = f'(\\d+:{pattern}-\\S+)'
    elif _yum() in ('dnf', 'dnf5'):
        lock_re = f'({pattern}-\\S+)'
    else:
        lock_re = f'\\d+:({pattern}-\\S+)'
    match = re.search(lock_re, line)
    if match:
        if not full:
            woarch = match.group(1).rsplit('.', 1)[0]
            worel = woarch.rsplit('-', 1)[0]
            return worel.rsplit('-', 1)[0]
        else:
            return match.group(1)
    return None

def _yum():
    if False:
        return 10
    '\n    Determine package manager name (yum or dnf[5]),\n    depending on the executable existence in $PATH.\n    '
    import os

    def _check(file):
        if False:
            i = 10
            return i + 15
        return os.path.exists(file) and os.access(file, os.F_OK | os.X_OK) and (not os.path.isdir(file))
    try:
        context = __context__
    except NameError:
        context = {}
    contextkey = 'yum_bin'
    if contextkey not in context:
        for dir in os.environ.get('PATH', os.defpath).split(os.pathsep):
            if _check(os.path.join(dir, 'dnf5')):
                context[contextkey] = 'dnf5'
                break
            elif _check(os.path.join(dir, 'dnf')):
                context[contextkey] = 'dnf'
                break
            elif _check(os.path.join(dir, 'tdnf')):
                context[contextkey] = 'tdnf'
                break
            elif _check(os.path.join(dir, 'yum')):
                context[contextkey] = 'yum'
                break
    return context.get(contextkey)

def _call_yum(args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Call yum/dnf.\n    '
    params = {'output_loglevel': 'trace', 'python_shell': False, 'env': salt.utils.environment.get_module_environment(globals())}
    params.update(kwargs)
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.append(_yum())
    cmd.extend(args)
    return __salt__['cmd.run_all'](cmd, **params)

def _yum_pkginfo(output):
    if False:
        return 10
    '\n    Parse yum/dnf output (which could contain irregular line breaks if package\n    names are long) retrieving the name, version, etc., and return a list of\n    pkginfo namedtuples.\n    '
    cur = {}
    keys = itertools.cycle(('name', 'version', 'repoid'))
    values = salt.utils.itertools.split(_strip_headers(output))
    osarch = __grains__['osarch']
    for (key, value) in zip(keys, values):
        if key == 'name':
            try:
                (cur['name'], cur['arch']) = value.rsplit('.', 1)
            except ValueError:
                cur['name'] = value
                cur['arch'] = osarch
            cur['name'] = salt.utils.pkg.rpm.resolve_name(cur['name'], cur['arch'], osarch)
        else:
            if key == 'version':
                value = value.rstrip('-')
            elif key == 'repoid':
                value = value.lstrip('@')
            cur[key] = value
            if key == 'repoid':
                pkginfo = salt.utils.pkg.rpm.pkginfo(**cur)
                cur = {}
                if pkginfo is not None:
                    yield pkginfo

def _versionlock_pkg(grains=None):
    if False:
        while True:
            i = 10
    '\n    Determine versionlock plugin package name\n    '
    if grains is None:
        grains = __grains__
    if _yum() in ('dnf', 'dnf5'):
        if grains['os'].lower() == 'fedora':
            return 'python3-dnf-plugin-versionlock' if int(grains.get('osrelease')) >= 26 else 'python3-dnf-plugins-extras-versionlock'
        if int(grains.get('osmajorrelease')) >= 8:
            return 'python3-dnf-plugin-versionlock'
        return 'python2-dnf-plugin-versionlock'
    elif _yum() == 'tdnf':
        raise SaltInvocationError('Cannot proceed, no versionlock for tdnf')
    else:
        return 'yum-plugin-versionlock'

def _check_versionlock():
    if False:
        return 10
    '\n    Ensure that the appropriate versionlock plugin is present\n    '
    vl_plugin = _versionlock_pkg()
    if vl_plugin not in list_pkgs():
        raise SaltInvocationError(f'Cannot proceed, {vl_plugin} is not installed.')

def _get_options(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of options to be used in the yum/dnf[5] command, based on the\n    kwargs passed.\n    '
    fromrepo = kwargs.pop('fromrepo', '')
    repo = kwargs.pop('repo', '')
    disablerepo = kwargs.pop('disablerepo', '')
    enablerepo = kwargs.pop('enablerepo', '')
    disableexcludes = kwargs.pop('disableexcludes', '')
    branch = kwargs.pop('branch', '')
    setopt = kwargs.pop('setopt', None)
    if setopt is None:
        setopt = []
    else:
        setopt = salt.utils.args.split_input(setopt)
    get_extra_options = kwargs.pop('get_extra_options', False)
    if repo and (not fromrepo):
        fromrepo = repo
    ret = []
    if fromrepo:
        log.info("Restricting to repo '%s'", fromrepo)
        ret.extend(['--disablerepo=*', f'--enablerepo={fromrepo}'])
    else:
        if disablerepo:
            targets = [disablerepo] if not isinstance(disablerepo, list) else disablerepo
            log.info('Disabling repo(s): %s', ', '.join(targets))
            ret.extend([f'--disablerepo={x}' for x in targets])
        if enablerepo:
            targets = [enablerepo] if not isinstance(enablerepo, list) else enablerepo
            log.info('Enabling repo(s): %s', ', '.join(targets))
            ret.extend([f'--enablerepo={x}' for x in targets])
    if disableexcludes:
        log.info("Disabling excludes for '%s'", disableexcludes)
        ret.append(f'--disableexcludes={disableexcludes}')
    if branch:
        log.info("Adding branch '%s'", branch)
        ret.append(f'--branch={branch}')
    for item in setopt:
        ret.extend(['--setopt', str(item)])
    if get_extra_options:
        for key in sorted(kwargs):
            if key.startswith('__'):
                continue
            value = kwargs[key]
            if isinstance(value, str):
                log.info('Found extra option --%s=%s', key, value)
                ret.append(f'--{key}={value}')
            elif value is True:
                log.info('Found extra option --%s', key)
                ret.append(f'--{key}')
        if ret:
            log.info('Adding extra options: %s', ret)
    return ret

def _get_yum_config(strict_parser=True):
    if False:
        while True:
            i = 10
    '\n    Returns a dict representing the yum config options and values.\n\n    We try to pull all of the yum config options into a standard dict object.\n    This is currently only used to get the reposdir settings, but could be used\n    for other things if needed.\n\n    If the yum python library is available, use that, which will give us all of\n    the options, including all of the defaults not specified in the yum config.\n    Additionally, they will all be of the correct object type.\n\n    If the yum library is not available, we try to read the yum.conf\n    directly ourselves with a minimal set of "defaults".\n    '
    conf = {'reposdir': ['/etc/yum/repos.d', '/etc/yum.repos.d']}
    if HAS_YUM:
        try:
            yb = yum.YumBase()
            yb.preconf.init_plugins = False
            for (name, value) in yb.conf.items():
                conf[name] = value
        except (AttributeError, yum.Errors.ConfigError) as exc:
            raise CommandExecutionError(f'Could not query yum config: {exc}')
        except yum.Errors.YumBaseError as yum_base_error:
            raise CommandExecutionError(f'Error accessing yum or rpmdb: {yum_base_error}')
    else:
        fn = None
        paths = ('/etc/yum/yum.conf', '/etc/yum.conf', '/etc/dnf/dnf.conf', '/etc/tdnf/tdnf.conf')
        for path in paths:
            if os.path.exists(path):
                fn = path
                break
        if not fn:
            raise CommandExecutionError(f'No suitable yum config file found in: {paths}')
        cp = configparser.ConfigParser(strict=strict_parser)
        try:
            cp.read(fn)
        except OSError as exc:
            raise CommandExecutionError(f'Unable to read from {fn}: {exc}')
        if cp.has_section('main'):
            for opt in cp.options('main'):
                if opt in ('reposdir', 'commands', 'excludes'):
                    conf[opt] = [x.strip() for x in cp.get('main', opt).split(',')]
                else:
                    conf[opt] = cp.get('main', opt)
        else:
            log.warning('Could not find [main] section in %s, using internal defaults', fn)
    return conf

def _get_yum_config_value(name, strict_config=True):
    if False:
        print('Hello World!')
    '\n    Look for a specific config variable and return its value\n    '
    conf = _get_yum_config(strict_config)
    if name in conf.keys():
        return conf.get(name)
    return None

def _normalize_basedir(basedir=None, strict_config=True):
    if False:
        return 10
    "\n    Takes a basedir argument as a string or a list. If the string or list is\n    empty, then look up the default from the 'reposdir' option in the yum\n    config.\n\n    Returns a list of directories.\n    "
    if isinstance(basedir, str):
        basedir = [x.strip() for x in basedir.split(',')]
    if basedir is None:
        basedir = []
    if not basedir:
        basedir = _get_yum_config_value('reposdir', strict_config)
    if not isinstance(basedir, list) or not basedir:
        raise SaltInvocationError('Could not determine any repo directories')
    return basedir

def normalize_name(name):
    if False:
        print('Hello World!')
    "\n    Strips the architecture from the specified package name, if necessary.\n    Circumstances where this would be done include:\n\n    * If the arch is 32 bit and the package name ends in a 32-bit arch.\n    * If the arch matches the OS arch, or is ``noarch``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.normalize_name zsh.x86_64\n    "
    try:
        arch = name.rsplit(PKG_ARCH_SEPARATOR, 1)[-1]
        if arch not in salt.utils.pkg.rpm.ARCHES + ('noarch',):
            return name
    except ValueError:
        return name
    if arch in (__grains__['osarch'], 'noarch') or salt.utils.pkg.rpm.check_32(arch, osarch=__grains__['osarch']):
        return name[:-(len(arch) + 1)]
    return name

def parse_arch(name):
    if False:
        print('Hello World!')
    "\n    Parse name and architecture from the specified package name.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.parse_arch zsh.x86_64\n    "
    (_name, _arch) = (None, None)
    try:
        (_name, _arch) = name.rsplit(PKG_ARCH_SEPARATOR, 1)
    except ValueError:
        pass
    if _arch not in salt.utils.pkg.rpm.ARCHES + ('noarch',):
        _name = name
        _arch = None
    return {'name': _name, 'arch': _arch}

def latest_version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    A specific repo can be requested using the ``fromrepo`` keyword argument,\n    and the ``disableexcludes`` option is also supported.\n\n    .. versionadded:: 2014.7.0\n        Support for the ``disableexcludes`` option\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package name> fromrepo=epel-testing\n        salt '*' pkg.latest_version <package name> disableexcludes=main\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n    "
    refresh = salt.utils.data.is_true(kwargs.pop('refresh', True))
    if not names:
        return ''
    options = _get_options(**kwargs)
    if refresh:
        refresh_db(**kwargs)
    cur_pkgs = list_pkgs(versions_as_list=True)
    cmd = ['--quiet']
    cmd.extend(options)
    cmd.extend(['list', 'available'])
    cmd.extend(names)
    out = _call_yum(cmd, ignore_retcode=True)
    if out['retcode'] != 0:
        if out['stderr']:
            if not all([x in cur_pkgs for x in names]):
                log.error('Problem encountered getting latest version for the following package(s): %s. Stderr follows: \n%s', ', '.join(names), out['stderr'])
        updates = []
    else:
        updates = sorted(_yum_pkginfo(out['stdout']), key=lambda pkginfo: LooseVersion(pkginfo.version), reverse=True)

    def _check_cur(pkg):
        if False:
            print('Hello World!')
        if pkg.name in cur_pkgs:
            for installed_version in cur_pkgs[pkg.name]:
                if salt.utils.versions.compare(ver1=installed_version, oper='>=', ver2=pkg.version, cmp_func=version_cmp):
                    return False
            return True
        else:
            return True
    ret = {}
    for name in names:
        try:
            arch = name.rsplit('.', 1)[-1]
            if arch not in salt.utils.pkg.rpm.ARCHES:
                arch = __grains__['osarch']
        except ValueError:
            arch = __grains__['osarch']
        for pkg in (x for x in updates if x.name == name):
            if pkg.arch == 'noarch' or pkg.arch == arch or salt.utils.pkg.rpm.check_32(pkg.arch):
                if _check_cur(pkg):
                    ret[name] = pkg.version
                    break
        else:
            ret[name] = ''
    if len(names) == 1:
        return ret[names[0]]
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def upgrade_available(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Check whether or not an upgrade is available for a given package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return latest_version(name, **kwargs) != ''

def version(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def version_cmp(pkg1, pkg2, ignore_epoch=False, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2015.5.4\n\n    Do a cmp-style comparison on two packages. Return -1 if pkg1 < pkg2, 0 if\n    pkg1 == pkg2, and 1 if pkg1 > pkg2. Return None if there was a problem\n    making the comparison.\n\n    ignore_epoch : False\n        Set to ``True`` to ignore the epoch when comparing versions\n\n        .. versionadded:: 2015.8.10,2016.3.2\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version_cmp '0.2-001' '0.2.0.1-002'\n    "
    return __salt__['lowpkg.version_cmp'](pkg1, pkg2, ignore_epoch=ignore_epoch)

def _list_pkgs_from_context(versions_as_list, contextkey, attr):
    if False:
        return 10
    '\n    Use pkg list from __context__\n    '
    return __salt__['pkg_resource.format_pkg_list'](__context__[contextkey], versions_as_list, attr)

def list_pkgs(versions_as_list=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    List the packages currently installed as a dict. By default, the dict\n    contains versions as a comma separated string::\n\n        {\'<package_name>\': \'<version>[,<version>...]\'}\n\n    versions_as_list:\n        If set to true, the versions are provided as a list\n\n        {\'<package_name>\': [\'<version>\', \'<version>\']}\n\n    attr:\n        If a list of package attributes is specified, returned value will\n        contain them in addition to version, eg.::\n\n        {\'<package_name>\': [{\'version\' : \'version\', \'arch\' : \'arch\'}]}\n\n        Valid attributes are: ``epoch``, ``version``, ``release``, ``arch``,\n        ``install_date``, ``install_date_time_t``.\n\n        If ``all`` is specified, all valid attributes will be returned.\n\n            .. versionadded:: 2018.3.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.list_pkgs\n        salt \'*\' pkg.list_pkgs attr=version,arch\n        salt \'*\' pkg.list_pkgs attr=\'["version", "arch"]\'\n    '
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    attr = kwargs.get('attr')
    if attr is not None and attr != 'all':
        attr = salt.utils.args.split_input(attr)
    contextkey = 'pkg.list_pkgs'
    if contextkey in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list, contextkey, attr)
    ret = {}
    cmd = ['rpm', '-qa', '--nodigest', '--nosignature', '--queryformat', salt.utils.pkg.rpm.QUERYFORMAT.replace('%{REPOID}', '(none)') + '\n']
    output = __salt__['cmd.run'](cmd, python_shell=False, output_loglevel='trace')
    for line in output.splitlines():
        pkginfo = salt.utils.pkg.rpm.parse_pkginfo(line, osarch=__grains__['osarch'])
        if pkginfo is not None:
            pkgver = pkginfo.version
            epoch = None
            release = None
            if ':' in pkgver:
                (epoch, pkgver) = pkgver.split(':', 1)
            if '-' in pkgver:
                (pkgver, release) = pkgver.split('-', 1)
            all_attr = {'epoch': epoch, 'version': pkgver, 'release': release, 'arch': pkginfo.arch, 'install_date': pkginfo.install_date, 'install_date_time_t': pkginfo.install_date_time_t}
            __salt__['pkg_resource.add_pkg'](ret, pkginfo.name, all_attr)
    for pkgname in ret:
        ret[pkgname] = sorted(ret[pkgname], key=lambda d: d['version'])
    __context__[contextkey] = ret
    return __salt__['pkg_resource.format_pkg_list'](__context__[contextkey], versions_as_list, attr)

def list_repo_pkgs(*args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.1.0\n    .. versionchanged:: 2014.7.0\n        All available versions of each package are now returned. This required\n        a slight modification to the structure of the return dict. The return\n        data shown below reflects the updated return dict structure. Note that\n        packages which are version-locked using :py:mod:`pkg.hold\n        <salt.modules.yumpkg.hold>` will only show the currently-installed\n        version, as locking a package will make other versions appear\n        unavailable to yum/dnf.\n    .. versionchanged:: 2017.7.0\n        By default, the versions for each package are no longer organized by\n        repository. To get results organized by repository, use\n        ``byrepo=True``.\n\n    Returns all available packages. Optionally, package names (and name globs)\n    can be passed and the results will be filtered to packages matching those\n    names. This is recommended as it speeds up the function considerably.\n\n    .. warning::\n        Running this function on RHEL/CentOS 6 and earlier will be more\n        resource-intensive, as the version of yum that ships with older\n        RHEL/CentOS has no yum subcommand for listing packages from a\n        repository. Thus, a ``yum list installed`` and ``yum list available``\n        are run, which generates a lot of output, which must then be analyzed\n        to determine which package information to include in the return data.\n\n    This function can be helpful in discovering the version or repo to specify\n    in a :mod:`pkg.installed <salt.states.pkg.installed>` state.\n\n    The return data will be a dictionary mapping package names to a list of\n    version numbers, ordered from newest to oldest. If ``byrepo`` is set to\n    ``True``, then the return dictionary will contain repository names at the\n    top level, and each repository will map packages to lists of version\n    numbers. For example:\n\n    .. code-block:: python\n\n        # With byrepo=False (default)\n        {\n            'bash': ['4.1.2-15.el6_5.2',\n                     '4.1.2-15.el6_5.1',\n                     '4.1.2-15.el6_4'],\n            'kernel': ['2.6.32-431.29.2.el6',\n                       '2.6.32-431.23.3.el6',\n                       '2.6.32-431.20.5.el6',\n                       '2.6.32-431.20.3.el6',\n                       '2.6.32-431.17.1.el6',\n                       '2.6.32-431.11.2.el6',\n                       '2.6.32-431.5.1.el6',\n                       '2.6.32-431.3.1.el6',\n                       '2.6.32-431.1.2.0.1.el6',\n                       '2.6.32-431.el6']\n        }\n        # With byrepo=True\n        {\n            'base': {\n                'bash': ['4.1.2-15.el6_4'],\n                'kernel': ['2.6.32-431.el6']\n            },\n            'updates': {\n                'bash': ['4.1.2-15.el6_5.2', '4.1.2-15.el6_5.1'],\n                'kernel': ['2.6.32-431.29.2.el6',\n                           '2.6.32-431.23.3.el6',\n                           '2.6.32-431.20.5.el6',\n                           '2.6.32-431.20.3.el6',\n                           '2.6.32-431.17.1.el6',\n                           '2.6.32-431.11.2.el6',\n                           '2.6.32-431.5.1.el6',\n                           '2.6.32-431.3.1.el6',\n                           '2.6.32-431.1.2.0.1.el6']\n            }\n        }\n\n    fromrepo : None\n        Only include results from the specified repo(s). Multiple repos can be\n        specified, comma-separated.\n\n    enablerepo (ignored if ``fromrepo`` is specified)\n        Specify a disabled package repository (or repositories) to enable.\n        (e.g., ``yum --enablerepo='somerepo'``)\n\n        .. versionadded:: 2017.7.0\n\n    disablerepo (ignored if ``fromrepo`` is specified)\n        Specify an enabled package repository (or repositories) to disable.\n        (e.g., ``yum --disablerepo='somerepo'``)\n\n        .. versionadded:: 2017.7.0\n\n    byrepo : False\n        When ``True``, the return data for each package will be organized by\n        repository.\n\n        .. versionadded:: 2017.7.0\n\n    cacheonly : False\n        When ``True``, the repo information will be retrieved from the cached\n        repo metadata. This is equivalent to passing the ``-C`` option to\n        yum/dnf.\n\n        .. versionadded:: 2017.7.0\n\n    setopt\n        A comma-separated or Python list of key=value options. This list will\n        be expanded and ``--setopt`` prepended to each in the yum/dnf command\n        that is run.\n\n        .. versionadded:: 2019.2.0\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_repo_pkgs\n        salt '*' pkg.list_repo_pkgs foo bar baz\n        salt '*' pkg.list_repo_pkgs 'samba4*' fromrepo=base,updates\n        salt '*' pkg.list_repo_pkgs 'python2-*' byrepo=True\n    "
    byrepo = kwargs.pop('byrepo', False)
    cacheonly = kwargs.pop('cacheonly', False)
    fromrepo = kwargs.pop('fromrepo', '') or ''
    disablerepo = kwargs.pop('disablerepo', '') or ''
    enablerepo = kwargs.pop('enablerepo', '') or ''
    repo_arg = _get_options(fromrepo=fromrepo, **kwargs)
    if fromrepo and (not isinstance(fromrepo, list)):
        try:
            fromrepo = [x.strip() for x in fromrepo.split(',')]
        except AttributeError:
            fromrepo = [x.strip() for x in str(fromrepo).split(',')]
    if disablerepo and (not isinstance(disablerepo, list)):
        try:
            disablerepo = [x.strip() for x in disablerepo.split(',') if x != '*']
        except AttributeError:
            disablerepo = [x.strip() for x in str(disablerepo).split(',') if x != '*']
    if enablerepo and (not isinstance(enablerepo, list)):
        try:
            enablerepo = [x.strip() for x in enablerepo.split(',') if x != '*']
        except AttributeError:
            enablerepo = [x.strip() for x in str(enablerepo).split(',') if x != '*']
    if fromrepo:
        repos = fromrepo
    else:
        repos = [repo_name for (repo_name, repo_info) in list_repos(**kwargs).items() if repo_name in enablerepo or (repo_name not in disablerepo and str(repo_info.get('enabled', '1')) == '1')]
    ret = {}

    def _check_args(args, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Do glob matching on args and return True if a match was found.\n        Otherwise, return False\n        '
        for arg in args:
            if fnmatch.fnmatch(name, arg):
                return True
        return False

    def _parse_output(output, strict=False):
        if False:
            return 10
        for pkg in _yum_pkginfo(output):
            if strict and (pkg.repoid not in repos or not _check_args(args, pkg.name)):
                continue
            repo_dict = ret.setdefault(pkg.repoid, {})
            version_list = repo_dict.setdefault(pkg.name, set())
            version_list.add(pkg.version)
    yum_version = None if _yum() != 'yum' else LooseVersion(__salt__['cmd.run'](['yum', '--version'], python_shell=False).splitlines()[0].strip())
    if yum_version and yum_version < LooseVersion('3.2.13'):
        cmd_prefix = ['--quiet']
        if cacheonly:
            cmd_prefix.append('-C')
        cmd_prefix.append('list')
        for pkg_src in ('installed', 'available'):
            out = _call_yum(cmd_prefix + [pkg_src], ignore_retcode=True)
            if out['retcode'] == 0:
                _parse_output(out['stdout'], strict=True)
    elif yum_version and yum_version < LooseVersion('3.4.3'):
        cmd_prefix = ['--quiet', '--showduplicates']
        if cacheonly:
            cmd_prefix.append('-C')
        cmd_prefix.append('list')
        for pkg_src in ('installed', 'available'):
            out = _call_yum(cmd_prefix + [pkg_src], ignore_retcode=True)
            if out['retcode'] == 0:
                _parse_output(out['stdout'], strict=True)
    else:
        for repo in repos:
            if _yum() == 'tdnf':
                cmd = ['--quiet', f'--enablerepo={repo}', 'list']
            else:
                cmd = ['--quiet', '--showduplicates', 'repository-packages', repo, 'list']
            if cacheonly:
                cmd.append('-C')
            cmd.extend(args)
            out = _call_yum(cmd, ignore_retcode=True)
            if out['retcode'] != 0 and 'Error:' in out['stdout']:
                continue
            _parse_output(out['stdout'])
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

def list_upgrades(refresh=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Check whether or not an upgrade is available for all packages\n\n    The ``fromrepo``, ``enablerepo``, and ``disablerepo`` arguments are\n    supported, as used in pkg states, and the ``disableexcludes`` option is\n    also supported.\n\n    .. versionadded:: 2014.7.0\n        Support for the ``disableexcludes`` option\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    options = _get_options(**kwargs)
    if salt.utils.data.is_true(refresh):
        refresh_db(check_update=False, **kwargs)
    cmd = ['--quiet']
    cmd.extend(options)
    cmd.extend(['list', 'upgrades' if _yum() in ('dnf', 'dnf5') else 'updates'])
    out = _call_yum(cmd, ignore_retcode=True)
    if out['retcode'] != 0 and 'Error:' in out:
        return {}
    return {x.name: x.version for x in _yum_pkginfo(out['stdout'])}
list_updates = salt.utils.functools.alias_function(list_upgrades, 'list_updates')

def list_downloaded(**kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2017.7.0\n\n    List prefetched packages downloaded by Yum in the local disk.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_downloaded\n    "
    CACHE_DIR = os.path.join('/var/cache/', _yum())
    ret = {}
    for (root, dirnames, filenames) in salt.utils.path.os_walk(CACHE_DIR):
        for filename in fnmatch.filter(filenames, '*.rpm'):
            package_path = os.path.join(root, filename)
            pkg_info = __salt__['lowpkg.bin_pkg_info'](package_path)
            pkg_timestamp = int(os.path.getctime(package_path))
            ret.setdefault(pkg_info['name'], {})[pkg_info['version']] = {'path': package_path, 'size': os.path.getsize(package_path), 'creation_date_time_t': pkg_timestamp, 'creation_date_time': datetime.datetime.fromtimestamp(pkg_timestamp).isoformat()}
    return ret

def info_installed(*names, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2015.8.1\n\n    Return the information of the named package(s), installed on the system.\n\n    :param all_versions:\n        Include information for all versions of the packages installed on the minion.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.info_installed <package1>\n        salt '*' pkg.info_installed <package1> <package2> <package3> ...\n        salt '*' pkg.info_installed <package1> <package2> <package3> all_versions=True\n    "
    all_versions = kwargs.get('all_versions', False)
    ret = dict()
    for (pkg_name, pkgs_nfo) in __salt__['lowpkg.info'](*names, **kwargs).items():
        pkg_nfo = pkgs_nfo if all_versions else [pkgs_nfo]
        for _nfo in pkg_nfo:
            t_nfo = dict()
            for (key, value) in _nfo.items():
                if key == 'source_rpm':
                    t_nfo['source'] = value
                else:
                    t_nfo[key] = value
            if not all_versions:
                ret[pkg_name] = t_nfo
            else:
                ret.setdefault(pkg_name, []).append(t_nfo)
    return ret

def refresh_db(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Check the yum repos for updated packages\n\n    Returns:\n\n    - ``True``: Updates are available\n    - ``False``: An error occurred\n    - ``None``: No updates are available\n\n    repo\n        Refresh just the specified repo\n\n    disablerepo\n        Do not refresh the specified repo\n\n    enablerepo\n        Refresh a disabled repo using this option\n\n    branch\n        Add the specified branch when refreshing\n\n    disableexcludes\n        Disable the excludes defined in your config files. Takes one of three\n        options:\n        - ``all`` - disable all excludes\n        - ``main`` - disable excludes defined in [main] in yum.conf\n        - ``repoid`` - disable excludes defined for that repo\n\n    setopt\n        A comma-separated or Python list of key=value options. This list will\n        be expanded and ``--setopt`` prepended to each in the yum/dnf command\n        that is run.\n\n        .. versionadded:: 2019.2.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n    "
    salt.utils.pkg.clear_rtag(__opts__)
    retcodes = {100: True, 0: None, 1: False}
    ret = True
    check_update_ = kwargs.pop('check_update', True)
    options = _get_options(**kwargs)
    clean_cmd = ['--quiet', '--assumeyes', 'clean', 'expire-cache']
    clean_cmd.extend(options)
    _call_yum(clean_cmd, ignore_retcode=True)
    if check_update_:
        update_cmd = ['--quiet', '--assumeyes', 'check-update']
        if __grains__.get('os_family') == 'RedHat' and __grains__.get('osmajorrelease') == 7:
            update_cmd.append('--setopt=autocheck_running_kernel=false')
        update_cmd.extend(options)
        ret = retcodes.get(_call_yum(update_cmd, ignore_retcode=True)['retcode'], False)
    return ret

def clean_metadata(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2014.1.0\n\n    Cleans local yum metadata. Functionally identical to :mod:`refresh_db()\n    <salt.modules.yumpkg.refresh_db>`.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.clean_metadata\n    "
    return refresh_db(**kwargs)

class AvailablePackages(salt.utils.lazy.LazyDict):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._args = args
        self._kwargs = kwargs

    def _load(self, key):
        if False:
            return 10
        self._load_all()
        return True

    def _load_all(self):
        if False:
            return 10
        self._dict = list_repo_pkgs(*self._args, **self._kwargs)
        self.loaded = True

def install(name=None, refresh=False, skip_verify=False, pkgs=None, sources=None, downloadonly=False, reinstall=False, normalize=True, update_holds=False, saltenv='base', ignore_epoch=False, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any yum/dnf commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Install the passed package(s), add refresh=True to clean the yum database\n    before package is installed.\n\n    name\n        The name of the package to be installed. Note that this parameter is\n        ignored if either "pkgs" or "sources" is passed. Additionally, please\n        note that this option can only be used to install packages from a\n        software repository. To install a package file manually, use the\n        "sources" option.\n\n        32-bit packages can be installed on 64-bit systems by appending the\n        architecture designation (``.i686``, ``.i586``, etc.) to the end of the\n        package name.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n\n    refresh\n        Whether or not to update the yum database before executing.\n\n    reinstall\n        Specifying reinstall=True will use ``yum reinstall`` rather than\n        ``yum install`` for requested packages that are already installed.\n\n        If a version is specified with the requested package, then\n        ``yum reinstall`` will only be used if the installed version\n        matches the requested version.\n\n        Works with ``sources`` when the package header of the source can be\n        matched to the name and version of an installed package.\n\n        .. versionadded:: 2014.7.0\n\n    skip_verify\n        Skip the GPG verification check (e.g., ``--nogpgcheck``)\n\n    downloadonly\n        Only download the packages, do not install.\n\n    version\n        Install a specific version of the package, e.g. 1.2.3-4.el5. Ignored\n        if "pkgs" or "sources" is passed.\n\n        .. versionchanged:: 2018.3.0\n            version can now contain comparison operators (e.g. ``>1.2.3``,\n            ``<=2.0``, etc.)\n\n    update_holds : False\n        If ``True``, and this function would update the package version, any\n        packages held using the yum/dnf "versionlock" plugin will be unheld so\n        that they can be updated. Otherwise, if this function attempts to\n        update a held package, the held package(s) will be skipped and an\n        error will be raised.\n\n        .. versionadded:: 2016.11.0\n\n    setopt\n        A comma-separated or Python list of key=value options. This list will\n        be expanded and ``--setopt`` prepended to each in the yum/dnf command\n        that is run.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install foo setopt=\'obsoletes=0,plugins=0\'\n\n        .. versionadded:: 2019.2.0\n\n    Repository Options:\n\n    fromrepo\n        Specify a package repository (or repositories) from which to install.\n        (e.g., ``yum --disablerepo=\'*\' --enablerepo=\'somerepo\'``)\n\n    enablerepo (ignored if ``fromrepo`` is specified)\n        Specify a disabled package repository (or repositories) to enable.\n        (e.g., ``yum --enablerepo=\'somerepo\'``)\n\n    disablerepo (ignored if ``fromrepo`` is specified)\n        Specify an enabled package repository (or repositories) to disable.\n        (e.g., ``yum --disablerepo=\'somerepo\'``)\n\n    disableexcludes\n        Disable exclude from main, for a repo or for everything.\n        (e.g., ``yum --disableexcludes=\'main\'``)\n\n        .. versionadded:: 2014.7.0\n\n    ignore_epoch : False\n        Only used when the version of a package is specified using a comparison\n        operator (e.g. ``>4.1``). If set to ``True``, then the epoch will be\n        ignored when comparing the currently-installed version to the desired\n        version.\n\n        .. versionadded:: 2018.3.0\n\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install from a software repository. Must be\n        passed as a python list. A specific version number can be specified\n        by using a single-element dict representing the package and its\n        version.\n\n        CLI Examples:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n            salt \'*\' pkg.install pkgs=\'["foo", {"bar": "1.2.3-4.el5"}]\'\n\n    sources\n        A list of RPM packages to install. Must be passed as a list of dicts,\n        with the keys being package names, and the values being the source URI\n        or local path to the package.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install sources=\'[{"foo": "salt://foo.rpm"}, {"bar": "salt://bar.rpm"}]\'\n\n    normalize : True\n        Normalize the package name by removing the architecture. This is useful\n        for poorly created packages which might include the architecture as an\n        actual part of the name such as kernel modules which match a specific\n        kernel version.\n\n        .. code-block:: bash\n\n            salt -G role:nsd pkg.install gpfs.gplbin-2.6.32-279.31.1.el6.x86_64 normalize=False\n\n        .. versionadded:: 2014.7.0\n\n    split_arch : True\n        If set to False it prevents package name normalization more strict way\n        than ``normalize`` set to ``False`` does.\n\n        .. versionadded:: 3006.0\n\n    diff_attr:\n        If a list of package attributes is specified, returned value will\n        contain them, eg.::\n\n            {\'<package>\': {\n                \'old\': {\n                    \'version\': \'<old-version>\',\n                    \'arch\': \'<old-arch>\'},\n\n                \'new\': {\n                    \'version\': \'<new-version>\',\n                    \'arch\': \'<new-arch>\'}}}\n\n        Valid attributes are: ``epoch``, ``version``, ``release``, ``arch``,\n        ``install_date``, ``install_date_time_t``.\n\n        If ``all`` is specified, all valid attributes will be returned.\n\n        .. versionadded:: 2018.3.0\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    If an attribute list in diff_attr is specified, the dict will also contain\n    any specified attribute, eg.::\n\n        {\'<package>\': {\n            \'old\': {\n                \'version\': \'<old-version>\',\n                \'arch\': \'<old-arch>\'},\n\n            \'new\': {\n                \'version\': \'<new-version>\',\n                \'arch\': \'<new-arch>\'}}}\n    '
    if 'version' in kwargs:
        kwargs['version'] = str(kwargs['version'])
    options = _get_options(**kwargs)
    if salt.utils.data.is_true(refresh):
        refresh_db(**kwargs)
    reinstall = salt.utils.data.is_true(reinstall)
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, sources, saltenv=saltenv, normalize=normalize and kwargs.get('split_arch', True), **kwargs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    diff_attr = kwargs.get('diff_attr')
    old = list_pkgs(versions_as_list=False, attr=diff_attr) if not downloadonly else list_downloaded()
    old_as_list = list_pkgs(versions_as_list=True) if not downloadonly else list_downloaded()
    to_install = []
    to_downgrade = []
    to_reinstall = []
    _available = {}
    if pkg_type == 'repository':
        has_wildcards = []
        has_comparison = []
        for (pkgname, pkgver) in pkg_params.items():
            try:
                if '*' in pkgver:
                    has_wildcards.append(pkgname)
                elif pkgver.startswith('<') or pkgver.startswith('>'):
                    has_comparison.append(pkgname)
            except (TypeError, ValueError):
                continue
        _available = AvailablePackages(*has_wildcards + has_comparison, byrepo=False, **kwargs)
        pkg_params_items = pkg_params.items()
    elif pkg_type == 'advisory':
        pkg_params_items = []
        cur_patches = list_patches()
        for advisory_id in pkg_params:
            if advisory_id not in cur_patches:
                raise CommandExecutionError(f'Advisory id "{advisory_id}" not found')
            else:
                pkg_params_items.append(advisory_id)
    else:
        pkg_params_items = []
        for pkg_source in pkg_params:
            if 'lowpkg.bin_pkg_info' in __salt__:
                rpm_info = __salt__['lowpkg.bin_pkg_info'](pkg_source)
            else:
                rpm_info = None
            if rpm_info is None:
                log.error('pkg.install: Unable to get rpm information for %s. Version comparisons will be unavailable, and return data may be inaccurate if reinstall=True.', pkg_source)
                pkg_params_items.append([pkg_source])
            else:
                pkg_params_items.append([rpm_info['name'], pkg_source, rpm_info['version']])
    errors = []
    for pkg_item_list in pkg_params_items:
        if pkg_type == 'repository':
            (pkgname, version_num) = pkg_item_list
        elif pkg_type == 'advisory':
            pkgname = pkg_item_list
            version_num = None
        else:
            try:
                (pkgname, pkgpath, version_num) = pkg_item_list
            except ValueError:
                pkgname = None
                pkgpath = pkg_item_list[0]
                version_num = None
        if version_num is None:
            if pkg_type == 'repository':
                if reinstall and pkgname in old:
                    to_reinstall.append((pkgname, pkgname))
                else:
                    to_install.append((pkgname, pkgname))
            elif pkg_type == 'advisory':
                to_install.append((pkgname, pkgname))
            else:
                to_install.append((pkgname, pkgpath))
        else:
            if pkg_type == 'repository':
                version_num = version_num.lstrip('=')
                if pkgname in has_comparison:
                    candidates = _available.get(pkgname, [])
                    target = salt.utils.pkg.match_version(version_num, candidates, cmp_func=version_cmp, ignore_epoch=ignore_epoch)
                    if target is None:
                        errors.append("No version matching '{}{}' could be found (available: {})".format(pkgname, version_num, ', '.join(candidates) if candidates else None))
                        continue
                    else:
                        version_num = target
                if _yum() == 'yum':
                    if ignore_epoch is True:
                        version_num = version_num.split(':', 1)[-1]
                arch = ''
                try:
                    (namepart, archpart) = pkgname.rsplit('.', 1)
                except ValueError:
                    pass
                else:
                    if archpart in salt.utils.pkg.rpm.ARCHES and (archpart != __grains__['osarch'] or kwargs.get('split_arch', True)):
                        arch = '.' + archpart
                        pkgname = namepart
                if '*' in version_num:
                    candidates = _available.get(pkgname, [])
                    match = salt.utils.itertools.fnmatch_multiple(candidates, version_num)
                    if match is not None:
                        version_num = match
                    else:
                        errors.append("No version matching '{}' found for package '{}' (available: {})".format(version_num, pkgname, ', '.join(candidates) if candidates else 'none'))
                        continue
                if ignore_epoch is True:
                    pkgstr = f'{pkgname}-{version_num}{arch}'
                else:
                    pkgstr = '{}-{}{}'.format(pkgname, version_num.split(':', 1)[-1], arch)
            else:
                pkgstr = pkgpath
            cver = old_as_list.get(pkgname, [])
            if reinstall and cver:
                for ver in cver:
                    if salt.utils.versions.compare(ver1=version_num, oper='==', ver2=ver, cmp_func=version_cmp, ignore_epoch=ignore_epoch):
                        to_reinstall.append((pkgname, pkgstr))
                        break
            elif not cver:
                to_install.append((pkgname, pkgstr))
            else:
                for ver in cver:
                    if salt.utils.versions.compare(ver1=version_num, oper='>=', ver2=ver, cmp_func=version_cmp, ignore_epoch=ignore_epoch):
                        to_install.append((pkgname, pkgstr))
                        break
                else:
                    if pkgname is not None:
                        if re.match('^kernel(|-devel)$', pkgname):
                            to_install.append((pkgname, pkgstr))
                        else:
                            to_downgrade.append((pkgname, pkgstr))

    def _add_common_args(cmd):
        if False:
            print('Hello World!')
        '\n        DRY function to add args common to all yum/dnf commands\n        '
        cmd.extend(options)
        if skip_verify:
            cmd.append('--nogpgcheck')
        if downloadonly:
            if _yum() != 'dnf5':
                cmd.append('--downloadonly')
    try:
        holds = list_holds(full=False)
    except SaltInvocationError:
        holds = []
        log.debug('Failed to get holds, versionlock plugin is probably not installed')
    unhold_prevented = []

    @contextlib.contextmanager
    def _temporarily_unhold(pkgs, targets):
        if False:
            return 10
        '\n        Temporarily unhold packages that need to be updated. Add any\n        successfully-removed ones (and any packages not in the list of current\n        holds) to the list of targets.\n        '
        to_unhold = {}
        for (pkgname, pkgstr) in pkgs:
            if pkgname in holds:
                if update_holds:
                    to_unhold[pkgname] = pkgstr
                else:
                    unhold_prevented.append(pkgname)
            else:
                targets.append(pkgstr)
        if not to_unhold:
            yield
        else:
            log.debug('Unholding packages: %s', ', '.join(to_unhold))
            try:
                unhold_names = list(to_unhold.keys())
                for (unheld_pkg, outcome) in unhold(pkgs=unhold_names).items():
                    if outcome['result']:
                        targets.append(to_unhold[unheld_pkg])
                    else:
                        errors.append(unheld_pkg)
                yield
            except Exception as exc:
                errors.append('Error encountered unholding packages {}: {}'.format(', '.join(to_unhold), exc))
            finally:
                hold(pkgs=unhold_names)
    targets = []
    with _temporarily_unhold(to_install, targets):
        if targets:
            if pkg_type == 'advisory':
                targets = [f'--advisory={t}' for t in targets]
            cmd = ['-y']
            if _yum() == 'dnf':
                cmd.extend(['--best', '--allowerasing'])
            _add_common_args(cmd)
            cmd.append('install' if pkg_type != 'advisory' else 'update')
            if _yum() == 'dnf5':
                cmd.extend(['--best', '--allowerasing'])
            cmd.extend(targets)
            out = _call_yum(cmd, ignore_retcode=False, redirect_stderr=True)
            if out['retcode'] != 0:
                errors.append(out['stdout'])
    targets = []
    with _temporarily_unhold(to_downgrade, targets):
        if targets:
            cmd = ['-y']
            _add_common_args(cmd)
            cmd.append('downgrade')
            cmd.extend(targets)
            out = _call_yum(cmd, redirect_stderr=True)
            if out['retcode'] != 0:
                errors.append(out['stdout'])
    targets = []
    with _temporarily_unhold(to_reinstall, targets):
        if targets:
            cmd = ['-y']
            _add_common_args(cmd)
            cmd.append('reinstall')
            cmd.extend(targets)
            out = _call_yum(cmd, redirect_stderr=True)
            if out['retcode'] != 0:
                errors.append(out['stdout'])
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs(versions_as_list=False, attr=diff_attr) if not downloadonly else list_downloaded()
    ret = salt.utils.data.compare_dicts(old, new)
    for (pkgname, _) in to_reinstall:
        if pkgname not in ret or pkgname in old:
            ret.update({pkgname: {'old': old.get(pkgname, ''), 'new': new.get(pkgname, '')}})
    if unhold_prevented:
        errors.append("The following package(s) could not be updated because they are being held: {}. Set 'update_holds' to True to temporarily unhold these packages so that they can be updated.".format(', '.join(unhold_prevented)))
    if errors:
        raise CommandExecutionError('Error occurred installing{} package(s)'.format('/reinstalling' if to_reinstall else ''), info={'errors': errors, 'changes': ret})
    return ret

def upgrade(name=None, pkgs=None, refresh=True, skip_verify=False, normalize=True, minimal=False, obsoletes=True, diff_attr=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Run a full system upgrade (a ``yum upgrade`` or ``dnf upgrade``), or\n    upgrade specified packages. If the packages aren\'t installed, they will\n    not be installed.\n\n    .. versionchanged:: 2014.7.0\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any yum/dnf commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    .. versionchanged:: 2019.2.0\n        Added ``obsoletes`` and ``minimal`` arguments\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {\'<package>\':  {\'old\': \'<old-version>\',\n                        \'new\': \'<new-version>\'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.upgrade\n        salt \'*\' pkg.upgrade name=openssl\n\n    Repository Options:\n\n    fromrepo\n        Specify a package repository (or repositories) from which to install.\n        (e.g., ``yum --disablerepo=\'*\' --enablerepo=\'somerepo\'``)\n\n    enablerepo (ignored if ``fromrepo`` is specified)\n        Specify a disabled package repository (or repositories) to enable.\n        (e.g., ``yum --enablerepo=\'somerepo\'``)\n\n    disablerepo (ignored if ``fromrepo`` is specified)\n        Specify an enabled package repository (or repositories) to disable.\n        (e.g., ``yum --disablerepo=\'somerepo\'``)\n\n    disableexcludes\n        Disable exclude from main, for a repo or for everything.\n        (e.g., ``yum --disableexcludes=\'main\'``)\n\n        .. versionadded:: 2014.7.0\n\n    name\n        The name of the package to be upgraded. Note that this parameter is\n        ignored if "pkgs" is passed.\n\n        32-bit packages can be upgraded on 64-bit systems by appending the\n        architecture designation (``.i686``, ``.i586``, etc.) to the end of the\n        package name.\n\n        Warning: if you forget \'name=\' and run pkg.upgrade openssl, ALL packages\n        are upgraded. This will be addressed in next releases.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.upgrade name=openssl\n\n        .. versionadded:: 2016.3.0\n\n    pkgs\n        A list of packages to upgrade from a software repository. Must be\n        passed as a python list. A specific version number can be specified\n        by using a single-element dict representing the package and its\n        version. If the package was not already installed on the system,\n        it will not be installed.\n\n        CLI Examples:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.upgrade pkgs=\'["foo", "bar"]\'\n            salt \'*\' pkg.upgrade pkgs=\'["foo", {"bar": "1.2.3-4.el5"}]\'\n\n        .. versionadded:: 2016.3.0\n\n    normalize : True\n        Normalize the package name by removing the architecture. This is useful\n        for poorly created packages which might include the architecture as an\n        actual part of the name such as kernel modules which match a specific\n        kernel version.\n\n        .. code-block:: bash\n\n            salt -G role:nsd pkg.upgrade gpfs.gplbin-2.6.32-279.31.1.el6.x86_64 normalize=False\n\n        .. versionadded:: 2016.3.0\n\n    minimal : False\n        Use upgrade-minimal instead of upgrade (e.g., ``yum upgrade-minimal``)\n        Goes to the \'newest\' package match which fixes a problem that affects your system.\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.upgrade minimal=True\n\n        .. versionadded:: 2019.2.0\n\n    obsoletes : True\n        Controls whether yum/dnf should take obsoletes into account and remove them.\n        If set to ``False`` yum will use ``update`` instead of ``upgrade``\n        and dnf will be run with ``--obsoletes=False``\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.upgrade obsoletes=False\n\n        .. versionadded:: 2019.2.0\n\n    setopt\n        A comma-separated or Python list of key=value options. This list will\n        be expanded and ``--setopt`` prepended to each in the yum/dnf command\n        that is run.\n\n        .. versionadded:: 2019.2.0\n\n    diff_attr:\n        If a list of package attributes is specified, returned value will\n        contain them, eg.::\n\n            {\'<package>\': {\n                \'old\': {\n                    \'version\': \'<old-version>\',\n                    \'arch\': \'<old-arch>\'},\n\n                \'new\': {\n                    \'version\': \'<new-version>\',\n                    \'arch\': \'<new-arch>\'}}}\n\n        Valid attributes are: ``epoch``, ``version``, ``release``, ``arch``,\n        ``install_date``, ``install_date_time_t``.\n\n        If ``all`` is specified, all valid attributes will be returned.\n\n        .. versionadded:: 3006.0\n\n    .. note::\n        To add extra arguments to the ``yum upgrade`` command, pass them as key\n        word arguments. For arguments without assignments, pass ``True``\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.upgrade security=True exclude=\'kernel*\'\n    '
    if _yum() in ('dnf', 'dnf5') and (not obsoletes):
        _setopt = [opt for opt in salt.utils.args.split_input(kwargs.pop('setopt', [])) if not opt.startswith('obsoletes=')]
        _setopt.append('obsoletes=False')
        kwargs['setopt'] = _setopt
    options = _get_options(get_extra_options=True, **kwargs)
    if salt.utils.data.is_true(refresh):
        refresh_db(**kwargs)
    old = list_pkgs(attr=diff_attr)
    targets = []
    if name or pkgs:
        try:
            pkg_params = __salt__['pkg_resource.parse_targets'](name=name, pkgs=pkgs, sources=None, normalize=normalize, **kwargs)[0]
        except MinionError as exc:
            raise CommandExecutionError(exc)
        if pkg_params:
            targets.extend(pkg_params)
    cmd = ['--quiet', '-y']
    cmd.extend(options)
    if skip_verify:
        cmd.append('--nogpgcheck')
    if obsoletes:
        cmd.append('upgrade' if not minimal else 'upgrade-minimal')
    elif _yum() in ('dnf', 'dnf5'):
        cmd.append('upgrade' if not minimal else 'upgrade-minimal')
    else:
        cmd.append('update' if not minimal else 'update-minimal')
    cmd.extend(targets)
    result = _call_yum(cmd)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs(attr=diff_attr)
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    return ret

def update(name=None, pkgs=None, refresh=True, skip_verify=False, normalize=True, minimal=False, obsoletes=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2019.2.0\n\n    Calls :py:func:`pkg.upgrade <salt.modules.yumpkg.upgrade>` with\n    ``obsoletes=False``. Mirrors the CLI behavior of ``yum update``.\n    See :py:func:`pkg.upgrade <salt.modules.yumpkg.upgrade>` for\n    further documentation.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.update\n    "
    return upgrade(name, pkgs, refresh, skip_verify, normalize, minimal, obsoletes, **kwargs)

def remove(name=None, pkgs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any yum/dnf commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Remove packages\n\n    name\n        The name of the package to be removed\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n    split_arch : True\n        If set to False it prevents package name normalization by removing arch.\n\n        .. versionadded:: 3006.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    old = list_pkgs()
    targets = []
    pkg_params = salt.utils.pkg.match_wildcard(old, pkg_params)
    for target in pkg_params:
        if target not in old:
            continue
        version_to_remove = pkg_params[target]
        if target in old and (not version_to_remove):
            targets.append(target)
        elif target in old and version_to_remove in old[target].split(','):
            arch = ''
            pkgname = target
            try:
                (namepart, archpart) = pkgname.rsplit('.', 1)
            except ValueError:
                pass
            else:
                if archpart in salt.utils.pkg.rpm.ARCHES and (archpart != __grains__['osarch'] or kwargs.get('split_arch', True)):
                    arch = '.' + archpart
                    pkgname = namepart
            targets.append('{}-{}{}'.format(pkgname, version_to_remove.split(':', 1)[-1], arch))
    if not targets:
        return {}
    out = _call_yum(['-y', 'remove'] + targets)
    if out['retcode'] != 0 and out['stderr']:
        errors = [out['stderr']]
    else:
        errors = []
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        raise CommandExecutionError('Error occurred removing package(s)', info={'errors': errors, 'changes': ret})
    return ret

def purge(name=None, pkgs=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any yum/dnf commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Package purges are not supported by yum, this function is identical to\n    :mod:`pkg.remove <salt.modules.yumpkg.remove>`.\n\n    name\n        The name of the package to be purged\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    return remove(name=name, pkgs=pkgs)

def hold(name=None, pkgs=None, sources=None, normalize=True, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2014.7.0\n\n    Version-lock packages\n\n    .. note::\n        Requires the appropriate ``versionlock`` plugin package to be installed:\n\n        - On RHEL 5: ``yum-versionlock``\n        - On RHEL 6 & 7: ``yum-plugin-versionlock``\n        - On Fedora: ``python-dnf-plugins-extras-versionlock``\n\n\n    name\n        The name of the package to be held.\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to hold. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.hold <package name>\n        salt \'*\' pkg.hold pkgs=\'["foo", "bar"]\'\n    '
    _check_versionlock()
    if not name and (not pkgs) and (not sources):
        raise SaltInvocationError('One of name, pkgs, or sources must be specified.')
    if pkgs and sources:
        raise SaltInvocationError('Only one of pkgs or sources can be specified.')
    targets = []
    if pkgs:
        targets.extend(pkgs)
    elif sources:
        for source in sources:
            targets.append(next(iter(source.keys())))
    else:
        targets.append(name)
    current_locks = list_holds(full=False)
    ret = {}
    for target in targets:
        if isinstance(target, dict):
            target = next(iter(target.keys()))
        ret[target] = {'name': target, 'changes': {}, 'result': False, 'comment': ''}
        if target not in current_locks:
            if 'test' in __opts__ and __opts__['test']:
                ret[target].update(result=None)
                ret[target]['comment'] = f'Package {target} is set to be held.'
            else:
                out = _call_yum(['versionlock', target])
                if out['retcode'] == 0:
                    ret[target].update(result=True)
                    ret[target]['comment'] = 'Package {} is now being held.'.format(target)
                    ret[target]['changes']['new'] = 'hold'
                    ret[target]['changes']['old'] = ''
                else:
                    ret[target]['comment'] = 'Package {} was unable to be held.'.format(target)
        else:
            ret[target].update(result=True)
            ret[target]['comment'] = 'Package {} is already set to be held.'.format(target)
    return ret

def unhold(name=None, pkgs=None, sources=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 2014.7.0\n\n    Remove version locks\n\n    .. note::\n        Requires the appropriate ``versionlock`` plugin package to be installed:\n\n        - On RHEL 5: ``yum-versionlock``\n        - On RHEL 6 & 7: ``yum-plugin-versionlock``\n        - On Fedora: ``python-dnf-plugins-extras-versionlock``\n\n\n    name\n        The name of the package to be unheld\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to unhold. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.unhold <package name>\n        salt \'*\' pkg.unhold pkgs=\'["foo", "bar"]\'\n    '
    _check_versionlock()
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
    current_locks = list_holds(full=_yum() == 'yum')
    ret = {}
    for target in targets:
        if isinstance(target, dict):
            target = next(iter(target.keys()))
        ret[target] = {'name': target, 'changes': {}, 'result': False, 'comment': ''}
        if _yum() in ('dnf', 'dnf5'):
            search_locks = [x for x in current_locks if x == target]
        else:
            search_locks = [x for x in current_locks if fnmatch.fnmatch(x, f'*{target}*') and target == _get_hold(x, full=False)]
        if search_locks:
            if __opts__['test']:
                ret[target].update(result=None)
                ret[target]['comment'] = 'Package {} is set to be unheld.'.format(target)
            else:
                out = _call_yum(['versionlock', 'delete'] + search_locks)
                if out['retcode'] == 0:
                    ret[target].update(result=True)
                    ret[target]['comment'] = 'Package {} is no longer held.'.format(target)
                    ret[target]['changes']['new'] = ''
                    ret[target]['changes']['old'] = 'hold'
                else:
                    ret[target]['comment'] = f'Package {target} was unable to be unheld.'
        else:
            ret[target].update(result=True)
            ret[target]['comment'] = f'Package {target} is not being held.'
    return ret

def list_holds(pattern=__HOLD_PATTERN, full=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionchanged:: 2015.5.10,2015.8.4,2016.3.0\n        Function renamed from ``pkg.get_locked_pkgs`` to ``pkg.list_holds``.\n\n    List information on locked packages\n\n    .. note::\n        Requires the appropriate ``versionlock`` plugin package to be installed:\n\n        - On RHEL 5: ``yum-versionlock``\n        - On RHEL 6 & 7: ``yum-plugin-versionlock``\n        - On Fedora: ``python-dnf-plugins-extras-versionlock``\n\n    pattern : \\w+(?:[.-][^-]+)*\n        Regular expression used to match the package name\n\n    full : True\n        Show the full hold definition including version and epoch. Set to\n        ``False`` to return just the name of the package(s) being held.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_holds\n        salt '*' pkg.list_holds full=False\n    "
    _check_versionlock()
    out = __salt__['cmd.run']([_yum(), 'versionlock', 'list'], python_shell=False)
    ret = []
    for line in salt.utils.itertools.split(out, '\n'):
        match = _get_hold(line, pattern=pattern, full=full)
        if match is not None:
            ret.append(match)
    return ret
get_locked_packages = salt.utils.functools.alias_function(list_holds, 'get_locked_packages')

def verify(*names, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.1.0\n\n    Runs an rpm -Va on a system, and returns the results in a dict\n\n    Pass options to modify rpm verify behavior using the ``verify_options``\n    keyword argument\n\n    Files with an attribute of config, doc, ghost, license or readme in the\n    package header can be ignored using the ``ignore_types`` keyword argument\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.verify\n        salt '*' pkg.verify httpd\n        salt '*' pkg.verify 'httpd postfix'\n        salt '*' pkg.verify 'httpd postfix' ignore_types=['config','doc']\n        salt '*' pkg.verify 'httpd postfix' verify_options=['nodeps','nosize']\n    "
    return __salt__['lowpkg.verify'](*names, **kwargs)

def group_list():
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2014.1.0\n\n    Lists all groups known by yum on this system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.group_list\n    "
    ret = {'installed': [], 'available': [], 'installed environments': [], 'available environments': [], 'available languages': {}}
    section_map = {'installed groups:': 'installed', 'available groups:': 'available', 'installed environment groups:': 'installed environments', 'available environment groups:': 'available environments', 'available language groups:': 'available languages'}
    out = __salt__['cmd.run_stdout']([_yum(), 'grouplist', 'hidden'], output_loglevel='trace', python_shell=False)
    key = None
    for line in salt.utils.itertools.split(out, '\n'):
        line_lc = line.lower()
        if line_lc == 'done':
            break
        section_lookup = section_map.get(line_lc)
        if section_lookup is not None and section_lookup != key:
            key = section_lookup
            continue
        if key is None:
            continue
        line = line.strip()
        if key != 'available languages':
            ret[key].append(line)
        else:
            match = re.match('(.+) \\[(.+)\\]', line)
            if match:
                (name, lang) = match.groups()
                ret[key][line] = {'name': name, 'language': lang}
    return ret

def group_info(name, expand=False, ignore_groups=None, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2014.1.0\n    .. versionchanged:: 2015.5.10,2015.8.4,2016.3.0,3001\n        The return data has changed. A new key ``type`` has been added to\n        distinguish environment groups from package groups. Also, keys for the\n        group name and group ID have been added. The ``mandatory packages``,\n        ``optional packages``, and ``default packages`` keys have been renamed\n        to ``mandatory``, ``optional``, and ``default`` for accuracy, as\n        environment groups include other groups, and not packages. Finally,\n        this function now properly identifies conditional packages.\n    .. versionchanged:: 3006.2\n        Support for ``fromrepo``, ``enablerepo``, and ``disablerepo`` (as used\n        in :py:func:`pkg.install <salt.modules.yumpkg.install>`) has been\n        added.\n\n    Lists packages belonging to a certain group\n\n    name\n        Name of the group to query\n\n    expand : False\n        If the specified group is an environment group, then the group will be\n        expanded and the return data will include package names instead of\n        group names.\n\n        .. versionadded:: 2016.3.0\n\n    ignore_groups : None\n        This parameter can be used to pass a list of groups to ignore when\n        expanding subgroups. It is used during recursion in order to prevent\n        expanding the same group multiple times.\n\n        .. versionadded:: 3001\n\n    fromrepo\n        Restrict ``yum groupinfo`` to the specified repo(s).\n        (e.g., ``yum --disablerepo='*' --enablerepo='somerepo'``)\n\n        .. versionadded:: 3006.2\n\n    enablerepo (ignored if ``fromrepo`` is specified)\n        Specify a disabled package repository (or repositories) to enable.\n        (e.g., ``yum --enablerepo='somerepo'``)\n\n        .. versionadded:: 3006.2\n\n    disablerepo (ignored if ``fromrepo`` is specified)\n        Specify an enabled package repository (or repositories) to disable.\n        (e.g., ``yum --disablerepo='somerepo'``)\n\n        .. versionadded:: 3006.2\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.group_info 'Perl Support'\n        salt '*' pkg.group_info 'Perl Support' fromrepo=base,updates\n        salt '*' pkg.group_info 'Perl Support' enablerepo=somerepo\n    "
    pkgtypes = ('mandatory', 'optional', 'default', 'conditional')
    ret = {}
    for pkgtype in pkgtypes:
        ret[pkgtype] = set()
    options = _get_options(**{key: val for (key, val) in kwargs.items() if key in ('fromrepo', 'enablerepo', 'disablerepo')})
    cmd = [_yum(), '--quiet'] + options + ['groupinfo', name]
    out = __salt__['cmd.run_stdout'](cmd, output_loglevel='trace', python_shell=False)
    g_info = {}
    for line in salt.utils.itertools.split(out, '\n'):
        try:
            (key, value) = (x.strip() for x in line.split(':'))
            g_info[key.lower()] = value
        except ValueError:
            continue
    if 'environment group' in g_info:
        ret['type'] = 'environment group'
    elif 'group' in g_info:
        ret['type'] = 'package group'
    ret['group'] = g_info.get('environment group') or g_info.get('group')
    ret['id'] = g_info.get('environment-id') or g_info.get('group-id')
    if not ret['group'] and (not ret['id']):
        raise CommandExecutionError(f"Group '{name}' not found")
    ret['description'] = g_info.get('description', '')
    completed_groups = ignore_groups or []
    pkgtypes_capturegroup = '(' + '|'.join(pkgtypes) + ')'
    for pkgtype in pkgtypes:
        target_found = False
        for line in salt.utils.itertools.split(out, '\n'):
            line = line.strip().lstrip(string.punctuation)
            match = re.match(pkgtypes_capturegroup + ' (?:groups|packages):\\s*$', line.lower())
            if match:
                if target_found:
                    break
                else:
                    if match.group(1) == pkgtype:
                        target_found = True
                    continue
            if target_found:
                if expand and ret['type'] == 'environment group':
                    if not line or line in completed_groups:
                        continue
                    log.trace('Adding group "%s" to completed list: %s', line, completed_groups)
                    completed_groups.append(line)
                    expanded = group_info('@' + line, expand=True, ignore_groups=completed_groups)
                    for p_type in pkgtypes:
                        ret[p_type].update(set(expanded[p_type]))
                else:
                    ret[pkgtype].add(line)
    for pkgtype in pkgtypes:
        ret[pkgtype] = sorted(ret[pkgtype])
    return ret

def group_diff(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.1.0\n    .. versionchanged:: 2015.5.10,2015.8.4,2016.3.0\n        Environment groups are now supported. The key names have been renamed,\n        similar to the changes made in :py:func:`pkg.group_info\n        <salt.modules.yumpkg.group_info>`.\n    .. versionchanged:: 3006.2\n        Support for ``fromrepo``, ``enablerepo``, and ``disablerepo`` (as used\n        in :py:func:`pkg.install <salt.modules.yumpkg.install>`) has been\n        added.\n\n    Lists which of a group's packages are installed and which are not\n    installed\n\n    name\n        The name of the group to check\n\n    fromrepo\n        Restrict ``yum groupinfo`` to the specified repo(s).\n        (e.g., ``yum --disablerepo='*' --enablerepo='somerepo'``)\n\n        .. versionadded:: 3006.2\n\n    enablerepo (ignored if ``fromrepo`` is specified)\n        Specify a disabled package repository (or repositories) to enable.\n        (e.g., ``yum --enablerepo='somerepo'``)\n\n        .. versionadded:: 3006.2\n\n    disablerepo (ignored if ``fromrepo`` is specified)\n        Specify an enabled package repository (or repositories) to disable.\n        (e.g., ``yum --disablerepo='somerepo'``)\n\n        .. versionadded:: 3006.2\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.group_diff 'Perl Support'\n        salt '*' pkg.group_diff 'Perl Support' fromrepo=base,updates\n        salt '*' pkg.group_diff 'Perl Support' enablerepo=somerepo\n    "
    pkgtypes = ('mandatory', 'optional', 'default', 'conditional')
    ret = {}
    for pkgtype in pkgtypes:
        ret[pkgtype] = {'installed': [], 'not installed': []}
    pkgs = list_pkgs()
    group_pkgs = group_info(name, expand=True, **kwargs)
    for pkgtype in pkgtypes:
        for member in group_pkgs.get(pkgtype, []):
            if member in pkgs:
                ret[pkgtype]['installed'].append(member)
            else:
                ret[pkgtype]['not installed'].append(member)
    return ret

def group_install(name, skip=(), include=(), **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2014.1.0\n\n    Install the passed package group(s). This is basically a wrapper around\n    :py:func:`pkg.install <salt.modules.yumpkg.install>`, which performs\n    package group resolution for the user. This function is currently\n    considered experimental, and should be expected to undergo changes.\n\n    name\n        Package group to install. To install more than one group, either use a\n        comma-separated list or pass the value as a python list.\n\n        CLI Examples:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.group_install \'Group 1\'\n            salt \'*\' pkg.group_install \'Group 1,Group 2\'\n            salt \'*\' pkg.group_install \'["Group 1", "Group 2"]\'\n\n    skip\n        Packages that would normally be installed by the package group\n        ("default" packages), which should not be installed. Can be passed\n        either as a comma-separated list or a python list.\n\n        CLI Examples:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.group_install \'My Group\' skip=\'foo,bar\'\n            salt \'*\' pkg.group_install \'My Group\' skip=\'["foo", "bar"]\'\n\n    include\n        Packages which are included in a group, which would not normally be\n        installed by a ``yum groupinstall`` ("optional" packages). Note that\n        this will not enforce group membership; if you include packages which\n        are not members of the specified groups, they will still be installed.\n        Can be passed either as a comma-separated list or a python list.\n\n        CLI Examples:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.group_install \'My Group\' include=\'foo,bar\'\n            salt \'*\' pkg.group_install \'My Group\' include=\'["foo", "bar"]\'\n\n    .. note::\n        Because this is essentially a wrapper around pkg.install, any argument\n        which can be passed to pkg.install may also be included here, and it\n        will be passed along wholesale.\n    '
    groups = name.split(',') if isinstance(name, str) else name
    if not groups:
        raise SaltInvocationError('no groups specified')
    elif not isinstance(groups, list):
        raise SaltInvocationError("'groups' must be a list")
    if isinstance(skip, str):
        skip = skip.split(',')
    if not isinstance(skip, (list, tuple)):
        raise SaltInvocationError("'skip' must be a list")
    if isinstance(include, str):
        include = include.split(',')
    if not isinstance(include, (list, tuple)):
        raise SaltInvocationError("'include' must be a list")
    targets = []
    for group in groups:
        group_detail = group_info(group)
        targets.extend(group_detail.get('mandatory', []))
        targets.extend([pkg for pkg in group_detail.get('default', []) if pkg not in skip])
    if include:
        targets.extend(include)
    pkgs = [x for x in targets if x not in list_pkgs()]
    if not pkgs:
        return {}
    return install(pkgs=pkgs, **kwargs)
groupinstall = salt.utils.functools.alias_function(group_install, 'groupinstall')

def list_repos(basedir=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Lists all repos in <basedir> (default: all dirs in `reposdir` yum option).\n\n    Strict parsing of configuration files is the default, this can be disabled\n    using the  ``strict_config`` keyword argument set to False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_repos\n        salt '*' pkg.list_repos basedir=/path/to/dir\n        salt '*' pkg.list_repos basedir=/path/to/dir,/path/to/another/dir strict_config=False\n    "
    strict_parser = kwargs.get('strict_config', True)
    basedirs = _normalize_basedir(basedir, strict_parser)
    repos = {}
    log.debug('Searching for repos in %s', basedirs)
    for bdir in basedirs:
        if not os.path.exists(bdir):
            continue
        for repofile in os.listdir(bdir):
            repopath = f'{bdir}/{repofile}'
            if not repofile.endswith('.repo'):
                continue
            filerepos = _parse_repo_file(repopath, strict_parser)[1]
            for reponame in filerepos:
                repo = filerepos[reponame]
                repo['file'] = repopath
                repos[reponame] = repo
    return repos

def get_repo(repo, basedir=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display a repo from <basedir> (default basedir: all dirs in ``reposdir``\n    yum option).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.get_repo myrepo\n        salt '*' pkg.get_repo myrepo basedir=/path/to/dir\n        salt '*' pkg.get_repo myrepo basedir=/path/to/dir,/path/to/another/dir\n    "
    repos = list_repos(basedir, **kwargs)
    if repo.startswith('copr:'):
        repo = _get_copr_repo(repo)
    repofile = ''
    for list_repo in repos:
        if list_repo == repo:
            repofile = repos[list_repo]['file']
    if repofile:
        strict_parser = kwargs.get('strict_config', True)
        filerepos = _parse_repo_file(repofile, strict_parser)[1]
        return filerepos[repo]
    return {}

def del_repo(repo, basedir=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Delete a repo from <basedir> (default basedir: all dirs in `reposdir` yum\n    option).\n\n    If the .repo file in which the repo exists does not contain any other repo\n    configuration, the file itself will be deleted.\n\n    Strict parsing of configuration files is the default, this can be disabled\n    using the  ``strict_config`` keyword argument set to False\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.del_repo myrepo\n        salt '*' pkg.del_repo myrepo basedir=/path/to/dir strict_config=False\n        salt '*' pkg.del_repo myrepo basedir=/path/to/dir,/path/to/another/dir\n    "
    if repo.startswith('copr:'):
        repo = _get_copr_repo(repo)
    strict_parser = kwargs.get('strict_config', True)
    basedirs = _normalize_basedir(basedir, strict_parser)
    repos = list_repos(basedirs, **kwargs)
    if repo not in repos:
        return f'Error: the {repo} repo does not exist in {basedirs}'
    repofile = ''
    for arepo in repos:
        if arepo == repo:
            repofile = repos[arepo]['file']
    onlyrepo = True
    for arepo in repos:
        if arepo == repo:
            continue
        if repos[arepo]['file'] == repofile:
            onlyrepo = False
    if onlyrepo:
        os.remove(repofile)
        return f'File {repofile} containing repo {repo} has been removed'
    (header, filerepos) = _parse_repo_file(repofile, strict_parser)
    content = header
    for stanza in filerepos.keys():
        if stanza == repo:
            continue
        comments = ''
        if 'comments' in filerepos[stanza].keys():
            comments = salt.utils.pkg.rpm.combine_comments(filerepos[stanza]['comments'])
            del filerepos[stanza]['comments']
        content += f'\n[{stanza}]'
        for line in filerepos[stanza]:
            value = filerepos[stanza][line]
            if isinstance(value, str) and '\n' in value:
                value = '\n '.join(value.split('\n'))
            content += f'\n{line}={value}'
        content += f'\n{comments}\n'
    with salt.utils.files.fopen(repofile, 'w') as fileout:
        fileout.write(salt.utils.stringutils.to_str(content))
    return f'Repo {repo} has been removed from {repofile}'

def mod_repo(repo, basedir=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Modify one or more values for a repo. If the repo does not exist, it will\n    be created, so long as the following values are specified:\n\n    repo\n        name by which the yum refers to the repo\n    name\n        a human-readable name for the repo\n    baseurl\n        the URL for yum to reference\n    mirrorlist\n        the URL for yum to reference\n\n    Key/Value pairs may also be removed from a repo's configuration by setting\n    a key to a blank value. Bear in mind that a name cannot be deleted, and a\n    baseurl can only be deleted if a mirrorlist is specified (or vice versa).\n\n    Strict parsing of configuration files is the default, this can be disabled\n    using the  ``strict_config`` keyword argument set to False\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.mod_repo reponame enabled=1 gpgcheck=1\n        salt '*' pkg.mod_repo reponame basedir=/path/to/dir enabled=1 strict_config=False\n        salt '*' pkg.mod_repo reponame baseurl= mirrorlist=http://host.com/\n    "
    repo_opts = {x: kwargs[x] for x in kwargs if not x.startswith('__') and x not in ('saltenv',)}
    if all((x in repo_opts for x in ('mirrorlist', 'baseurl'))):
        raise SaltInvocationError("Only one of 'mirrorlist' and 'baseurl' can be specified")
    use_copr = False
    if repo.startswith('copr:'):
        copr_name = repo.split(':', 1)[1]
        repo = _get_copr_repo(repo)
        use_copr = True
    todelete = []
    for key in list(repo_opts):
        if repo_opts[key] != 0 and (not repo_opts[key]):
            del repo_opts[key]
            todelete.append(key)
    if 'mirrorlist' in repo_opts:
        todelete.append('baseurl')
    elif 'baseurl' in repo_opts:
        todelete.append('mirrorlist')
    if 'name' in todelete:
        raise SaltInvocationError('The repo name cannot be deleted')
    repos = {}
    strict_parser = kwargs.get('strict_config', True)
    basedirs = _normalize_basedir(basedir, strict_parser)
    repos = list_repos(basedirs, **kwargs)
    repofile = ''
    header = ''
    filerepos = {}
    if repo not in repos:
        newdir = None
        for d in basedirs:
            if os.path.exists(d):
                newdir = d
                break
        if not newdir:
            raise SaltInvocationError('The repo does not exist and needs to be created, but none of the following basedir directories exist: {}'.format(basedirs))
        repofile = f'{newdir}/{repo}.repo'
        if use_copr:
            copr_plugin_name = ''
            if _yum() in ('dnf', 'dnf5'):
                copr_plugin_name = 'dnf-plugins-core'
            else:
                copr_plugin_name = 'yum-plugin-copr'
            if not __salt__['pkg_resource.version'](copr_plugin_name):
                raise SaltInvocationError(f'{copr_plugin_name} must be installed to use COPR')
            out = _call_yum(['copr', 'enable', copr_name, '-y'])
            if out['retcode']:
                raise CommandExecutionError("Unable to add COPR '{}'. '{}' exited with status {!s}: '{}' ".format(copr_name, _yum(), out['retcode'], out['stderr']))
            repos = list_repos(basedirs, **kwargs)
            repofile = repos[repo]['file']
            (header, filerepos) = _parse_repo_file(repofile, strict_parser)
        else:
            repofile = f'{newdir}/{repo}.repo'
            if 'name' not in repo_opts:
                raise SaltInvocationError('The repo does not exist and needs to be created, but a name was not given')
            if 'baseurl' not in repo_opts and 'mirrorlist' not in repo_opts:
                raise SaltInvocationError('The repo does not exist and needs to be created, but either a baseurl or a mirrorlist needs to be given')
            filerepos[repo] = {}
    else:
        repofile = repos[repo]['file']
        (header, filerepos) = _parse_repo_file(repofile, strict_parser)
    if 'baseurl' in todelete:
        if 'mirrorlist' not in repo_opts and 'mirrorlist' not in filerepos[repo]:
            raise SaltInvocationError('Cannot delete baseurl without specifying mirrorlist')
    if 'mirrorlist' in todelete:
        if 'baseurl' not in repo_opts and 'baseurl' not in filerepos[repo]:
            raise SaltInvocationError('Cannot delete mirrorlist without specifying baseurl')
    for key in todelete:
        if key in filerepos[repo].copy().keys():
            del filerepos[repo][key]
    _bool_to_str = lambda x: '1' if x else '0'
    filerepos[repo].update(repo_opts)
    content = header
    for stanza in filerepos.keys():
        comments = salt.utils.pkg.rpm.combine_comments(filerepos[stanza].pop('comments', []))
        content += f'[{stanza}]\n'
        for line in filerepos[stanza].keys():
            value = filerepos[stanza][line]
            if isinstance(value, str) and '\n' in value:
                value = '\n '.join(value.split('\n'))
            content += '{}={}\n'.format(line, value if not isinstance(value, bool) else _bool_to_str(value))
        content += comments + '\n'
    with salt.utils.files.fopen(repofile, 'w') as fileout:
        fileout.write(salt.utils.stringutils.to_str(content))
    return {repofile: filerepos}

def _parse_repo_file(filename, strict_config=True):
    if False:
        return 10
    '\n    Turn a single repo file into a dict\n    '
    parsed = configparser.ConfigParser(strict=strict_config)
    config = {}
    try:
        parsed.read(filename)
    except configparser.MissingSectionHeaderError as err:
        log.error('Failed to parse file %s, error: %s', filename, err.message)
        return ('', {})
    for section in parsed._sections:
        section_dict = dict(parsed._sections[section])
        section_dict.pop('__name__', None)
        config[section] = section_dict
    headers = ''
    section = None
    with salt.utils.files.fopen(filename, 'r') as repofile:
        for line in repofile:
            line = salt.utils.stringutils.to_unicode(line)
            line = line.strip()
            if line.startswith('#'):
                if section is None:
                    headers += line + '\n'
                else:
                    try:
                        comments = config[section].setdefault('comments', [])
                        comments.append(line[1:].lstrip())
                    except KeyError:
                        log.debug('Found comment in %s which does not appear to belong to any repo section: %s', filename, line)
            elif line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
    return (headers, salt.utils.data.decode(config))

def file_list(*packages, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2014.1.0\n\n    List the files that belong to a package. Not specifying any packages will\n    return a list of *every* file on the system's rpm database (not generally\n    recommended).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list httpd\n        salt '*' pkg.file_list httpd postfix\n        salt '*' pkg.file_list\n    "
    return __salt__['lowpkg.file_list'](*packages)

def file_dict(*packages, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2014.1.0\n\n    List the files that belong to a package, grouped by package. Not\n    specifying any packages will return a list of *every* file on the system's\n    rpm database (not generally recommended).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list httpd\n        salt '*' pkg.file_list httpd postfix\n        salt '*' pkg.file_list\n    "
    return __salt__['lowpkg.file_dict'](*packages)

def owner(*paths, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2014.7.0\n\n    Return the name of the package that owns the file. Multiple file paths can\n    be passed. Like :mod:`pkg.version <salt.modules.yumpkg.version>`, if a\n    single path is passed, a string will be returned, and if multiple paths are\n    passed, a dictionary of file/package name pairs will be returned.\n\n    If the file is not owned by a package, or is not present on the minion,\n    then an empty string will be returned for that path.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.owner /usr/bin/apachectl\n        salt '*' pkg.owner /usr/bin/apachectl /etc/httpd/conf/httpd.conf\n    "
    if not paths:
        return ''
    ret = {}
    cmd_prefix = ['rpm', '-qf', '--queryformat', '%{name}']
    for path in paths:
        ret[path] = __salt__['cmd.run_stdout'](cmd_prefix + [path], output_loglevel='trace', python_shell=False)
        if 'not owned' in ret[path].lower():
            ret[path] = ''
    if len(ret) == 1:
        return next(iter(ret.values()))
    return ret

def modified(*packages, **flags):
    if False:
        i = 10
        return i + 15
    "\n    List the modified files that belong to a package. Not specifying any packages\n    will return a list of _all_ modified files on the system's RPM database.\n\n    .. versionadded:: 2015.5.0\n\n    Filtering by flags (True or False):\n\n    size\n        Include only files where size changed.\n\n    mode\n        Include only files which file's mode has been changed.\n\n    checksum\n        Include only files which MD5 checksum has been changed.\n\n    device\n        Include only files which major and minor numbers has been changed.\n\n    symlink\n        Include only files which are symbolic link contents.\n\n    owner\n        Include only files where owner has been changed.\n\n    group\n        Include only files where group has been changed.\n\n    time\n        Include only files where modification time of the file has been\n        changed.\n\n    capabilities\n        Include only files where capabilities differ or not. Note: supported\n        only on newer RPM versions.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.modified\n        salt '*' pkg.modified httpd\n        salt '*' pkg.modified httpd postfix\n        salt '*' pkg.modified httpd owner=True group=False\n    "
    return __salt__['lowpkg.modified'](*packages, **flags)

@salt.utils.decorators.path.which('yumdownloader')
def download(*packages, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2015.5.0\n\n    Download packages to the local disk. Requires ``yumdownloader`` from\n    ``yum-utils`` package.\n\n    .. note::\n\n        ``yum-utils`` will already be installed on the minion if the package\n        was installed from the Fedora / EPEL repositories.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.download httpd\n        salt '*' pkg.download httpd postfix\n    "
    if not packages:
        raise SaltInvocationError('No packages were specified')
    CACHE_DIR = '/var/cache/yum/packages'
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cached_pkgs = os.listdir(CACHE_DIR)
    to_purge = []
    for pkg in packages:
        to_purge.extend([os.path.join(CACHE_DIR, x) for x in cached_pkgs if x.startswith(f'{pkg}-')])
    for purge_target in set(to_purge):
        log.debug('Removing cached package %s', purge_target)
        try:
            os.unlink(purge_target)
        except OSError as exc:
            log.error('Unable to remove %s: %s', purge_target, exc)
    cmd = ['yumdownloader', '-q', f'--destdir={CACHE_DIR}']
    cmd.extend(packages)
    __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False)
    ret = {}
    for dld_result in os.listdir(CACHE_DIR):
        if not dld_result.endswith('.rpm'):
            continue
        pkg_name = None
        pkg_file = None
        for query_pkg in packages:
            if dld_result.startswith(f'{query_pkg}-'):
                pkg_name = query_pkg
                pkg_file = dld_result
                break
        if pkg_file is not None:
            ret[pkg_name] = os.path.join(CACHE_DIR, pkg_file)
    if not ret:
        raise CommandExecutionError('Unable to download any of the following packages: {}'.format(', '.join(packages)))
    failed = [x for x in packages if x not in ret]
    if failed:
        ret['_error'] = 'The following package(s) failed to download: {}'.format(', '.join(failed))
    return ret

def diff(*paths, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return a formatted diff between current files and original in a package.\n    NOTE: this function includes all files (configuration and not), but does\n    not work on binary content.\n\n    :param path: Full path to the installed file\n    :return: Difference string or raises and exception if examined file is binary.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.diff /etc/apache2/httpd.conf /etc/sudoers\n    "
    ret = {}
    pkg_to_paths = {}
    for pth in paths:
        pth_pkg = __salt__['lowpkg.owner'](pth)
        if not pth_pkg:
            ret[pth] = os.path.exists(pth) and 'Not managed' or 'N/A'
        else:
            if pkg_to_paths.get(pth_pkg) is None:
                pkg_to_paths[pth_pkg] = []
            pkg_to_paths[pth_pkg].append(pth)
    if pkg_to_paths:
        local_pkgs = __salt__['pkg.download'](*pkg_to_paths.keys())
        for (pkg, files) in pkg_to_paths.items():
            for path in files:
                ret[path] = __salt__['lowpkg.diff'](local_pkgs[pkg]['path'], path) or 'Unchanged'
    return ret

def _get_patches(installed_only=False):
    if False:
        while True:
            i = 10
    '\n    List all known patches in repos.\n    '
    patches = {}
    cmd = [_yum(), '--quiet', 'updateinfo', 'list', 'all']
    ret = __salt__['cmd.run_stdout'](cmd, python_shell=False)
    parsing_errors = False
    for line in salt.utils.itertools.split(ret, os.linesep):
        try:
            (inst, advisory_id, sev, pkg) = re.match('([i|\\s]) ([^\\s]+) +([^\\s]+) +([^\\s]+)', line).groups()
        except Exception:
            parsing_errors = True
            continue
        if advisory_id not in patches:
            patches[advisory_id] = {'installed': True if inst == 'i' else False, 'summary': [pkg]}
        else:
            patches[advisory_id]['summary'].append(pkg)
            if inst != 'i':
                patches[advisory_id]['installed'] = False
    if parsing_errors:
        log.warning("Skipped some unexpected output while running '%s' to list patches. Please check output", ' '.join(cmd))
    if installed_only:
        patches = {k: v for (k, v) in patches.items() if v['installed']}
    return patches

def list_patches(refresh=False, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2017.7.0\n\n    List all known advisory patches from available repos.\n\n    refresh\n        force a refresh if set to True.\n        If set to False (default) it depends on yum if a refresh is\n        executed.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_patches\n    "
    if refresh:
        refresh_db()
    return _get_patches()

def list_installed_patches(**kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2017.7.0\n\n    List installed advisory patches on the system.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_installed_patches\n    "
    return _get_patches(installed_only=True)

def services_need_restart(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 3003\n\n    List services that use files which have been changed by the\n    package manager. It might be needed to restart them.\n\n    Requires systemd.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.services_need_restart\n    "
    if _yum() != 'dnf':
        raise CommandExecutionError('dnf is required to list outdated services.')
    if not salt.utils.systemd.booted(__context__):
        raise CommandExecutionError('systemd is required to list outdated services.')
    cmd = ['dnf', '--quiet', 'needs-restarting']
    dnf_output = __salt__['cmd.run_stdout'](cmd, python_shell=False)
    if not dnf_output:
        return []
    services = set()
    for line in dnf_output.split('\n'):
        (pid, has_delim, _) = line.partition(':')
        if has_delim:
            service = salt.utils.systemd.pid_to_service(pid.strip())
            if service:
                services.add(service)
    return list(services)