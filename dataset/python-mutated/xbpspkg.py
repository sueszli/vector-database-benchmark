"""
Package support for XBPS package manager (used by VoidLinux)

.. versionadded:: 2016.11.0
"""
import glob
import logging
import os
import re
import salt.utils.data
import salt.utils.decorators as decorators
import salt.utils.files
import salt.utils.path
import salt.utils.pkg
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError, MinionError
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Set the virtual pkg module if the os is Void and xbps-install found\n    '
    if __grains__['os'] in 'Void' and _check_xbps():
        return __virtualname__
    return (False, 'Missing dependency: xbps-install')

@decorators.memoize
def _check_xbps():
    if False:
        i = 10
        return i + 15
    '\n    Looks to see if xbps-install is present on the system, return full path\n    '
    return salt.utils.path.which('xbps-install')

@decorators.memoize
def _get_version():
    if False:
        i = 10
        return i + 15
    '\n    Get the xbps version\n    '
    version_string = __salt__['cmd.run']([_check_xbps(), '--version'], output_loglevel='trace')
    if version_string is None:
        return False
    VERSION_MATCH = re.compile('(?:XBPS:[\\s]+)([\\d.]+)(?:[\\s]+.*)')
    version_match = VERSION_MATCH.search(version_string)
    if not version_match:
        return False
    return version_match.group(1).split('.')

def _rehash():
    if False:
        while True:
            i = 10
    '\n    Recomputes internal hash table for the PATH variable.\n    Used whenever a new command is created during the current\n    session.\n    '
    shell = __salt__['environ.get']('SHELL')
    if shell.split('/')[-1] in ('csh', 'tcsh'):
        __salt__['cmd.run']('rehash', output_loglevel='trace')

def list_pkgs(versions_as_list=False, **kwargs):
    if False:
        while True:
            i = 10
    "\n    List the packages currently installed as a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    cmd = 'xbps-query -l'
    ret = {}
    out = __salt__['cmd.run'](cmd, output_loglevel='trace')
    for line in out.splitlines():
        if not line:
            continue
        try:
            (pkg, ver) = line.split(None)[1].rsplit('-', 1)
        except ValueError:
            log.error('xbps-query: Unexpected formatting in line: "%s"', line)
        __salt__['pkg_resource.add_pkg'](ret, pkg, ver)
    __salt__['pkg_resource.sort_pkglist'](ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def list_upgrades(refresh=True, **kwargs):
    if False:
        print('Hello World!')
    "\n    Check whether or not an upgrade is available for all packages\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    refresh = salt.utils.data.is_true(refresh)
    if refresh:
        refresh_db()
    ret = {}
    cmd = 'xbps-install -un'
    out = __salt__['cmd.run'](cmd, output_loglevel='trace')
    for line in out.splitlines():
        if not line:
            continue
        pkg = 'base-system'
        ver = 'NonNumericValueIsError'
        try:
            (pkg, ver) = line.split()[0].rsplit('-', 1)
        except (ValueError, IndexError):
            log.error('xbps-query: Unexpected formatting in line: "%s"', line)
            continue
        log.trace('pkg=%s version=%s', pkg, ver)
        ret[pkg] = ver
    return ret

def latest_version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    string will be returned for that package.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n    "
    refresh = salt.utils.data.is_true(kwargs.pop('refresh', True))
    if not names:
        return ''
    if refresh:
        refresh_db()
    ret = {}
    for name in names:
        ret[name] = ''
    cmd = ['xbps-install', '-un']
    cmd.extend(names)
    out = __salt__['cmd.run'](cmd, ignore_retcode=True, output_loglevel='trace')
    for line in out.splitlines():
        if not line:
            continue
        if line.find(' is up to date.') != -1:
            continue
        try:
            (pkg, ver) = line.split()[0].rsplit('-', 1)
        except (ValueError, IndexError):
            log.error('xbps-query: Unexpected formatting in line: "%s"', line)
            continue
        log.trace('pkg=%s version=%s', pkg, ver)
        if pkg in names:
            ret[pkg] = ver
    if len(names) == 1:
        return ret[names[0]]
    return ret
available_version = latest_version

def upgrade_available(name, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Check whether or not an upgrade is available for a given package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return latest_version(name) != ''

def refresh_db(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Update list of available packages from installed repos\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n    "
    salt.utils.pkg.clear_rtag(__opts__)
    cmd = 'xbps-install -Sy'
    call = __salt__['cmd.run_all'](cmd, output_loglevel='trace')
    if call['retcode'] != 0:
        comment = ''
        if 'stderr' in call:
            comment += call['stderr']
        raise CommandExecutionError(comment)
    return True

def version(*names, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def upgrade(refresh=True, **kwargs):
    if False:
        return 10
    "\n    Run a full system upgrade\n\n    refresh\n        Whether or not to refresh the package database before installing.\n        Default is `True`.\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {'<package>':  {'old': '<old-version>',\n                        'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n    "
    old = list_pkgs()
    cmd = ['xbps-install', '-{}yu'.format('S' if refresh else '')]
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    return ret

def install(name=None, refresh=False, fromrepo=None, pkgs=None, sources=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Install the passed package\n\n    name\n        The name of the package to be installed.\n\n    refresh\n        Whether or not to refresh the package database before installing.\n\n    fromrepo\n        Specify a package repository (url) to install from.\n\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install from a software repository. Must be\n        passed as a python list.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo","bar"]\'\n\n    sources\n        A list of packages to install. Must be passed as a list of dicts,\n        with the keys being package names, and the values being the source URI\n        or local path to the package.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install sources=\'[{"foo": "salt://foo.deb"},{"bar": "salt://bar.deb"}]\'\n\n    Return a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install <package name>\n    '
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, sources, **kwargs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    if pkg_type != 'repository':
        log.error('xbps: pkg_type "%s" not supported.', pkg_type)
        return {}
    cmd = ['xbps-install']
    if refresh:
        cmd.append('-S')
    if fromrepo:
        cmd.append('--repository={}'.format(fromrepo))
    cmd.append('-y')
    cmd.extend(pkg_params)
    old = list_pkgs()
    __salt__['cmd.run'](cmd, output_loglevel='trace')
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    _rehash()
    return salt.utils.data.compare_dicts(old, new)

def remove(name=None, pkgs=None, recursive=True, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    name\n        The name of the package to be deleted.\n\n    recursive\n        Also remove dependent packages (not required elsewhere).\n        Default mode: enabled.\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    Returns a list containing the removed packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name> [recursive=False]\n        salt \'*\' pkg.remove <package1>,<package2>,<package3> [recursive=False]\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\' [recursive=False]\n    '
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    old = list_pkgs()
    targets = [x for x in pkg_params if x in old]
    if not targets:
        return {}
    cmd = ['xbps-remove', '-y']
    if recursive:
        cmd.append('-R')
    cmd.extend(targets)
    __salt__['cmd.run'](cmd, output_loglevel='trace')
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    return salt.utils.data.compare_dicts(old, new)

def list_repos(**kwargs):
    if False:
        print('Hello World!')
    "\n    List all repos known by XBPS\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' pkg.list_repos\n    "
    repos = {}
    out = __salt__['cmd.run']('xbps-query -L', output_loglevel='trace')
    for line in out.splitlines():
        repo = {}
        if not line:
            continue
        try:
            (nb, url, rsa) = line.strip().split(' ', 2)
        except ValueError:
            log.error('Problem parsing xbps-query: Unexpected formatting in line: "%s"', line)
        repo['nbpkg'] = int(nb) if nb.isdigit() else 0
        repo['url'] = url
        repo['rsasigned'] = True if rsa == '(RSA signed)' else False
        repos[repo['url']] = repo
    return repos

def get_repo(repo, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Display information about the repo.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.get_repo 'repo-url'\n    "
    repos = list_repos()
    if repo in repos:
        return repos[repo]
    return {}

def _locate_repo_files(repo, rewrite=False):
    if False:
        i = 10
        return i + 15
    '\n    Find what file a repo is called in.\n\n    Helper function for add_repo() and del_repo()\n\n    repo\n        url of the repo to locate (persistent).\n\n    rewrite\n        Whether to remove matching repository settings during this process.\n\n    Returns a list of absolute paths.\n    '
    ret_val = []
    files = []
    conf_dirs = ['/etc/xbps.d/', '/usr/share/xbps.d/']
    name_glob = '*.conf'
    regex = re.compile('\\s*repository\\s*=\\s*' + repo + '/?\\s*(#.*)?$')
    for cur_dir in conf_dirs:
        files.extend(glob.glob(cur_dir + name_glob))
    for filename in files:
        write_buff = []
        with salt.utils.files.fopen(filename, 'r') as cur_file:
            for line in cur_file:
                if regex.match(salt.utils.stringutils.to_unicode(line)):
                    ret_val.append(filename)
                else:
                    write_buff.append(line)
        if rewrite and filename in ret_val:
            if write_buff:
                with salt.utils.files.fopen(filename, 'w') as rewrite_file:
                    rewrite_file.writelines(write_buff)
            else:
                os.remove(filename)
    return ret_val

def add_repo(repo, conffile='/usr/share/xbps.d/15-saltstack.conf'):
    if False:
        i = 10
        return i + 15
    "\n    Add an XBPS repository to the system.\n\n    repo\n        url of repo to add (persistent).\n\n    conffile\n        path to xbps conf file to add this repo\n        default: /usr/share/xbps.d/15-saltstack.conf\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.add_repo <repo url> [conffile=/path/to/xbps/repo.conf]\n    "
    if not _locate_repo_files(repo):
        try:
            with salt.utils.files.fopen(conffile, 'a+') as conf_file:
                conf_file.write(salt.utils.stringutils.to_str('repository={}\n'.format(repo)))
        except OSError:
            return False
    return True

def del_repo(repo, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove an XBPS repository from the system.\n\n    repo\n        url of repo to remove (persistent).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.del_repo <repo url>\n    "
    try:
        _locate_repo_files(repo, rewrite=True)
    except OSError:
        return False
    else:
        return True