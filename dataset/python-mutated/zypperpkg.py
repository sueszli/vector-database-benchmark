"""
Package support for openSUSE via the zypper package manager

:depends: - ``rpm`` Python module.  Install with ``zypper install rpm-python``

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.

"""
import configparser
import datetime
import errno
import fnmatch
import logging
import os
import re
import time
import urllib.parse
from xml.dom import minidom as dom
from xml.parsers.expat import ExpatError
import salt.utils.data
import salt.utils.environment
import salt.utils.event
import salt.utils.files
import salt.utils.functools
import salt.utils.path
import salt.utils.pkg
import salt.utils.pkg.rpm
import salt.utils.stringutils
import salt.utils.systemd
import salt.utils.versions
from salt.exceptions import CommandExecutionError, MinionError, SaltInvocationError
from salt.utils.versions import LooseVersion
if salt.utils.files.is_fcntl_available():
    import fcntl
log = logging.getLogger(__name__)
HAS_ZYPP = False
ZYPP_HOME = '/etc/zypp'
LOCKS = f'{ZYPP_HOME}/locks'
REPOS = f'{ZYPP_HOME}/repos.d'
DEFAULT_PRIORITY = 99
PKG_ARCH_SEPARATOR = '.'
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Set the virtual pkg module if the os is openSUSE\n    '
    if __grains__.get('os_family', '') != 'Suse':
        return (False, 'Module zypper: non SUSE OS not supported by zypper package manager')
    if not salt.utils.path.which('zypper'):
        return (False, 'Module zypper: zypper package manager not found')
    return __virtualname__

class _Zypper:
    """
    Zypper parallel caller.
    Validates the result and either raises an exception or reports an error.
    Allows serial zypper calls (first came, first won).
    """
    SUCCESS_EXIT_CODES = {0: 'Successful run of zypper with no special info.', 100: 'Patches are available for installation.', 101: 'Security patches are available for installation.', 102: 'Installation successful, reboot required.', 103: 'Installation successful, restart of the package manager itself required.'}
    WARNING_EXIT_CODES = {6: 'No repositories are defined.', 7: 'The ZYPP library is locked.', 106: 'Some repository had to be disabled temporarily because it failed to refresh. You should check your repository configuration (e.g. zypper ref -f).', 107: 'Installation basically succeeded, but some of the packages %post install scripts returned an error. These packages were successfully unpacked to disk and are registered in the rpm database, but due to the failed install script they may not work as expected. The failed scripts output might reveal what actually went wrong. Any scripts output is also logged to /var/log/zypp/history.'}
    LOCK_EXIT_CODE = 7
    XML_DIRECTIVES = ['-x', '--xmlout']
    ZYPPER_LOCK = '/var/run/zypp.pid'
    RPM_LOCK = '/var/lib/rpm/.rpm.lock'
    TAG_RELEASED = 'zypper/released'
    TAG_BLOCKED = 'zypper/blocked'

    def __init__(self):
        if False:
            return 10
        '\n        Constructor\n        '
        self._reset()

    def _reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Resets values of the call setup.\n\n        :return:\n        '
        self.__cmd = ['zypper', '--non-interactive']
        self.__exit_code = 0
        self.__call_result = dict()
        self.__error_msg = ''
        self.__env = salt.utils.environment.get_module_environment(globals())
        self.__xml = False
        self.__no_lock = False
        self.__no_raise = False
        self.__refresh = False
        self.__ignore_repo_failure = False
        self.__systemd_scope = False
        self.__root = None
        self.__called = False

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        :param args:\n        :param kwargs:\n        :return:\n        '
        if self.__called:
            self._reset()
        if 'no_repo_failure' in kwargs:
            self.__ignore_repo_failure = kwargs['no_repo_failure']
        if 'systemd_scope' in kwargs:
            self.__systemd_scope = kwargs['systemd_scope']
        if 'root' in kwargs:
            self.__root = kwargs['root']
        return self

    def __getattr__(self, item):
        if False:
            i = 10
            return i + 15
        '\n        Call configurator.\n\n        :param item:\n        :return:\n        '
        if self.__called:
            self._reset()
        if item == 'xml':
            self.__xml = True
        elif item == 'nolock':
            self.__no_lock = True
        elif item == 'noraise':
            self.__no_raise = True
        elif item == 'refreshable':
            self.__refresh = True
        elif item == 'call':
            return self.__call
        else:
            return self.__dict__[item]
        if self.__no_lock:
            self.__no_lock = not self.__refresh
        return self

    @property
    def exit_code(self):
        if False:
            return 10
        return self.__exit_code

    @exit_code.setter
    def exit_code(self, exit_code):
        if False:
            for i in range(10):
                print('nop')
        self.__exit_code = int(exit_code or '0')

    @property
    def error_msg(self):
        if False:
            print('Hello World!')
        return self.__error_msg

    @error_msg.setter
    def error_msg(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if self._is_error():
            self.__error_msg = msg and os.linesep.join(msg) or "Check Zypper's logs."

    @property
    def stdout(self):
        if False:
            print('Hello World!')
        return self.__call_result.get('stdout', '')

    @property
    def stderr(self):
        if False:
            print('Hello World!')
        return self.__call_result.get('stderr', '')

    @property
    def pid(self):
        if False:
            print('Hello World!')
        return self.__call_result.get('pid', '')

    def _is_error(self):
        if False:
            while True:
                i = 10
        '\n        Is this is an error code?\n\n        :return:\n        '
        if self.exit_code:
            msg = self.SUCCESS_EXIT_CODES.get(self.exit_code)
            if msg:
                log.info(msg)
            msg = self.WARNING_EXIT_CODES.get(self.exit_code)
            if msg:
                log.warning(msg)
        return self.exit_code not in self.SUCCESS_EXIT_CODES and self.exit_code not in self.WARNING_EXIT_CODES

    def _is_zypper_lock(self):
        if False:
            return 10
        '\n        Is this is a lock error code?\n\n        :return:\n        '
        return self.exit_code == self.LOCK_EXIT_CODE

    def _is_rpm_lock(self):
        if False:
            while True:
                i = 10
        '\n        Is this an RPM lock error?\n        '
        if salt.utils.files.is_fcntl_available():
            if self.exit_code > 0 and os.path.exists(self.RPM_LOCK):
                with salt.utils.files.fopen(self.RPM_LOCK, mode='w+') as rfh:
                    try:
                        fcntl.lockf(rfh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except OSError as err:
                        if err.errno == errno.EAGAIN:
                            return True
                    else:
                        fcntl.lockf(rfh, fcntl.LOCK_UN)
        return False

    def _is_xml_mode(self):
        if False:
            print('Hello World!')
        "\n        Is Zypper's output is in XML format?\n\n        :return:\n        "
        return [itm for itm in self.XML_DIRECTIVES if itm in self.__cmd] and True or False

    def _check_result(self):
        if False:
            while True:
                i = 10
        '\n        Check and set the result of a zypper command. In case of an error,\n        either raise a CommandExecutionError or extract the error.\n\n        result\n            The result of a zypper command called with cmd.run_all\n        '
        if not self.__call_result:
            raise CommandExecutionError('No output result from Zypper?')
        self.exit_code = self.__call_result['retcode']
        if self._is_zypper_lock() or self._is_rpm_lock():
            return False
        if self._is_error():
            _error_msg = list()
            if not self._is_xml_mode():
                msg = self.__call_result['stderr'] and self.__call_result['stderr'].strip() or ''
                msg += self.__call_result['stdout'] and self.__call_result['stdout'].strip() or ''
                if msg:
                    _error_msg.append(msg)
            else:
                try:
                    doc = dom.parseString(self.__call_result['stdout'])
                except ExpatError as err:
                    log.error(err)
                    doc = None
                if doc:
                    msg_nodes = doc.getElementsByTagName('message')
                    for node in msg_nodes:
                        if node.getAttribute('type') == 'error':
                            _error_msg.append(node.childNodes[0].nodeValue)
                elif self.__call_result['stderr'].strip():
                    _error_msg.append(self.__call_result['stderr'].strip())
            self.error_msg = _error_msg
        return True

    def __call(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Call Zypper.\n\n        :param state:\n        :return:\n        '
        self.__called = True
        if self.__xml:
            self.__cmd.append('--xmlout')
        if not self.__refresh and '--no-refresh' not in args:
            self.__cmd.append('--no-refresh')
        if self.__root:
            self.__cmd.extend(['--root', self.__root])
        self.__cmd.extend(args)
        kwargs['output_loglevel'] = 'trace'
        kwargs['python_shell'] = False
        kwargs['env'] = self.__env.copy()
        if self.__no_lock:
            kwargs['env']['ZYPP_READONLY_HACK'] = '1'
        was_blocked = False
        while True:
            cmd = []
            if self.__systemd_scope:
                cmd.extend(['systemd-run', '--scope'])
            cmd.extend(self.__cmd)
            log.debug('Calling Zypper: %s', ' '.join(cmd))
            self.__call_result = __salt__['cmd.run_all'](cmd, **kwargs)
            if self._check_result():
                break
            if self._is_zypper_lock():
                self._handle_zypper_lock_file()
            if self._is_rpm_lock():
                self._handle_rpm_lock_file()
            was_blocked = True
        if was_blocked:
            __salt__['event.fire_master']({'success': not self.error_msg, 'info': self.error_msg or 'Zypper has been released'}, self.TAG_RELEASED)
        if self.error_msg and (not self.__no_raise) and (not self.__ignore_repo_failure):
            raise CommandExecutionError(f'Zypper command failure: {self.error_msg}')
        return self._is_xml_mode() and dom.parseString(salt.utils.stringutils.to_str(self.__call_result['stdout'])) or self.__call_result['stdout']

    def _handle_zypper_lock_file(self):
        if False:
            return 10
        if os.path.exists(self.ZYPPER_LOCK):
            try:
                with salt.utils.files.fopen(self.ZYPPER_LOCK) as rfh:
                    data = __salt__['ps.proc_info'](int(rfh.readline()), attrs=['pid', 'name', 'cmdline', 'create_time'])
                    data['cmdline'] = ' '.join(data['cmdline'])
                    data['info'] = 'Blocking process created at {}.'.format(datetime.datetime.utcfromtimestamp(data['create_time']).isoformat())
                    data['success'] = True
            except Exception as err:
                data = {'info': 'Unable to retrieve information about blocking process: {}'.format(err), 'success': False}
        else:
            data = {'info': 'Zypper is locked, but no Zypper lock has been found.', 'success': False}
        if not data['success']:
            log.debug('Unable to collect data about blocking process.')
        else:
            log.debug('Collected data about blocking process.')
        __salt__['event.fire_master'](data, self.TAG_BLOCKED)
        log.debug('Fired a Zypper blocked event to the master with the data: %s', data)
        log.debug('Waiting 5 seconds for Zypper gets released...')
        time.sleep(5)

    def _handle_rpm_lock_file(self):
        if False:
            for i in range(10):
                print('nop')
        data = {'info': 'RPM is temporarily locked.', 'success': True}
        __salt__['event.fire_master'](data, self.TAG_BLOCKED)
        log.debug('Fired an RPM blocked event to the master with the data: %s', data)
        log.debug('Waiting 5 seconds for RPM to get released...')
        time.sleep(5)
__zypper__ = _Zypper()

class Wildcard:
    """
    .. versionadded:: 2017.7.0

    Converts string wildcard to a zypper query.
    Example:
       '1.2.3.4*' is '1.2.3.4.whatever.is.here' and is equal to:
       '1.2.3.4 >= and < 1.2.3.5'

    :param ptn: Pattern
    :return: Query range
    """
    Z_OP = ['<', '<=', '=', '>=', '>']

    def __init__(self, zypper):
        if False:
            while True:
                i = 10
        '\n        :type zypper: a reference to an instance of a _Zypper class.\n        '
        self.name = None
        self.version = None
        self.zypper = zypper
        self._attr_solvable_version = 'edition'
        self._op = None

    def __call__(self, pkg_name, pkg_version):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert a string wildcard to a zypper query.\n\n        :param pkg_name:\n        :param pkg_version:\n        :return:\n        '
        if pkg_version:
            self.name = pkg_name
            self._set_version(pkg_version)
            versions = sorted((LooseVersion(vrs) for vrs in self._get_scope_versions(self._get_available_versions())))
            return versions and '{}{}'.format(self._op or '', versions[-1]) or None

    def _get_available_versions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get available versions of the package.\n        :return:\n        '
        solvables = self.zypper.nolock.xml.call('se', '-xv', self.name).getElementsByTagName('solvable')
        if not solvables:
            raise CommandExecutionError(f"No packages found matching '{self.name}'")
        return sorted({slv.getAttribute(self._attr_solvable_version) for slv in solvables if slv.getAttribute(self._attr_solvable_version)})

    def _get_scope_versions(self, pkg_versions):
        if False:
            return 10
        '\n        Get available difference between next possible matches.\n\n        :return:\n        '
        get_in_versions = []
        for p_version in pkg_versions:
            if fnmatch.fnmatch(p_version, self.version):
                get_in_versions.append(p_version)
        return get_in_versions

    def _set_version(self, version):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stash operator from the version, if any.\n\n        :return:\n        '
        if not version:
            return
        exact_version = re.sub('[<>=+]*', '', version)
        self._op = version.replace(exact_version, '') or None
        if self._op and self._op not in self.Z_OP:
            raise CommandExecutionError(f'Zypper do not supports operator "{self._op}".')
        self.version = exact_version

def _systemd_scope():
    if False:
        while True:
            i = 10
    return salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True)

def _clean_cache():
    if False:
        while True:
            i = 10
    '\n    Clean cached results\n    '
    keys = []
    for cache_name in ['pkg.list_pkgs', 'pkg.list_provides']:
        for contextkey in __context__:
            if contextkey.startswith(cache_name):
                keys.append(contextkey)
    for key in keys:
        __context__.pop(key, None)

def list_upgrades(refresh=True, root=None, **kwargs):
    if False:
        return 10
    "\n    List all available package upgrades on this system\n\n    refresh\n        force a refresh if set to True (default).\n        If set to False it depends on zypper if a refresh is\n        executed.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    if refresh:
        refresh_db(root)
    ret = dict()
    cmd = ['list-updates']
    if 'fromrepo' in kwargs:
        repos = kwargs['fromrepo']
        if isinstance(repos, str):
            repos = [repos]
        for repo in repos:
            cmd.extend(['--repo', repo if isinstance(repo, str) else str(repo)])
        log.debug('Targeting repos: %s', repos)
    for update_node in __zypper__(root=root).nolock.xml.call(*cmd).getElementsByTagName('update'):
        if update_node.getAttribute('kind') == 'package':
            ret[update_node.getAttribute('name')] = update_node.getAttribute('edition')
    return ret
list_updates = salt.utils.functools.alias_function(list_upgrades, 'list_updates')

def info_installed(*names, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the information of the named package(s), installed on the system.\n\n    :param names:\n        Names of the packages to get information about.\n\n    :param attr:\n        Comma-separated package attributes. If no 'attr' is specified, all available attributes returned.\n\n        Valid attributes are:\n            version, vendor, release, build_date, build_date_time_t, install_date, install_date_time_t,\n            build_host, group, source_rpm, arch, epoch, size, license, signature, packager, url,\n            summary, description.\n\n    :param errors:\n        Handle RPM field errors. If 'ignore' is chosen, then various mistakes are simply ignored and omitted\n        from the texts or strings. If 'report' is chonen, then a field with a mistake is not returned, instead\n        a 'N/A (broken)' (not available, broken) text is placed.\n\n        Valid attributes are:\n            ignore, report\n\n    :param all_versions:\n        Include information for all versions of the packages installed on the minion.\n\n    :param root:\n        Operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.info_installed <package1>\n        salt '*' pkg.info_installed <package1> <package2> <package3> ...\n        salt '*' pkg.info_installed <package1> <package2> <package3> all_versions=True\n        salt '*' pkg.info_installed <package1> attr=version,vendor all_versions=True\n        salt '*' pkg.info_installed <package1> <package2> <package3> ... attr=version,vendor\n        salt '*' pkg.info_installed <package1> <package2> <package3> ... attr=version,vendor errors=ignore\n        salt '*' pkg.info_installed <package1> <package2> <package3> ... attr=version,vendor errors=report\n    "
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

def info_available(*names, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the information of the named package available for the system.\n\n    refresh\n        force a refresh if set to True (default).\n        If set to False it depends on zypper if a refresh is\n        executed or not.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.info_available <package1>\n        salt '*' pkg.info_available <package1> <package2> <package3> ...\n    "
    ret = {}
    if not names:
        return ret
    else:
        names = sorted(list(set(names)))
    root = kwargs.get('root', None)
    if kwargs.get('refresh', True):
        refresh_db(root)
    pkg_info = []
    batch = names[:]
    batch_size = 200
    while batch:
        pkg_info.extend(re.split('Information for package*', __zypper__(root=root).nolock.call('info', '-t', 'package', *batch[:batch_size])))
        batch = batch[batch_size:]
    for pkg_data in pkg_info:
        nfo = {}
        for line in [data for data in pkg_data.split('\n') if ':' in data]:
            if line.startswith('-----'):
                continue
            kw = [data.strip() for data in line.split(':', 1)]
            if len(kw) == 2 and kw[1]:
                nfo[kw[0].lower()] = kw[1]
        if nfo.get('name'):
            name = nfo.pop('name')
            ret[name] = nfo
        if nfo.get('status'):
            nfo['status'] = nfo.get('status')
        if nfo.get('installed'):
            nfo['installed'] = nfo.get('installed').lower().startswith('yes')
    return ret

def parse_arch(name):
    if False:
        return 10
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
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    If the latest version of a given package is already installed, an empty\n    dict will be returned for that package.\n\n    refresh\n        force a refresh if set to True (default).\n        If set to False it depends on zypper if a refresh is\n        executed or not.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n    "
    ret = dict()
    if not names:
        return ret
    names = sorted(list(set(names)))
    package_info = info_available(*names, **kwargs)
    for name in names:
        pkg_info = package_info.get(name, {})
        status = pkg_info.get('status', '').lower()
        if status.find('not installed') > -1 or status.find('out-of-date') > -1:
            ret[name] = pkg_info.get('version')
        else:
            ret[name] = ''
    if len(names) == 1 and ret:
        return ret[names[0]]
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def upgrade_available(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check whether or not an upgrade is available for a given package\n\n    refresh\n        force a refresh if set to True (default).\n        If set to False it depends on zypper if a refresh is\n        executed or not.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return not not latest_version(name, **kwargs)

def version(*names, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Returns a string representing the package version or an empty dict if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs) or {}

def version_cmp(ver1, ver2, ignore_epoch=False, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2015.5.4\n\n    Do a cmp-style comparison on two packages. Return -1 if ver1 < ver2, 0 if\n    ver1 == ver2, and 1 if ver1 > ver2. Return None if there was a problem\n    making the comparison.\n\n    ignore_epoch : False\n        Set to ``True`` to ignore the epoch when comparing versions\n\n        .. versionadded:: 2015.8.10,2016.3.2\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version_cmp '0.2-001' '0.2.0.1-002'\n    "
    return __salt__['lowpkg.version_cmp'](ver1, ver2, ignore_epoch=ignore_epoch)

def _list_pkgs_from_context(versions_as_list, contextkey, attr):
    if False:
        return 10
    '\n    Use pkg list from __context__\n    '
    return __salt__['pkg_resource.format_pkg_list'](__context__[contextkey], versions_as_list, attr)

def list_pkgs(versions_as_list=False, root=None, includes=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    List the packages currently installed as a dict. By default, the dict\n    contains versions as a comma separated string::\n\n        {\'<package_name>\': \'<version>[,<version>...]\'}\n\n    versions_as_list:\n        If set to true, the versions are provided as a list\n\n        {\'<package_name>\': [\'<version>\', \'<version>\']}\n\n    root:\n        operate on a different root directory.\n\n    includes:\n        List of types of packages to include (package, patch, pattern, product)\n        By default packages are always included\n\n    attr:\n        If a list of package attributes is specified, returned value will\n        contain them in addition to version, eg.::\n\n        {\'<package_name>\': [{\'version\' : \'version\', \'arch\' : \'arch\'}]}\n\n        Valid attributes are: ``epoch``, ``version``, ``release``, ``arch``,\n        ``install_date``, ``install_date_time_t``.\n\n        If ``all`` is specified, all valid attributes will be returned.\n\n            .. versionadded:: 2018.3.0\n\n    removed:\n        not supported\n\n    purge_desired:\n        not supported\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.list_pkgs\n        salt \'*\' pkg.list_pkgs attr=version,arch\n        salt \'*\' pkg.list_pkgs attr=\'["version", "arch"]\'\n    '
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    attr = kwargs.get('attr')
    if attr is not None and attr != 'all':
        attr = salt.utils.args.split_input(attr)
    includes = includes if includes else []
    contextkey = f'pkg.list_pkgs_{root}_{includes}'
    if contextkey in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list, contextkey, attr)
    ret = {}
    cmd = ['rpm']
    if root:
        cmd.extend(['--root', root])
    cmd.extend(['-qa', '--queryformat', salt.utils.pkg.rpm.QUERYFORMAT.replace('%{REPOID}', '(none)') + '\n'])
    output = __salt__['cmd.run'](cmd, python_shell=False, output_loglevel='trace')
    for line in output.splitlines():
        pkginfo = salt.utils.pkg.rpm.parse_pkginfo(line, osarch=__grains__['osarch'])
        if pkginfo:
            pkgver = pkginfo.version
            epoch = None
            release = None
            if ':' in pkgver:
                (epoch, pkgver) = pkgver.split(':', 1)
            if '-' in pkgver:
                (pkgver, release) = pkgver.split('-', 1)
            all_attr = {'epoch': epoch, 'version': pkgver, 'release': release, 'arch': pkginfo.arch, 'install_date': pkginfo.install_date, 'install_date_time_t': pkginfo.install_date_time_t}
            __salt__['pkg_resource.add_pkg'](ret, pkginfo.name, all_attr)
    _ret = {}
    for pkgname in ret:
        if pkgname.startswith('gpg-pubkey'):
            continue
        _ret[pkgname] = sorted(ret[pkgname], key=lambda d: d['version'])
    for include in includes:
        if include == 'product':
            products = list_products(all=False, root=root)
            for product in products:
                extended_name = '{}:{}'.format(include, product['name'])
                _ret[extended_name] = [{'epoch': product['epoch'], 'version': product['version'], 'release': product['release'], 'arch': product['arch'], 'install_date': None, 'install_date_time_t': None}]
        if include in ('pattern', 'patch'):
            if include == 'pattern':
                elements = list_installed_patterns(root=root)
            elif include == 'patch':
                elements = list_installed_patches(root=root)
            else:
                elements = []
            for element in elements:
                extended_name = f'{include}:{element}'
                info = info_available(extended_name, refresh=False, root=root)
                _ret[extended_name] = [{'epoch': None, 'version': info[element]['version'], 'release': None, 'arch': info[element]['arch'], 'install_date': None, 'install_date_time_t': None}]
    __context__[contextkey] = _ret
    return __salt__['pkg_resource.format_pkg_list'](__context__[contextkey], versions_as_list, attr)

def list_repo_pkgs(*args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2017.7.5,2018.3.1\n\n    Returns all available packages. Optionally, package names (and name globs)\n    can be passed and the results will be filtered to packages matching those\n    names. This is recommended as it speeds up the function considerably.\n\n    This function can be helpful in discovering the version or repo to specify\n    in a :mod:`pkg.installed <salt.states.pkg.installed>` state.\n\n    The return data will be a dictionary mapping package names to a list of\n    version numbers, ordered from newest to oldest. If ``byrepo`` is set to\n    ``True``, then the return dictionary will contain repository names at the\n    top level, and each repository will map packages to lists of version\n    numbers. For example:\n\n    .. code-block:: python\n\n        # With byrepo=False (default)\n        {\n            'bash': ['4.3-83.3.1',\n                     '4.3-82.6'],\n            'vim': ['7.4.326-12.1']\n        }\n        {\n            'OSS': {\n                'bash': ['4.3-82.6'],\n                'vim': ['7.4.326-12.1']\n            },\n            'OSS Update': {\n                'bash': ['4.3-83.3.1']\n            }\n        }\n\n    fromrepo : None\n        Only include results from the specified repo(s). Multiple repos can be\n        specified, comma-separated.\n\n    byrepo : False\n        When ``True``, the return data for each package will be organized by\n        repository.\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_repo_pkgs\n        salt '*' pkg.list_repo_pkgs foo bar baz\n        salt '*' pkg.list_repo_pkgs 'python2-*' byrepo=True\n        salt '*' pkg.list_repo_pkgs 'python2-*' fromrepo='OSS Updates'\n    "
    byrepo = kwargs.pop('byrepo', False)
    fromrepo = kwargs.pop('fromrepo', '') or ''
    ret = {}
    targets = [arg if isinstance(arg, str) else str(arg) for arg in args]

    def _is_match(pkgname):
        if False:
            while True:
                i = 10
        '\n        When package names are passed to a zypper search, they will be matched\n        anywhere in the package name. This makes sure that only exact or\n        fnmatch matches are identified.\n        '
        if not args:
            return True
        for target in targets:
            if fnmatch.fnmatch(pkgname, target):
                return True
        return False
    root = kwargs.get('root') or None
    for node in __zypper__(root=root).xml.call('se', '-s', *targets).getElementsByTagName('solvable'):
        pkginfo = dict(node.attributes.items())
        try:
            if pkginfo['kind'] != 'package':
                continue
            reponame = pkginfo['repository']
            if fromrepo and reponame != fromrepo:
                continue
            pkgname = pkginfo['name']
            pkgversion = pkginfo['edition']
        except KeyError:
            continue
        else:
            if _is_match(pkgname):
                repo_dict = ret.setdefault(reponame, {})
                version_list = repo_dict.setdefault(pkgname, set())
                version_list.add(pkgversion)
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

def _get_configured_repos(root=None):
    if False:
        print('Hello World!')
    '\n    Get all the info about repositories from the configurations.\n    '
    repos = os.path.join(root, os.path.relpath(REPOS, os.path.sep)) if root else REPOS
    repos_cfg = configparser.ConfigParser()
    if os.path.exists(repos):
        repos_cfg.read([repos + '/' + fname for fname in os.listdir(repos) if fname.endswith('.repo')])
    else:
        log.warning('Repositories not found in %s', repos)
    return repos_cfg

def _get_repo_info(alias, repos_cfg=None, root=None):
    if False:
        print('Hello World!')
    '\n    Get one repo meta-data.\n    '
    try:
        meta = dict((repos_cfg or _get_configured_repos(root=root)).items(alias))
        meta['alias'] = alias
        for (key, val) in meta.items():
            if val in ['0', '1']:
                meta[key] = int(meta[key]) == 1
            elif val == 'NONE':
                meta[key] = None
        return meta
    except (ValueError, configparser.NoSectionError):
        return {}

def get_repo(repo, root=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Display a repo.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.get_repo alias\n    "
    return _get_repo_info(repo, root=root)

def list_repos(root=None, **kwargs):
    if False:
        return 10
    "\n    Lists all repos.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n       salt '*' pkg.list_repos\n    "
    repos_cfg = _get_configured_repos(root=root)
    all_repos = {}
    for alias in repos_cfg.sections():
        all_repos[alias] = _get_repo_info(alias, repos_cfg=repos_cfg, root=root)
    return all_repos

def del_repo(repo, root=None):
    if False:
        return 10
    "\n    Delete a repo.\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.del_repo alias\n    "
    repos_cfg = _get_configured_repos(root=root)
    for alias in repos_cfg.sections():
        if alias == repo:
            doc = __zypper__(root=root).xml.call('rr', '--loose-auth', '--loose-query', alias)
            msg = doc.getElementsByTagName('message')
            if doc.getElementsByTagName('progress') and msg:
                return {repo: True, 'message': msg[0].childNodes[0].nodeValue}
    raise CommandExecutionError(f"Repository '{repo}' not found.")

def mod_repo(repo, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Modify one or more values for a repo. If the repo does not exist, it will\n    be created, so long as the following values are specified:\n\n    repo or alias\n        alias by which Zypper refers to the repo\n\n    url, mirrorlist or baseurl\n        the URL for Zypper to reference\n\n    enabled\n        Enable or disable (True or False) repository,\n        but do not remove if disabled.\n\n    name\n        This is used as the descriptive name value in the repo file.\n\n    refresh\n        Enable or disable (True or False) auto-refresh of the repository.\n\n    cache\n        Enable or disable (True or False) RPM files caching.\n\n    gpgcheck\n        Enable or disable (True or False) GPG check for this repository.\n\n    gpgautoimport : False\n        If set to True, automatically trust and import public GPG key for\n        the repository.\n\n    root\n        operate on a different root directory.\n\n    Key/Value pairs may also be removed from a repo's configuration by setting\n    a key to a blank value. Bear in mind that a name cannot be deleted, and a\n    URL can only be deleted if a ``mirrorlist`` is specified (or vice versa).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.mod_repo alias alias=new_alias\n        salt '*' pkg.mod_repo alias url= mirrorlist=http://host.com/\n    "
    root = kwargs.get('root') or None
    repos_cfg = _get_configured_repos(root=root)
    added = False
    if repo not in repos_cfg.sections():
        url = kwargs.get('url', kwargs.get('mirrorlist', kwargs.get('baseurl')))
        if not url:
            raise CommandExecutionError("Repository '{}' not found, and neither 'baseurl' nor 'mirrorlist' was specified".format(repo))
        if not urllib.parse.urlparse(url).scheme:
            raise CommandExecutionError("Repository '{}' not found and URL for baseurl/mirrorlist is malformed".format(repo))
        for alias in repos_cfg.sections():
            repo_meta = _get_repo_info(alias, repos_cfg=repos_cfg, root=root)
            new_url = urllib.parse.urlparse(url)
            if not new_url.path:
                new_url = urllib.parse.urlparse.ParseResult(scheme=new_url.scheme, netloc=new_url.netloc, path='/', params=new_url.params, query=new_url.query, fragment=new_url.fragment)
            base_url = urllib.parse.urlparse(repo_meta['baseurl'])
            if new_url == base_url:
                raise CommandExecutionError(f"Repository '{repo}' already exists as '{alias}'.")
        __zypper__(root=root).xml.call('ar', url, repo)
        repos_cfg = _get_configured_repos(root=root)
        if repo not in repos_cfg.sections():
            raise CommandExecutionError("Failed add new repository '{}' for unspecified reason. Please check zypper logs.".format(repo))
        added = True
    repo_info = _get_repo_info(repo, root=root)
    if not added and 'baseurl' in kwargs and (not kwargs['baseurl'] == repo_info['baseurl']):
        repo_info.update(kwargs)
        repo_info.setdefault('cache', False)
        del_repo(repo, root=root)
        return mod_repo(repo, root=root, **repo_info)
    cmd_opt = []
    global_cmd_opt = []
    call_refresh = False
    if 'enabled' in kwargs:
        cmd_opt.append(kwargs['enabled'] and '--enable' or '--disable')
    if 'refresh' in kwargs:
        cmd_opt.append(kwargs['refresh'] and '--refresh' or '--no-refresh')
    if 'cache' in kwargs:
        cmd_opt.append(kwargs['cache'] and '--keep-packages' or '--no-keep-packages')
    if 'gpgcheck' in kwargs:
        cmd_opt.append(kwargs['gpgcheck'] and '--gpgcheck' or '--no-gpgcheck')
    if 'priority' in kwargs:
        cmd_opt.append('--priority={}'.format(kwargs.get('priority', DEFAULT_PRIORITY)))
    if 'humanname' in kwargs:
        salt.utils.versions.warn_until(3009, "Passing 'humanname' to 'mod_repo' is deprecated, slated for removal in {version}. Please use 'name' instead.")
        cmd_opt.append("--name='{}'".format(kwargs.get('humanname')))
    if 'name' in kwargs:
        cmd_opt.append('--name')
        cmd_opt.append(kwargs.get('name'))
    if kwargs.get('gpgautoimport') is True:
        global_cmd_opt.append('--gpg-auto-import-keys')
        call_refresh = True
    if cmd_opt:
        cmd_opt = global_cmd_opt + ['mr'] + cmd_opt + [repo]
        __zypper__(root=root).refreshable.xml.call(*cmd_opt)
    comment = None
    if call_refresh:
        refresh_opts = global_cmd_opt + ['refresh'] + [repo]
        __zypper__(root=root).xml.call(*refresh_opts)
    elif not added and (not cmd_opt):
        comment = 'Specified arguments did not result in modification of repo'
    repo = get_repo(repo, root=root)
    if comment:
        repo['comment'] = comment
    return repo

def refresh_db(force=None, root=None):
    if False:
        i = 10
        return i + 15
    '\n    Trigger a repository refresh by calling ``zypper refresh``. Refresh will run\n    with ``--force`` if the "force=True" flag is passed on the CLI or\n    ``refreshdb_force`` is set to ``true`` in the pillar. The CLI option\n    overrides the pillar setting.\n\n    It will return a dict::\n\n        {\'<database name>\': Bool}\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.refresh_db [force=true|false]\n\n    Pillar Example:\n\n    .. code-block:: yaml\n\n       zypper:\n         refreshdb_force: false\n    '
    salt.utils.pkg.clear_rtag(__opts__)
    ret = {}
    refresh_opts = ['refresh']
    if force is None:
        force = __pillar__.get('zypper', {}).get('refreshdb_force', True)
    if force:
        refresh_opts.append('--force')
    out = __zypper__(root=root).refreshable.call(*refresh_opts)
    for line in out.splitlines():
        if not line:
            continue
        if line.strip().startswith('Repository') and "'" in line:
            try:
                key = line.split("'")[1].strip()
                if 'is up to date' in line:
                    ret[key] = False
            except IndexError:
                continue
        elif line.strip().startswith('Building') and "'" in line:
            key = line.split("'")[1].strip()
            if 'done' in line:
                ret[key] = True
    return ret

def _find_types(pkgs):
    if False:
        for i in range(10):
            print('nop')
    'Form a package names list, find prefixes of packages types.'
    return sorted({pkg.split(':', 1)[0] for pkg in pkgs if len(pkg.split(':', 1)) == 2})

def install(name=None, refresh=False, fromrepo=None, pkgs=None, sources=None, downloadonly=None, skip_verify=False, version=None, ignore_repo_failure=False, no_recommends=False, root=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any zypper commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Install the passed package(s), add refresh=True to force a \'zypper refresh\'\n    before package is installed.\n\n    name\n        The name of the package to be installed. Note that this parameter is\n        ignored if either ``pkgs`` or ``sources`` is passed. Additionally,\n        please note that this option can only be used to install packages from\n        a software repository. To install a package file manually, use the\n        ``sources`` option.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n\n    refresh\n        force a refresh if set to True.\n        If set to False (default) it depends on zypper if a refresh is\n        executed.\n\n    fromrepo\n        Specify a package repository to install from.\n\n    downloadonly\n        Only download the packages, do not install.\n\n    skip_verify\n        Skip the GPG verification check (e.g., ``--no-gpg-checks``)\n\n    version\n        Can be either a version number, or the combination of a comparison\n        operator (<, >, <=, >=, =) and a version number (ex. \'>1.2.3-4\').\n        This parameter is ignored if ``pkgs`` or ``sources`` is passed.\n\n    resolve_capabilities\n        If this option is set to True zypper will take capabilities into\n        account. In this case names which are just provided by a package\n        will get installed. Default is False.\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install from a software repository. Must be\n        passed as a python list. A specific version number can be specified\n        by using a single-element dict representing the package and its\n        version. As with the ``version`` parameter above, comparison operators\n        can be used to target a specific version of a package.\n\n        CLI Examples:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n            salt \'*\' pkg.install pkgs=\'["foo", {"bar": "1.2.3-4"}]\'\n            salt \'*\' pkg.install pkgs=\'["foo", {"bar": "<1.2.3-4"}]\'\n\n    sources\n        A list of RPM packages to install. Must be passed as a list of dicts,\n        with the keys being package names, and the values being the source URI\n        or local path to the package.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install sources=\'[{"foo": "salt://foo.rpm"},{"bar": "salt://bar.rpm"}]\'\n\n    ignore_repo_failure\n        Zypper returns error code 106 if one of the repositories are not available for various reasons.\n        In case to set strict check, this parameter needs to be set to True. Default: False.\n\n    no_recommends\n        Do not install recommended packages, only required ones.\n\n    root\n        operate on a different root directory.\n\n    diff_attr:\n        If a list of package attributes is specified, returned value will\n        contain them, eg.::\n\n            {\'<package>\': {\n                \'old\': {\n                    \'version\': \'<old-version>\',\n                    \'arch\': \'<old-arch>\'},\n\n                \'new\': {\n                    \'version\': \'<new-version>\',\n                    \'arch\': \'<new-arch>\'}}}\n\n        Valid attributes are: ``epoch``, ``version``, ``release``, ``arch``,\n        ``install_date``, ``install_date_time_t``.\n\n        If ``all`` is specified, all valid attributes will be returned.\n\n        .. versionadded:: 2018.3.0\n\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    If an attribute list is specified in ``diff_attr``, the dict will also contain\n    any specified attribute, eg.::\n\n        {\'<package>\': {\n            \'old\': {\n                \'version\': \'<old-version>\',\n                \'arch\': \'<old-arch>\'},\n\n            \'new\': {\n                \'version\': \'<new-version>\',\n                \'arch\': \'<new-arch>\'}}}\n    '
    if refresh:
        refresh_db(root)
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, sources, **kwargs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if not pkg_params:
        return {}
    version_num = Wildcard(__zypper__(root=root))(name, version)
    if version_num:
        if pkgs is None and sources is None:
            pkg_params = {name: version_num}
        else:
            log.warning('"version" parameter will be ignored for multiple package targets')
    if pkg_type == 'repository':
        targets = []
        for (param, version_num) in pkg_params.items():
            if version_num is None:
                log.debug('targeting package: %s', param)
                targets.append(param)
            else:
                (prefix, verstr) = salt.utils.pkg.split_comparison(version_num)
                if not prefix:
                    prefix = '='
                target = f'{param}{prefix}{verstr}'
                log.debug('targeting package: %s', target)
                targets.append(target)
    elif pkg_type == 'advisory':
        targets = []
        cur_patches = list_patches(root=root)
        for advisory_id in pkg_params:
            if advisory_id not in cur_patches:
                raise CommandExecutionError(f'Advisory id "{advisory_id}" not found')
            else:
                targets.append(advisory_id)
    else:
        targets = pkg_params
    diff_attr = kwargs.get('diff_attr')
    includes = _find_types(targets)
    old = list_pkgs(attr=diff_attr, root=root, includes=includes) if not downloadonly else list_downloaded(root)
    downgrades = []
    if fromrepo:
        fromrepoopt = ['--force', '--force-resolution', '--from', fromrepo]
        log.info("Targeting repo '%s'", fromrepo)
    else:
        fromrepoopt = ''
    cmd_install = ['install', '--auto-agree-with-licenses']
    cmd_install.append(kwargs.get('resolve_capabilities') and '--capability' or '--name')
    if not refresh:
        cmd_install.insert(0, '--no-refresh')
    if skip_verify:
        cmd_install.insert(0, '--no-gpg-checks')
    if downloadonly:
        cmd_install.append('--download-only')
    if fromrepo:
        cmd_install.extend(fromrepoopt)
    if no_recommends:
        cmd_install.append('--no-recommends')
    errors = []
    if pkg_type == 'advisory':
        targets = [f'patch:{t}' for t in targets]
    systemd_scope = _systemd_scope()
    while targets:
        cmd = cmd_install + targets[:500]
        targets = targets[500:]
        for line in __zypper__(no_repo_failure=ignore_repo_failure, systemd_scope=systemd_scope, root=root).call(*cmd).splitlines():
            match = re.match("^The selected package '([^']+)'.+has lower version", line)
            if match:
                downgrades.append(match.group(1))
    while downgrades:
        cmd = cmd_install + ['--force'] + downgrades[:500]
        downgrades = downgrades[500:]
        __zypper__(no_repo_failure=ignore_repo_failure, root=root).call(*cmd)
    _clean_cache()
    new = list_pkgs(attr=diff_attr, root=root, includes=includes) if not downloadonly else list_downloaded(root)
    ret = salt.utils.data.compare_dicts(old, new)
    if includes:
        _clean_cache()
    if errors:
        raise CommandExecutionError('Problem encountered {} package(s)'.format('downloading' if downloadonly else 'installing'), info={'errors': errors, 'changes': ret})
    return ret

def upgrade(name=None, pkgs=None, refresh=True, dryrun=False, dist_upgrade=False, fromrepo=None, novendorchange=False, skip_verify=False, no_recommends=False, root=None, diff_attr=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any zypper commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Run a full system upgrade, a zypper upgrade\n\n    name\n        The name of the package to be installed. Note that this parameter is\n        ignored if ``pkgs`` is passed or if ``dryrun`` is set to True.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install name=<package name>\n\n    pkgs\n        A list of packages to install from a software repository. Must be\n        passed as a python list. Note that this parameter is ignored if\n        ``dryrun`` is set to True.\n\n        CLI Examples:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n\n    refresh\n        force a refresh if set to True (default).\n        If set to False it depends on zypper if a refresh is\n        executed.\n\n    dryrun\n        If set to True, it creates a debug solver log file and then perform\n        a dry-run upgrade (no changes are made). Default: False\n\n    dist_upgrade\n        Perform a system dist-upgrade. Default: False\n\n    fromrepo\n        Specify a list of package repositories to upgrade from. Default: None\n\n    novendorchange\n        If set to True, no allow vendor changes. Default: False\n\n    skip_verify\n        Skip the GPG verification check (e.g., ``--no-gpg-checks``)\n\n    no_recommends\n        Do not install recommended packages, only required ones.\n\n    root\n        Operate on a different root directory.\n\n    diff_attr:\n        If a list of package attributes is specified, returned value will\n        contain them, eg.::\n\n            {\'<package>\': {\n                \'old\': {\n                    \'version\': \'<old-version>\',\n                    \'arch\': \'<old-arch>\'},\n\n                \'new\': {\n                    \'version\': \'<new-version>\',\n                    \'arch\': \'<new-arch>\'}}}\n\n        Valid attributes are: ``epoch``, ``version``, ``release``, ``arch``,\n        ``install_date``, ``install_date_time_t``.\n\n        If ``all`` is specified, all valid attributes will be returned.\n\n        .. versionadded:: 3006.0\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {\'<package>\':  {\'old\': \'<old-version>\',\n                        \'new\': \'<new-version>\'}}\n\n    If an attribute list is specified in ``diff_attr``, the dict will also contain\n    any specified attribute, eg.::\n\n    .. code-block:: python\n\n        {\'<package>\': {\n            \'old\': {\n                \'version\': \'<old-version>\',\n                \'arch\': \'<old-arch>\'},\n\n            \'new\': {\n                \'version\': \'<new-version>\',\n                \'arch\': \'<new-arch>\'}}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.upgrade\n        salt \'*\' pkg.upgrade name=mypackage\n        salt \'*\' pkg.upgrade pkgs=\'["package1", "package2"]\'\n        salt \'*\' pkg.upgrade dist_upgrade=True fromrepo=\'["MyRepoName"]\' novendorchange=True\n        salt \'*\' pkg.upgrade dist_upgrade=True dryrun=True\n    '
    cmd_update = (['dist-upgrade'] if dist_upgrade else ['update']) + ['--auto-agree-with-licenses']
    if skip_verify:
        cmd_update.insert(0, '--no-gpg-checks')
    if refresh:
        refresh_db(root)
    if dryrun:
        cmd_update.append('--dry-run')
    if fromrepo:
        if isinstance(fromrepo, str):
            fromrepo = [fromrepo]
        for repo in fromrepo:
            cmd_update.extend(['--from' if dist_upgrade else '--repo', repo])
        log.info('Targeting repos: %s', fromrepo)
    if dist_upgrade:
        if novendorchange:
            if __grains__['osrelease_info'][0] > 11:
                cmd_update.append('--no-allow-vendor-change')
                log.info('Disabling vendor changes')
            else:
                log.warning('Disabling vendor changes is not supported on this Zypper version')
        if no_recommends:
            cmd_update.append('--no-recommends')
            log.info('Disabling recommendations')
        if dryrun:
            log.info('Executing debugsolver and performing a dry-run dist-upgrade')
            __zypper__(systemd_scope=_systemd_scope(), root=root).noraise.call(*cmd_update + ['--debug-solver'])
    elif name or pkgs:
        try:
            (pkg_params, _) = __salt__['pkg_resource.parse_targets'](name=name, pkgs=pkgs, sources=None, **kwargs)
            if pkg_params:
                cmd_update.extend(pkg_params.keys())
        except MinionError as exc:
            raise CommandExecutionError(exc)
    old = list_pkgs(root=root, attr=diff_attr)
    __zypper__(systemd_scope=_systemd_scope(), root=root).noraise.call(*cmd_update)
    _clean_cache()
    new = list_pkgs(root=root, attr=diff_attr)
    ret = salt.utils.data.compare_dicts(old, new)
    if __zypper__.exit_code not in __zypper__.SUCCESS_EXIT_CODES:
        result = {'retcode': __zypper__.exit_code, 'stdout': __zypper__.stdout, 'stderr': __zypper__.stderr, 'pid': __zypper__.pid}
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    if dryrun:
        ret = (__zypper__.stdout + os.linesep + __zypper__.stderr).strip()
    return ret

def _uninstall(name=None, pkgs=None, root=None):
    if False:
        i = 10
        return i + 15
    '\n    Remove and purge do identical things but with different Zypper commands,\n    this function performs the common logic.\n    '
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    includes = _find_types(pkg_params.keys())
    old = list_pkgs(root=root, includes=includes)
    targets = []
    for target in pkg_params:
        if target in old and pkg_params[target] in old[target].split(','):
            targets.append(target + '-' + pkg_params[target])
        elif target in old and (not pkg_params[target]):
            targets.append(target)
    if not targets:
        return {}
    systemd_scope = _systemd_scope()
    errors = []
    while targets:
        __zypper__(systemd_scope=systemd_scope, root=root).call('remove', *targets[:500])
        targets = targets[500:]
    _clean_cache()
    new = list_pkgs(root=root, includes=includes)
    ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        raise CommandExecutionError('Problem encountered removing package(s)', info={'errors': errors, 'changes': ret})
    return ret

def normalize_name(name):
    if False:
        print('Hello World!')
    "\n    Strips the architecture from the specified package name, if necessary.\n    Circumstances where this would be done include:\n\n    * If the arch is 32 bit and the package name ends in a 32-bit arch.\n    * If the arch matches the OS arch, or is ``noarch``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.normalize_name zsh.x86_64\n    "
    try:
        arch = name.rsplit('.', 1)[-1]
        if arch not in salt.utils.pkg.rpm.ARCHES + ('noarch',):
            return name
    except ValueError:
        return name
    if arch in (__grains__['osarch'], 'noarch') or salt.utils.pkg.rpm.check_32(arch, osarch=__grains__['osarch']):
        return name[:-(len(arch) + 1)]
    return name

def remove(name=None, pkgs=None, root=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any zypper commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Remove packages with ``zypper -n remove``\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    root\n        Operate on a different root directory.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    return _uninstall(name=name, pkgs=pkgs, root=root)

def purge(name=None, pkgs=None, root=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any zypper commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Recursively remove a package and all dependencies which were installed\n    with it, this will call a ``zypper -n remove -u``\n\n    name\n        The name of the package to be deleted.\n\n\n    Multiple Package Options:\n\n    pkgs\n        A list of packages to delete. Must be passed as a python list. The\n        ``name`` parameter will be ignored if this option is passed.\n\n    root\n        Operate on a different root directory.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    return _uninstall(name=name, pkgs=pkgs, root=root)

def list_holds(pattern=None, full=True, root=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 3005\n\n    List information on locked packages.\n\n    .. note::\n        This function returns the computed output of ``list_locks``\n        to show exact locked packages.\n\n    pattern\n        Regular expression used to match the package name\n\n    full : True\n        Show the full hold definition including version and epoch. Set to\n        ``False`` to return just the name of the package(s) being held.\n\n    root\n        Operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_holds\n        salt '*' pkg.list_holds full=False\n    "
    locks = list_locks(root=root)
    ret = []
    inst_pkgs = {}
    for (solv_name, lock) in locks.items():
        if lock.get('type', 'package') != 'package':
            continue
        try:
            found_pkgs = search(solv_name, root=root, match=None if '*' in solv_name else 'exact', case_sensitive=lock.get('case_sensitive', 'on') == 'on', installed_only=True, details=True)
        except CommandExecutionError:
            continue
        if found_pkgs:
            for pkg in found_pkgs:
                if pkg not in inst_pkgs:
                    inst_pkgs.update(info_installed(pkg, root=root, attr='edition,epoch', all_versions=True))
    ptrn_re = re.compile(f'{pattern}-\\S+') if pattern else None
    for (pkg_name, pkg_editions) in inst_pkgs.items():
        for pkg_info in pkg_editions:
            pkg_ret = '{}-{}:{}.*'.format(pkg_name, pkg_info.get('epoch', 0), pkg_info.get('edition')) if full else pkg_name
            if pkg_ret not in ret and (not ptrn_re or ptrn_re.match(pkg_ret)):
                ret.append(pkg_ret)
    return ret

def list_locks(root=None):
    if False:
        while True:
            i = 10
    "\n    List current package locks.\n\n    root\n        operate on a different root directory.\n\n    Return a dict containing the locked package with attributes::\n\n        {'<package>': {'case_sensitive': '<case_sensitive>',\n                       'match_type': '<match_type>'\n                       'type': '<type>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_locks\n    "
    locks = {}
    _locks = os.path.join(root, os.path.relpath(LOCKS, os.path.sep)) if root else LOCKS
    try:
        with salt.utils.files.fopen(_locks) as fhr:
            items = salt.utils.stringutils.to_unicode(fhr.read()).split('\n\n')
            for meta in [item.split('\n') for item in items]:
                lock = {}
                for element in [el for el in meta if el]:
                    if ':' in element:
                        lock.update(dict([tuple((i.strip() for i in element.split(':', 1)))]))
                if lock.get('solvable_name'):
                    locks[lock.pop('solvable_name')] = lock
    except OSError:
        pass
    except Exception:
        log.warning('Detected a problem when accessing %s', _locks)
    return locks

def clean_locks(root=None):
    if False:
        while True:
            i = 10
    "\n    Remove unused locks that do not currently (with regard to repositories\n    used) lock any package.\n\n    root\n        Operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.clean_locks\n    "
    LCK = 'removed'
    out = {LCK: 0}
    locks = os.path.join(root, os.path.relpath(LOCKS, os.path.sep)) if root else LOCKS
    if not os.path.exists(locks):
        return out
    for node in __zypper__(root=root).xml.call('cl').getElementsByTagName('message'):
        text = node.childNodes[0].nodeValue.lower()
        if text.startswith(LCK):
            out[LCK] = text.split(' ')[1]
            break
    return out

def unhold(name=None, pkgs=None, root=None, **kwargs):
    if False:
        return 10
    '\n    .. versionadded:: 3003\n\n    Remove a package hold.\n\n    name\n        A package name to unhold, or a comma-separated list of package names to\n        unhold.\n\n    pkgs\n        A list of packages to unhold.  The ``name`` parameter will be ignored if\n        this option is passed.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.unhold <package name>\n        salt \'*\' pkg.unhold <package1>,<package2>,<package3>\n        salt \'*\' pkg.unhold pkgs=\'["foo", "bar"]\'\n    '
    ret = {}
    if not name and (not pkgs):
        raise CommandExecutionError('Name or packages must be specified.')
    targets = []
    if pkgs:
        targets.extend(pkgs)
    else:
        targets.append(name)
    locks = list_locks(root=root)
    removed = []
    for target in targets:
        version = None
        if isinstance(target, dict):
            (target, version) = next(iter(target.items()))
        ret[target] = {'name': target, 'changes': {}, 'result': True, 'comment': ''}
        if locks.get(target):
            lock_ver = None
            if 'version' in locks.get(target):
                lock_ver = locks.get(target)['version']
                lock_ver = lock_ver.lstrip('= ')
            if version and lock_ver != version:
                ret[target]['result'] = False
                ret[target]['comment'] = 'Unable to unhold package {} as it is held with the other version.'.format(target)
            else:
                removed.append(target if not lock_ver else f'{target}={lock_ver}')
                ret[target]['changes']['new'] = ''
                ret[target]['changes']['old'] = 'hold'
                ret[target]['comment'] = f'Package {target} is no longer held.'
        else:
            ret[target]['comment'] = f'Package {target} was already unheld.'
    if removed:
        __zypper__(root=root).call('rl', *removed)
    return ret

def hold(name=None, pkgs=None, root=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionadded:: 3003\n\n    Add a package hold.  Specify one of ``name`` and ``pkgs``.\n\n    name\n        A package name to hold, or a comma-separated list of package names to\n        hold.\n\n    pkgs\n        A list of packages to hold.  The ``name`` parameter will be ignored if\n        this option is passed.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.hold <package name>\n        salt \'*\' pkg.hold <package1>,<package2>,<package3>\n        salt \'*\' pkg.hold pkgs=\'["foo", "bar"]\'\n    '
    ret = {}
    if not name and (not pkgs):
        raise CommandExecutionError('Name or packages must be specified.')
    targets = []
    if pkgs:
        targets.extend(pkgs)
    else:
        targets.append(name)
    locks = list_locks(root=root)
    added = []
    for target in targets:
        version = None
        if isinstance(target, dict):
            (target, version) = next(iter(target.items()))
        ret[target] = {'name': target, 'changes': {}, 'result': True, 'comment': ''}
        if not locks.get(target):
            added.append(target if not version else f'{target}={version}')
            ret[target]['changes']['new'] = 'hold'
            ret[target]['changes']['old'] = ''
            ret[target]['comment'] = f'Package {target} is now being held.'
        else:
            ret[target]['comment'] = 'Package {} is already set to be held.'.format(target)
    if added:
        __zypper__(root=root).call('al', *added)
    return ret

def verify(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    Runs an rpm -Va on a system, and returns the results in a dict\n\n    Files with an attribute of config, doc, ghost, license or readme in the\n    package header can be ignored using the ``ignore_types`` keyword argument.\n\n    The root parameter can also be passed via the keyword argument.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.verify\n        salt '*' pkg.verify httpd\n        salt '*' pkg.verify 'httpd postfix'\n        salt '*' pkg.verify 'httpd postfix' ignore_types=['config','doc']\n    "
    return __salt__['lowpkg.verify'](*names, **kwargs)

def file_list(*packages, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List the files that belong to a package. Not specifying any packages will\n    return a list of *every* file on the system's rpm database (not generally\n    recommended).\n\n    The root parameter can also be passed via the keyword argument.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list httpd\n        salt '*' pkg.file_list httpd postfix\n        salt '*' pkg.file_list\n    "
    return __salt__['lowpkg.file_list'](*packages, **kwargs)

def file_dict(*packages, **kwargs):
    if False:
        print('Hello World!')
    "\n    List the files that belong to a package, grouped by package. Not\n    specifying any packages will return a list of *every* file on the system's\n    rpm database (not generally recommended).\n\n    The root parameter can also be passed via the keyword argument.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.file_list httpd\n        salt '*' pkg.file_list httpd postfix\n        salt '*' pkg.file_list\n    "
    return __salt__['lowpkg.file_dict'](*packages, **kwargs)

def modified(*packages, **flags):
    if False:
        while True:
            i = 10
    "\n    List the modified files that belong to a package. Not specifying any packages\n    will return a list of _all_ modified files on the system's RPM database.\n\n    .. versionadded:: 2015.5.0\n\n    Filtering by flags (True or False):\n\n    size\n        Include only files where size changed.\n\n    mode\n        Include only files which file's mode has been changed.\n\n    checksum\n        Include only files which MD5 checksum has been changed.\n\n    device\n        Include only files which major and minor numbers has been changed.\n\n    symlink\n        Include only files which are symbolic link contents.\n\n    owner\n        Include only files where owner has been changed.\n\n    group\n        Include only files where group has been changed.\n\n    time\n        Include only files where modification time of the file has been changed.\n\n    capabilities\n        Include only files where capabilities differ or not. Note: supported only on newer RPM versions.\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.modified\n        salt '*' pkg.modified httpd\n        salt '*' pkg.modified httpd postfix\n        salt '*' pkg.modified httpd owner=True group=False\n    "
    return __salt__['lowpkg.modified'](*packages, **flags)

def owner(*paths, **kwargs):
    if False:
        print('Hello World!')
    "\n    Return the name of the package that owns the file. Multiple file paths can\n    be passed. If a single path is passed, a string will be returned,\n    and if multiple paths are passed, a dictionary of file/package name\n    pairs will be returned.\n\n    If the file is not owned by a package, or is not present on the minion,\n    then an empty string will be returned for that path.\n\n    The root parameter can also be passed via the keyword argument.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.owner /usr/bin/apachectl\n        salt '*' pkg.owner /usr/bin/apachectl /etc/httpd/conf/httpd.conf\n    "
    return __salt__['lowpkg.owner'](*paths, **kwargs)

def _get_visible_patterns(root=None):
    if False:
        i = 10
        return i + 15
    'Get all available patterns in the repo that are visible.'
    patterns = {}
    search_patterns = __zypper__(root=root).nolock.xml.call('se', '-t', 'pattern')
    for element in search_patterns.getElementsByTagName('solvable'):
        installed = element.getAttribute('status') == 'installed'
        patterns[element.getAttribute('name')] = {'installed': installed, 'summary': element.getAttribute('summary')}
    return patterns

def _get_installed_patterns(root=None):
    if False:
        print('Hello World!')
    '\n    List all installed patterns.\n    '

    def _pattern_name(capability):
        if False:
            return 10
        'Return from a suitable capability the pattern name.'
        return capability.split('=')[-1].strip()
    cmd = ['rpm']
    if root:
        cmd.extend(['--root', root])
    cmd.extend(['-q', '--provides', '--whatprovides', 'pattern()'])
    output = __salt__['cmd.run'](cmd, ignore_retcode=True)
    installed_patterns = {_pattern_name(line) for line in output.splitlines() if line.startswith('pattern() = ') and (not _pattern_name(line).startswith('.'))}
    patterns = {k: v for (k, v) in _get_visible_patterns(root=root).items() if v['installed']}
    for pattern in installed_patterns:
        if pattern not in patterns:
            patterns[pattern] = {'installed': True, 'summary': 'Non-visible pattern'}
    return patterns

def list_patterns(refresh=False, root=None):
    if False:
        while True:
            i = 10
    "\n    List all known patterns from available repos.\n\n    refresh\n        force a refresh if set to True.\n        If set to False (default) it depends on zypper if a refresh is\n        executed.\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_patterns\n    "
    if refresh:
        refresh_db(root)
    return _get_visible_patterns(root=root)

def list_installed_patterns(root=None):
    if False:
        return 10
    "\n    List installed patterns on the system.\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_installed_patterns\n    "
    return _get_installed_patterns(root=root)

def search(criteria, refresh=False, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List known packages, available to the system.\n\n    refresh\n        force a refresh if set to True.\n        If set to False (default) it depends on zypper if a refresh is\n        executed.\n\n    match (str)\n        One of `exact`, `words`, `substrings`. Search for an `exact` match\n        or for the whole `words` only. Default to `substrings` to patch\n        partial words.\n\n    provides (bool)\n        Search for packages which provide the search strings.\n\n    recommends (bool)\n        Search for packages which recommend the search strings.\n\n    requires (bool)\n        Search for packages which require the search strings.\n\n    suggests (bool)\n        Search for packages which suggest the search strings.\n\n    conflicts (bool)\n        Search packages conflicting with search strings.\n\n    obsoletes (bool)\n        Search for packages which obsolete the search strings.\n\n    file_list (bool)\n        Search for a match in the file list of packages.\n\n    search_descriptions (bool)\n        Search also in package summaries and descriptions.\n\n    case_sensitive (bool)\n        Perform case-sensitive search.\n\n    installed_only (bool)\n        Show only installed packages.\n\n    not_installed_only (bool)\n        Show only packages which are not installed.\n\n    details (bool)\n        Show version and repository\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.search <criteria>\n    "
    ALLOWED_SEARCH_OPTIONS = {'provides': '--provides', 'recommends': '--recommends', 'requires': '--requires', 'suggests': '--suggests', 'conflicts': '--conflicts', 'obsoletes': '--obsoletes', 'file_list': '--file-list', 'search_descriptions': '--search-descriptions', 'case_sensitive': '--case-sensitive', 'installed_only': '--installed-only', 'not_installed_only': '-u', 'details': '--details'}
    root = kwargs.get('root', None)
    if refresh:
        refresh_db(root)
    cmd = ['search']
    if kwargs.get('match') == 'exact':
        cmd.append('--match-exact')
    elif kwargs.get('match') == 'words':
        cmd.append('--match-words')
    elif kwargs.get('match') == 'substrings':
        cmd.append('--match-substrings')
    for opt in kwargs:
        if opt in ALLOWED_SEARCH_OPTIONS:
            cmd.append(ALLOWED_SEARCH_OPTIONS.get(opt))
    cmd.append(criteria)
    solvables = __zypper__(root=root).nolock.noraise.xml.call(*cmd).getElementsByTagName('solvable')
    if not solvables:
        raise CommandExecutionError(f"No packages found matching '{criteria}'")
    out = {}
    for solvable in solvables:
        out[solvable.getAttribute('name')] = dict()
        for (k, v) in solvable.attributes.items():
            out[solvable.getAttribute('name')][k] = v
    return out

def _get_first_aggregate_text(node_list):
    if False:
        i = 10
        return i + 15
    '\n    Extract text from the first occurred DOM aggregate.\n    '
    if not node_list:
        return ''
    out = []
    for node in node_list[0].childNodes:
        if node.nodeType == dom.Document.TEXT_NODE:
            out.append(node.nodeValue)
    return '\n'.join(out)

def list_products(all=False, refresh=False, root=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all available or installed SUSE products.\n\n    all\n        List all products available or only installed. Default is False.\n\n    refresh\n        force a refresh if set to True.\n        If set to False (default) it depends on zypper if a refresh is\n        executed.\n\n    root\n        operate on a different root directory.\n\n    Includes handling for OEM products, which read the OEM productline file\n    and overwrite the release value.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_products\n        salt '*' pkg.list_products all=True\n    "
    if refresh:
        refresh_db(root)
    ret = list()
    OEM_PATH = '/var/lib/suseRegister/OEM'
    if root:
        OEM_PATH = os.path.join(root, os.path.relpath(OEM_PATH, os.path.sep))
    cmd = list()
    if not all:
        cmd.append('--disable-repositories')
    cmd.append('products')
    if not all:
        cmd.append('-i')
    product_list = __zypper__(root=root).nolock.xml.call(*cmd).getElementsByTagName('product-list')
    if not product_list:
        return ret
    for prd in product_list[0].getElementsByTagName('product'):
        p_nfo = dict()
        for (k_p_nfo, v_p_nfo) in prd.attributes.items():
            if k_p_nfo in ['isbase', 'installed']:
                p_nfo[k_p_nfo] = bool(v_p_nfo in ['true', '1'])
            elif v_p_nfo:
                p_nfo[k_p_nfo] = v_p_nfo
        eol = prd.getElementsByTagName('endoflife')
        if eol:
            p_nfo['eol'] = eol[0].getAttribute('text')
            p_nfo['eol_t'] = int(eol[0].getAttribute('time_t') or 0)
        p_nfo['description'] = ' '.join([line.strip() for line in _get_first_aggregate_text(prd.getElementsByTagName('description')).split(os.linesep)])
        if 'productline' in p_nfo and p_nfo['productline']:
            oem_file = os.path.join(OEM_PATH, p_nfo['productline'])
            if os.path.isfile(oem_file):
                with salt.utils.files.fopen(oem_file, 'r') as rfile:
                    oem_release = salt.utils.stringutils.to_unicode(rfile.readline()).strip()
                    if oem_release:
                        p_nfo['release'] = oem_release
        ret.append(p_nfo)
    return ret

def download(*packages, **kwargs):
    if False:
        return 10
    "\n    Download packages to the local disk.\n\n    refresh\n        force a refresh if set to True.\n        If set to False (default) it depends on zypper if a refresh is\n        executed.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.download httpd\n        salt '*' pkg.download httpd postfix\n    "
    if not packages:
        raise SaltInvocationError('No packages specified')
    root = kwargs.get('root', None)
    refresh = kwargs.get('refresh', False)
    if refresh:
        refresh_db(root)
    pkg_ret = {}
    for dld_result in __zypper__(root=root).xml.call('download', *packages).getElementsByTagName('download-result'):
        repo = dld_result.getElementsByTagName('repository')[0]
        path = dld_result.getElementsByTagName('localfile')[0].getAttribute('path')
        pkg_info = {'repository-name': repo.getAttribute('name'), 'repository-alias': repo.getAttribute('alias'), 'path': path}
        key = _get_first_aggregate_text(dld_result.getElementsByTagName('name'))
        if __salt__['lowpkg.checksum'](pkg_info['path'], root=root):
            pkg_ret[key] = pkg_info
    if pkg_ret:
        failed = [pkg for pkg in packages if pkg not in pkg_ret]
        if failed:
            pkg_ret['_error'] = 'The following package(s) failed to download: {}'.format(', '.join(failed))
        return pkg_ret
    raise CommandExecutionError('Unable to download packages: {}'.format(', '.join(packages)))

def list_downloaded(root=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2017.7.0\n\n    List prefetched packages downloaded by Zypper in the local disk.\n\n    root\n        operate on a different root directory.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_downloaded\n    "
    CACHE_DIR = '/var/cache/zypp/packages/'
    if root:
        CACHE_DIR = os.path.join(root, os.path.relpath(CACHE_DIR, os.path.sep))
    ret = {}
    for (root, dirnames, filenames) in salt.utils.path.os_walk(CACHE_DIR):
        for filename in fnmatch.filter(filenames, '*.rpm'):
            package_path = os.path.join(root, filename)
            pkg_info = __salt__['lowpkg.bin_pkg_info'](package_path)
            pkg_timestamp = int(os.path.getctime(package_path))
            ret.setdefault(pkg_info['name'], {})[pkg_info['version']] = {'path': package_path, 'size': os.path.getsize(package_path), 'creation_date_time_t': pkg_timestamp, 'creation_date_time': datetime.datetime.utcfromtimestamp(pkg_timestamp).isoformat()}
    return ret

def diff(*paths, **kwargs):
    if False:
        print('Hello World!')
    "\n    Return a formatted diff between current files and original in a package.\n    NOTE: this function includes all files (configuration and not), but does\n    not work on binary content.\n\n    The root parameter can also be passed via the keyword argument.\n\n    :param path: Full path to the installed file\n    :return: Difference string or raises and exception if examined file is binary.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.diff /etc/apache2/httpd.conf /etc/sudoers\n    "
    ret = {}
    pkg_to_paths = {}
    for pth in paths:
        pth_pkg = __salt__['lowpkg.owner'](pth, **kwargs)
        if not pth_pkg:
            ret[pth] = os.path.exists(pth) and 'Not managed' or 'N/A'
        else:
            if pkg_to_paths.get(pth_pkg) is None:
                pkg_to_paths[pth_pkg] = []
            pkg_to_paths[pth_pkg].append(pth)
    if pkg_to_paths:
        local_pkgs = __salt__['pkg.download'](*pkg_to_paths.keys(), **kwargs)
        for (pkg, files) in pkg_to_paths.items():
            for path in files:
                ret[path] = __salt__['lowpkg.diff'](local_pkgs[pkg]['path'], path) or 'Unchanged'
    return ret

def _get_patches(installed_only=False, root=None):
    if False:
        print('Hello World!')
    '\n    List all known patches in repos.\n    '
    patches = {}
    for element in __zypper__(root=root).nolock.xml.call('se', '-t', 'patch').getElementsByTagName('solvable'):
        installed = element.getAttribute('status') == 'installed'
        if installed_only and installed or not installed_only:
            patches[element.getAttribute('name')] = {'installed': installed, 'summary': element.getAttribute('summary')}
    return patches

def list_patches(refresh=False, root=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2017.7.0\n\n    List all known advisory patches from available repos.\n\n    refresh\n        force a refresh if set to True.\n        If set to False (default) it depends on zypper if a refresh is\n        executed.\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_patches\n    "
    if refresh:
        refresh_db(root)
    return _get_patches(root=root)

def list_installed_patches(root=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2017.7.0\n\n    List installed advisory patches on the system.\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_installed_patches\n    "
    return _get_patches(installed_only=True, root=root)

def list_provides(root=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2018.3.0\n\n    List package provides of installed packages as a dict.\n    {'<provided_name>': ['<package_name>', '<package_name>', ...]}\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_provides\n    "
    ret = __context__.get('pkg.list_provides')
    if not ret:
        cmd = ['rpm']
        if root:
            cmd.extend(['--root', root])
        cmd.extend(['-qa', '--queryformat', '%{PROVIDES}_|-%{NAME}\n'])
        ret = dict()
        for line in __salt__['cmd.run'](cmd, output_loglevel='trace', python_shell=False).splitlines():
            (provide, realname) = line.split('_|-')
            if provide == realname:
                continue
            if provide not in ret:
                ret[provide] = list()
            ret[provide].append(realname)
        __context__['pkg.list_provides'] = ret
    return ret

def resolve_capabilities(pkgs, refresh=False, root=None, **kwargs):
    if False:
        return 10
    "\n    .. versionadded:: 2018.3.0\n\n    Convert name provides in ``pkgs`` into real package names if\n    ``resolve_capabilities`` parameter is set to True. In case of\n    ``resolve_capabilities`` is set to False the package list\n    is returned unchanged.\n\n    refresh\n        force a refresh if set to True.\n        If set to False (default) it depends on zypper if a refresh is\n        executed.\n\n    root\n        operate on a different root directory.\n\n    resolve_capabilities\n        If this option is set to True the input will be checked if\n        a package with this name exists. If not, this function will\n        search for a package which provides this name. If one is found\n        the output is exchanged with the real package name.\n        In case this option is set to False (Default) the input will\n        be returned unchanged.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.resolve_capabilities resolve_capabilities=True w3m_ssl\n    "
    if refresh:
        refresh_db(root)
    ret = list()
    for pkg in pkgs:
        if isinstance(pkg, dict):
            name = next(iter(pkg))
            version = pkg[name]
        else:
            name = pkg
            version = None
        if kwargs.get('resolve_capabilities', False):
            try:
                search(name, root=root, match='exact')
            except CommandExecutionError:
                try:
                    result = search(name, root=root, provides=True, match='exact')
                    if len(result) == 1:
                        name = next(iter(result.keys()))
                    elif len(result) > 1:
                        log.warning("Found ambiguous match for capability '%s'.", pkg)
                except CommandExecutionError as exc:
                    log.debug('Search failed with: %s', exc)
        if version:
            ret.append({name: version})
        else:
            ret.append(name)
    return ret

def services_need_restart(root=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 3003\n\n    List services that use files which have been changed by the\n    package manager. It might be needed to restart them.\n\n    root\n        operate on a different root directory.\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.services_need_restart\n    "
    cmd = ['ps', '-sss']
    zypper_output = __zypper__(root=root).nolock.call(*cmd)
    services = zypper_output.split()
    return services