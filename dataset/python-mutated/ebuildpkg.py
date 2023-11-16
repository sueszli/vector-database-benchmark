"""
Support for Portage

.. important::
    If you feel that Salt should be using this module to manage packages on a
    minion, and it is using a different module (or gives an error similar to
    *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.

:optdepends:    - portage Python adapter

For now all package names *MUST* include the package category,
i.e. ``'vim'`` will not work, ``'app-editors/vim'`` will.
"""
import copy
import datetime
import logging
import os
import re
import salt.utils.args
import salt.utils.compat
import salt.utils.data
import salt.utils.functools
import salt.utils.path
import salt.utils.pkg
import salt.utils.systemd
import salt.utils.versions
from salt.exceptions import CommandExecutionError, MinionError
HAS_PORTAGE = False
try:
    import portage
    HAS_PORTAGE = True
except ImportError:
    import os
    import sys
    if os.path.isdir('/usr/lib/portage/pym'):
        try:
            sys.path.insert(0, '/usr/lib/portage/pym')
            import portage
            HAS_PORTAGE = True
        except ImportError:
            pass
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Confirm this module is on a Gentoo based system\n    '
    if HAS_PORTAGE and __grains__['os'] == 'Gentoo':
        return __virtualname__
    return (False, 'The ebuild execution module cannot be loaded: either the system is not Gentoo or the portage python library is not available.')

def _vartree():
    if False:
        for i in range(10):
            print('nop')
    import portage
    portage = salt.utils.compat.reload(portage)
    return portage.db[portage.root]['vartree']

def _porttree():
    if False:
        for i in range(10):
            print('nop')
    import portage
    portage = salt.utils.compat.reload(portage)
    return portage.db[portage.root]['porttree']

def _p_to_cp(p):
    if False:
        print('Hello World!')
    try:
        ret = portage.dep_getkey(p)
        if ret:
            return ret
    except portage.exception.InvalidAtom:
        pass
    try:
        ret = _porttree().dbapi.xmatch('bestmatch-visible', p)
        if ret:
            return portage.dep_getkey(ret)
    except portage.exception.InvalidAtom:
        pass
    try:
        ret = _porttree().dbapi.xmatch('match-all', p)
        if ret:
            return portage.cpv_getkey(ret[0])
    except portage.exception.InvalidAtom:
        pass
    return None

def _allnodes():
    if False:
        print('Hello World!')
    if 'portage._allnodes' in __context__:
        return __context__['portage._allnodes']
    else:
        ret = _porttree().getallnodes()
        __context__['portage._allnodes'] = ret
        return ret

def _cpv_to_cp(cpv):
    if False:
        print('Hello World!')
    try:
        ret = portage.dep_getkey(cpv)
        if ret:
            return ret
    except portage.exception.InvalidAtom:
        pass
    try:
        ret = portage.cpv_getkey(cpv)
        if ret:
            return ret
    except portage.exception.InvalidAtom:
        pass
    return cpv

def _cpv_to_version(cpv):
    if False:
        return 10
    return portage.versions.cpv_getversion(cpv)

def _process_emerge_err(stdout, stderr):
    if False:
        for i in range(10):
            print('nop')
    '\n    Used to parse emerge output to provide meaningful output when emerge fails\n    '
    ret = {}
    rexp = re.compile('^[<>=][^ ]+/[^ ]+ [^\\n]+', re.M)
    slot_conflicts = re.compile('^[^ \\n]+/[^ ]+:[^ ]', re.M).findall(stderr)
    if slot_conflicts:
        ret['slot conflicts'] = slot_conflicts
    blocked = re.compile('(?m)^\\[blocks .+\\] ([^ ]+/[^ ]+-[0-9]+[^ ]+).*$').findall(stdout)
    unsatisfied = re.compile('Error: The above package list contains').findall(stderr)
    if blocked and unsatisfied:
        ret['blocked'] = blocked
    sections = re.split('\n\n', stderr)
    for section in sections:
        if 'The following keyword changes' in section:
            ret['keywords'] = rexp.findall(section)
        elif 'The following license changes' in section:
            ret['license'] = rexp.findall(section)
        elif 'The following USE changes' in section:
            ret['use'] = rexp.findall(section)
        elif 'The following mask changes' in section:
            ret['mask'] = rexp.findall(section)
    return ret

def check_db(*names, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 0.17.0\n\n    Returns a dict containing the following information for each specified\n    package:\n\n    1. A key ``found``, which will be a boolean value denoting if a match was\n       found in the package database.\n    2. If ``found`` is ``False``, then a second key called ``suggestions`` will\n       be present, which will contain a list of possible matches. This list\n       will be empty if the package name was specified in ``category/pkgname``\n       format, since the suggestions are only intended to disambiguate\n       ambiguous package names (ones submitted without a category).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' pkg.check_db <package1> <package2> <package3>\n    "
    ret = {}
    for name in names:
        if name in ret:
            log.warning("pkg.check_db: Duplicate package name '%s' submitted", name)
            continue
        if '/' not in name:
            ret.setdefault(name, {})['found'] = False
            ret[name]['suggestions'] = porttree_matches(name)
        else:
            ret.setdefault(name, {})['found'] = name in _allnodes()
            if ret[name]['found'] is False:
                ret[name]['suggestions'] = []
    return ret

def ex_mod_init(low):
    if False:
        return 10
    "\n    If the config option ``ebuild.enforce_nice_config`` is set to True, this\n    module will enforce a nice tree structure for /etc/portage/package.*\n    configuration files.\n\n    .. versionadded:: 0.17.0\n       Initial automatic enforcement added when pkg is used on a Gentoo system.\n\n    .. versionchanged:: 2014.7.0\n       Configure option added to make this behaviour optional, defaulting to\n       off.\n\n    .. seealso::\n       ``ebuild.ex_mod_init`` is called automatically when a state invokes a\n       pkg state on a Gentoo system.\n       :py:func:`salt.states.pkg.mod_init`\n\n       ``ebuild.ex_mod_init`` uses ``portage_config.enforce_nice_config`` to do\n       the lifting.\n       :py:func:`salt.modules.portage_config.enforce_nice_config`\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.ex_mod_init\n    "
    if __salt__['config.get']('ebuild.enforce_nice_config', False):
        __salt__['portage_config.enforce_nice_config']()
    return True

def latest_version(*names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the latest version of the named package available for upgrade or\n    installation. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n    "
    refresh = salt.utils.data.is_true(kwargs.pop('refresh', True))
    if not names:
        return ''
    if refresh:
        refresh_db()
    ret = {}
    for name in names:
        ret[name] = ''
        installed = _cpv_to_version(_vartree().dep_bestmatch(name))
        avail = _cpv_to_version(_porttree().dep_bestmatch(name))
        if avail and (not installed or salt.utils.versions.compare(ver1=installed, oper='<', ver2=avail, cmp_func=version_cmp)):
            ret[name] = avail
    if len(names) == 1:
        return ret[names[0]]
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def _get_upgradable(backtrack=3):
    if False:
        for i in range(10):
            print('nop')
    "\n    Utility function to get upgradable packages\n\n    Sample return data:\n    { 'pkgname': '1.2.3-45', ... }\n    "
    cmd = ['emerge', '--ask', 'n', '--backtrack', '{}'.format(backtrack), '--pretend', '--update', '--newuse', '--deep', '@world']
    call = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if call['retcode'] != 0:
        msg = 'Failed to get upgrades'
        for key in ('stderr', 'stdout'):
            if call[key]:
                msg += ': ' + call[key]
                break
        raise CommandExecutionError(msg)
    else:
        out = call['stdout']
    rexp = re.compile('(?m)^\\[.+\\] ([^ ]+/[^ ]+)-([0-9]+[^ ]+).*$')
    keys = ['name', 'version']
    _get = lambda l, k: l[keys.index(k)]
    upgrades = rexp.findall(out)
    ret = {}
    for line in upgrades:
        name = _get(line, 'name')
        version_num = _get(line, 'version')
        ret[name] = version_num
    return ret

def list_upgrades(refresh=True, backtrack=3, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all available package upgrades.\n\n    refresh\n        Whether or not to sync the portage tree before checking for upgrades.\n\n    backtrack\n        Specifies an integer number of times to backtrack if dependency\n        calculation fails due to a conflict or an unsatisfied dependency\n        (default: ´3´).\n\n        .. versionadded:: 2015.8.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_upgrades\n    "
    if salt.utils.data.is_true(refresh):
        refresh_db()
    return _get_upgradable(backtrack)

def upgrade_available(name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check whether or not an upgrade is available for a given package\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n    "
    return latest_version(name) != ''

def version(*names, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Returns a string representing the package version or an empty string if not\n    installed. If more than one package name is specified, a dict of\n    name/version pairs is returned.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version <package name>\n        salt '*' pkg.version <package1> <package2> <package3> ...\n    "
    return __salt__['pkg_resource.version'](*names, **kwargs)

def porttree_matches(name):
    if False:
        print('Hello World!')
    '\n    Returns a list containing the matches for a given package name from the\n    portage tree. Note that the specific version of the package will not be\n    provided for packages that have several versions in the portage tree, but\n    rather the name of the package (i.e. "dev-python/paramiko").\n    '
    matches = []
    for category in _porttree().dbapi.categories:
        if _porttree().dbapi.cp_list(category + '/' + name):
            matches.append(category + '/' + name)
    return matches

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
        print('Hello World!')
    "\n    List the packages currently installed in a dict::\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return {}
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    ret = {}
    pkgs = _vartree().dbapi.cpv_all()
    for cpv in pkgs:
        __salt__['pkg_resource.add_pkg'](ret, _cpv_to_cp(cpv), _cpv_to_version(cpv))
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def refresh_db(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Update the portage tree using the first available method from the following\n    list:\n\n    - emaint sync\n    - eix-sync\n    - emerge-webrsync\n    - emerge --sync\n\n    To prevent the portage tree from being synced within one day of the\n    previous sync, add the following pillar data for this minion:\n\n    .. code-block:: yaml\n\n        portage:\n          sync_wait_one_day: True\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.refresh_db\n    "
    has_emaint = os.path.isdir('/etc/portage/repos.conf')
    has_eix = True if 'eix.sync' in __salt__ else False
    has_webrsync = True if __salt__['makeconf.features_contains']('webrsync-gpg') else False
    salt.utils.pkg.clear_rtag(__opts__)
    if __salt__['pillar.get']('portage:sync_wait_one_day', False):
        main_repo_root = __salt__['cmd.run']('portageq get_repo_path / gentoo')
        day = datetime.timedelta(days=1)
        now = datetime.datetime.now()
        timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(main_repo_root))
        if now - timestamp < day:
            log.info('Did not sync package tree since last sync was done at %s, less than 1 day ago', timestamp)
            return False
    if has_emaint:
        return __salt__['cmd.retcode']('emaint sync -a') == 0
    elif has_eix:
        return __salt__['eix.sync']()
    elif has_webrsync:
        cmd = 'emerge-webrsync -q'
        if salt.utils.path.which('emerge-delta-webrsync'):
            cmd = 'emerge-delta-webrsync -q'
        return __salt__['cmd.retcode'](cmd) == 0
    else:
        return __salt__['cmd.retcode']('emerge --ask n --quiet --sync') == 0

def _flags_changed(inst_flags, conf_flags):
    if False:
        i = 10
        return i + 15
    '\n    @type inst_flags: list\n    @param inst_flags: list of use flags which were used\n        when package was installed\n    @type conf_flags: list\n    @param conf_flags: list of use flags form portage/package.use\n    @rtype: bool\n    @return: True, if lists have changes\n    '
    conf_flags = conf_flags[:]
    for i in inst_flags:
        try:
            conf_flags.remove(i)
        except ValueError:
            return True
    return True if conf_flags else False

def install(name=None, refresh=False, pkgs=None, sources=None, slot=None, fromrepo=None, uses=None, binhost=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any emerge commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Install the passed package(s), add refresh=True to sync the portage tree\n    before package is installed.\n\n    name\n        The name of the package to be installed. Note that this parameter is\n        ignored if either "pkgs" or "sources" is passed. Additionally, please\n        note that this option can only be used to emerge a package from the\n        portage tree. To install a tbz2 package manually, use the "sources"\n        option described below.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install <package name>\n\n    refresh\n        Whether or not to sync the portage tree before installing.\n\n    version\n        Install a specific version of the package, e.g. 1.0.9-r1. Ignored\n        if "pkgs" or "sources" is passed.\n\n    slot\n        Similar to version, but specifies a valid slot to be installed. It\n        will install the latest available version in the specified slot.\n        Ignored if "pkgs" or "sources" or "version" is passed.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install sys-devel/gcc slot=\'4.4\'\n\n    fromrepo\n        Similar to slot, but specifies the repository from the package will be\n        installed. It will install the latest available version in the\n        specified repository.\n        Ignored if "pkgs" or "sources" or "version" is passed.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install salt fromrepo=\'gentoo\'\n\n    uses\n        Similar to slot, but specifies a list of use flag.\n        Ignored if "pkgs" or "sources" or "version" is passed.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install sys-devel/gcc uses=\'["nptl","-nossp"]\'\n\n\n    Multiple Package Installation Options:\n\n    pkgs\n        A list of packages to install from the portage tree. Must be passed as\n        a python list.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install pkgs=\'["foo","bar","~category/package:slot::repository[use]"]\'\n\n    sources\n        A list of tbz2 packages to install. Must be passed as a list of dicts,\n        with the keys being package names, and the values being the source URI\n        or local path to the package.\n\n        CLI Example:\n\n        .. code-block:: bash\n\n            salt \'*\' pkg.install sources=\'[{"foo": "salt://foo.tbz2"},{"bar": "salt://bar.tbz2"}]\'\n    binhost\n        has two options try and force.\n        try - tells emerge to try and install the package from a configured binhost.\n        force - forces emerge to install the package from a binhost otherwise it fails out.\n\n    Returns a dict containing the new package names and versions::\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n    '
    log.debug('Called modules.pkg.install: %s', {'name': name, 'refresh': refresh, 'pkgs': pkgs, 'sources': sources, 'kwargs': kwargs, 'binhost': binhost})
    if salt.utils.data.is_true(refresh):
        refresh_db()
    try:
        (pkg_params, pkg_type) = __salt__['pkg_resource.parse_targets'](name, pkgs, sources, **kwargs)
    except MinionError as exc:
        raise CommandExecutionError(exc)
    if pkgs is None and sources is None:
        version_num = kwargs.get('version')
        if not version_num:
            version_num = ''
            if slot is not None:
                version_num += ':{}'.format(slot)
            if fromrepo is not None:
                version_num += '::{}'.format(fromrepo)
            if uses is not None:
                version_num += '[{}]'.format(','.join(uses))
            pkg_params = {name: version_num}
    if not pkg_params:
        return {}
    elif pkg_type == 'file':
        emerge_opts = ['tbz2file']
    else:
        emerge_opts = []
    if binhost == 'try':
        bin_opts = ['-g']
    elif binhost == 'force':
        bin_opts = ['-G']
    else:
        bin_opts = []
    changes = {}
    if pkg_type == 'repository':
        targets = list()
        for (param, version_num) in pkg_params.items():
            original_param = param
            param = _p_to_cp(param)
            if param is None:
                raise portage.dep.InvalidAtom(original_param)
            if version_num is None:
                targets.append(param)
            else:
                keyword = None
                match = re.match('^(~)?([<>])?(=)?([^<>=]*)$', version_num)
                if match:
                    (keyword, gt_lt, eq, verstr) = match.groups()
                    prefix = gt_lt or ''
                    prefix += eq or ''
                    if len(verstr) > 0 and verstr[0] != ':' and (verstr[0] != '['):
                        prefix = prefix or '='
                        target = '{}{}-{}'.format(prefix, param, verstr)
                    else:
                        target = '{}{}'.format(param, verstr)
                else:
                    target = '{}'.format(param)
                if '[' in target:
                    old = __salt__['portage_config.get_flags_from_package_conf']('use', target)
                    __salt__['portage_config.append_use_flags'](target)
                    new = __salt__['portage_config.get_flags_from_package_conf']('use', target)
                    if old != new:
                        changes[param + '-USE'] = {'old': old, 'new': new}
                    target = target[:target.rfind('[')]
                if keyword is not None:
                    __salt__['portage_config.append_to_package_conf']('accept_keywords', target, ['~ARCH'])
                    changes[param + '-ACCEPT_KEYWORD'] = {'old': '', 'new': '~ARCH'}
                if not changes:
                    inst_v = version(param)
                    if latest_version(param, refresh=False) == inst_v:
                        all_uses = __salt__['portage_config.get_cleared_flags'](param)
                        if _flags_changed(*all_uses):
                            changes[param] = {'version': inst_v, 'old': {'use': all_uses[0]}, 'new': {'use': all_uses[1]}}
                targets.append(target)
    else:
        targets = pkg_params
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.extend(['emerge', '--ask', 'n', '--quiet'])
    cmd.extend(bin_opts)
    cmd.extend(emerge_opts)
    cmd.extend(targets)
    old = list_pkgs()
    call = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if call['retcode'] != 0:
        needed_changes = _process_emerge_err(call['stdout'], call['stderr'])
    else:
        needed_changes = []
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    changes.update(salt.utils.data.compare_dicts(old, new))
    if needed_changes:
        raise CommandExecutionError('Error occurred installing package(s)', info={'needed changes': needed_changes, 'changes': changes})
    return changes

def update(pkg, slot=None, fromrepo=None, refresh=False, binhost=None, **kwargs):
    if False:
        return 10
    "\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon's control group. This is done to keep systemd\n        from killing any emerge commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Updates the passed package (emerge --update package)\n\n    slot\n        Restrict the update to a particular slot. It will update to the\n        latest version within the slot.\n\n    fromrepo\n        Restrict the update to a particular repository. It will update to the\n        latest version within the repository.\n    binhost\n        has two options try and force.\n        try - tells emerge to try and install the package from a configured binhost.\n        force - forces emerge to install the package from a binhost otherwise it fails out.\n\n    Return a dict containing the new package names and versions::\n\n        {'<package>': {'old': '<old-version>',\n                       'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.update <package name>\n    "
    if salt.utils.data.is_true(refresh):
        refresh_db()
    full_atom = pkg
    if slot is not None:
        full_atom = '{}:{}'.format(full_atom, slot)
    if fromrepo is not None:
        full_atom = '{}::{}'.format(full_atom, fromrepo)
    if binhost == 'try':
        bin_opts = ['-g']
    elif binhost == 'force':
        bin_opts = ['-G']
    else:
        bin_opts = []
    old = list_pkgs()
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.extend(['emerge', '--ask', 'n', '--quiet', '--update', '--newuse', '--oneshot'])
    cmd.extend(bin_opts)
    cmd.append(full_atom)
    call = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    if call['retcode'] != 0:
        needed_changes = _process_emerge_err(call['stdout'], call['stderr'])
    else:
        needed_changes = []
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if needed_changes:
        raise CommandExecutionError('Problem encountered updating package(s)', info={'needed_changes': needed_changes, 'changes': ret})
    return ret

def upgrade(refresh=True, binhost=None, backtrack=3, **kwargs):
    if False:
        return 10
    "\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon's control group. This is done to keep systemd\n        from killing any emerge commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Run a full system upgrade (emerge -uDN @world)\n\n    binhost\n        has two options try and force.\n        try - tells emerge to try and install the package from a configured binhost.\n        force - forces emerge to install the package from a binhost otherwise it fails out.\n\n    backtrack\n        Specifies an integer number of times to backtrack if dependency\n        calculation fails due to a conflict or an unsatisfied dependency\n        (default: ´3´).\n\n        .. versionadded:: 2015.8.0\n\n    Returns a dictionary containing the changes:\n\n    .. code-block:: python\n\n        {'<package>':  {'old': '<old-version>',\n                        'new': '<new-version>'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade\n    "
    ret = {'changes': {}, 'result': True, 'comment': ''}
    if salt.utils.data.is_true(refresh):
        refresh_db()
    if binhost == 'try':
        bin_opts = ['--getbinpkg']
    elif binhost == 'force':
        bin_opts = ['--getbinpkgonly']
    else:
        bin_opts = []
    old = list_pkgs()
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.extend(['emerge', '--ask', 'n', '--quiet', '--backtrack', '{}'.format(backtrack), '--update', '--newuse', '--deep'])
    if bin_opts:
        cmd.extend(bin_opts)
    cmd.append('@world')
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if result['retcode'] != 0:
        raise CommandExecutionError('Problem encountered upgrading packages', info={'changes': ret, 'result': result})
    return ret

def remove(name=None, slot=None, fromrepo=None, pkgs=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any emerge commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Remove packages via emerge --unmerge.\n\n    name\n        The name of the package to be deleted.\n\n    slot\n        Restrict the remove to a specific slot. Ignored if ``name`` is None.\n\n    fromrepo\n        Restrict the remove to a specific slot. Ignored if ``name`` is None.\n\n    Multiple Package Options:\n\n    pkgs\n        Uninstall multiple packages. ``slot`` and ``fromrepo`` arguments are\n        ignored if this argument is present. Must be passed as a python list.\n\n    .. versionadded:: 0.16.0\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <package name>\n        salt \'*\' pkg.remove <package name> slot=4.4 fromrepo=gentoo\n        salt \'*\' pkg.remove <package1>,<package2>,<package3>\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    old = list_pkgs()
    if name and (not pkgs) and (slot is not None or fromrepo is not None) and (len(pkg_params) == 1):
        fullatom = name
        if slot is not None:
            targets = ['{}:{}'.format(fullatom, slot)]
        if fromrepo is not None:
            targets = ['{}::{}'.format(fullatom, fromrepo)]
        targets = [fullatom]
    else:
        targets = [x for x in pkg_params if x in old]
    if not targets:
        return {}
    cmd = []
    if salt.utils.systemd.has_scope(__context__) and __salt__['config.get']('systemd.scope', True):
        cmd.extend(['systemd-run', '--scope'])
    cmd.extend(['emerge', '--ask', 'n', '--quiet', '--unmerge', '--quiet-unmerge-warn'])
    cmd.extend(targets)
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

def purge(name=None, slot=None, fromrepo=None, pkgs=None, **kwargs):
    if False:
        return 10
    '\n    .. versionchanged:: 2015.8.12,2016.3.3,2016.11.0\n        On minions running systemd>=205, `systemd-run(1)`_ is now used to\n        isolate commands which modify installed packages from the\n        ``salt-minion`` daemon\'s control group. This is done to keep systemd\n        from killing any emerge commands spawned by Salt when the\n        ``salt-minion`` service is restarted. (see ``KillMode`` in the\n        `systemd.kill(5)`_ manpage for more information). If desired, usage of\n        `systemd-run(1)`_ can be suppressed by setting a :mod:`config option\n        <salt.modules.config.get>` called ``systemd.scope``, with a value of\n        ``False`` (no quotes).\n\n    .. _`systemd-run(1)`: https://www.freedesktop.org/software/systemd/man/systemd-run.html\n    .. _`systemd.kill(5)`: https://www.freedesktop.org/software/systemd/man/systemd.kill.html\n\n    Portage does not have a purge, this function calls remove followed\n    by depclean to emulate a purge process\n\n    name\n        The name of the package to be deleted.\n\n    slot\n        Restrict the remove to a specific slot. Ignored if name is None.\n\n    fromrepo\n        Restrict the remove to a specific slot. Ignored if ``name`` is None.\n\n    Multiple Package Options:\n\n    pkgs\n        Uninstall multiple packages. ``slot`` and ``fromrepo`` arguments are\n        ignored if this argument is present. Must be passed as a python list.\n\n    .. versionadded:: 0.16.0\n\n\n    Returns a dict containing the changes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.purge <package name>\n        salt \'*\' pkg.purge <package name> slot=4.4\n        salt \'*\' pkg.purge <package1>,<package2>,<package3>\n        salt \'*\' pkg.purge pkgs=\'["foo", "bar"]\'\n    '
    ret = remove(name=name, slot=slot, fromrepo=fromrepo, pkgs=pkgs)
    ret.update(depclean(name=name, slot=slot, fromrepo=fromrepo, pkgs=pkgs))
    return ret

def depclean(name=None, slot=None, fromrepo=None, pkgs=None):
    if False:
        while True:
            i = 10
    "\n    Portage has a function to remove unused dependencies. If a package\n    is provided, it will only removed the package if no other package\n    depends on it.\n\n    name\n        The name of the package to be cleaned.\n\n    slot\n        Restrict the remove to a specific slot. Ignored if ``name`` is None.\n\n    fromrepo\n        Restrict the remove to a specific slot. Ignored if ``name`` is None.\n\n    pkgs\n        Clean multiple packages. ``slot`` and ``fromrepo`` arguments are\n        ignored if this argument is present. Must be passed as a python list.\n\n    Return a list containing the removed packages:\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.depclean <package name>\n    "
    try:
        pkg_params = __salt__['pkg_resource.parse_targets'](name, pkgs)[0]
    except MinionError as exc:
        raise CommandExecutionError(exc)
    old = list_pkgs()
    if name and (not pkgs) and (slot is not None or fromrepo is not None) and (len(pkg_params) == 1):
        fullatom = name
        if slot is not None:
            targets = ['{}:{}'.format(fullatom, slot)]
        if fromrepo is not None:
            targets = ['{}::{}'.format(fullatom, fromrepo)]
        targets = [fullatom]
    else:
        targets = [x for x in pkg_params if x in old]
    cmd = ['emerge', '--ask', 'n', '--quiet', '--depclean'] + targets
    __salt__['cmd.run_all'](cmd, output_loglevel='trace', python_shell=False)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    return salt.utils.data.compare_dicts(old, new)

def version_cmp(pkg1, pkg2, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Do a cmp-style comparison on two packages. Return -1 if pkg1 < pkg2, 0 if\n    pkg1 == pkg2, and 1 if pkg1 > pkg2. Return None if there was a problem\n    making the comparison.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version_cmp '0.2.4-0' '0.2.4.1-0'\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    kwargs.pop('ignore_epoch', None)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    regex = '^~?([^:\\[]+):?[^\\[]*\\[?.*$'
    ver1 = re.match(regex, pkg1)
    ver2 = re.match(regex, pkg2)
    if ver1 and ver2:
        return portage.versions.vercmp(ver1.group(1), ver2.group(1))
    return None

def version_clean(version):
    if False:
        print('Hello World!')
    "\n    Clean the version string removing extra data.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.version_clean <version_string>\n    "
    return re.match('^~?[<>]?=?([^<>=:\\[]+).*$', version)

def check_extra_requirements(pkgname, pkgver):
    if False:
        while True:
            i = 10
    "\n    Check if the installed package already has the given requirements.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.check_extra_requirements 'sys-devel/gcc' '~>4.1.2:4.1::gentoo[nls,fortran]'\n    "
    keyword = None
    match = re.match('^(~)?([<>])?(=)?([^<>=]*)$', pkgver)
    if match:
        (keyword, gt_lt, eq, verstr) = match.groups()
        prefix = gt_lt or ''
        prefix += eq or ''
        verstr = verstr.replace("'", '')
        if verstr[0] != ':' and verstr[0] != '[':
            prefix = prefix or '='
            atom = '{}{}-{}'.format(prefix, pkgname, verstr)
        else:
            atom = '{}{}'.format(pkgname, verstr)
    else:
        return True
    try:
        cpv = _porttree().dbapi.xmatch('bestmatch-visible', atom)
    except portage.exception.InvalidAtom as iae:
        log.error('Unable to find a matching package for %s: (%s)', atom, iae)
        return False
    if cpv == '':
        return False
    try:
        (cur_repo, cur_use) = _vartree().dbapi.aux_get(cpv, ['repository', 'USE'])
    except KeyError:
        return False
    des_repo = re.match('^.+::([^\\[]+).*$', atom)
    if des_repo and des_repo.group(1) != cur_repo:
        return False
    des_uses = set(portage.dep.dep_getusedeps(atom))
    cur_use = cur_use.split()
    if len([x for x in des_uses.difference(cur_use) if x[0] != '-' or x[1:] in cur_use]) > 0:
        return False
    if keyword:
        if not __salt__['portage_config.has_flag']('accept_keywords', atom, '~ARCH'):
            return False
    return True