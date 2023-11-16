"""
Package support for AIX

.. important::
    If you feel that Salt should be using this module to manage filesets or
    rpm packages on a minion, and it is using a different module (or gives an
    error similar to *'pkg.install' is not available*), see :ref:`here
    <module-provider-override>`.
"""
import copy
import logging
import os
import pathlib
import salt.utils.data
import salt.utils.functools
import salt.utils.path
import salt.utils.pkg
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'pkg'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Set the virtual pkg module if the os is AIX\n    '
    if __grains__['os_family'] == 'AIX':
        return __virtualname__
    return (False, 'Did not load AIX module on non-AIX OS.')

def _check_pkg(target):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return name, version and if rpm package for specified target\n    '
    ret = {}
    cmd = ['/usr/bin/lslpp', '-Lc', target]
    result = __salt__['cmd.run_all'](cmd, python_shell=False)
    if 0 == result['retcode']:
        name = ''
        version_num = ''
        rpmpkg = False
        lines = result['stdout'].splitlines()
        for line in lines:
            if line.startswith('#'):
                continue
            comps = line.split(':')
            if len(comps) < 7:
                raise CommandExecutionError('Error occurred finding fileset/package', info={'errors': comps[1].strip()})
            if 'R' in comps[6]:
                name = comps[0]
                rpmpkg = True
            else:
                name = comps[1]
            version_num = comps[2]
            break
        return (name, version_num, rpmpkg)
    else:
        raise CommandExecutionError('Error occurred finding fileset/package', info={'errors': result['stderr'].strip()})

def _is_installed_rpm(name):
    if False:
        while True:
            i = 10
    '\n    Returns True if the rpm package is installed. Otherwise returns False.\n    '
    cmd = ['/usr/bin/rpm', '-q', name]
    return __salt__['cmd.retcode'](cmd) == 0

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
        return 10
    "\n    List the filesets/rpm packages currently installed as a dict:\n\n    .. code-block:: python\n\n        {'<package_name>': '<version>'}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.list_pkgs\n    "
    ret = {}
    versions_as_list = salt.utils.data.is_true(versions_as_list)
    if any([salt.utils.data.is_true(kwargs.get(x)) for x in ('removed', 'purge_desired')]):
        return ret
    if 'pkg.list_pkgs' in __context__ and kwargs.get('use_context', True):
        return _list_pkgs_from_context(versions_as_list)
    cmd = '/usr/bin/lslpp -Lc'
    lines = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    for line in lines:
        if line.startswith('#'):
            continue
        comps = line.split(':')
        if len(comps) < 7:
            continue
        if 'R' in comps[6]:
            name = comps[0]
        else:
            name = comps[1]
        version_num = comps[2]
        __salt__['pkg_resource.add_pkg'](ret, name, version_num)
    __salt__['pkg_resource.sort_pkglist'](ret)
    __context__['pkg.list_pkgs'] = copy.deepcopy(ret)
    if not versions_as_list:
        __salt__['pkg_resource.stringify'](ret)
    return ret

def version(*names, **kwargs):
    if False:
        print('Hello World!')
    "\n    Return the current installed version of the named fileset/rpm package\n    If more than one fileset/rpm package name is specified a dict of\n    name/version pairs is returned.\n\n    .. versionchanged:: 3005\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n\n    "
    kwargs.pop('refresh', True)
    ret = {}
    if not names:
        return ''
    for name in names:
        version_found = ''
        cmd = 'lslpp -Lq {}'.format(name)
        aix_info = __salt__['cmd.run_all'](cmd, python_shell=False)
        if 0 == aix_info['retcode']:
            aix_info_list = aix_info['stdout'].split('\n')
            log.debug('Returned AIX packaging information aix_info_list %s for name %s', aix_info_list, name)
            for aix_line in aix_info_list:
                if name in aix_line:
                    aix_ver_list = aix_line.split()
                    log.debug('Processing name %s with AIX packaging version information %s', name, aix_ver_list)
                    version_found = aix_ver_list[1]
                    if version_found:
                        log.debug('Found name %s in AIX packaging information, version %s', name, version_found)
                        break
        else:
            log.debug('Could not find name %s in AIX packaging information', name)
        ret[name] = version_found
    if len(names) == 1:
        return ret[names[0]]
    return ret

def _is_installed(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Returns True if the fileset/rpm package is installed. Otherwise returns False.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg._is_installed bash\n    "
    cmd = ['/usr/bin/lslpp', '-Lc', name]
    return __salt__['cmd.retcode'](cmd) == 0

def install(name=None, refresh=False, pkgs=None, version=None, test=False, **kwargs):
    if False:
        return 10
    '\n    Install the named fileset(s)/rpm package(s).\n\n    .. versionchanged:: 3005\n\n        preference to install rpm packages are to use in the following order:\n            /opt/freeware/bin/dnf\n            /opt/freeware/bin/yum\n            /usr/bin/yum\n            /usr/bin/rpm\n\n    .. note:\n        use of rpm to install implies that rpm\'s dependencies must have been previously installed.\n        dnf and yum automatically install rpm\'s dependencies as part of the install process\n\n        Alogrithm to install filesets or rpms is as follows:\n            if ends with \'.rte\' or \'.bff\'\n                process as fileset\n            if ends with \'.rpm\'\n                process as rpm\n            if unrecognised or no file extension\n                attempt process with dnf | yum\n                failure implies attempt process as fileset\n\n        Fileset needs to be available as a single path and filename\n        compound filesets are not handled and are not supported.\n        An example is bos.adt.insttools which is part of bos.adt.other and is installed as follows\n        /usr/bin/installp -acXYg /cecc/repos/aix72/TL4/BASE/installp/ppc/bos.adt.other bos.adt.insttools\n\n    name\n        The name of the fileset or rpm package to be installed.\n\n    refresh\n        Whether or not to update the yum database before executing.\n\n\n    pkgs\n        A list of filesets and/or rpm packages to install.\n        Must be passed as a python list. The ``name`` parameter will be\n        ignored if this option is passed.\n\n    version\n        Install a specific version of a fileset/rpm package.\n        (Unused at present).\n\n    test\n        Verify that command functions correctly.\n\n    Returns a dict containing the new fileset(s)/rpm package(s) names and versions:\n\n        {\'<package>\': {\'old\': \'<old-version>\',\n                       \'new\': \'<new-version>\'}}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.install /stage/middleware/AIX/bash-4.2-3.aix6.1.ppc.rpm\n        salt \'*\' pkg.install /stage/middleware/AIX/bash-4.2-3.aix6.1.ppc.rpm refresh=True\n        salt \'*\' pkg.install /stage/middleware/AIX/VIOS2211_update/tpc_4.1.1.85.bff\n        salt \'*\' pkg.install /cecc/repos/aix72/TL3/BASE/installp/ppc/bos.rte.printers_7.2.2.0.bff\n        salt \'*\' pkg.install /stage/middleware/AIX/Xlc/usr/sys/inst.images/xlC.rte\n        salt \'*\' pkg.install /stage/middleware/AIX/Firefox/ppc-AIX53/Firefox.base\n        salt \'*\' pkg.install /cecc/repos/aix72/TL3/BASE/installp/ppc/bos.net\n        salt \'*\' pkg.install pkgs=\'["foo", "bar"]\'\n        salt \'*\' pkg.install libxml2\n    '
    targets = salt.utils.args.split_input(pkgs) if pkgs else [name]
    if not targets:
        return {}
    if pkgs:
        log.debug('Installing these fileset(s)/rpm package(s) %s: %s', name, targets)
    old = list_pkgs()
    errors = []
    for target in targets:
        filename = os.path.basename(target)
        flag_fileset = False
        flag_actual_rpm = False
        flag_try_rpm_failed = False
        cmd = ''
        out = {}
        if filename.endswith('.bff') or filename.endswith('.rte'):
            flag_fileset = True
            log.debug('install identified %s as fileset', filename)
        else:
            if filename.endswith('.rpm'):
                flag_actual_rpm = True
                log.debug('install identified %s as rpm', filename)
            else:
                log.debug('install filename %s trying install as rpm', filename)
            cmdflags = 'install '
            libpathenv = {'LIBPATH': '/opt/freeware/lib:/usr/lib'}
            if pathlib.Path('/opt/freeware/bin/dnf').is_file():
                cmdflags += '--allowerasing '
                cmdexe = '/opt/freeware/bin/dnf'
                if test:
                    cmdflags += '--assumeno '
                else:
                    cmdflags += '--assumeyes '
                if refresh:
                    cmdflags += '--refresh '
                cmd = '{} {} {}'.format(cmdexe, cmdflags, target)
                out = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
            elif pathlib.Path('/usr/bin/yum').is_file():
                cmdexe = '/usr/bin/yum'
                if test:
                    cmdflags += '--assumeno '
                else:
                    cmdflags += '--assumeyes '
                cmd = '{} {} {}'.format(cmdexe, cmdflags, target)
                out = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
            elif pathlib.Path('/opt/freeware/bin/yum').is_file():
                cmdflags += '--allowerasing '
                cmdexe = '/opt/freeware/bin/yum'
                if test:
                    cmdflags += '--assumeno '
                else:
                    cmdflags += '--assumeyes '
                if refresh:
                    cmdflags += '--refresh '
                cmd = '{} {} {}'.format(cmdexe, cmdflags, target)
                out = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
            else:
                cmdexe = '/usr/bin/rpm'
                cmdflags = '-Uivh '
                if test:
                    cmdflags += '--test'
                cmd = '{} {} {}'.format(cmdexe, cmdflags, target)
                out = __salt__['cmd.run_all'](cmd, python_shell=False)
        if 'retcode' in out and (not (0 == out['retcode'] or 100 == out['retcode'])):
            if not flag_actual_rpm:
                flag_try_rpm_failed = True
                log.debug('install tried filename %s as rpm and failed, trying as fileset', filename)
            else:
                errors.append(out['stderr'])
                log.debug('install error rpm path, returned result %s, resultant errors %s', out, errors)
        if flag_fileset or flag_try_rpm_failed:
            cmd = '/usr/sbin/installp -acYXg'
            if test:
                cmd += 'p'
            cmd += ' -d '
            dirpath = os.path.dirname(target)
            cmd += dirpath + ' ' + filename
            log.debug('install fileset commanda to attempt %s', cmd)
            out = __salt__['cmd.run_all'](cmd, python_shell=False)
            if 0 != out['retcode']:
                errors.append(out['stderr'])
                log.debug('install error fileset path, returned result %s, resultant errors %s', out, errors)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        raise CommandExecutionError('Problems encountered installing filesets(s)/package(s)', info={'changes': ret, 'errors': errors})
    if test:
        return 'Test succeeded.'
    return ret

def remove(name=None, pkgs=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove specified fileset(s)/rpm package(s).\n\n    name\n        The name of the fileset or rpm package to be deleted.\n\n    .. versionchanged:: 3005\n\n        preference to install rpm packages are to use in the following order:\n            /opt/freeware/bin/dnf\n            /opt/freeware/bin/yum\n            /usr/bin/yum\n            /usr/bin/rpm\n\n    pkgs\n        A list of filesets and/or rpm packages to delete.\n        Must be passed as a python list. The ``name`` parameter will be\n        ignored if this option is passed.\n\n\n    Returns a list containing the removed packages.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pkg.remove <fileset/rpm package name>\n        salt \'*\' pkg.remove tcsh\n        salt \'*\' pkg.remove xlC.rte\n        salt \'*\' pkg.remove Firefox.base.adt\n        salt \'*\' pkg.remove pkgs=\'["foo", "bar"]\'\n    '
    targets = salt.utils.args.split_input(pkgs) if pkgs else [name]
    if not targets:
        return {}
    if pkgs:
        log.debug('Removing these fileset(s)/rpm package(s) %s: %s', name, targets)
    errors = []
    old = list_pkgs()
    for target in targets:
        cmd = ''
        out = {}
        try:
            (named, versionpkg, rpmpkg) = _check_pkg(target)
        except CommandExecutionError as exc:
            if exc.info:
                errors.append(exc.info['errors'])
            continue
        if rpmpkg:
            cmdflags = '-y remove'
            libpathenv = {'LIBPATH': '/opt/freeware/lib:/usr/lib'}
            if pathlib.Path('/opt/freeware/bin/dnf').is_file():
                cmdexe = '/opt/freeware/bin/dnf'
                cmd = '{} {} {}'.format(cmdexe, cmdflags, target)
                out = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
            elif pathlib.Path('/opt/freeware/bin/yum').is_file():
                cmdexe = '/opt/freeware/bin/yum'
                cmd = '{} {} {}'.format(cmdexe, cmdflags, target)
                out = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
            elif pathlib.Path('/usr/bin/yum').is_file():
                cmdexe = '/usr/bin/yum'
                cmd = '{} {} {}'.format(cmdexe, cmdflags, target)
                out = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
            else:
                cmdexe = '/usr/bin/rpm'
                cmdflags = '-e'
                cmd = '{} {} {}'.format(cmdexe, cmdflags, target)
                out = __salt__['cmd.run_all'](cmd, python_shell=False)
        else:
            cmd = ['/usr/sbin/installp', '-u', named]
            out = __salt__['cmd.run_all'](cmd, python_shell=False)
        log.debug('result of removal command %s, returned result %s', cmd, out)
    __context__.pop('pkg.list_pkgs', None)
    new = list_pkgs()
    ret = salt.utils.data.compare_dicts(old, new)
    if errors:
        raise CommandExecutionError('Problems encountered removing filesets(s)/package(s)', info={'changes': ret, 'errors': errors})
    return ret

def latest_version(*names, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Return the latest available version of the named fileset/rpm package available for\n    upgrade or installation. If more than one fileset/rpm package name is\n    specified, a dict of name/version pairs is returned.\n\n    If the latest version of a given fileset/rpm package is already installed,\n    an empty string will be returned for that package.\n\n    .. versionchanged:: 3005\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.latest_version <package name>\n        salt '*' pkg.latest_version <package1> <package2> <package3> ...\n\n    Note: currently only functional for rpm packages due to filesets do not have a specific location to check\n        Requires yum of dnf available in order to query a repository\n\n    This function will always return an empty string for unfound fileset/rpm package.\n    "
    kwargs.pop('refresh', True)
    ret = {}
    if not names:
        return ''
    for name in names:
        version_found = ''
        libpathenv = {'LIBPATH': '/opt/freeware/lib:/usr/lib'}
        if pathlib.Path('/opt/freeware/bin/dnf').is_file():
            cmdexe = '/opt/freeware/bin/dnf'
            cmd = '{} check-update {}'.format(cmdexe, name)
            available_info = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
        elif pathlib.Path('/opt/freeware/bin/yum').is_file():
            cmdexe = '/opt/freeware/bin/yum'
            cmd = '{} check-update {}'.format(cmdexe, name)
            available_info = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
        elif pathlib.Path('/usr/bin/yum').is_file():
            cmdexe = '/usr/bin/yum'
            cmd = '{} check-update {}'.format(cmdexe, name)
            available_info = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
        else:
            available_info = None
        log.debug('latest_version dnf|yum check-update command returned information %s', available_info)
        if available_info and (0 == available_info['retcode'] or 100 == available_info['retcode']):
            available_output = available_info['stdout']
            if available_output:
                available_list = available_output.split()
                flag_found = False
                for name_chk in available_list:
                    if name_chk.startswith(name):
                        pkg_label = name_chk.split('.')
                        if name == pkg_label[0]:
                            flag_found = True
                    elif flag_found:
                        version_found = name_chk
                        break
        if version_found:
            log.debug('latest_version result for name %s found version %s', name, version_found)
        else:
            log.debug('Could not find AIX / RPM packaging version for %s', name)
        ret[name] = version_found
    if len(names) == 1:
        return ret[names[0]]
    return ret
available_version = salt.utils.functools.alias_function(latest_version, 'available_version')

def upgrade_available(name, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Check whether or not an upgrade is available for a given package\n\n    .. versionchanged:: 3005\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pkg.upgrade_available <package name>\n\n    Note: currently only functional for rpm packages due to filesets do not have a specific location to check\n        Requires yum of dnf available in order to query a repository\n\n    "
    rpm_found = False
    version_found = ''
    libpathenv = {'LIBPATH': '/opt/freeware/lib:/usr/lib'}
    if pathlib.Path('/opt/freeware/bin/dnf').is_file():
        cmdexe = '/opt/freeware/bin/dnf'
        cmd = '{} check-update {}'.format(cmdexe, name)
        available_info = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
    elif pathlib.Path('/opt/freeware/bin/yum').is_file():
        cmdexe = '/opt/freeware/bin/yum'
        cmd = '{} check-update {}'.format(cmdexe, name)
        available_info = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
    elif pathlib.Path('/usr/bin/yum').is_file():
        cmdexe = '/usr/bin/yum'
        cmd = '{} check-update {}'.format(cmdexe, name)
        available_info = __salt__['cmd.run_all'](cmd, python_shell=False, env=libpathenv, ignore_retcode=True)
    else:
        return False
    log.debug('upgrade_available yum check-update command %s, returned information %s', cmd, available_info)
    if 0 == available_info['retcode'] or 100 == available_info['retcode']:
        available_output = available_info['stdout']
        if available_output:
            available_list = available_output.split()
            flag_found = False
            for name_chk in available_list:
                if name_chk.startswith(name):
                    pkg_label = name_chk.split('.')
                    if name == pkg_label[0]:
                        flag_found = True
                elif flag_found:
                    version_found = name_chk
                    break
        current_version = version(name)
        log.debug('upgrade_available result for name %s, found current version %s, available version %s', name, current_version, version_found)
    if version_found:
        return current_version != version_found
    else:
        log.debug('upgrade_available information for name %s was not found', name)
        return False