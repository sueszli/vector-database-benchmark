"""
Support for DEB packages
"""
import datetime
import logging
import os
import re
import salt.utils.args
import salt.utils.data
import salt.utils.files
import salt.utils.path
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError, SaltInvocationError
log = logging.getLogger(__name__)
__virtualname__ = 'lowpkg'

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Confirm this module is on a Debian based system\n    '
    if __grains__['os_family'] == 'Debian':
        return __virtualname__
    return (False, 'The dpkg execution module cannot be loaded: only works on Debian family systems.')

def bin_pkg_info(path, saltenv='base'):
    if False:
        return 10
    "\n    .. versionadded:: 2015.8.0\n\n    Parses DEB metadata and returns a dictionary of information about the\n    package (name, version, etc.).\n\n    path\n        Path to the file. Can either be an absolute path to a file on the\n        minion, or a salt fileserver URL (e.g. ``salt://path/to/file.deb``).\n        If a salt fileserver URL is passed, the file will be cached to the\n        minion so that it can be examined.\n\n    saltenv : base\n        Salt fileserver environment from which to retrieve the package. Ignored\n        if ``path`` is a local file path on the minion.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lowpkg.bin_pkg_info /root/foo-1.2.3-1ubuntu1_all.deb\n        salt '*' lowpkg.bin_pkg_info salt://foo-1.2.3-1ubuntu1_all.deb\n    "
    if __salt__['config.valid_fileproto'](path):
        newpath = __salt__['cp.cache_file'](path, saltenv)
        if not newpath:
            raise CommandExecutionError("Unable to retrieve {} from saltenv '{}'".format(path, saltenv))
        path = newpath
    elif not os.path.exists(path):
        raise CommandExecutionError('{} does not exist on minion'.format(path))
    elif not os.path.isabs(path):
        raise SaltInvocationError('{} does not exist on minion'.format(path))
    cmd = ['dpkg', '-I', path]
    result = __salt__['cmd.run_all'](cmd, output_loglevel='trace')
    if result['retcode'] != 0:
        msg = 'Unable to get info for ' + path
        if result['stderr']:
            msg += ': ' + result['stderr']
        raise CommandExecutionError(msg)
    ret = {}
    for line in result['stdout'].splitlines():
        line = line.strip()
        if re.match('^Package[ ]*:', line):
            ret['name'] = line.split()[-1]
        elif re.match('^Version[ ]*:', line):
            ret['version'] = line.split()[-1]
        elif re.match('^Architecture[ ]*:', line):
            ret['arch'] = line.split()[-1]
    missing = [x for x in ('name', 'version', 'arch') if x not in ret]
    if missing:
        raise CommandExecutionError('Unable to get {} for {}'.format(', '.join(missing), path))
    if __grains__.get('cpuarch', '') == 'x86_64':
        osarch = __grains__.get('osarch', '')
        arch = ret['arch']
        if arch != 'all' and osarch == 'amd64' and (osarch != arch):
            ret['name'] += ':{}'.format(arch)
    return ret

def unpurge(*packages):
    if False:
        return 10
    "\n    Change package selection for each package specified to 'install'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lowpkg.unpurge curl\n    "
    if not packages:
        return {}
    old = __salt__['pkg.list_pkgs'](purge_desired=True)
    ret = {}
    __salt__['cmd.run'](['dpkg', '--set-selections'], stdin='\\n'.join(['{} install'.format(x) for x in packages]), python_shell=False, output_loglevel='trace')
    __context__.pop('pkg.list_pkgs', None)
    new = __salt__['pkg.list_pkgs'](purge_desired=True)
    return salt.utils.data.compare_dicts(old, new)

def list_pkgs(*packages, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List the packages currently installed in a dict::\n\n        {'<package_name>': '<version>'}\n\n    External dependencies::\n\n        Virtual package resolution requires aptitude. Because this function\n        uses dpkg, virtual packages will be reported as not installed.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lowpkg.list_pkgs\n        salt '*' lowpkg.list_pkgs hostname\n        salt '*' lowpkg.list_pkgs hostname mount\n    "
    cmd = ['dpkg-query', '-f=${db:Status-Status}\t${binary:Package}\t${Version}\n', '-W'] + list(packages)
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode'] != 0:
        msg = 'Error:  ' + out['stderr']
        log.error(msg)
        return msg
    lines = [line.split('\t', 1) for line in out['stdout'].splitlines()]
    pkgs = dict([line.split('\t') for (status, line) in lines if status == 'installed'])
    return pkgs

def file_list(*packages, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List the files that belong to a package. Not specifying any packages will\n    return a list of _every_ file on the system's package database (not\n    generally recommended).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lowpkg.file_list hostname\n        salt '*' lowpkg.file_list hostname mount\n        salt '*' lowpkg.file_list\n    "
    errors = []
    ret = set()
    cmd = ['dpkg-query', '-f=${db:Status-Status}\t${binary:Package}\n', '-W'] + list(packages)
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode'] != 0:
        msg = 'Error:  ' + out['stderr']
        log.error(msg)
        return msg
    lines = [line.split('\t') for line in out['stdout'].splitlines()]
    pkgs = [package for (status, package) in lines if status == 'installed']
    for pkg in pkgs:
        output = __salt__['cmd.run'](['dpkg', '-L', pkg], python_shell=False)
        fileset = set(output.splitlines())
        ret = ret.union(fileset)
    return {'errors': errors, 'files': sorted(ret)}

def file_dict(*packages, **kwargs):
    if False:
        print('Hello World!')
    "\n    List the files that belong to a package, grouped by package. Not\n    specifying any packages will return a list of _every_ file on the system's\n    package database (not generally recommended).\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' lowpkg.file_dict hostname\n        salt '*' lowpkg.file_dict hostname mount\n        salt '*' lowpkg.file_dict\n    "
    errors = []
    ret = {}
    cmd = ['dpkg-query', '-f=${db:Status-Status}\t${binary:Package}\n', '-W'] + list(packages)
    out = __salt__['cmd.run_all'](cmd, python_shell=False)
    if out['retcode'] != 0:
        msg = 'Error:  ' + out['stderr']
        log.error(msg)
        return msg
    lines = [line.split('\t') for line in out['stdout'].splitlines()]
    pkgs = [package for (status, package) in lines if status == 'installed']
    for pkg in pkgs:
        cmd = ['dpkg', '-L', pkg]
        ret[pkg] = __salt__['cmd.run'](cmd, python_shell=False).splitlines()
    return {'errors': errors, 'packages': ret}

def _get_pkg_info(*packages, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Return list of package information. If 'packages' parameter is empty,\n    then data about all installed packages will be returned.\n\n    :param packages: Specified packages.\n    :param failhard: Throw an exception if no packages found.\n    :return:\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    failhard = kwargs.pop('failhard', True)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    if __grains__['os'] == 'Ubuntu' and __grains__['osrelease_info'] < (12, 4):
        bin_var = '${binary}'
    else:
        bin_var = '${Package}'
    ret = []
    cmd = "dpkg-query -W -f='package:" + bin_var + "\\nrevision:${binary:Revision}\\narchitecture:${Architecture}\\nmaintainer:${Maintainer}\\nsummary:${Summary}\\nsource:${source:Package}\\nversion:${Version}\\nsection:${Section}\\ninstalled_size:${Installed-size}\\nsize:${Size}\\nMD5:${MD5sum}\\nSHA1:${SHA1}\\nSHA256:${SHA256}\\norigin:${Origin}\\nhomepage:${Homepage}\\nstatus:${db:Status-Abbrev}\\ndescription:${Description}\\n\\n*/~^\\\\*\\n'"
    cmd += ' {}'.format(' '.join(packages))
    cmd = cmd.strip()
    call = __salt__['cmd.run_all'](cmd, python_shell=False)
    if call['retcode']:
        if failhard:
            raise CommandExecutionError('Error getting packages information: {}'.format(call['stderr']))
        else:
            return ret
    for pkg_info in [elm for elm in re.split('\\r?\\n\\*/~\\^\\\\\\*(\\r?\\n|)', call['stdout']) if elm.strip()]:
        pkg_data = {}
        (pkg_info, pkg_descr) = pkg_info.split('\ndescription:', 1)
        for pkg_info_line in [el.strip() for el in pkg_info.split(os.linesep) if el.strip()]:
            (key, value) = pkg_info_line.split(':', 1)
            if value:
                pkg_data[key] = value
            install_date = _get_pkg_install_time(pkg_data.get('package'))
            if install_date:
                pkg_data['install_date'] = install_date
        pkg_data['description'] = pkg_descr
        ret.append(pkg_data)
    return ret

def _get_pkg_license(pkg):
    if False:
        i = 10
        return i + 15
    '\n    Try to get a license from the package.\n    Based on https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n\n    :param pkg:\n    :return:\n    '
    licenses = set()
    cpr = '/usr/share/doc/{}/copyright'.format(pkg)
    if os.path.exists(cpr):
        with salt.utils.files.fopen(cpr, errors='ignore') as fp_:
            for line in salt.utils.stringutils.to_unicode(fp_.read()).split(os.linesep):
                if line.startswith('License:'):
                    licenses.add(line.split(':', 1)[1].strip())
    return ', '.join(sorted(licenses))

def _get_pkg_install_time(pkg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return package install time, based on the /var/lib/dpkg/info/<package>.list\n\n    :return:\n    '
    iso_time = None
    if pkg is not None:
        location = '/var/lib/dpkg/info/{}.list'.format(pkg)
        if os.path.exists(location):
            iso_time = datetime.datetime.utcfromtimestamp(int(os.path.getmtime(location))).isoformat() + 'Z'
    return iso_time

def _get_pkg_ds_avail():
    if False:
        return 10
    "\n    Get the package information of the available packages, maintained by dselect.\n    Note, this will be not very useful, if dselect isn't installed.\n\n    :return:\n    "
    avail = '/var/lib/dpkg/available'
    if not salt.utils.path.which('dselect') or not os.path.exists(avail):
        return dict()
    ret = dict()
    pkg_mrk = 'Package:'
    pkg_name = 'package'
    with salt.utils.files.fopen(avail) as fp_:
        for pkg_info in salt.utils.stringutils.to_unicode(fp_.read()).split(pkg_mrk):
            nfo = dict()
            for line in (pkg_mrk + pkg_info).split(os.linesep):
                line = line.split(': ', 1)
                if len(line) != 2:
                    continue
                (key, value) = line
                if value.strip():
                    nfo[key.lower()] = value
            if nfo.get(pkg_name):
                ret[nfo[pkg_name]] = nfo
    return ret

def info(*packages, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Returns a detailed summary of package information for provided package names.\n    If no packages are specified, all packages will be returned.\n\n    .. versionadded:: 2015.8.1\n\n    packages\n        The names of the packages for which to return information.\n\n    failhard\n        Whether to throw an exception if none of the packages are installed.\n        Defaults to True.\n\n        .. versionadded:: 2016.11.3\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' lowpkg.info\n        salt '*' lowpkg.info apache2 bash\n        salt '*' lowpkg.info 'php5*' failhard=false\n    "
    dselect_pkg_avail = _get_pkg_ds_avail()
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    failhard = kwargs.pop('failhard', True)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    ret = dict()
    for pkg in _get_pkg_info(*packages, failhard=failhard):
        for (pkg_ext_k, pkg_ext_v) in dselect_pkg_avail.get(pkg['package'], {}).items():
            if pkg_ext_k not in pkg:
                pkg[pkg_ext_k] = pkg_ext_v
        for t_key in ['installed_size', 'depends', 'recommends', 'provides', 'replaces', 'conflicts', 'bugs', 'description-md5', 'task']:
            if t_key in pkg:
                del pkg[t_key]
        lic = _get_pkg_license(pkg['package'])
        if lic:
            pkg['license'] = lic
        ret[pkg['package']] = pkg
    return ret