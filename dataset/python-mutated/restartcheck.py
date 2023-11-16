"""
checkrestart functionality for Debian and Red Hat Based systems

Identifies services (processes) that are linked against deleted files (for example after downloading an updated
binary of a shared library).

Based on checkrestart script from debian-goodies (written  by Matt Zimmerman for the Debian GNU/Linux distribution,
https://packages.debian.org/debian-goodies) and psdel by Sam Morris.

:codeauthor: Jiri Kotlin <jiri.kotlin@ultimum.io>
"""
import os
import re
import subprocess
import sys
import time
import salt.exceptions
import salt.utils.args
import salt.utils.files
import salt.utils.path
NILRT_FAMILY_NAME = 'NILinuxRT'
HAS_PSUTIL = False
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    pass
LIST_DIRS = ['^/var/log/', '^/var/local/log/', '^/var/run/', '^/var/local/run/', '^/tmp/', '^/dev/shm/', '^/run/', '^/drm', '^/var/tmp/', '^/var/local/tmp/', '^/dev/zero', '^/dev/pts/', '^/usr/lib/locale/', '^/home/', '^.*icon-theme.cache', '^/var/cache/fontconfig/', '^/var/lib/nagios3/spool/', '^/var/lib/nagios3/spool/checkresults/', '^/var/lib/postgresql/', '^/var/lib/vdr/', '^/[aio]', '^/SYSV']

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only run this module if the psutil python module is installed (package python-psutil).\n    '
    if HAS_PSUTIL:
        return HAS_PSUTIL
    else:
        return (False, 'Missing dependency: psutil')

def _valid_deleted_file(path):
    if False:
        return 10
    '\n    Filters file path against unwanted directories and decides whether file is marked as deleted.\n\n    Returns:\n        True if file is desired deleted file, else False.\n\n    Args:\n        path: A string - path to file.\n    '
    ret = False
    if path.endswith(' (deleted)'):
        ret = True
    if re.compile('\\(path inode=[0-9]+\\)$').search(path):
        ret = True
    regex = re.compile('|'.join(LIST_DIRS))
    if regex.match(path):
        ret = False
    return ret

def _deleted_files():
    if False:
        while True:
            i = 10
    '\n    Iterates over /proc/PID/maps and /proc/PID/fd links and returns list of desired deleted files.\n\n    Returns:\n        List of deleted files to analyze, False on failure.\n\n    '
    deleted_files = []
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
            try:
                with salt.utils.files.fopen('/proc/{}/maps'.format(pinfo['pid'])) as maps:
                    dirpath = '/proc/' + str(pinfo['pid']) + '/fd/'
                    listdir = os.listdir(dirpath)
                    maplines = maps.readlines()
            except OSError:
                yield False
            mapline = re.compile('^[\\da-f]+-[\\da-f]+ [r-][w-][x-][sp-] [\\da-f]+ [\\da-f]{2}:[\\da-f]{2} (\\d+) *(.+)( \\(deleted\\))?\\n$')
            for line in maplines:
                line = salt.utils.stringutils.to_unicode(line)
                matched = mapline.match(line)
                if not matched:
                    continue
                path = matched.group(2)
                if not path:
                    continue
                valid = _valid_deleted_file(path)
                if not valid:
                    continue
                val = (pinfo['name'], pinfo['pid'], path[0:-10])
                if val not in deleted_files:
                    deleted_files.append(val)
                    yield val
            try:
                for link in listdir:
                    path = dirpath + link
                    readlink = os.readlink(path)
                    filenames = []
                    if os.path.isfile(readlink):
                        filenames.append(readlink)
                    elif os.path.isdir(readlink) and readlink != '/':
                        for (root, dummy_dirs, files) in salt.utils.path.os_walk(readlink, followlinks=True):
                            for name in files:
                                filenames.append(os.path.join(root, name))
                    for filename in filenames:
                        valid = _valid_deleted_file(filename)
                        if not valid:
                            continue
                        val = (pinfo['name'], pinfo['pid'], filename)
                        if val not in deleted_files:
                            deleted_files.append(val)
                            yield val
            except OSError:
                pass
        except psutil.NoSuchProcess:
            pass

def _format_output(kernel_restart, packages, verbose, restartable, nonrestartable, restartservicecommands, restartinitcommands):
    if False:
        return 10
    '\n    Formats the output of the restartcheck module.\n\n    Returns:\n        String - formatted output.\n\n    Args:\n        kernel_restart: indicates that newer kernel is instaled.\n        packages: list of packages that should be restarted.\n        verbose: enables extensive output.\n        restartable: list of restartable packages.\n        nonrestartable: list of non-restartable packages.\n        restartservicecommands: list of commands to restart services.\n        restartinitcommands: list of commands to restart init.d scripts.\n\n    '
    if not verbose:
        packages = restartable + nonrestartable
        if kernel_restart:
            packages.append('System restart required.')
        return packages
    else:
        ret = ''
        if kernel_restart:
            ret = 'System restart required.\n\n'
        if packages:
            ret += 'Found {} processes using old versions of upgraded files.\n'.format(len(packages))
            ret += 'These are the packages:\n'
        if restartable:
            ret += 'Of these, {} seem to contain systemd service definitions or init scripts which can be used to restart them:\n'.format(len(restartable))
            for package in restartable:
                ret += package + ':\n'
                for program in packages[package]['processes']:
                    ret += program + '\n'
            if restartservicecommands:
                ret += '\n\nThese are the systemd services:\n'
                ret += '\n'.join(restartservicecommands)
            if restartinitcommands:
                ret += '\n\nThese are the initd scripts:\n'
                ret += '\n'.join(restartinitcommands)
        if nonrestartable:
            ret += '\n\nThese processes {} do not seem to have an associated init script to restart them:\n'.format(len(nonrestartable))
            for package in nonrestartable:
                ret += package + ':\n'
                for program in packages[package]['processes']:
                    ret += program + '\n'
    return ret

def _kernel_versions_debian():
    if False:
        for i in range(10):
            print('nop')
    '\n    Last installed kernel name, for Debian based systems.\n\n    Returns:\n            List with possible names of last installed kernel\n            as they are probably interpreted in output of `uname -a` command.\n    '
    kernel_get_selections = __salt__['cmd.run']('dpkg --get-selections linux-image-*')
    kernels = []
    kernel_versions = []
    for line in kernel_get_selections.splitlines():
        kernels.append(line)
    try:
        kernel = kernels[-2]
    except IndexError:
        kernel = kernels[0]
    kernel = kernel.rstrip('\t\tinstall')
    kernel_get_version = __salt__['cmd.run']('apt-cache policy ' + kernel)
    for line in kernel_get_version.splitlines():
        if line.startswith('  Installed: '):
            kernel_v = line.strip('  Installed: ')
            kernel_versions.append(kernel_v)
            break
    if __grains__['os'] == 'Ubuntu':
        kernel_v = kernel_versions[0].rsplit('.', 1)
        kernel_ubuntu_generic = kernel_v[0] + '-generic #' + kernel_v[1]
        kernel_ubuntu_lowlatency = kernel_v[0] + '-lowlatency #' + kernel_v[1]
        kernel_versions.extend([kernel_ubuntu_generic, kernel_ubuntu_lowlatency])
    return kernel_versions

def _kernel_versions_redhat():
    if False:
        print('Hello World!')
    '\n    Name of the last installed kernel, for Red Hat based systems.\n\n    Returns:\n            List with name of last installed kernel as it is interpreted in output of `uname -a` command.\n    '
    kernel_get_last = __salt__['cmd.run']('rpm -q --last kernel')
    kernels = []
    kernel_versions = []
    for line in kernel_get_last.splitlines():
        if 'kernel-' in line:
            kernels.append(line)
    kernel = kernels[0].split(' ', 1)[0]
    kernel = kernel.strip('kernel-')
    kernel_versions.append(kernel)
    return kernel_versions

def _kernel_versions_nilrt():
    if False:
        while True:
            i = 10
    '\n    Last installed kernel name, for Debian based systems.\n\n    Returns:\n            List with possible names of last installed kernel\n            as they are probably interpreted in output of `uname -a` command.\n    '
    kver = None

    def _get_kver_from_bin(kbin):
        if False:
            return 10
        '\n        Get kernel version from a binary image or None if detection fails\n        '
        kvregex = '[0-9]+\\.[0-9]+\\.[0-9]+-rt\\S+'
        kernel_strings = __salt__['cmd.run']('strings {}'.format(kbin))
        re_result = re.search(kvregex, kernel_strings)
        return None if re_result is None else re_result.group(0)
    if __grains__.get('lsb_distrib_id') == 'nilrt':
        if 'arm' in __grains__.get('cpuarch'):
            itb_path = '/boot/linux_runmode.itb'
            compressed_kernel = '/var/volatile/tmp/uImage.gz'
            uncompressed_kernel = '/var/volatile/tmp/uImage'
            __salt__['cmd.run']('dumpimage -i {} -T flat_dt -p0 kernel -o {}'.format(itb_path, compressed_kernel))
            __salt__['cmd.run']('gunzip -f {}'.format(compressed_kernel))
            kver = _get_kver_from_bin(uncompressed_kernel)
        else:
            kver = _get_kver_from_bin('/boot/runmode/bzImage')
    elif 'arm' in __grains__.get('cpuarch'):
        kver = os.path.basename(os.readlink('/boot/uImage')).strip('uImage-')
    else:
        kver = os.path.basename(os.readlink('/boot/bzImage')).strip('bzImage-')
    return [] if kver is None else [kver]

def _check_timeout(start_time, timeout):
    if False:
        i = 10
        return i + 15
    '\n    Name of the last installed kernel, for Red Hat based systems.\n\n    Returns:\n            List with name of last installed kernel as it is interpreted in output of `uname -a` command.\n    '
    timeout_milisec = timeout * 60000
    if timeout_milisec < int(round(time.time() * 1000)) - start_time:
        raise salt.exceptions.TimeoutError('Timeout expired.')

def _file_changed_nilrt(full_filepath):
    if False:
        while True:
            i = 10
    '\n    Detect whether a file changed in an NILinuxRT system using md5sum and timestamp\n    files from a state directory.\n\n    Returns:\n             - True if either md5sum/timestamp state files do not exist, or\n               the file at ``full_filepath`` was touched or modified.\n             - False otherwise.\n    '
    rs_state_dir = '/var/lib/salt/restartcheck_state'
    base_filename = os.path.basename(full_filepath)
    timestamp_file = os.path.join(rs_state_dir, '{}.timestamp'.format(base_filename))
    md5sum_file = os.path.join(rs_state_dir, '{}.md5sum'.format(base_filename))
    if not os.path.exists(timestamp_file) or not os.path.exists(md5sum_file):
        return True
    prev_timestamp = __salt__['file.read'](timestamp_file).rstrip()
    cur_timestamp = str(int(os.path.getmtime(full_filepath)))
    if prev_timestamp != cur_timestamp:
        return True
    return bool(__salt__['cmd.retcode']('md5sum -cs {}'.format(md5sum_file), output_loglevel='quiet'))

def _kernel_modules_changed_nilrt(kernelversion):
    if False:
        return 10
    "\n    Once a NILRT kernel module is inserted, it can't be rmmod so systems need\n    rebooting (some modules explicitly ask for reboots even on first install),\n    hence this functionality of determining if the module state got modified by\n    testing if depmod was run.\n\n    Returns:\n             - True if modules.dep was modified/touched, False otherwise.\n    "
    if kernelversion is not None:
        return _file_changed_nilrt('/lib/modules/{}/modules.dep'.format(kernelversion))
    return False

def _sysapi_changed_nilrt():
    if False:
        for i in range(10):
            print('nop')
    '\n    Besides the normal Linux kernel driver interfaces, NILinuxRT-supported hardware features an\n    extensible, plugin-based device enumeration and configuration interface named "System API".\n    When an installed package is extending the API it is very hard to know all repercurssions and\n    actions to be taken, so reboot making sure all drivers are reloaded, hardware reinitialized,\n    daemons restarted, etc.\n\n    Returns:\n             - True if nisysapi .ini files were modified/touched.\n             - False if no nisysapi .ini files exist.\n    '
    nisysapi_path = '/usr/local/natinst/share/nisysapi.ini'
    if os.path.exists(nisysapi_path) and _file_changed_nilrt(nisysapi_path):
        return True
    restartcheck_state_dir = '/var/lib/salt/restartcheck_state'
    nisysapi_conf_d_path = '/usr/lib/{}/nisysapi/conf.d/experts/'.format('arm-linux-gnueabi' if 'arm' in __grains__.get('cpuarch') else 'x86_64-linux-gnu')
    if os.path.exists(nisysapi_conf_d_path):
        rs_count_file = '{}/sysapi.conf.d.count'.format(restartcheck_state_dir)
        if not os.path.exists(rs_count_file):
            return True
        with salt.utils.files.fopen(rs_count_file, 'r') as fcount:
            current_nb_files = len(os.listdir(nisysapi_conf_d_path))
            rs_stored_nb_files = int(fcount.read())
            if current_nb_files != rs_stored_nb_files:
                return True
        for fexpert in os.listdir(nisysapi_conf_d_path):
            if _file_changed_nilrt('{}/{}'.format(nisysapi_conf_d_path, fexpert)):
                return True
    return False

def restartcheck(ignorelist=None, blacklist=None, excludepid=None, **kwargs):
    if False:
        return 10
    "\n    Analyzes files openeded by running processes and seeks for packages which need to be restarted.\n\n    Args:\n        ignorelist: string or list of packages to be ignored.\n        blacklist: string or list of file paths to be ignored.\n        excludepid: string or list of process IDs to be ignored.\n        verbose: boolean, enables extensive output.\n        timeout: int, timeout in minute.\n\n    Returns:\n        Dict on error: ``{ 'result': False, 'comment': '<reason>' }``.\n        String with checkrestart output if some package seems to need to be restarted or\n        if no packages need restarting.\n\n    .. versionadded:: 2015.8.3\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' restartcheck.restartcheck\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    start_time = int(round(time.time() * 1000))
    kernel_restart = True
    verbose = kwargs.pop('verbose', True)
    timeout = kwargs.pop('timeout', 5)
    if __grains__.get('os_family') == 'Debian':
        cmd_pkg_query = ['dpkg-query', '--listfiles']
        systemd_folder = '/lib/systemd/system/'
        systemd = '/bin/systemd'
        kernel_versions = _kernel_versions_debian()
    elif __grains__.get('os_family') == 'RedHat':
        cmd_pkg_query = ['repoquery', '-l']
        systemd_folder = '/usr/lib/systemd/system/'
        systemd = '/usr/bin/systemctl'
        kernel_versions = _kernel_versions_redhat()
    elif __grains__.get('os_family') == NILRT_FAMILY_NAME:
        cmd_pkg_query = ['opkg', 'files']
        systemd = ''
        kernel_versions = _kernel_versions_nilrt()
    else:
        return {'result': False, 'comment': 'Only available on Debian, Red Hat and NI Linux Real-Time based systems.'}
    kernel_current = __salt__['cmd.run']('uname -a')
    for kernel in kernel_versions:
        _check_timeout(start_time, timeout)
        if kernel in kernel_current:
            if __grains__.get('os_family') == 'NILinuxRT':
                if not _kernel_modules_changed_nilrt(kernel) and (not _sysapi_changed_nilrt()) and (not __salt__['system.get_reboot_required_witnessed']()):
                    kernel_restart = False
                    break
            else:
                kernel_restart = False
                break
    packages = {}
    running_services = {}
    restart_services = []
    if ignorelist:
        if not isinstance(ignorelist, list):
            ignorelist = [ignorelist]
    else:
        ignorelist = ['screen', 'systemd']
    if blacklist:
        if not isinstance(blacklist, list):
            blacklist = [blacklist]
    else:
        blacklist = []
    if excludepid:
        if not isinstance(excludepid, list):
            excludepid = [excludepid]
    else:
        excludepid = []
    for service in __salt__['service.get_running']():
        _check_timeout(start_time, timeout)
        service_show = __salt__['service.show'](service)
        if 'ExecMainPID' in service_show:
            running_services[service] = int(service_show['ExecMainPID'])
    owners_cache = {}
    for deleted_file in _deleted_files():
        if deleted_file is False:
            return {'result': False, 'comment': 'Could not get list of processes. (Do you have root access?)'}
        _check_timeout(start_time, timeout)
        (name, pid, path) = (deleted_file[0], deleted_file[1], deleted_file[2])
        if path in blacklist or pid in excludepid:
            continue
        try:
            readlink = os.readlink('/proc/{}/exe'.format(pid))
        except OSError:
            excludepid.append(pid)
            continue
        try:
            packagename = owners_cache[readlink]
        except KeyError:
            packagename = __salt__['pkg.owner'](readlink)
            if not packagename:
                packagename = name
            owners_cache[readlink] = packagename
        for running_service in running_services:
            _check_timeout(start_time, timeout)
            if running_service not in restart_services and pid == running_services[running_service]:
                if packagename and packagename not in ignorelist:
                    restart_services.append(running_service)
                    name = running_service
        if packagename and packagename not in ignorelist:
            program = '\t' + str(pid) + ' ' + readlink + ' (file: ' + str(path) + ')'
            if packagename not in packages:
                packages[packagename] = {'initscripts': [], 'systemdservice': [], 'processes': [program], 'process_name': name}
            elif program not in packages[packagename]['processes']:
                packages[packagename]['processes'].append(program)
    if not packages and (not kernel_restart):
        return 'No packages seem to need to be restarted.'
    for package in packages:
        _check_timeout(start_time, timeout)
        cmd = cmd_pkg_query[:]
        cmd.append(package)
        paths = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        while True:
            _check_timeout(start_time, timeout)
            line = salt.utils.stringutils.to_unicode(paths.stdout.readline())
            if not line:
                break
            pth = line[:-1]
            if pth.startswith('/etc/init.d/') and (not pth.endswith('.sh')):
                packages[package]['initscripts'].append(pth[12:])
            if os.path.exists(systemd) and pth.startswith(systemd_folder) and pth.endswith('.service') and (pth.find('.wants') == -1):
                is_oneshot = False
                try:
                    servicefile = salt.utils.files.fopen(pth)
                except OSError:
                    continue
                sysfold_len = len(systemd_folder)
                for line in servicefile.readlines():
                    line = salt.utils.stringutils.to_unicode(line)
                    if line.find('Type=oneshot') > 0:
                        is_oneshot = True
                    continue
                servicefile.close()
                if not is_oneshot:
                    packages[package]['systemdservice'].append(pth[sysfold_len:])
            sys.stdout.flush()
        paths.stdout.close()
    for package in packages:
        _check_timeout(start_time, timeout)
        if not packages[package]['systemdservice'] and (not packages[package]['initscripts']):
            service = __salt__['service.available'](packages[package]['process_name'])
            if service:
                if os.path.exists('/etc/init.d/' + packages[package]['process_name']):
                    packages[package]['initscripts'].append(packages[package]['process_name'])
                else:
                    packages[package]['systemdservice'].append(packages[package]['process_name'])
    restartable = []
    nonrestartable = []
    restartinitcommands = []
    restartservicecommands = []
    for package in packages:
        _check_timeout(start_time, timeout)
        if packages[package]['initscripts']:
            restartable.append(package)
            restartinitcommands.extend(['service ' + s + ' restart' for s in packages[package]['initscripts']])
        elif packages[package]['systemdservice']:
            restartable.append(package)
            restartservicecommands.extend(['systemctl restart ' + s for s in packages[package]['systemdservice']])
        else:
            nonrestartable.append(package)
        if packages[package]['process_name'] in restart_services:
            restart_services.remove(packages[package]['process_name'])
    for restart_service in restart_services:
        _check_timeout(start_time, timeout)
        restartservicecommands.extend(['systemctl restart ' + restart_service])
    ret = _format_output(kernel_restart, packages, verbose, restartable, nonrestartable, restartservicecommands, restartinitcommands)
    return ret