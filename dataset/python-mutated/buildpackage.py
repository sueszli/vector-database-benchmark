import errno
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
from optparse import OptionGroup, OptionParser
logging.QUIET = 0
logging.GARBAGE = 1
logging.TRACE = 5
logging.addLevelName(logging.QUIET, 'QUIET')
logging.addLevelName(logging.TRACE, 'TRACE')
logging.addLevelName(logging.GARBAGE, 'GARBAGE')
LOG_LEVELS = {'all': logging.NOTSET, 'debug': logging.DEBUG, 'error': logging.ERROR, 'critical': logging.CRITICAL, 'garbage': logging.GARBAGE, 'info': logging.INFO, 'quiet': logging.QUIET, 'trace': logging.TRACE, 'warning': logging.WARNING}
log = logging.getLogger(__name__)

def _abort(msgs):
    if False:
        print('Hello World!')
    '\n    Unrecoverable error, pull the plug\n    '
    if not isinstance(msgs, list):
        msgs = [msgs]
    for msg in msgs:
        log.error(msg)
        sys.stderr.write(msg + '\n\n')
    sys.stderr.write('Build failed. See log file for further details.\n')
    sys.exit(1)

def _init():
    if False:
        i = 10
        return i + 15
    '\n    Parse CLI options.\n    '
    parser = OptionParser()
    parser.add_option('--platform', dest='platform', help="Platform ('os' grain)")
    parser.add_option('--log-level', dest='log_level', default='warning', help='Control verbosity of logging. Default: %default')
    path_group = OptionGroup(parser, 'File/Directory Options')
    path_group.add_option('--source-dir', default='/testing', help='Source directory. Must be a git checkout. (default: %default)')
    path_group.add_option('--build-dir', default='/tmp/salt-buildpackage', help='Build root, will be removed if it exists prior to running script. (default: %default)')
    path_group.add_option('--artifact-dir', default='/tmp/salt-packages', help='Location where build artifacts should be placed for Jenkins to retrieve them (default: %default)')
    parser.add_option_group(path_group)
    rpm_group = OptionGroup(parser, 'RPM-specific File/Directory Options')
    rpm_group.add_option('--spec', dest='spec_file', default='/tmp/salt.spec', help='Spec file to use as a template to build RPM. (default: %default)')
    parser.add_option_group(rpm_group)
    opts = parser.parse_args()[0]
    for group in (path_group, rpm_group):
        for path_opt in [opt.dest for opt in group.option_list]:
            path = getattr(opts, path_opt)
            if not os.path.isabs(path):
                path = os.path.expanduser(path)
                if not os.path.isabs(path):
                    path = os.path.realpath(path)
                setattr(opts, path_opt, path)
    problems = []
    if not opts.platform:
        problems.append("Platform ('os' grain) required")
    if not os.path.isdir(opts.source_dir):
        problems.append('Source directory {} not found'.format(opts.source_dir))
    try:
        shutil.rmtree(opts.build_dir)
    except OSError as exc:
        if exc.errno not in (errno.ENOENT, errno.ENOTDIR):
            problems.append('Unable to remove pre-existing destination directory {}: {}'.format(opts.build_dir, exc))
    finally:
        try:
            os.makedirs(opts.build_dir)
        except OSError as exc:
            problems.append('Unable to create destination directory {}: {}'.format(opts.build_dir, exc))
    try:
        shutil.rmtree(opts.artifact_dir)
    except OSError as exc:
        if exc.errno not in (errno.ENOENT, errno.ENOTDIR):
            problems.append('Unable to remove pre-existing artifact directory {}: {}'.format(opts.artifact_dir, exc))
    finally:
        try:
            os.makedirs(opts.artifact_dir)
        except OSError as exc:
            problems.append('Unable to create artifact directory {}: {}'.format(opts.artifact_dir, exc))
    opts.log_file = os.path.join(opts.artifact_dir, 'salt-buildpackage.log')
    if problems:
        _abort(problems)
    return opts

def _move(src, dst):
    if False:
        i = 10
        return i + 15
    '\n    Wrapper around shutil.move()\n    '
    try:
        os.remove(os.path.join(dst, os.path.basename(src)))
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            _abort(exc)
    try:
        shutil.move(src, dst)
    except shutil.Error as exc:
        _abort(exc)

def _run_command(args):
    if False:
        return 10
    log.info('Running command: %s', args)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = proc.communicate()
    if stdout:
        log.debug('Command output: \n%s', stdout)
    if stderr:
        log.error(stderr)
    log.info('Return code: %s', proc.returncode)
    return (stdout, stderr, proc.returncode)

def _make_sdist(opts, python_bin='python'):
    if False:
        i = 10
        return i + 15
    os.chdir(opts.source_dir)
    (stdout, stderr, rcode) = _run_command([python_bin, 'setup.py', 'sdist'])
    if rcode == 0:
        sdist_path = max(glob.iglob(os.path.join(opts.source_dir, 'dist', 'salt-*.tar.gz')), key=os.path.getctime)
        log.info('sdist is located at %s', sdist_path)
        return sdist_path
    else:
        _abort('Failed to create sdist')

def build_centos(opts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build an RPM\n    '
    log.info('Building CentOS RPM')
    log.info('Detecting major release')
    try:
        with open('/etc/redhat-release') as fp_:
            redhat_release = fp_.read().strip()
            major_release = int(redhat_release.split()[2].split('.')[0])
    except (ValueError, IndexError):
        _abort("Unable to determine major release from /etc/redhat-release contents: '{}'".format(redhat_release))
    except OSError as exc:
        _abort('{}'.format(exc))
    log.info('major_release: %s', major_release)
    define_opts = ['--define', '_topdir {}'.format(os.path.join(opts.build_dir))]
    build_reqs = ['rpm-build']
    if major_release == 5:
        python_bin = 'python26'
        define_opts.extend(['--define', 'dist .el5'])
        if os.path.exists('/etc/yum.repos.d/saltstack.repo'):
            build_reqs.extend(['--enablerepo=saltstack'])
        build_reqs.extend(['python26-devel'])
    elif major_release == 6:
        build_reqs.extend(['python-devel'])
    elif major_release == 7:
        build_reqs.extend(['python-devel', 'systemd-units'])
    else:
        _abort('Unsupported major release: {}'.format(major_release))
    _run_command(['yum', '-y', 'install'] + build_reqs)
    try:
        sdist = _make_sdist(opts, python_bin=python_bin)
    except NameError:
        sdist = _make_sdist(opts)
    tarball_re = re.compile('^salt-([^-]+)(?:-(\\d+)-(g[0-9a-f]+))?\\.tar\\.gz$')
    try:
        (base, offset, oid) = tarball_re.match(os.path.basename(sdist)).groups()
    except AttributeError:
        _abort("Unable to extract version info from sdist filename '{}'".format(sdist))
    if offset is None:
        salt_pkgver = salt_srcver = base
    else:
        salt_pkgver = '.'.join((base, offset, oid))
        salt_srcver = '-'.join((base, offset, oid))
    log.info('salt_pkgver: %s', salt_pkgver)
    log.info('salt_srcver: %s', salt_srcver)
    for build_dir in 'BUILD BUILDROOT RPMS SOURCES SPECS SRPMS'.split():
        path = os.path.join(opts.build_dir, build_dir)
        try:
            os.makedirs(path)
        except OSError:
            pass
        if not os.path.isdir(path):
            _abort('Unable to make directory: {}'.format(path))
    build_sources_path = os.path.join(opts.build_dir, 'SOURCES')
    rpm_sources_path = os.path.join(opts.source_dir, 'pkg', 'rpm')
    _move(sdist, build_sources_path)
    for src in ('salt-master', 'salt-syndic', 'salt-minion', 'salt-api', 'salt-master.service', 'salt-syndic.service', 'salt-minion.service', 'salt-api.service', 'README.fedora', 'logrotate.salt', 'salt.bash'):
        shutil.copy(os.path.join(rpm_sources_path, src), build_sources_path)
    spec_path = os.path.join(opts.build_dir, 'SPECS', 'salt.spec')
    with open(opts.spec_file) as spec:
        spec_lines = spec.read().splitlines()
    with open(spec_path, 'w') as fp_:
        for line in spec_lines:
            if line.startswith('%global srcver '):
                line = '%global srcver {}'.format(salt_srcver)
            elif line.startswith('Version: '):
                line = 'Version: {}'.format(salt_pkgver)
            fp_.write(line + '\n')
    cmd = ['rpmbuild', '-ba']
    cmd.extend(define_opts)
    cmd.append(spec_path)
    (stdout, stderr, rcode) = _run_command(cmd)
    if rcode != 0:
        _abort('Build failed.')
    packages = glob.glob(os.path.join(opts.build_dir, 'RPMS', 'noarch', 'salt-*{}*.noarch.rpm'.format(salt_pkgver)))
    packages.extend(glob.glob(os.path.join(opts.build_dir, 'SRPMS', 'salt-{}*.src.rpm'.format(salt_pkgver))))
    return packages
if __name__ == '__main__':
    opts = _init()
    print('Starting {} build. Progress will be logged to {}.'.format(opts.platform, opts.log_file))
    log_format = '%(asctime)s.%(msecs)03d %(levelname)s: %(message)s'
    log_datefmt = '%H:%M:%S'
    log_level = LOG_LEVELS[opts.log_level] if opts.log_level in LOG_LEVELS else LOG_LEVELS['warning']
    logging.basicConfig(filename=opts.log_file, format=log_format, datefmt=log_datefmt, level=LOG_LEVELS[opts.log_level])
    if opts.log_level not in LOG_LEVELS:
        log.error("Invalid log level '%s', falling back to 'warning'", opts.log_level)
    if not opts.platform:
        _abort('Platform required')
    elif opts.platform.lower() == 'centos':
        artifacts = build_centos(opts)
    else:
        _abort("Unsupported platform '{}'".format(opts.platform))
    msg = 'Build complete. Artifacts will be stored in {}'.format(opts.artifact_dir)
    log.info(msg)
    print(msg)
    for artifact in artifacts:
        shutil.copy(artifact, opts.artifact_dir)
        log.info('Copied %s to artifact directory', artifact)
    log.info('Done!')