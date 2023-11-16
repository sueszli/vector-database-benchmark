"""
Build steps for the windows binary packages.

The script is designed to be called by appveyor. Subcommands map the steps in
'appveyor.yml'.

"""
import re
import os
import sys
import json
import shutil
import logging
import subprocess as sp
from glob import glob
from pathlib import Path
from zipfile import ZipFile
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
opt = None
STEP_PREFIX = 'step_'
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    if False:
        while True:
            i = 10
    global opt
    opt = parse_cmdline()
    logger.setLevel(opt.loglevel)
    cmd = globals()[STEP_PREFIX + opt.step]
    cmd()

def setup_build_env():
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the environment variables according to the build environment\n    '
    setenv('VS_VER', opt.vs_ver)
    path = [str(opt.py_dir), str(opt.py_dir / 'Scripts'), 'C:\\Strawberry\\Perl\\bin', 'C:\\Program Files\\Git\\mingw64\\bin', str(opt.ssl_build_dir / 'bin'), os.environ['PATH']]
    setenv('PATH', os.pathsep.join(path))
    logger.info('Configuring compiler')
    bat_call([opt.vc_dir / 'vcvarsall.bat', 'x86' if opt.arch_32 else 'amd64'])

def python_info():
    if False:
        print('Hello World!')
    logger.info('Python Information')
    run_python(['--version'], stderr=sp.STDOUT)
    run_python(['-c', "import sys; print('64bit: %s' % (sys.maxsize > 2**32))"])

def step_install():
    if False:
        i = 10
        return i + 15
    python_info()
    configure_sdk()
    configure_postgres()
    install_python_build_tools()

def install_python_build_tools():
    if False:
        for i in range(10):
            print('nop')
    '\n    Install or upgrade pip and build tools.\n    '
    run_python('-m pip install --upgrade pip setuptools wheel'.split())

def configure_sdk():
    if False:
        for i in range(10):
            print('nop')
    if opt.arch_64:
        for fn in glob('C:\\Program Files\\Microsoft SDKs\\Windows\\v7.0\\Bin\\x64\\rc*'):
            copy_file(fn, 'C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\Bin')

def configure_postgres():
    if False:
        return 10
    '\n    Set up PostgreSQL config before the service starts.\n    '
    logger.info('Configuring Postgres')
    with (opt.pg_data_dir / 'postgresql.conf').open('a') as f:
        print('max_prepared_transactions = 10', file=f)
        print('ssl = on', file=f)
    cwd = os.getcwd()
    os.chdir(opt.pg_data_dir)
    run_openssl('req -new -x509 -days 365 -nodes -text -out server.crt -keyout server.key -subj /CN=initd.org'.split())
    run_openssl('req -new -nodes -text -out root.csr -keyout root.key -subj /CN=initd.org'.split())
    run_openssl('x509 -req -in root.csr -text -days 3650 -extensions v3_ca -signkey root.key -out root.crt'.split())
    run_openssl('req -new -nodes -text -out server.csr -keyout server.key -subj /CN=initd.org'.split())
    run_openssl('x509 -req -in server.csr -text -days 365 -CA root.crt -CAkey root.key -CAcreateserial -out server.crt'.split())
    os.chdir(cwd)

def run_openssl(args):
    if False:
        while True:
            i = 10
    'Run the appveyor-installed openssl with some args.'
    openssl = Path('C:\\OpenSSL-v111-Win64') / 'bin' / 'openssl'
    return run_command([openssl] + args)

def step_build_script():
    if False:
        return 10
    setup_build_env()
    build_openssl()
    build_libpq()
    build_psycopg()
    if opt.is_wheel:
        build_binary_packages()

def build_openssl():
    if False:
        while True:
            i = 10
    top = opt.ssl_build_dir
    if (top / 'lib' / 'libssl.lib').exists():
        return
    logger.info('Building OpenSSL')
    ensure_dir(top / 'include' / 'openssl')
    ensure_dir(top / 'lib')
    if opt.arch_32:
        target = 'VC-WIN32'
        setenv('VCVARS_PLATFORM', 'x86')
    else:
        target = 'VC-WIN64A'
        setenv('VCVARS_PLATFORM', 'amd64')
        setenv('CPU', 'AMD64')
    ver = os.environ['OPENSSL_VERSION']
    zipname = f'OpenSSL_{ver}.zip'
    zipfile = opt.cache_dir / zipname
    if not zipfile.exists():
        download(f'https://github.com/openssl/openssl/archive/{zipname}', zipfile)
    with ZipFile(zipfile) as z:
        z.extractall(path=opt.build_dir)
    sslbuild = opt.build_dir / f'openssl-OpenSSL_{ver}'
    os.chdir(sslbuild)
    run_command(['perl', 'Configure', target, 'no-asm'] + ['no-shared', 'no-zlib', f'--prefix={top}', f'--openssldir={top}'])
    run_command('nmake build_libs install_sw'.split())
    assert (top / 'lib' / 'libssl.lib').exists()
    os.chdir(opt.clone_dir)
    shutil.rmtree(sslbuild)

def build_libpq():
    if False:
        return 10
    top = opt.pg_build_dir
    if (top / 'lib' / 'libpq.lib').exists():
        return
    logger.info('Building libpq')
    ensure_dir(top / 'include')
    ensure_dir(top / 'lib')
    ensure_dir(top / 'bin')
    ver = os.environ['POSTGRES_VERSION']
    zipname = f'postgres-REL_{ver}.zip'
    zipfile = opt.cache_dir / zipname
    if not zipfile.exists():
        download(f'https://github.com/postgres/postgres/archive/REL_{ver}.zip', zipfile)
    with ZipFile(zipfile) as z:
        z.extractall(path=opt.build_dir)
    pgbuild = opt.build_dir / f'postgres-REL_{ver}'
    os.chdir(pgbuild)
    os.chdir('src/tools/msvc')
    with open('config.pl', 'w') as f:
        print('$config->{ldap} = 0;\n$config->{openssl} = "%s";\n\n1;\n' % str(opt.ssl_build_dir).replace('\\', '\\\\'), file=f)
    file_replace('Mkvcbuild.pm', "'libpq', 'dll'", "'libpq', 'lib'")
    run_command([which('build'), 'libpgport'])
    run_command([which('build'), 'libpgcommon'])
    run_command([which('build'), 'libpq'])
    with (pgbuild / 'src/backend/parser/gram.h').open('w') as f:
        print('', file=f)
    file_replace('Install.pm', 'qw(Install)', 'qw(Install CopyIncludeFiles)')
    run_command(['perl', '-MInstall=CopyIncludeFiles', '-e'] + [f"chdir('../../..'); CopyIncludeFiles('{top}')"])
    for lib in ('libpgport', 'libpgcommon', 'libpq'):
        copy_file(pgbuild / f'Release/{lib}/{lib}.lib', top / 'lib')
    for dir in ('win32', 'win32_msvc'):
        merge_dir(pgbuild / f'src/include/port/{dir}', pgbuild / 'src/include')
    os.chdir(pgbuild / 'src/bin/pg_config')
    run_command(['cl', 'pg_config.c', '/MT', '/nologo', f'/I{pgbuild}\\src\\include'] + ['/link', f'/LIBPATH:{top}\\lib'] + ['libpgcommon.lib', 'libpgport.lib', 'advapi32.lib'] + ['/NODEFAULTLIB:libcmt.lib'] + [f'/OUT:{top}\\bin\\pg_config.exe'])
    assert (top / 'lib' / 'libpq.lib').exists()
    assert (top / 'bin' / 'pg_config.exe').exists()
    os.chdir(opt.clone_dir)
    shutil.rmtree(pgbuild)

def build_psycopg():
    if False:
        print('Hello World!')
    os.chdir(opt.package_dir)
    patch_package_name()
    add_pg_config_path()
    run_python(['setup.py', 'build_ext', '--have-ssl'] + ['-l', 'libpgcommon libpgport'] + ['-L', opt.ssl_build_dir / 'lib'] + ['-I', opt.ssl_build_dir / 'include'])
    run_python(['setup.py', 'build_py'])

def patch_package_name():
    if False:
        print('Hello World!')
    'Change the psycopg2 package name in the setup.py if required.'
    if opt.package_name == 'psycopg2':
        return
    logger.info('changing package name to %s', opt.package_name)
    with (opt.package_dir / 'setup.py').open() as f:
        data = f.read()
    rex = re.compile('name=["\']psycopg2["\']')
    assert len(rex.findall(data)) == 1, rex.findall(data)
    data = rex.sub(f'name="{opt.package_name}"', data)
    with (opt.package_dir / 'setup.py').open('w') as f:
        f.write(data)

def build_binary_packages():
    if False:
        return 10
    'Create wheel binary packages.'
    os.chdir(opt.package_dir)
    add_pg_config_path()
    run_python(['setup.py', 'bdist_wheel', '-d', opt.dist_dir])

def step_after_build():
    if False:
        print('Hello World!')
    if not opt.is_wheel:
        install_built_package()
    else:
        install_binary_package()

def install_built_package():
    if False:
        while True:
            i = 10
    'Install the package just built by setup build.'
    os.chdir(opt.package_dir)
    add_pg_config_path()
    run_python(['setup.py', 'install'])
    shutil.rmtree('psycopg2.egg-info')

def install_binary_package():
    if False:
        while True:
            i = 10
    'Install the package from a packaged wheel.'
    run_python(['-m', 'pip', 'install', '--no-index', '-f', opt.dist_dir] + [opt.package_name])

def add_pg_config_path():
    if False:
        for i in range(10):
            print('nop')
    'Allow finding in the path the pg_config just built.'
    pg_path = str(opt.pg_build_dir / 'bin')
    if pg_path not in os.environ['PATH'].split(os.pathsep):
        setenv('PATH', os.pathsep.join([pg_path, os.environ['PATH']]))

def step_before_test():
    if False:
        while True:
            i = 10
    print_psycopg2_version()
    run_command([opt.pg_bin_dir / 'createdb', os.environ['PSYCOPG2_TESTDB']])
    run_command([opt.pg_bin_dir / 'psql', '-d', os.environ['PSYCOPG2_TESTDB']] + ['-c', 'CREATE EXTENSION hstore'])

def print_psycopg2_version():
    if False:
        print('Hello World!')
    'Print psycopg2 and libpq versions installed.'
    for expr in ('psycopg2.__version__', 'psycopg2.__libpq_version__', 'psycopg2.extensions.libpq_version()'):
        out = out_python(['-c', f'import psycopg2; print({expr})'])
        logger.info('built %s: %s', expr, out.decode('ascii'))

def step_test_script():
    if False:
        i = 10
        return i + 15
    check_libpq_version()
    run_test_suite()

def check_libpq_version():
    if False:
        i = 10
        return i + 15
    '\n    Fail if the package installed is not using the expected libpq version.\n    '
    want_ver = tuple(map(int, os.environ['POSTGRES_VERSION'].split('_')))
    want_ver = '%d%04d' % want_ver
    got_ver = out_python(['-c'] + ['import psycopg2; print(psycopg2.extensions.libpq_version())']).decode('ascii').rstrip()
    assert want_ver == got_ver, f'libpq version mismatch: {want_ver!r} != {got_ver!r}'

def run_test_suite():
    if False:
        while True:
            i = 10
    os.environ.pop('OPENSSL_CONF', None)
    args = ['-c', "import tests; tests.unittest.main(defaultTest='tests.test_suite')"]
    if opt.is_wheel:
        os.environ['PSYCOPG2_TEST_FAST'] = '1'
    else:
        args.append('--verbose')
    os.chdir(opt.package_dir)
    run_python(args)

def step_on_success():
    if False:
        while True:
            i = 10
    print_sha1_hashes()
    if setup_ssh():
        upload_packages()

def print_sha1_hashes():
    if False:
        print('Hello World!')
    '\n    Print the packages sha1 so their integrity can be checked upon signing.\n    '
    logger.info('artifacts SHA1 hashes:')
    os.chdir(opt.package_dir / 'dist')
    run_command([which('sha1sum'), '-b', 'psycopg2-*/*'])

def setup_ssh():
    if False:
        for i in range(10):
            print('nop')
    "\n    Configure ssh to upload built packages where they can be retrieved.\n\n    Return False if can't configure and upload shoould be skipped.\n    "
    if os.environ['APPVEYOR_ACCOUNT_NAME'] != 'psycopg':
        logger.warn('skipping artifact upload: you are not psycopg')
        return False
    pkey = os.environ.get('REMOTE_KEY', None)
    if not pkey:
        logger.warn('skipping artifact upload: no remote key')
        return False
    pkey = pkey.replace(' ', '\n')
    with (opt.clone_dir / 'data/id_rsa-psycopg-upload').open('w') as f:
        f.write(f'-----BEGIN RSA PRIVATE KEY-----\n{pkey}\n-----END RSA PRIVATE KEY-----\n')
    ensure_dir('C:\\MinGW\\msys\\1.0\\home\\appveyor\\.ssh')
    return True

def upload_packages():
    if False:
        return 10
    logger.info('uploading artifacts')
    os.chdir(opt.clone_dir)
    run_command(['C:\\MinGW\\msys\\1.0\\bin\\rsync', '-avr'] + ['-e', 'C:\\MinGW\\msys\\1.0\\bin\\ssh -F data/ssh_config'] + ['psycopg2/dist/', 'upload:'])

def download(url, fn):
    if False:
        for i in range(10):
            print('nop')
    'Download a file locally'
    logger.info('downloading %s', url)
    with open(fn, 'wb') as fo, urlopen(url) as fi:
        while 1:
            data = fi.read(8192)
            if not data:
                break
            fo.write(data)
    logger.info('file downloaded: %s', fn)

def file_replace(fn, s1, s2):
    if False:
        while True:
            i = 10
    '\n    Replace all the occurrences of the string s1 into s2 in the file fn.\n    '
    assert os.path.exists(fn)
    with open(fn, 'r+') as f:
        data = f.read()
        f.seek(0)
        f.write(data.replace(s1, s2))
        f.truncate()

def merge_dir(src, tgt):
    if False:
        print('Hello World!')
    '\n    Merge the content of the directory src into the directory tgt\n\n    Reproduce the semantic of "XCOPY /Y /S src/* tgt"\n    '
    src = str(src)
    for (dp, _dns, fns) in os.walk(src):
        logger.debug('dirpath %s', dp)
        if not fns:
            continue
        assert dp.startswith(src)
        subdir = dp[len(src):].lstrip(os.sep)
        tgtdir = ensure_dir(os.path.join(tgt, subdir))
        for fn in fns:
            copy_file(os.path.join(dp, fn), tgtdir)

def bat_call(cmdline):
    if False:
        print('Hello World!')
    "\n    Simulate 'CALL' from a batch file\n\n    Execute CALL *cmdline* and export the changed environment to the current\n    environment.\n\n    nana-nana-nana-nana...\n\n    "
    if not isinstance(cmdline, str):
        cmdline = map(str, cmdline)
        cmdline = ' '.join((c if ' ' not in c else '"%s"' % c for c in cmdline))
    data = f'CALL {cmdline}\n{opt.py_exe} -c "import os, sys, json; json.dump(dict(os.environ), sys.stdout, indent=2)"\n'
    logger.debug('preparing file to batcall:\n\n%s', data)
    with NamedTemporaryFile(suffix='.bat') as tmp:
        fn = tmp.name
    with open(fn, 'w') as f:
        f.write(data)
    try:
        out = out_command(fn)
        m = list(re.finditer(b'^{', out, re.MULTILINE))[-1]
        out = out[m.start():]
        env = json.loads(out)
        for (k, v) in env.items():
            if os.environ.get(k) != v:
                setenv(k, v)
    finally:
        os.remove(fn)

def ensure_dir(dir):
    if False:
        i = 10
        return i + 15
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.is_dir():
        logger.info('creating directory %s', dir)
        dir.mkdir(parents=True)
    return dir

def run_command(cmdline, **kwargs):
    if False:
        return 10
    'Run a command, raise on error.'
    if not isinstance(cmdline, str):
        cmdline = list(map(str, cmdline))
    logger.info('running command: %s', cmdline)
    sp.check_call(cmdline, **kwargs)

def out_command(cmdline, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Run a command, return its output, raise on error.'
    if not isinstance(cmdline, str):
        cmdline = list(map(str, cmdline))
    logger.info('running command: %s', cmdline)
    data = sp.check_output(cmdline, **kwargs)
    return data

def run_python(args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Run a script in the target Python.\n    '
    return run_command([opt.py_exe] + args, **kwargs)

def out_python(args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Return the output of a script run in the target Python.\n    '
    return out_command([opt.py_exe] + args, **kwargs)

def copy_file(src, dst):
    if False:
        return 10
    logger.info('copying file %s -> %s', src, dst)
    shutil.copy(src, dst)

def setenv(k, v):
    if False:
        for i in range(10):
            print('nop')
    logger.debug('setting %s=%s', k, v)
    os.environ[k] = v

def which(name):
    if False:
        return 10
    '\n    Return the full path of a command found on the path\n    '
    (base, ext) = os.path.splitext(name)
    if not ext:
        exts = ('.com', '.exe', '.bat', '.cmd')
    else:
        exts = (ext,)
    for dir in ['.'] + os.environ['PATH'].split(os.pathsep):
        for ext in exts:
            fn = os.path.join(dir, base + ext)
            if os.path.isfile(fn):
                return fn
    raise Exception(f"couldn't find program on path: {name}")

class Options:
    """
    An object exposing the script configuration from env vars and command line.
    """

    @property
    def py_ver(self):
        if False:
            return 10
        'The Python version to build as 2 digits string.\n\n        For large values of 2, occasionally.\n        '
        rv = os.environ['PY_VER']
        assert rv in ('37', '38', '39', '310', '311', '312'), rv
        return rv

    @property
    def py_arch(self):
        if False:
            while True:
                i = 10
        'The Python architecture to build, 32 or 64.'
        rv = os.environ['PY_ARCH']
        assert rv in ('32', '64'), rv
        return int(rv)

    @property
    def arch_32(self):
        if False:
            print('Hello World!')
        'True if the Python architecture to build is 32 bits.'
        return self.py_arch == 32

    @property
    def arch_64(self):
        if False:
            i = 10
            return i + 15
        'True if the Python architecture to build is 64 bits.'
        return self.py_arch == 64

    @property
    def package_name(self):
        if False:
            for i in range(10):
                print('nop')
        return os.environ.get('CONFIGURATION', 'psycopg2')

    @property
    def package_version(self):
        if False:
            return 10
        'The psycopg2 version number to build.'
        with (self.package_dir / 'setup.py').open() as f:
            data = f.read()
        m = re.search('^PSYCOPG_VERSION\\s*=\\s*[\'"](.*)[\'"]', data, re.MULTILINE)
        return m.group(1)

    @property
    def is_wheel(self):
        if False:
            for i in range(10):
                print('nop')
        'Are we building the wheel packages or just the extension?'
        workflow = os.environ['WORKFLOW']
        return workflow == 'packages'

    @property
    def py_dir(self):
        if False:
            print('Hello World!')
        '\n        The path to the target python binary to execute.\n        '
        dirname = ''.join(['C:\\Python', self.py_ver, '-x64' if self.arch_64 else ''])
        return Path(dirname)

    @property
    def py_exe(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The full path of the target python executable.\n        '
        return self.py_dir / 'python.exe'

    @property
    def vc_dir(self):
        if False:
            i = 10
            return i + 15
        '\n        The path of the Visual C compiler.\n        '
        if self.vs_ver == '16.0':
            path = Path('C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build')
        else:
            path = Path('C:\\Program Files (x86)\\Microsoft Visual Studio %s\\VC' % self.vs_ver)
        return path

    @property
    def vs_ver(self):
        if False:
            for i in range(10):
                print('nop')
        vsvers = {'37': '14.0', '38': '14.0', '39': '16.0', '310': '16.0', '311': '16.0', '312': '16.0'}
        return vsvers[self.py_ver]

    @property
    def clone_dir(self):
        if False:
            return 10
        'The directory where the repository is cloned.'
        return Path('C:\\Project')

    @property
    def appveyor_pg_dir(self):
        if False:
            print('Hello World!')
        'The directory of the postgres service made available by Appveyor.'
        return Path(os.environ['POSTGRES_DIR'])

    @property
    def pg_data_dir(self):
        if False:
            print('Hello World!')
        'The data dir of the appveyor postgres service.'
        return self.appveyor_pg_dir / 'data'

    @property
    def pg_bin_dir(self):
        if False:
            while True:
                i = 10
        'The bin dir of the appveyor postgres service.'
        return self.appveyor_pg_dir / 'bin'

    @property
    def pg_build_dir(self):
        if False:
            print('Hello World!')
        'The directory where to build the postgres libraries for psycopg.'
        return self.cache_arch_dir / 'postgresql'

    @property
    def ssl_build_dir(self):
        if False:
            while True:
                i = 10
        'The directory where to build the openssl libraries for psycopg.'
        return self.cache_arch_dir / 'openssl'

    @property
    def cache_arch_dir(self):
        if False:
            while True:
                i = 10
        rv = self.cache_dir / str(self.py_arch) / self.vs_ver
        return ensure_dir(rv)

    @property
    def cache_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return Path('C:\\Others')

    @property
    def build_dir(self):
        if False:
            return 10
        rv = self.cache_arch_dir / 'Builds'
        return ensure_dir(rv)

    @property
    def package_dir(self):
        if False:
            while True:
                i = 10
        return self.clone_dir

    @property
    def dist_dir(self):
        if False:
            return 10
        'The directory where to build packages to distribute.'
        return self.package_dir / 'dist' / f'psycopg2-{self.package_version}'

def parse_cmdline():
    if False:
        while True:
            i = 10
    parser = ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('-q', '--quiet', help='Talk less', dest='loglevel', action='store_const', const=logging.WARN, default=logging.INFO)
    g.add_argument('-v', '--verbose', help='Talk more', dest='loglevel', action='store_const', const=logging.DEBUG, default=logging.INFO)
    steps = [n[len(STEP_PREFIX):] for n in globals() if n.startswith(STEP_PREFIX) and callable(globals()[n])]
    parser.add_argument('step', choices=steps, help='the appveyor step to execute')
    opt = parser.parse_args(namespace=Options())
    return opt
if __name__ == '__main__':
    sys.exit(main())