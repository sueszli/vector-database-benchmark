import datetime
import os
import re
import shutil
import subprocess
import sys
from subprocess import PIPE
from subprocess import STDOUT
from subprocess import Popen
import setuptools.command.build_py
import setuptools.command.egg_info
from setuptools import setup
import distutils.cmd
old_listdir = os.listdir

def listdir(path):
    if False:
        i = 10
        return i + 15
    l = old_listdir(path)
    if 'node_modules' in l:
        l.remove('node_modules')
    return l
os.listdir = listdir

def check_output(cmd, shell):
    if False:
        print('Hello World!')
    'Version of check_output which does not throw error'
    popen = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE)
    out = popen.communicate()[0].strip()
    if not isinstance(out, str):
        out = out.decode(sys.stdout.encoding)
    return out

def gitDescribeToPep440(version):
    if False:
        for i in range(10):
            print('nop')
    VERSION_MATCH = re.compile('(?P<major>\\d+)\\.(?P<minor>\\d+)\\.(?P<patch>\\d+)(\\.post(?P<post>\\d+))?(-(?P<dev>\\d+))?(-g(?P<commit>.+))?')
    v = VERSION_MATCH.search(version)
    if v:
        major = int(v.group('major'))
        minor = int(v.group('minor'))
        patch = int(v.group('patch'))
        if v.group('dev'):
            patch += 1
            dev = int(v.group('dev'))
            return '{}.{}.{}-dev{}'.format(major, minor, patch, dev)
        if v.group('post'):
            return '{}.{}.{}.post{}'.format(major, minor, patch, v.group('post'))
        return '{}.{}.{}'.format(major, minor, patch)
    return v

def mTimeVersion(init_file):
    if False:
        while True:
            i = 10
    cwd = os.path.dirname(os.path.abspath(init_file))
    m = 0
    for (root, dirs, files) in os.walk(cwd):
        for f in files:
            m = max(os.path.getmtime(os.path.join(root, f)), m)
    d = datetime.datetime.utcfromtimestamp(m)
    return d.strftime('%Y.%m.%d')

def getVersionFromArchiveId(git_archive_id='$Format:%ct %d$'):
    if False:
        print('Hello World!')
    ' Extract the tag if a source is from git archive.\n\n        When source is exported via `git archive`, the git_archive_id init value is modified\n        and placeholders are expanded to the "archived" revision:\n\n            %ct: committer date, UNIX timestamp\n            %d: ref names, like the --decorate option of git-log\n\n        See man gitattributes(5) and git-log(1) (PRETTY FORMATS) for more details.\n    '
    if not git_archive_id.startswith('$Format:'):
        match = re.search('tag:\\s*v([^,)]+)', git_archive_id)
        if match:
            return gitDescribeToPep440(match.group(1))
        tstamp = git_archive_id.strip().split()[0]
        d = datetime.datetime.utcfromtimestamp(int(tstamp))
        return d.strftime('%Y.%m.%d')
    return None

def getVersion(init_file):
    if False:
        return 10
    "\n    Return BUILDBOT_VERSION environment variable, content of VERSION file, git\n    tag or 'latest'\n    "
    try:
        return os.environ['BUILDBOT_VERSION']
    except KeyError:
        pass
    try:
        cwd = os.path.dirname(os.path.abspath(init_file))
        fn = os.path.join(cwd, 'VERSION')
        with open(fn) as f:
            return f.read().strip()
    except IOError:
        pass
    version = getVersionFromArchiveId()
    if version is not None:
        return version
    try:
        p = Popen(['git', 'describe', '--tags', '--always'], stdout=PIPE, stderr=STDOUT, cwd=cwd)
        out = p.communicate()[0]
        if not p.returncode and out:
            v = gitDescribeToPep440(str(out))
            if v:
                return v
    except OSError:
        pass
    try:
        return mTimeVersion(init_file)
    except Exception:
        return 'latest'

class BuildJsCommand(distutils.cmd.Command):
    """A custom command to run JS build."""
    description = 'run JS build'
    already_run = False

    def initialize_options(self):
        if False:
            while True:
                i = 10
        'Set default values for options.'

    def finalize_options(self):
        if False:
            for i in range(10):
                print('nop')
        'Post-process options.'

    def run(self):
        if False:
            while True:
                i = 10
        'Run command.'
        if self.already_run:
            return
        if os.path.isdir('build'):
            shutil.rmtree('build')
        package = self.distribution.packages[0]
        if os.path.exists('package.json'):
            shell = bool(os.name == 'nt')
            yarn_program = None
            for program in ['yarnpkg', 'yarn']:
                try:
                    yarn_version = check_output([program, '--version'], shell=shell)
                    if yarn_version != '':
                        yarn_program = program
                        break
                except subprocess.CalledProcessError:
                    pass
            assert yarn_program is not None, 'need nodejs and yarn installed in current PATH'
            yarn_bin = check_output([yarn_program, 'bin'], shell=shell).strip()
            commands = [[yarn_program, 'install', '--pure-lockfile'], [yarn_program, 'run', 'build']]
            for command in commands:
                self.announce('Running command: {}'.format(str(' '.join(command))), level=distutils.log.INFO)
                subprocess.check_call(command, shell=shell)
        self.copy_tree(os.path.join(package, 'static'), os.path.join('build', 'lib', package, 'static'))
        with open(os.path.join('build', 'lib', package, 'VERSION'), 'w') as f:
            f.write(self.distribution.metadata.version)
        with open(os.path.join(package, 'VERSION'), 'w') as f:
            f.write(self.distribution.metadata.version)
        self.already_run = True

class BuildPyCommand(setuptools.command.build_py.build_py):
    """Custom build command."""

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_command('build_js')
        super().run()

class EggInfoCommand(setuptools.command.egg_info.egg_info):
    """Custom egginfo command."""

    def run(self):
        if False:
            while True:
                i = 10
        self.run_command('build_js')
        super().run()

def setup_www_plugin(**kw):
    if False:
        while True:
            i = 10
    package = kw['packages'][0]
    if 'version' not in kw:
        kw['version'] = getVersion(os.path.join(package, '__init__.py'))
    setup(cmdclass=dict(egg_info=EggInfoCommand, build_py=BuildPyCommand, build_js=BuildJsCommand), **kw)