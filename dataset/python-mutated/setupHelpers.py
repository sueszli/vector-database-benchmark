from contextlib import suppress
import json
import os
import re
import shutil
import subprocess
import sys
from distutils import core
from typing import Dict, Any
from .generateChangelog import generateDebianChangelog
MERGE_SIZE_LIMIT = 100
FLAKE_CHECK_PATHS = ['pyqtgraph', 'examples', 'tools']
FLAKE_MANDATORY = set(['E101', 'E112', 'E122', 'E125', 'E133', 'E223', 'E224', 'E242', 'E273', 'E274', 'E901', 'E902', 'W191', 'W601', 'W602', 'W603', 'W604'])
FLAKE_RECOMMENDED = set(['E124', 'E231', 'E211', 'E261', 'E271', 'E272', 'E304', 'F401', 'F402', 'F403', 'F404', 'E501', 'E502', 'E702', 'E703', 'E711', 'E712', 'E721', 'F811', 'F812', 'F821', 'F822', 'F823', 'F831', 'F841', 'W292'])
FLAKE_OPTIONAL = set(['E121', 'E123', 'E126', 'E127', 'E128', 'E201', 'E202', 'E203', 'E221', 'E222', 'E225', 'E227', 'E226', 'E228', 'E241', 'E251', 'E262', 'E301', 'E302', 'E303', 'E401', 'E701', 'W291', 'W293', 'W391'])
FLAKE_IGNORE = set(['E111', 'E113'])

def checkStyle():
    if False:
        while True:
            i = 10
    ' Run flake8, checking only lines that are modified since the last\n    git commit. '
    print('flake8: check all code against mandatory error set...')
    errors = ','.join(FLAKE_MANDATORY)
    cmd = ['flake8', '--select=' + errors] + FLAKE_CHECK_PATHS
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = proc.stdout.read().decode('utf-8')
    ret = proc.wait()
    printFlakeOutput(output)
    print('check line endings in all files...')
    count = 0
    allowedEndings = set([None, '\n'])
    for (path, dirs, files) in os.walk('.'):
        if path.startswith('.' + os.path.sep + '.tox'):
            continue
        for f in files:
            if os.path.splitext(f)[1] not in ('.py', '.rst'):
                continue
            filename = os.path.join(path, f)
            with open(filename, 'U') as fh:
                _ = fh.readlines()
                endings = set(fh.newlines if isinstance(fh.newlines, tuple) else (fh.newlines,))
                endings -= allowedEndings
                if len(endings) > 0:
                    print('\x1b[0;31m' + 'File has invalid line endings: ' + '%s' % filename + '\x1b[0m')
                    ret = ret | 2
                count += 1
    print('checked line endings in %d files' % count)
    print('flake8: check new code against recommended error set...')
    diff = subprocess.check_output(['git', 'diff'])
    proc = subprocess.Popen(['flake8', '--diff', '--ignore=' + errors], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(diff)
    proc.stdin.close()
    output = proc.stdout.read().decode('utf-8')
    ret |= printFlakeOutput(output)
    if ret == 0:
        print('style test passed.')
    else:
        print('style test failed: %d' % ret)
    return ret

def printFlakeOutput(text):
    if False:
        while True:
            i = 10
    ' Print flake output, colored by error category.\n    Return 2 if there were any mandatory errors,\n    1 if only recommended / optional errors, and\n    0 if only optional errors.\n    '
    ret = 0
    gotError = False
    for line in text.split('\n'):
        m = re.match('[^\\:]+\\:\\d+\\:\\d+\\: (\\w+) .*', line)
        if m is None:
            print(line)
        else:
            gotError = True
            error = m.group(1)
            if error in FLAKE_MANDATORY:
                print('\x1b[0;31m' + line + '\x1b[0m')
                ret |= 2
            elif error in FLAKE_RECOMMENDED:
                print('\x1b[0;33m' + line + '\x1b[0m')
            elif error in FLAKE_OPTIONAL:
                print('\x1b[0;32m' + line + '\x1b[0m')
            elif error in FLAKE_IGNORE:
                continue
            else:
                print('\x1b[0;36m' + line + '\x1b[0m')
    if not gotError:
        print('    [ no errors ]\n')
    return ret

def unitTests():
    if False:
        for i in range(10):
            print('nop')
    '\n    Run all unit tests (using py.test)\n    Return the exit code.\n    '
    try:
        if sys.version[0] == '3':
            out = subprocess.check_output('PYTHONPATH=. py.test-3', shell=True)
        else:
            out = subprocess.check_output('PYTHONPATH=. py.test', shell=True)
        ret = 0
    except Exception as e:
        out = e.output
        ret = e.returncode
    print(out.decode('utf-8'))
    return ret

def checkMergeSize(sourceBranch=None, targetBranch=None, sourceRepo=None, targetRepo=None):
    if False:
        i = 10
        return i + 15
    '\n    Check that a git merge would not increase the repository size by\n    MERGE_SIZE_LIMIT.\n    '
    if sourceBranch is None:
        sourceBranch = getGitBranch()
        sourceRepo = '..'
    if targetBranch is None:
        if sourceBranch == 'master':
            targetBranch = 'master'
            targetRepo = 'https://github.com/pyqtgraph/pyqtgraph.git'
        else:
            targetBranch = 'master'
            targetRepo = '..'
    workingDir = '__merge-test-clone'
    env = dict(TARGET_BRANCH=targetBranch, SOURCE_BRANCH=sourceBranch, TARGET_REPO=targetRepo, SOURCE_REPO=sourceRepo, WORKING_DIR=workingDir)
    print('Testing merge size difference:\n  SOURCE: {SOURCE_REPO} {SOURCE_BRANCH}\n  TARGET: {TARGET_BRANCH} {TARGET_REPO}'.format(**env))
    setup = '\n        mkdir {WORKING_DIR} && cd {WORKING_DIR} &&\n        git init && git remote add -t {TARGET_BRANCH} target {TARGET_REPO} &&\n        git fetch target {TARGET_BRANCH} &&\n        git checkout -qf target/{TARGET_BRANCH} &&\n        git gc -q --aggressive\n        '.format(**env)
    checkSize = '\n        cd {WORKING_DIR} &&\n        du -s . | sed -e "s/\t.*//"\n        '.format(**env)
    merge = '\n        cd {WORKING_DIR} &&\n        git pull -q {SOURCE_REPO} {SOURCE_BRANCH} &&\n        git gc -q --aggressive\n        '.format(**env)
    try:
        print('Check out target branch:\n' + setup)
        subprocess.check_call(setup, shell=True)
        targetSize = int(subprocess.check_output(checkSize, shell=True))
        print('TARGET SIZE: %d kB' % targetSize)
        print('Merge source branch:\n' + merge)
        subprocess.check_call(merge, shell=True)
        mergeSize = int(subprocess.check_output(checkSize, shell=True))
        print('MERGE SIZE: %d kB' % mergeSize)
        diff = mergeSize - targetSize
        if diff <= MERGE_SIZE_LIMIT:
            print('DIFFERENCE: %d kB  [OK]' % diff)
            return 0
        else:
            print('\x1b[0;31m' + 'DIFFERENCE: %d kB  [exceeds %d kB]' % (diff, MERGE_SIZE_LIMIT) + '\x1b[0m')
            return 2
    finally:
        if os.path.isdir(workingDir):
            shutil.rmtree(workingDir)

def mergeTests():
    if False:
        print('Hello World!')
    ret = checkMergeSize()
    ret |= unitTests()
    ret |= checkStyle()
    if ret == 0:
        print('\x1b[0;32m' + '\nAll merge tests passed.' + '\x1b[0m')
    else:
        print('\x1b[0;31m' + '\nMerge tests failed.' + '\x1b[0m')
    return ret

def getInitVersion(pkgroot):
    if False:
        for i in range(10):
            print('nop')
    'Return the version string defined in __init__.py'
    path = os.getcwd()
    initfile = os.path.join(path, pkgroot, '__init__.py')
    init = open(initfile).read()
    m = re.search('__version__ = (\\S+)\\n', init)
    if m is None or len(m.groups()) != 1:
        raise Exception('Cannot determine __version__ from init file: ' + "'%s'!" % initfile)
    version = m.group(1).strip('\'"')
    return version

def gitCommit(name):
    if False:
        while True:
            i = 10
    'Return the commit ID for the given name.'
    commit = subprocess.check_output(['git', 'show', name], universal_newlines=True).split('\n')[0]
    assert commit[:7] == 'commit '
    return commit[7:]

def getGitVersion(tagPrefix):
    if False:
        i = 10
        return i + 15
    'Return a version string with information about this git checkout.\n    If the checkout is an unmodified, tagged commit, then return the tag\n    version\n\n    If this is not a tagged commit, return the output of\n    ``git describe --tags``\n\n    If this checkout has been modified, append "+" to the version.\n    '
    path = os.getcwd()
    if not os.path.isdir(os.path.join(path, '.git')):
        return None
    try:
        v = subprocess.check_output(['git', 'describe', '--tags', '--dirty', '--match="%s*"' % tagPrefix], stderr=subprocess.DEVNULL).strip().decode('utf-8')
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    assert v.startswith(tagPrefix)
    v = v[len(tagPrefix):]
    parts = v.split('-')
    modified = False
    if parts[-1] == 'dirty':
        modified = True
        parts = parts[:-1]
    local = None
    if len(parts) > 2 and re.match('\\d+', parts[-2]) and re.match('g[0-9a-f]{7}', parts[-1]):
        local = parts[-1]
        parts = parts[:-2]
    gitVersion = '-'.join(parts)
    if local is not None:
        gitVersion += '+' + local
    if modified:
        gitVersion += 'm'
    return gitVersion

def getGitBranch():
    if False:
        while True:
            i = 10
    m = re.search('\\* (.*)', subprocess.check_output(['git', 'branch'], universal_newlines=True))
    if m is None:
        return ''
    else:
        return m.group(1)

def getVersionStrings(pkg):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns 4 version strings:\n\n      * the version string to use for this build,\n      * version string requested with --force-version (or None)\n      * version string that describes the current git checkout (or None).\n      * version string in the pkg/__init__.py,\n\n    The first return value is (forceVersion or gitVersion or initVersion).\n    '
    initVersion = getInitVersion(pkgroot=pkg)
    try:
        gitVersion = getGitVersion(tagPrefix=pkg + '-')
    except:
        gitVersion = None
        sys.stderr.write('This appears to be a git checkout, but an error occurred while attempting to determine a version string for the current commit.\n')
        sys.excepthook(*sys.exc_info())
    forcedVersion = None
    for (i, arg) in enumerate(sys.argv):
        if arg.startswith('--force-version'):
            if arg == '--force-version':
                forcedVersion = sys.argv[i + 1]
                sys.argv.pop(i)
                sys.argv.pop(i)
            elif arg.startswith('--force-version='):
                forcedVersion = sys.argv[i].replace('--force-version=', '')
                sys.argv.pop(i)
    if forcedVersion is not None:
        version = forcedVersion
    else:
        version = initVersion
        if gitVersion is not None:
            (_, _, local) = gitVersion.partition('+')
            if local != '':
                version = version + '+' + local
                sys.stderr.write('Detected git commit; ' + "will use version string: '%s'\n" % version)
    return (version, forcedVersion, gitVersion, initVersion)
DEFAULT_ASV: Dict[str, Any] = {'version': 1, 'project': 'pyqtgraph', 'project_url': 'http://pyqtgraph.org/', 'repo': '.', 'branches': ['master'], 'environment_type': 'virtualenv', 'show_commit_url': 'http://github.com/pyqtgraph/pyqtgraph/commit/', 'matrix': {'numpy': '', 'pyqt5': ['', None], 'pyside2': ['', None]}, 'exclude': [{'pyqt5': '', 'pyside2': ''}, {'pyqt5': None, 'pyside2': None}], 'benchmark_dir': 'benchmarks', 'env_dir': '.asv/env', 'results_dir': '.asv/results', 'html_dir': '.asv/html', 'build_cache_size': 5}

class ASVConfigCommand(core.Command):
    description = 'Setup the ASV benchmarking config for this system'
    user_options = []

    def initialize_options(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def finalize_options(self) -> None:
        if False:
            print('Hello World!')
        pass

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        config = DEFAULT_ASV
        with suppress(FileNotFoundError, subprocess.CalledProcessError):
            cuda_check = subprocess.check_output(['nvcc', '--version'])
            match = re.search('release (\\d{1,2}\\.\\d)', cuda_check.decode('utf-8'))
            ver = match.groups()[0]
            ver_str = ver.replace('.', '')
            config['matrix'][f'cupy-cuda{ver_str}'] = ''
        with open('asv.conf.json', 'w') as conf_file:
            conf_file.write(json.dumps(config, indent=2))

class DebCommand(core.Command):
    description = 'build .deb package using `debuild -us -uc`'
    maintainer = 'Luke Campagnola <luke.campagnola@gmail.com>'
    debTemplate = 'debian'
    debDir = 'deb_build'
    user_options = []

    def initialize_options(self):
        if False:
            while True:
                i = 10
        self.cwd = None

    def finalize_options(self):
        if False:
            while True:
                i = 10
        self.cwd = os.getcwd()

    def run(self):
        if False:
            i = 10
            return i + 15
        version = self.distribution.get_version()
        pkgName = self.distribution.get_name()
        debName = 'python-' + pkgName
        debDir = self.debDir
        assert os.getcwd() == self.cwd, 'Must be in package root: %s' % self.cwd
        if os.path.isdir(debDir):
            raise Exception('DEB build dir already exists: "%s"' % debDir)
        sdist = 'dist/%s-%s.tar.gz' % (pkgName, version)
        if not os.path.isfile(sdist):
            raise Exception('No source distribution; ' + 'run `setup.py sdist` first.')
        os.mkdir(debDir)
        renamedSdist = '%s_%s.orig.tar.gz' % (debName, version)
        print('copy %s => %s' % (sdist, os.path.join(debDir, renamedSdist)))
        shutil.copy(sdist, os.path.join(debDir, renamedSdist))
        print('cd %s; tar -xzf %s' % (debDir, renamedSdist))
        if os.system('cd %s; tar -xzf %s' % (debDir, renamedSdist)) != 0:
            raise Exception('Error extracting source distribution.')
        buildDir = '%s/%s-%s' % (debDir, pkgName, version)
        print('copytree %s => %s' % (self.debTemplate, buildDir + '/debian'))
        shutil.copytree(self.debTemplate, buildDir + '/debian')
        chlog = generateDebianChangelog(pkgName, 'CHANGELOG', version, self.maintainer)
        print('write changelog %s' % buildDir + '/debian/changelog')
        open(buildDir + '/debian/changelog', 'w').write(chlog)
        print('cd %s; debuild -us -uc' % buildDir)
        if os.system('cd %s; debuild -us -uc' % buildDir) != 0:
            raise Exception('Error during debuild.')

class DebugCommand(core.Command):
    """Just for learning about distutils."""
    description = ''
    user_options = []

    def initialize_options(self):
        if False:
            i = 10
            return i + 15
        pass

    def finalize_options(self):
        if False:
            print('Hello World!')
        pass

    def run(self):
        if False:
            print('Hello World!')
        global cmd
        cmd = self
        print(self.distribution.name)
        print(self.distribution.version)

class TestCommand(core.Command):
    description = ('Run all package tests and exit immediately with ', 'informative return code.')
    user_options = []

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        sys.exit(unitTests())

    def initialize_options(self):
        if False:
            i = 10
            return i + 15
        pass

    def finalize_options(self):
        if False:
            i = 10
            return i + 15
        pass

class StyleCommand(core.Command):
    description = ('Check all code for style, exit immediately with ', 'informative return code.')
    user_options = []

    def run(self):
        if False:
            while True:
                i = 10
        sys.exit(checkStyle())

    def initialize_options(self):
        if False:
            print('Hello World!')
        pass

    def finalize_options(self):
        if False:
            return 10
        pass

class MergeTestCommand(core.Command):
    description = ('Run all tests needed to determine whether the current ', 'code is suitable for merge.')
    user_options = []

    def run(self):
        if False:
            return 10
        sys.exit(mergeTests())

    def initialize_options(self):
        if False:
            i = 10
            return i + 15
        pass

    def finalize_options(self):
        if False:
            for i in range(10):
                print('nop')
        pass