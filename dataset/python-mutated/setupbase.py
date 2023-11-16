"""
This module defines the things that are used in setup.py for building IPython

This includes:

    * The basic arguments to setup
    * Functions for finding things like packages, package data, etc.
    * A function for checking dependencies.
"""
import os
import re
import sys
from glob import glob
from logging import log
from setuptools import Command
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.install_scripts import install_scripts
isfile = os.path.isfile
pjoin = os.path.join
repo_root = os.path.dirname(os.path.abspath(__file__))

def execfile(fname, globs, locs=None):
    if False:
        return 10
    locs = locs or globs
    with open(fname, encoding='utf-8') as f:
        exec(compile(f.read(), fname, 'exec'), globs, locs)

def file_doesnt_endwith(test, endings):
    if False:
        print('Hello World!')
    'Return true if test is a file and its name does NOT end with any\n    of the strings listed in endings.'
    if not isfile(test):
        return False
    for e in endings:
        if test.endswith(e):
            return False
    return True
execfile(pjoin(repo_root, 'IPython', 'core', 'release.py'), globals())
setup_args = dict(author=author, author_email=author_email, license=license)

def find_packages():
    if False:
        i = 10
        return i + 15
    "\n    Find all of IPython's packages.\n    "
    excludes = ['deathrow', 'quarantine']
    packages = []
    for (directory, subdirs, files) in os.walk('IPython'):
        package = directory.replace(os.path.sep, '.')
        if any((package.startswith('IPython.' + exc) for exc in excludes)):
            continue
        if '__init__.py' not in files:
            continue
        packages.append(package)
    return packages

def find_package_data():
    if False:
        return 10
    "\n    Find IPython's package_data.\n    "
    package_data = {'IPython.core': ['profile/README*'], 'IPython.core.tests': ['*.png', '*.jpg', 'daft_extension/*.py'], 'IPython.lib.tests': ['*.wav'], 'IPython.testing.plugin': ['*.txt']}
    return package_data

def check_package_data(package_data):
    if False:
        i = 10
        return i + 15
    'verify that package_data globs make sense'
    print('checking package data')
    for (pkg, data) in package_data.items():
        pkg_root = pjoin(*pkg.split('.'))
        for d in data:
            path = pjoin(pkg_root, d)
            if '*' in path:
                assert len(glob(path)) > 0, 'No files match pattern %s' % path
            else:
                assert os.path.exists(path), 'Missing package data: %s' % path

def check_package_data_first(command):
    if False:
        for i in range(10):
            print('nop')
    'decorator for checking package_data before running a given command\n\n    Probably only needs to wrap build_py\n    '

    class DecoratedCommand(command):

        def run(self):
            if False:
                return 10
            check_package_data(self.package_data)
            command.run(self)
    return DecoratedCommand

def find_data_files():
    if False:
        return 10
    "\n    Find IPython's data_files.\n\n    Just man pages at this point.\n    "
    if 'freebsd' in sys.platform:
        manpagebase = pjoin('man', 'man1')
    else:
        manpagebase = pjoin('share', 'man', 'man1')
    manpages = [f for f in glob(pjoin('docs', 'man', '*.1.gz')) if isfile(f)]
    if not manpages:
        manpages = [f for f in glob(pjoin('docs', 'man', '*.1')) if isfile(f)]
    data_files = [(manpagebase, manpages)]
    return data_files

def target_outdated(target, deps):
    if False:
        return 10
    "Determine whether a target is out of date.\n\n    target_outdated(target,deps) -> 1/0\n\n    deps: list of filenames which MUST exist.\n    target: single filename which may or may not exist.\n\n    If target doesn't exist or is older than any file listed in deps, return\n    true, otherwise return false.\n    "
    try:
        target_time = os.path.getmtime(target)
    except os.error:
        return 1
    for dep in deps:
        dep_time = os.path.getmtime(dep)
        if dep_time > target_time:
            return 1
    return 0

def target_update(target, deps, cmd):
    if False:
        i = 10
        return i + 15
    'Update a target with a given command given a list of dependencies.\n\n    target_update(target,deps,cmd) -> runs cmd if target is outdated.\n\n    This is just a wrapper around target_outdated() which calls the given\n    command if target is outdated.'
    if target_outdated(target, deps):
        os.system(cmd)

def find_entry_points():
    if False:
        i = 10
        return i + 15
    'Defines the command line entry points for IPython\n\n    This always uses setuptools-style entry points. When setuptools is not in\n    use, our own build_scripts_entrypt class below parses these and builds\n    command line scripts.\n\n    Each of our entry points gets a plain name, e.g. ipython, and a name\n    suffixed with the Python major version number, e.g. ipython3.\n    '
    ep = ['ipython%s = IPython:start_ipython']
    major_suffix = str(sys.version_info[0])
    return [e % '' for e in ep] + [e % major_suffix for e in ep]

class install_lib_symlink(Command):
    user_options = [('install-dir=', 'd', 'directory to install to')]

    def initialize_options(self):
        if False:
            return 10
        self.install_dir = None

    def finalize_options(self):
        if False:
            while True:
                i = 10
        self.set_undefined_options('symlink', ('install_lib', 'install_dir'))

    def run(self):
        if False:
            return 10
        if sys.platform == 'win32':
            raise Exception("This doesn't work on Windows.")
        pkg = os.path.join(os.getcwd(), 'IPython')
        dest = os.path.join(self.install_dir, 'IPython')
        if os.path.islink(dest):
            print('removing existing symlink at %s' % dest)
            os.unlink(dest)
        print('symlinking %s -> %s' % (pkg, dest))
        os.symlink(pkg, dest)

class unsymlink(install):

    def run(self):
        if False:
            while True:
                i = 10
        dest = os.path.join(self.install_lib, 'IPython')
        if os.path.islink(dest):
            print('removing symlink at %s' % dest)
            os.unlink(dest)
        else:
            print('No symlink exists at %s' % dest)

class install_symlinked(install):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if sys.platform == 'win32':
            raise Exception("This doesn't work on Windows.")
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)
    sub_commands = [('install_lib_symlink', lambda self: True), ('install_scripts_sym', lambda self: True)]

class install_scripts_for_symlink(install_scripts):
    """Redefined to get options from 'symlink' instead of 'install'.

    I love distutils almost as much as I love setuptools.
    """

    def finalize_options(self):
        if False:
            while True:
                i = 10
        self.set_undefined_options('build', ('build_scripts', 'build_dir'))
        self.set_undefined_options('symlink', ('install_scripts', 'install_dir'), ('force', 'force'), ('skip_build', 'skip_build'))

def git_prebuild(pkg_dir, build_cmd=build_py):
    if False:
        while True:
            i = 10
    'Return extended build or sdist command class for recording commit\n\n    records git commit in IPython.utils._sysinfo.commit\n\n    for use in IPython.utils.sysinfo.sys_info() calls after installation.\n    '

    class MyBuildPy(build_cmd):
        """ Subclass to write commit data into installation tree """

        def run(self):
            if False:
                while True:
                    i = 10
            print('check version number')
            loose_pep440re = re.compile('^(\\d+)\\.(\\d+)\\.(\\d+((a|b|rc)\\d+)?)(\\.post\\d+)?(\\.dev\\d*)?$')
            if not loose_pep440re.match(version):
                raise ValueError("Version number '%s' is not valid (should match [N!]N(.N)*[{a|b|rc}N][.postN][.devN])" % version)
            build_cmd.run(self)
            if hasattr(self, 'build_lib'):
                self._record_commit(self.build_lib)

        def make_release_tree(self, base_dir, files):
            if False:
                while True:
                    i = 10
            build_cmd.make_release_tree(self, base_dir, files)
            self._record_commit(base_dir)

        def _record_commit(self, base_dir):
            if False:
                while True:
                    i = 10
            import subprocess
            proc = subprocess.Popen('git rev-parse --short HEAD', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (repo_commit, _) = proc.communicate()
            repo_commit = repo_commit.strip().decode('ascii')
            out_pth = pjoin(base_dir, pkg_dir, 'utils', '_sysinfo.py')
            if os.path.isfile(out_pth) and (not repo_commit):
                return
            print("writing git commit '%s' to %s" % (repo_commit, out_pth))
            try:
                os.remove(out_pth)
            except (IOError, OSError):
                pass
            with open(out_pth, 'w', encoding='utf-8') as out_file:
                out_file.writelines(['# GENERATED BY setup.py\n', 'commit = "%s"\n' % repo_commit])
    return MyBuildPy