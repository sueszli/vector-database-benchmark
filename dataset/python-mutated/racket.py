import os
from typing import Optional, Tuple
import llnl.util.filesystem as fs
import llnl.util.lang as lang
import llnl.util.tty as tty
import spack.builder
from spack.build_environment import SPACK_NO_PARALLEL_MAKE
from spack.directives import build_system, extends, maintainers
from spack.package_base import PackageBase
from spack.util.cpus import determine_number_of_jobs
from spack.util.environment import env_flag
from spack.util.executable import Executable, ProcessError

class RacketPackage(PackageBase):
    """Specialized class for packages that are built using Racket's
    `raco pkg install` and `raco setup` commands.
    """
    maintainers('elfprince13')
    build_system_class = 'RacketPackage'
    legacy_buildsystem = 'racket'
    build_system('racket')
    extends('racket', when='build_system=racket')
    racket_name: Optional[str] = None
    parallel = True

    @lang.classproperty
    def homepage(cls):
        if False:
            print('Hello World!')
        if cls.racket_name:
            return 'https://pkgs.racket-lang.org/package/{0}'.format(cls.racket_name)
        return None

@spack.builder.builder('racket')
class RacketBuilder(spack.builder.Builder):
    """The Racket builder provides an ``install`` phase that can be overridden."""
    phases = ('install',)
    legacy_methods: Tuple[str, ...] = tuple()
    legacy_attributes = ('build_directory', 'build_time_test_callbacks', 'subdirectory')
    build_time_test_callbacks = ['check']
    racket_name: Optional[str] = None

    @property
    def subdirectory(self):
        if False:
            while True:
                i = 10
        if self.pkg.racket_name:
            return 'pkgs/{0}'.format(self.pkg.racket_name)
        return None

    @property
    def build_directory(self):
        if False:
            for i in range(10):
                print('nop')
        ret = os.getcwd()
        if self.subdirectory:
            ret = os.path.join(ret, self.subdirectory)
        return ret

    def install(self, pkg, spec, prefix):
        if False:
            while True:
                i = 10
        'Install everything from build directory.'
        raco = Executable('raco')
        with fs.working_dir(self.build_directory):
            parallel = self.pkg.parallel and (not env_flag(SPACK_NO_PARALLEL_MAKE))
            args = ['pkg', 'install', '-t', 'dir', '-n', self.pkg.racket_name, '--deps', 'fail', '--ignore-implies', '--copy', '-i', '-j', str(determine_number_of_jobs(parallel=parallel)), '--', os.getcwd()]
            try:
                raco(*args)
            except ProcessError:
                args.insert(-2, '--skip-installed')
                raco(*args)
                msg = 'Racket package {0} was already installed, uninstalling via Spack may make someone unhappy!'
                tty.warn(msg.format(self.pkg.racket_name))