import inspect
from llnl.util.filesystem import working_dir
import spack.builder
import spack.package_base
from spack.directives import build_system, depends_on
from ._checks import BaseBuilder, execute_build_time_tests

class QMakePackage(spack.package_base.PackageBase):
    """Specialized class for packages built using qmake.

    For more information on the qmake build system, see:
    http://doc.qt.io/qt-5/qmake-manual.html
    """
    build_system_class = 'QMakePackage'
    legacy_buildsystem = 'qmake'
    build_system('qmake')
    depends_on('qmake', type='build', when='build_system=qmake')

@spack.builder.builder('qmake')
class QMakeBuilder(BaseBuilder):
    """The qmake builder provides three phases that can be overridden:

    1. :py:meth:`~.QMakeBuilder.qmake`
    2. :py:meth:`~.QMakeBuilder.build`
    3. :py:meth:`~.QMakeBuilder.install`

    They all have sensible defaults and for many packages the only thing
    necessary will be to override :py:meth:`~.QMakeBuilder.qmake_args`.
    """
    phases = ('qmake', 'build', 'install')
    legacy_methods = ('qmake_args', 'check')
    legacy_attributes = ('build_directory', 'build_time_test_callbacks')
    build_time_test_callbacks = ['check']

    @property
    def build_directory(self):
        if False:
            while True:
                i = 10
        'The directory containing the ``*.pro`` file.'
        return self.stage.source_path

    def qmake_args(self):
        if False:
            while True:
                i = 10
        'List of arguments passed to qmake.'
        return []

    def qmake(self, pkg, spec, prefix):
        if False:
            for i in range(10):
                print('nop')
        'Run ``qmake`` to configure the project and generate a Makefile.'
        with working_dir(self.build_directory):
            inspect.getmodule(self.pkg).qmake(*self.qmake_args())

    def build(self, pkg, spec, prefix):
        if False:
            for i in range(10):
                print('nop')
        'Make the build targets'
        with working_dir(self.build_directory):
            inspect.getmodule(self.pkg).make()

    def install(self, pkg, spec, prefix):
        if False:
            while True:
                i = 10
        'Make the install targets'
        with working_dir(self.build_directory):
            inspect.getmodule(self.pkg).make('install')

    def check(self):
        if False:
            i = 10
            return i + 15
        'Search the Makefile for a ``check:`` target and runs it if found.'
        with working_dir(self.build_directory):
            self.pkg._if_make_target_execute('check')
    spack.builder.run_after('build')(execute_build_time_tests)