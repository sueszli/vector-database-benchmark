import inspect
import os
from typing import List
import llnl.util.filesystem as fs
import spack.builder
import spack.package_base
from spack.directives import build_system, conflicts, depends_on, variant
from spack.multimethod import when
from ._checks import BaseBuilder, execute_build_time_tests

class MesonPackage(spack.package_base.PackageBase):
    """Specialized class for packages built using Meson. For more information
    on the Meson build system, see https://mesonbuild.com/
    """
    build_system_class = 'MesonPackage'
    legacy_buildsystem = 'meson'
    build_system('meson')
    with when('build_system=meson'):
        variant('buildtype', default='release', description='Meson build type', values=('plain', 'debug', 'debugoptimized', 'release', 'minsize'))
        variant('default_library', default='shared', values=('shared', 'static'), multi=True, description='Build shared libs, static libs or both')
        variant('strip', default=False, description='Strip targets on install')
        depends_on('meson', type='build')
        depends_on('ninja', type='build')
        conflicts('^python@3.12:', when='^meson@:1.0')

    def flags_to_build_system_args(self, flags):
        if False:
            i = 10
            return i + 15
        'Produces a list of all command line arguments to pass the specified\n        compiler flags to meson.'
        setattr(self, 'meson_flag_args', [])

@spack.builder.builder('meson')
class MesonBuilder(BaseBuilder):
    """The Meson builder encodes the default way to build software with Meson.
    The builder has three phases that can be overridden, if need be:

            1. :py:meth:`~.MesonBuilder.meson`
            2. :py:meth:`~.MesonBuilder.build`
            3. :py:meth:`~.MesonBuilder.install`

    They all have sensible defaults and for many packages the only thing
    necessary will be to override :py:meth:`~.MesonBuilder.meson_args`.

    For a finer tuning you may also override:

        +-----------------------------------------------+--------------------+
        | **Method**                                    | **Purpose**        |
        +===============================================+====================+
        | :py:meth:`~.MesonBuilder.root_mesonlists_dir` | Location of the    |
        |                                               | root MesonLists.txt|
        +-----------------------------------------------+--------------------+
        | :py:meth:`~.MesonBuilder.build_directory`     | Directory where to |
        |                                               | build the package  |
        +-----------------------------------------------+--------------------+
    """
    phases = ('meson', 'build', 'install')
    legacy_methods = ('meson_args', 'check')
    legacy_attributes = ('build_targets', 'install_targets', 'build_time_test_callbacks', 'root_mesonlists_dir', 'std_meson_args', 'build_directory')
    build_targets: List[str] = []
    install_targets = ['install']
    build_time_test_callbacks = ['check']

    @property
    def archive_files(self):
        if False:
            for i in range(10):
                print('nop')
        'Files to archive for packages based on Meson'
        return [os.path.join(self.build_directory, 'meson-logs', 'meson-log.txt')]

    @property
    def root_mesonlists_dir(self):
        if False:
            for i in range(10):
                print('nop')
        'Relative path to the directory containing meson.build\n\n        This path is relative to the root of the extracted tarball,\n        not to the ``build_directory``. Defaults to the current directory.\n        '
        return self.pkg.stage.source_path

    @property
    def std_meson_args(self):
        if False:
            while True:
                i = 10
        'Standard meson arguments provided as a property for convenience\n        of package writers.\n        '
        std_meson_args = MesonBuilder.std_args(self.pkg)
        std_meson_args += getattr(self, 'meson_flag_args', [])
        return std_meson_args

    @staticmethod
    def std_args(pkg):
        if False:
            print('Hello World!')
        'Standard meson arguments for a generic package.'
        try:
            build_type = pkg.spec.variants['buildtype'].value
        except KeyError:
            build_type = 'release'
        strip = 'true' if '+strip' in pkg.spec else 'false'
        if 'default_library=static,shared' in pkg.spec:
            default_library = 'both'
        elif 'default_library=static' in pkg.spec:
            default_library = 'static'
        else:
            default_library = 'shared'
        args = ['-Dprefix={0}'.format(pkg.prefix), '-Dlibdir={0}'.format(pkg.prefix.lib), '-Dbuildtype={0}'.format(build_type), '-Dstrip={0}'.format(strip), '-Ddefault_library={0}'.format(default_library), '-Dwrap_mode=nodownload']
        return args

    @property
    def build_dirname(self):
        if False:
            return 10
        'Returns the directory name to use when building the package.'
        return 'spack-build-{}'.format(self.spec.dag_hash(7))

    @property
    def build_directory(self):
        if False:
            return 10
        'Directory to use when building the package.'
        return os.path.join(self.pkg.stage.path, self.build_dirname)

    def meson_args(self):
        if False:
            i = 10
            return i + 15
        'List of arguments that must be passed to meson, except:\n\n        * ``--prefix``\n        * ``--libdir``\n        * ``--buildtype``\n        * ``--strip``\n        * ``--default_library``\n\n        which will be set automatically.\n        '
        return []

    def meson(self, pkg, spec, prefix):
        if False:
            print('Hello World!')
        'Run ``meson`` in the build directory'
        options = []
        if self.spec['meson'].satisfies('@0.64:'):
            options.append('setup')
        options.append(os.path.abspath(self.root_mesonlists_dir))
        options += self.std_meson_args
        options += self.meson_args()
        with fs.working_dir(self.build_directory, create=True):
            inspect.getmodule(self.pkg).meson(*options)

    def build(self, pkg, spec, prefix):
        if False:
            while True:
                i = 10
        'Make the build targets'
        options = ['-v']
        options += self.build_targets
        with fs.working_dir(self.build_directory):
            inspect.getmodule(self.pkg).ninja(*options)

    def install(self, pkg, spec, prefix):
        if False:
            return 10
        'Make the install targets'
        with fs.working_dir(self.build_directory):
            inspect.getmodule(self.pkg).ninja(*self.install_targets)
    spack.builder.run_after('build')(execute_build_time_tests)

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        'Search Meson-generated files for the target ``test`` and run it if found.'
        with fs.working_dir(self.build_directory):
            self.pkg._if_ninja_target_execute('test')
            self.pkg._if_ninja_target_execute('check')