import inspect
from typing import List
import llnl.util.filesystem as fs
import spack.builder
import spack.package_base
from spack.directives import build_system, conflicts
from ._checks import BaseBuilder

class NMakePackage(spack.package_base.PackageBase):
    """Specialized class for packages built using a Makefiles."""
    build_system_class = 'NMakePackage'
    build_system('nmake')
    conflicts('platform=linux', when='build_system=nmake')
    conflicts('platform=darwin', when='build_system=nmake')
    conflicts('platform=cray', when='build_system=nmake')

@spack.builder.builder('nmake')
class NMakeBuilder(BaseBuilder):
    """The NMake builder encodes the most common way of building software with
    Mircosoft's NMake tool. It has two phases that can be overridden, if need be:

            1. :py:meth:`~.NMakeBuilder.build`
            2. :py:meth:`~.NMakeBuilder.install`

    It is usually necessary to override the :py:meth:`~.NMakeBuilder.install`
    phase as many packages with NMake systems neglect to provide an install
    target. The default install phase will attempt to invoke an install target
    from NMake. If none exists, this will result in a build failure

    For a finer tuning you may override:

        +-----------------------------------------------+---------------------+
        | **Method**                                    | **Purpose**         |
        +===============================================+=====================+
        | :py:attr:`~.NMakeBuilder.build_targets`       | Specify ``nmake``   |
        |                                               | targets for the     |
        |                                               | build phase         |
        +-----------------------------------------------+---------------------+
        | :py:attr:`~.NMakeBuilder.install_targets`     | Specify ``nmake``   |
        |                                               | targets for the     |
        |                                               | install phase       |
        +-----------------------------------------------+---------------------+
        | :py:meth:`~.NMakeBuilder.build_directory`     | Directory where the |
        |                                               | project makefile    |
        |                                               | is located          |
        +-----------------------------------------------+---------------------+
    """
    phases = ('build', 'install')
    build_targets: List[str] = []
    install_targets: List[str] = ['INSTALL']

    @property
    def ignore_quotes(self):
        if False:
            i = 10
            return i + 15
        'Control whether or not Spack warns about quoted arguments passed to\n        build utilities. If this is True, spack will not warn about quotes.\n        This is useful in cases with a space in the path or when build scripts\n        require quoted arugments.'
        return False

    @property
    def build_directory(self):
        if False:
            while True:
                i = 10
        'Return the directory containing the makefile.'
        return self.pkg.stage.source_path if not self.makefile_root else self.makefile_root

    @property
    def std_nmake_args(self):
        if False:
            while True:
                i = 10
        'Returns list of standards arguments provided to NMake\n        Currently is only /NOLOGO'
        return ['/NOLOGO']

    @property
    def makefile_root(self):
        if False:
            while True:
                i = 10
        'The relative path to the directory containing nmake makefile\n\n        This path is relative to the root of the extracted tarball,\n        not to the ``build_directory``. Defaults to the current directory.\n        '
        return self.stage.source_path

    @property
    def makefile_name(self):
        if False:
            i = 10
            return i + 15
        'Name of the current makefile. This is currently an empty value.\n        If a project defines this value, it will be used with the /f argument\n        to provide nmake an explicit makefile. This is usefule in scenarios where\n        there are multiple nmake files in the same directory.'
        return ''

    def define(self, nmake_arg, value):
        if False:
            print('Hello World!')
        'Helper method to format arguments to nmake command line'
        return '{}={}'.format(nmake_arg, value)

    def override_env(self, var_name, new_value):
        if False:
            for i in range(10):
                print('nop')
        'Helper method to format arguments for overridding env variables on the\n        nmake command line. Returns properly formatted argument'
        return '/E{}={}'.format(var_name, new_value)

    def nmake_args(self):
        if False:
            i = 10
            return i + 15
        'Define build arguments to NMake. This is an empty list by default.\n        Individual packages should override to specify NMake args to command line'
        return []

    def nmake_install_args(self):
        if False:
            while True:
                i = 10
        'Define arguments appropriate only for install phase to NMake.\n        This is an empty list by default.\n        Individual packages should override to specify NMake args to command line'
        return []

    def build(self, pkg, spec, prefix):
        if False:
            print('Hello World!')
        'Run "nmake" on the build targets specified by the builder.'
        opts = self.std_nmake_args
        opts += self.nmake_args()
        if self.makefile_name:
            opts.append('/F{}'.format(self.makefile_name))
        with fs.working_dir(self.build_directory):
            inspect.getmodule(self.pkg).nmake(*opts, *self.build_targets, ignore_quotes=self.ignore_quotes)

    def install(self, pkg, spec, prefix):
        if False:
            return 10
        'Run "nmake" on the install targets specified by the builder.\n        This is INSTALL by default'
        opts = self.std_nmake_args
        opts += self.nmake_args()
        opts += self.nmake_install_args()
        if self.makefile_name:
            opts.append('/F{}'.format(self.makefile_name))
        opts.append(self.define('PREFIX', prefix))
        with fs.working_dir(self.build_directory):
            inspect.getmodule(self.pkg).nmake(*opts, *self.install_targets, ignore_quotes=self.ignore_quotes)