import inspect
from typing import List
import llnl.util.filesystem as fs
import spack.builder
import spack.package_base
from spack.directives import build_system, conflicts
from ._checks import BaseBuilder

class MSBuildPackage(spack.package_base.PackageBase):
    """Specialized class for packages built using Visual Studio project files or solutions."""
    build_system_class = 'MSBuildPackage'
    build_system('msbuild')
    conflicts('platform=linux', when='build_system=msbuild')
    conflicts('platform=darwin', when='build_system=msbuild')
    conflicts('platform=cray', when='build_system=msbuild')

@spack.builder.builder('msbuild')
class MSBuildBuilder(BaseBuilder):
    """The MSBuild builder encodes the most common way of building software with
    Mircosoft's MSBuild tool. It has two phases that can be overridden, if need be:

            1. :py:meth:`~.MSBuildBuilder.build`
            2. :py:meth:`~.MSBuildBuilder.install`

    It is usually necessary to override the :py:meth:`~.MSBuildBuilder.install`
    phase as many packages with MSBuild systems neglect to provide an install
    target. The default install phase will attempt to invoke an install target
    from MSBuild. If none exists, this will result in a build failure

    For a finer tuning you may override:

        +-----------------------------------------------+---------------------+
        | **Method**                                    | **Purpose**         |
        +===============================================+=====================+
        | :py:attr:`~.MSBuildBuilder.build_targets`     | Specify ``msbuild`` |
        |                                               | targets for the     |
        |                                               | build phase         |
        +-----------------------------------------------+---------------------+
        | :py:attr:`~.MSBuildBuilder.install_targets`   | Specify ``msbuild`` |
        |                                               | targets for the     |
        |                                               | install phase       |
        +-----------------------------------------------+---------------------+
        | :py:meth:`~.MSBuildBuilder.build_directory`   | Directory where the |
        |                                               | project sln/vcxproj |
        |                                               | is located          |
        +-----------------------------------------------+---------------------+
    """
    phases = ('build', 'install')
    build_targets: List[str] = []
    install_targets: List[str] = ['INSTALL']

    @property
    def build_directory(self):
        if False:
            return 10
        'Return the directory containing the MSBuild solution or vcxproj.'
        return self.pkg.stage.source_path

    @property
    def toolchain_version(self):
        if False:
            return 10
        'Return currently targeted version of MSVC toolchain\n        Override this method to select a specific version of the toolchain or change\n        selection heuristics.\n        Default is whatever version of msvc has been selected by concretization'
        return 'v' + self.pkg.compiler.platform_toolset_ver

    @property
    def std_msbuild_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Return common msbuild cl arguments, for now just toolchain'
        return [self.define('PlatformToolset', self.toolchain_version)]

    def define_targets(self, *targets):
        if False:
            return 10
        return '/target:' + ';'.join(targets) if targets else ''

    def define(self, msbuild_arg, value):
        if False:
            i = 10
            return i + 15
        return '/p:{}={}'.format(msbuild_arg, value)

    def msbuild_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Define build arguments to MSbuild. This is an empty list by default.\n        Individual packages should override to specify MSBuild args to command line\n        PlatformToolset is already defined an can be controlled via the `toolchain_version`\n        property'
        return []

    def msbuild_install_args(self):
        if False:
            print('Hello World!')
        'Define install arguments to MSBuild outside of the INSTALL target. This is the same\n        as `msbuild_args` by default.'
        return self.msbuild_args()

    def build(self, pkg, spec, prefix):
        if False:
            return 10
        'Run "msbuild" on the build targets specified by the builder.'
        with fs.working_dir(self.build_directory):
            inspect.getmodule(self.pkg).msbuild(*self.std_msbuild_args, *self.msbuild_args(), self.define_targets(*self.build_targets))

    def install(self, pkg, spec, prefix):
        if False:
            i = 10
            return i + 15
        'Run "msbuild" on the install targets specified by the builder.\n        This is INSTALL by default'
        with fs.working_dir(self.build_directory):
            inspect.getmodule(self.pkg).msbuild(*self.msbuild_install_args(), self.define_targets(*self.install_targets))