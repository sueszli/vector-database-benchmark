"""Common utilities for managing intel oneapi packages."""
import getpass
import os
import platform
import shutil
from os.path import basename, dirname, isdir
from llnl.util.filesystem import find_headers, find_libraries, join_path, mkdirp
from llnl.util.link_tree import LinkTree
from spack.directives import conflicts, variant
from spack.util.environment import EnvironmentModifications
from spack.util.executable import Executable
from .generic import Package

class IntelOneApiPackage(Package):
    """Base class for Intel oneAPI packages."""
    homepage = 'https://software.intel.com/oneapi'
    redistribute_source = False
    for c in ['target=ppc64:', 'target=ppc64le:', 'target=aarch64:', 'platform=darwin:', 'platform=cray:', 'platform=windows:']:
        conflicts(c, msg='This package in only available for x86_64 and Linux')
    variant('envmods', default=True, description='Toggles environment modifications')

    @staticmethod
    def update_description(cls):
        if False:
            return 10
        'Updates oneapi package descriptions with common text.'
        text = ' LICENSE INFORMATION: By downloading and using this software, you agree to the terms\n        and conditions of the software license agreements at https://intel.ly/393CijO.'
        cls.__doc__ = cls.__doc__ + text
        return cls

    @property
    def component_dir(self):
        if False:
            return 10
        'Subdirectory for this component in the install prefix.'
        raise NotImplementedError

    @property
    def component_prefix(self):
        if False:
            i = 10
            return i + 15
        'Path to component <prefix>/<component>/<version>.'
        return self.prefix.join(join_path(self.component_dir, self.spec.version))

    @property
    def env_script_args(self):
        if False:
            return 10
        'Additional arguments to pass to vars.sh script.'
        return ()

    def install(self, spec, prefix):
        if False:
            i = 10
            return i + 15
        self.install_component(basename(self.url_for_version(spec.version)))

    def install_component(self, installer_path):
        if False:
            while True:
                i = 10
        'Shared install method for all oneapi packages.'
        if platform.system() == 'Linux':
            if getpass.getuser() == 'root':
                shutil.rmtree('/var/intel/installercache', ignore_errors=True)
            bash = Executable('bash')
            bash.add_default_env('HOME', self.prefix)
            bash.add_default_env('XDG_RUNTIME_DIR', join_path(self.stage.path, 'runtime'))
            bash(installer_path, '-s', '-a', '-s', '--action', 'install', '--eula', 'accept', '--install-dir', self.prefix)
            if getpass.getuser() == 'root':
                shutil.rmtree('/var/intel/installercache', ignore_errors=True)
        if not isdir(join_path(self.prefix, self.component_dir)):
            raise RuntimeError('install failed')

    def setup_run_environment(self, env):
        if False:
            for i in range(10):
                print('nop')
        'Adds environment variables to the generated module file.\n\n        These environment variables come from running:\n\n        .. code-block:: console\n\n           $ source {prefix}/{component}/{version}/env/vars.sh\n        '
        if '~envmods' not in self.spec:
            env.extend(EnvironmentModifications.from_sourcing_file(join_path(self.component_prefix, 'env', 'vars.sh'), *self.env_script_args))

    def symlink_dir(self, src, dest):
        if False:
            i = 10
            return i + 15
        mkdirp(dest)
        for entry in os.listdir(src):
            src_path = join_path(src, entry)
            dest_path = join_path(dest, entry)
            if isdir(src_path) and os.access(src_path, os.X_OK):
                link_tree = LinkTree(src_path)
                link_tree.merge(dest_path)
            else:
                os.symlink(src_path, dest_path)

class IntelOneApiLibraryPackage(IntelOneApiPackage):
    """Base class for Intel oneAPI library packages.

    Contains some convenient default implementations for libraries.
    Implement the method directly in the package if something
    different is needed.

    """

    @property
    def headers(self):
        if False:
            print('Hello World!')
        include_path = join_path(self.component_prefix, 'include')
        return find_headers('*', include_path, recursive=True)

    @property
    def libs(self):
        if False:
            i = 10
            return i + 15
        lib_path = join_path(self.component_prefix, 'lib', 'intel64')
        lib_path = lib_path if isdir(lib_path) else dirname(lib_path)
        return find_libraries('*', root=lib_path, shared=True, recursive=True)

class IntelOneApiStaticLibraryList:
    """Provides ld_flags when static linking is needed

    Oneapi puts static and dynamic libraries in the same directory, so
    -l will default to finding the dynamic library. Use absolute
    paths, as recommended by oneapi documentation.

    Allow both static and dynamic libraries to be supplied by the
    package.
    """

    def __init__(self, static_libs, dynamic_libs):
        if False:
            print('Hello World!')
        self.static_libs = static_libs
        self.dynamic_libs = dynamic_libs

    @property
    def directories(self):
        if False:
            return 10
        return self.dynamic_libs.directories

    @property
    def search_flags(self):
        if False:
            while True:
                i = 10
        return self.dynamic_libs.search_flags

    @property
    def link_flags(self):
        if False:
            while True:
                i = 10
        return '-Wl,--start-group {0} -Wl,--end-group {1}'.format(' '.join(self.static_libs.libraries), self.dynamic_libs.link_flags)

    @property
    def ld_flags(self):
        if False:
            i = 10
            return i + 15
        return '{0} {1}'.format(self.search_flags, self.link_flags)
INTEL_MATH_LIBRARIES = ('intel-mkl', 'intel-oneapi-mkl', 'intel-parallel-studio')