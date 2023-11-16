import collections.abc
import os
from typing import Tuple
import llnl.util.filesystem as fs
import llnl.util.tty as tty
import spack.builder
from .cmake import CMakeBuilder, CMakePackage

def cmake_cache_path(name, value, comment='', force=False):
    if False:
        return 10
    'Generate a string for a cmake cache variable'
    force_str = ' FORCE' if force else ''
    return 'set({0} "{1}" CACHE PATH "{2}"{3})\n'.format(name, value, comment, force_str)

def cmake_cache_string(name, value, comment='', force=False):
    if False:
        i = 10
        return i + 15
    'Generate a string for a cmake cache variable'
    force_str = ' FORCE' if force else ''
    return 'set({0} "{1}" CACHE STRING "{2}"{3})\n'.format(name, value, comment, force_str)

def cmake_cache_option(name, boolean_value, comment='', force=False):
    if False:
        while True:
            i = 10
    'Generate a string for a cmake configuration option'
    value = 'ON' if boolean_value else 'OFF'
    force_str = ' FORCE' if force else ''
    return 'set({0} {1} CACHE BOOL "{2}"{3})\n'.format(name, value, comment, force_str)

def cmake_cache_filepath(name, value, comment=''):
    if False:
        return 10
    'Generate a string for a cmake cache variable of type FILEPATH'
    return 'set({0} "{1}" CACHE FILEPATH "{2}")\n'.format(name, value, comment)

class CachedCMakeBuilder(CMakeBuilder):
    phases: Tuple[str, ...] = ('initconfig', 'cmake', 'build', 'install')
    legacy_methods: Tuple[str, ...] = CMakeBuilder.legacy_methods + ('initconfig_compiler_entries', 'initconfig_mpi_entries', 'initconfig_hardware_entries', 'std_initconfig_entries', 'initconfig_package_entries')
    legacy_attributes: Tuple[str, ...] = CMakeBuilder.legacy_attributes + ('cache_name', 'cache_path')

    @property
    def cache_name(self):
        if False:
            return 10
        return '{0}-{1}-{2}@{3}.cmake'.format(self.pkg.name, self.pkg.spec.architecture, self.pkg.spec.compiler.name, self.pkg.spec.compiler.version)

    @property
    def cache_path(self):
        if False:
            print('Hello World!')
        return os.path.join(self.pkg.stage.source_path, self.cache_name)

    def define_cmake_cache_from_variant(self, cmake_var, variant=None, comment=''):
        if False:
            for i in range(10):
                print('nop')
        "Return a Cached CMake field from the given variant's value.\n        See define_from_variant in lib/spack/spack/build_systems/cmake.py package\n        "
        if variant is None:
            variant = cmake_var.lower()
        if variant not in self.pkg.variants:
            raise KeyError('"{0}" is not a variant of "{1}"'.format(variant, self.pkg.name))
        if variant not in self.pkg.spec.variants:
            return ''
        value = self.pkg.spec.variants[variant].value
        field = None
        if isinstance(value, bool):
            field = cmake_cache_option(cmake_var, value, comment)
        else:
            if isinstance(value, collections.abc.Sequence) and (not isinstance(value, str)):
                value = ';'.join((str(v) for v in value))
            else:
                value = str(value)
            field = cmake_cache_string(cmake_var, value, comment)
        return field

    def initconfig_compiler_entries(self):
        if False:
            for i in range(10):
                print('nop')
        spec = self.pkg.spec
        if 'FC' in os.environ:
            spack_fc_entry = cmake_cache_path('CMAKE_Fortran_COMPILER', os.environ['FC'])
            system_fc_entry = cmake_cache_path('CMAKE_Fortran_COMPILER', self.pkg.compiler.fc)
        else:
            spack_fc_entry = '# No Fortran compiler defined in spec'
            system_fc_entry = '# No Fortran compiler defined in spec'
        entries = ['#------------------{0}'.format('-' * 60), '# Compilers', '#------------------{0}'.format('-' * 60), '# Compiler Spec: {0}'.format(spec.compiler), '#------------------{0}'.format('-' * 60), 'if(DEFINED ENV{SPACK_CC})\n', '  ' + cmake_cache_path('CMAKE_C_COMPILER', os.environ['CC']), '  ' + cmake_cache_path('CMAKE_CXX_COMPILER', os.environ['CXX']), '  ' + spack_fc_entry, 'else()\n', '  ' + cmake_cache_path('CMAKE_C_COMPILER', self.pkg.compiler.cc), '  ' + cmake_cache_path('CMAKE_CXX_COMPILER', self.pkg.compiler.cxx), '  ' + system_fc_entry, 'endif()\n']
        flags = spec.compiler_flags
        cppflags = ' '.join(flags['cppflags'])
        if cppflags:
            cppflags += ' '
        cflags = cppflags + ' '.join(flags['cflags'])
        if cflags:
            entries.append(cmake_cache_string('CMAKE_C_FLAGS', cflags))
        cxxflags = cppflags + ' '.join(flags['cxxflags'])
        if cxxflags:
            entries.append(cmake_cache_string('CMAKE_CXX_FLAGS', cxxflags))
        fflags = ' '.join(flags['fflags'])
        if fflags:
            entries.append(cmake_cache_string('CMAKE_Fortran_FLAGS', fflags))
        if flags['ldflags']:
            ld_flags = ' '.join(flags['ldflags'])
            ld_format_string = 'CMAKE_{0}_LINKER_FLAGS'
            for ld_type in ['EXE', 'MODULE', 'SHARED', 'STATIC']:
                ld_string = ld_format_string.format(ld_type)
                entries.append(cmake_cache_string(ld_string, ld_flags))
        if flags['ldlibs']:
            libs_flags = ' '.join(flags['ldlibs'])
            libs_format_string = 'CMAKE_{0}_STANDARD_LIBRARIES'
            langs = ['C', 'CXX', 'Fortran']
            for lang in langs:
                libs_string = libs_format_string.format(lang)
                entries.append(cmake_cache_string(libs_string, libs_flags))
        return entries

    def initconfig_mpi_entries(self):
        if False:
            i = 10
            return i + 15
        spec = self.pkg.spec
        if not spec.satisfies('^mpi'):
            return []
        entries = ['#------------------{0}'.format('-' * 60), '# MPI', '#------------------{0}\n'.format('-' * 60)]
        entries.append(cmake_cache_path('MPI_C_COMPILER', spec['mpi'].mpicc))
        entries.append(cmake_cache_path('MPI_CXX_COMPILER', spec['mpi'].mpicxx))
        entries.append(cmake_cache_path('MPI_Fortran_COMPILER', spec['mpi'].mpifc))
        using_slurm = False
        slurm_checks = ['+slurm', 'schedulers=slurm', 'process_managers=slurm']
        if any((spec['mpi'].satisfies(variant) for variant in slurm_checks)):
            using_slurm = True
        if using_slurm:
            if spec['mpi'].external:
                mpiexec = '/usr/bin/srun'
            else:
                mpiexec = os.path.join(spec['slurm'].prefix.bin, 'srun')
        else:
            mpiexec = os.path.join(spec['mpi'].prefix.bin, 'mpirun')
            if not os.path.exists(mpiexec):
                mpiexec = os.path.join(spec['mpi'].prefix.bin, 'mpiexec')
        if not os.path.exists(mpiexec):
            msg = 'Unable to determine MPIEXEC, %s tests may fail' % self.pkg.name
            entries.append('# {0}\n'.format(msg))
            tty.warn(msg)
        elif self.pkg.spec['cmake'].satisfies('@3.10:'):
            entries.append(cmake_cache_path('MPIEXEC_EXECUTABLE', mpiexec))
        else:
            entries.append(cmake_cache_path('MPIEXEC', mpiexec))
        if using_slurm:
            entries.append(cmake_cache_string('MPIEXEC_NUMPROC_FLAG', '-n'))
        else:
            entries.append(cmake_cache_string('MPIEXEC_NUMPROC_FLAG', '-np'))
        return entries

    def initconfig_hardware_entries(self):
        if False:
            for i in range(10):
                print('nop')
        spec = self.pkg.spec
        entries = ['#------------------{0}'.format('-' * 60), '# Hardware', '#------------------{0}\n'.format('-' * 60)]
        if spec.satisfies('^cuda'):
            entries.append('#------------------{0}'.format('-' * 30))
            entries.append('# Cuda')
            entries.append('#------------------{0}\n'.format('-' * 30))
            cudatoolkitdir = spec['cuda'].prefix
            entries.append(cmake_cache_path('CUDAToolkit_ROOT', cudatoolkitdir))
            entries.append(cmake_cache_path('CMAKE_CUDA_COMPILER', '${CUDAToolkit_ROOT}/bin/nvcc'))
            entries.append(cmake_cache_path('CMAKE_CUDA_HOST_COMPILER', '${CMAKE_CXX_COMPILER}'))
            entries.append(cmake_cache_path('CUDA_TOOLKIT_ROOT_DIR', cudatoolkitdir))
            archs = spec.variants['cuda_arch'].value
            if archs[0] != 'none':
                arch_str = ';'.join(archs)
                entries.append(cmake_cache_string('CMAKE_CUDA_ARCHITECTURES', '{0}'.format(arch_str)))
        if '+rocm' in spec:
            entries.append('#------------------{0}'.format('-' * 30))
            entries.append('# ROCm')
            entries.append('#------------------{0}\n'.format('-' * 30))
            entries.append(cmake_cache_path('HIP_ROOT_DIR', '{0}'.format(spec['hip'].prefix)))
            entries.append(cmake_cache_path('HIP_CXX_COMPILER', '{0}'.format(self.spec['hip'].hipcc)))
            llvm_bin = spec['llvm-amdgpu'].prefix.bin
            llvm_prefix = spec['llvm-amdgpu'].prefix
            if os.path.basename(os.path.normpath(llvm_prefix)) != 'llvm':
                llvm_bin = os.path.join(llvm_prefix, 'llvm/bin/')
            entries.append(cmake_cache_filepath('CMAKE_HIP_COMPILER', os.path.join(llvm_bin, 'clang++')))
            archs = self.spec.variants['amdgpu_target'].value
            if archs[0] != 'none':
                arch_str = ';'.join(archs)
                entries.append(cmake_cache_string('CMAKE_HIP_ARCHITECTURES', '{0}'.format(arch_str)))
                entries.append(cmake_cache_string('AMDGPU_TARGETS', '{0}'.format(arch_str)))
                entries.append(cmake_cache_string('GPU_TARGETS', '{0}'.format(arch_str)))
        return entries

    def std_initconfig_entries(self):
        if False:
            return 10
        cmake_prefix_path_env = os.environ['CMAKE_PREFIX_PATH']
        cmake_prefix_path = cmake_prefix_path_env.replace(os.pathsep, ';')
        return ['#------------------{0}'.format('-' * 60), '# !!!! This is a generated file, edit at own risk !!!!', '#------------------{0}'.format('-' * 60), '# CMake executable path: {0}'.format(self.pkg.spec['cmake'].command.path), '#------------------{0}\n'.format('-' * 60), cmake_cache_string('CMAKE_PREFIX_PATH', cmake_prefix_path), self.define_cmake_cache_from_variant('CMAKE_BUILD_TYPE', 'build_type')]

    def initconfig_package_entries(self):
        if False:
            print('Hello World!')
        'This method is to be overwritten by the package'
        return []

    def initconfig(self, pkg, spec, prefix):
        if False:
            return 10
        cache_entries = self.std_initconfig_entries() + self.initconfig_compiler_entries() + self.initconfig_mpi_entries() + self.initconfig_hardware_entries() + self.initconfig_package_entries()
        with open(self.cache_name, 'w') as f:
            for entry in cache_entries:
                f.write('%s\n' % entry)
            f.write('\n')

    @property
    def std_cmake_args(self):
        if False:
            while True:
                i = 10
        args = super().std_cmake_args
        args.extend(['-C', self.cache_path])
        return args

    @spack.builder.run_after('install')
    def install_cmake_cache(self):
        if False:
            while True:
                i = 10
        fs.mkdirp(self.pkg.spec.prefix.share.cmake)
        fs.install(self.cache_path, self.pkg.spec.prefix.share.cmake)

class CachedCMakePackage(CMakePackage):
    """Specialized class for packages built using CMake initial cache.

    This feature of CMake allows packages to increase reproducibility,
    especially between Spack- and manual builds. It also allows packages to
    sidestep certain parsing bugs in extremely long ``cmake`` commands, and to
    avoid system limits on the length of the command line.
    """
    CMakeBuilder = CachedCMakeBuilder

    def flag_handler(self, name, flags):
        if False:
            return 10
        if name in ('cflags', 'cxxflags', 'cppflags', 'fflags'):
            return (None, None, None)
        return (flags, None, None)