import distutils.ccompiler
import os
import os.path
import platform
import shutil
import sys
import subprocess
from typing import Any, Optional, List
import setuptools
import setuptools.msvc
from setuptools import Extension
from cupy_builder._context import Context
import cupy_builder.install_build as build

def _nvcc_gencode_options(cuda_version: int) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Returns NVCC GPU code generation options.'
    if sys.argv == ['setup.py', 'develop']:
        return []
    envcfg = os.getenv('CUPY_NVCC_GENERATE_CODE', None)
    if envcfg is not None and envcfg != 'current':
        return ['--generate-code={}'.format(arch) for arch in envcfg.split(';') if len(arch) > 0]
    if envcfg == 'current' and build.get_compute_capabilities() is not None:
        ccs = build.get_compute_capabilities()
        arch_list = [f'compute_{cc}' if cc < 60 else (f'compute_{cc}', f'sm_{cc}') for cc in ccs]
    else:
        aarch64 = platform.machine() == 'aarch64'
        if cuda_version >= 12000:
            arch_list = [('compute_50', 'sm_50'), ('compute_52', 'sm_52'), ('compute_60', 'sm_60'), ('compute_61', 'sm_61'), ('compute_70', 'sm_70'), ('compute_75', 'sm_75'), ('compute_80', 'sm_80'), ('compute_86', 'sm_86'), ('compute_89', 'sm_89'), ('compute_90', 'sm_90'), 'compute_90']
            if aarch64:
                arch_list += [('compute_72', 'sm_72'), ('compute_87', 'sm_87')]
        elif cuda_version >= 11080:
            arch_list = [('compute_35', 'sm_35'), ('compute_37', 'sm_37'), ('compute_50', 'sm_50'), ('compute_52', 'sm_52'), ('compute_60', 'sm_60'), ('compute_61', 'sm_61'), ('compute_70', 'sm_70'), ('compute_75', 'sm_75'), ('compute_80', 'sm_80'), ('compute_86', 'sm_86'), ('compute_89', 'sm_89'), ('compute_90', 'sm_90'), 'compute_90']
            if aarch64:
                arch_list += [('compute_72', 'sm_72'), ('compute_87', 'sm_87')]
        elif cuda_version >= 11040:
            arch_list = [('compute_35', 'sm_35'), ('compute_37', 'sm_37'), ('compute_50', 'sm_50'), ('compute_52', 'sm_52'), ('compute_60', 'sm_60'), ('compute_61', 'sm_61'), ('compute_70', 'sm_70'), ('compute_75', 'sm_75'), ('compute_80', 'sm_80'), ('compute_86', 'sm_86'), 'compute_86']
            if aarch64:
                arch_list += [('compute_72', 'sm_72'), ('compute_87', 'sm_87')]
        elif cuda_version >= 11010:
            arch_list = ['compute_35', 'compute_50', ('compute_60', 'sm_60'), ('compute_61', 'sm_61'), ('compute_70', 'sm_70'), ('compute_75', 'sm_75'), ('compute_80', 'sm_80'), ('compute_86', 'sm_86'), 'compute_86']
        elif cuda_version >= 11000:
            arch_list = ['compute_35', 'compute_50', ('compute_60', 'sm_60'), ('compute_61', 'sm_61'), ('compute_70', 'sm_70'), ('compute_75', 'sm_75'), ('compute_80', 'sm_80'), 'compute_80']
        elif cuda_version >= 10000:
            arch_list = ['compute_30', 'compute_50', ('compute_60', 'sm_60'), ('compute_61', 'sm_61'), ('compute_70', 'sm_70'), ('compute_75', 'sm_75'), 'compute_70']
        else:
            assert False
    options = []
    for arch in arch_list:
        if type(arch) is tuple:
            (virtual_arch, real_arch) = arch
            options.append('--generate-code=arch={},code={}'.format(virtual_arch, real_arch))
        else:
            options.append('--generate-code=arch={},code={}'.format(arch, arch))
    return options

class DeviceCompilerBase:
    """A class that invokes NVCC or HIPCC."""

    def __init__(self, ctx: Context):
        if False:
            while True:
                i = 10
        self._context = ctx

    def _get_preprocess_options(self, ext: Extension) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        incdirs = ext.include_dirs[:]
        macros: List[Any] = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))
        return distutils.ccompiler.gen_preprocess_options(macros, incdirs)

    def spawn(self, commands: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        print('Command:', commands)
        subprocess.check_call(commands)

class DeviceCompilerUnix(DeviceCompilerBase):

    def compile(self, obj: str, src: str, ext: Extension) -> None:
        if False:
            while True:
                i = 10
        if self._context.use_hip:
            self._compile_unix_hipcc(obj, src, ext)
        else:
            self._compile_unix_nvcc(obj, src, ext)

    def _compile_unix_nvcc(self, obj: str, src: str, ext: Extension) -> None:
        if False:
            return 10
        cc_args = self._get_preprocess_options(ext) + ['-c']
        nvcc_path = build.get_nvcc_path()
        base_opts = build.get_compiler_base_options(nvcc_path)
        compiler_so = nvcc_path
        cuda_version = self._context.features['cuda'].get_version()
        postargs = _nvcc_gencode_options(cuda_version) + ['-Xfatbin=-compress-all', '-O2', '--compiler-options="-fPIC"']
        if cuda_version >= 11020:
            postargs += ['--std=c++14']
            num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
            postargs += [f'-t{num_threads}']
        else:
            postargs += ['--std=c++11']
        postargs += ['-Xcompiler=-fno-gnu-unique']
        print('NVCC options:', postargs)
        self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] + postargs)

    def _compile_unix_hipcc(self, obj: str, src: str, ext: Extension) -> None:
        if False:
            while True:
                i = 10
        cc_args = self._get_preprocess_options(ext) + ['-c']
        rocm_path = build.get_hipcc_path()
        base_opts = build.get_compiler_base_options(rocm_path)
        compiler_so = rocm_path
        hip_version = build.get_hip_version()
        postargs = ['-O2', '-fPIC', '--include', 'hip_runtime.h']
        if hip_version >= 402:
            postargs += ['--std=c++14']
        else:
            postargs += ['--std=c++11']
        print('HIPCC options:', postargs)
        self.spawn(compiler_so + base_opts + cc_args + [src, '-o', obj] + postargs)

class DeviceCompilerWin32(DeviceCompilerBase):

    def compile(self, obj: str, src: str, ext: Extension) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._context.use_hip:
            raise RuntimeError('ROCm is not supported on Windows')
        compiler_so = build.get_nvcc_path()
        cc_args = self._get_preprocess_options(ext) + ['-c']
        cuda_version = self._context.features['cuda'].get_version()
        postargs = _nvcc_gencode_options(cuda_version) + ['-Xfatbin=-compress-all', '-O2']
        if cuda_version >= 11020:
            postargs += ['-allow-unsupported-compiler']
        postargs += ['-Xcompiler', '/MD', '-D_USE_MATH_DEFINES']
        if cuda_version >= 11020:
            postargs += ['--std=c++14']
            num_threads = int(os.environ.get('CUPY_NUM_NVCC_THREADS', '2'))
            postargs += [f'-t{num_threads}']
        cl_exe_path = self._find_host_compiler_path()
        if cl_exe_path is not None:
            print(f'Using host compiler at {cl_exe_path}')
            postargs += ['--compiler-bindir', cl_exe_path]
        print('NVCC options:', postargs)
        self.spawn(compiler_so + cc_args + [src, '-o', obj] + postargs)

    def _find_host_compiler_path(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        cl_exe = shutil.which('cl.exe')
        if cl_exe:
            return None
        vctools: List[str] = setuptools.msvc.EnvironmentInfo(platform.machine()).VCTools
        for path in vctools:
            cl_exe = os.path.join(path, 'cl.exe')
            if os.path.exists(cl_exe):
                return path
        print(f'Warning: cl.exe could not be found in {vctools}')
        return None