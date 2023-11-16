"""Manages CMake."""
import multiprocessing
import os
import platform
import sys
import sysconfig
from distutils.version import LooseVersion
from subprocess import CalledProcessError, check_call, check_output
from typing import Any, cast, Dict, List, Optional
from . import which
from .cmake_utils import CMakeValue, get_cmake_cache_variables_from_file
from .env import BUILD_DIR, check_negative_env_flag, IS_64BIT, IS_DARWIN, IS_WINDOWS
from .numpy_ import NUMPY_INCLUDE_DIR, USE_NUMPY

def _mkdir_p(d: str) -> None:
    if False:
        while True:
            i = 10
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f'Failed to create folder {os.path.abspath(d)}: {e.strerror}') from e
USE_NINJA = not check_negative_env_flag('USE_NINJA') and which('ninja') is not None
if 'CMAKE_GENERATOR' in os.environ:
    USE_NINJA = os.environ['CMAKE_GENERATOR'].lower() == 'ninja'

class CMake:
    """Manages cmake."""

    def __init__(self, build_dir: str=BUILD_DIR) -> None:
        if False:
            print('Hello World!')
        self._cmake_command = CMake._get_cmake_command()
        self.build_dir = build_dir

    @property
    def _cmake_cache_file(self) -> str:
        if False:
            return 10
        'Returns the path to CMakeCache.txt.\n\n        Returns:\n          string: The path to CMakeCache.txt.\n        '
        return os.path.join(self.build_dir, 'CMakeCache.txt')

    @staticmethod
    def _get_cmake_command() -> str:
        if False:
            print('Hello World!')
        'Returns cmake command.'
        cmake_command = 'cmake'
        if IS_WINDOWS:
            return cmake_command
        cmake3_version = CMake._get_version(which('cmake3'))
        cmake_version = CMake._get_version(which('cmake'))
        _cmake_min_version = LooseVersion('3.18.0')
        if all((ver is None or ver < _cmake_min_version for ver in [cmake_version, cmake3_version])):
            raise RuntimeError('no cmake or cmake3 with version >= 3.18.0 found')
        if cmake3_version is None:
            cmake_command = 'cmake'
        elif cmake_version is None:
            cmake_command = 'cmake3'
        elif cmake3_version >= cmake_version:
            cmake_command = 'cmake3'
        else:
            cmake_command = 'cmake'
        return cmake_command

    @staticmethod
    def _get_version(cmd: Optional[str]) -> Any:
        if False:
            while True:
                i = 10
        'Returns cmake version.'
        if cmd is None:
            return None
        for line in check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')

    def run(self, args: List[str], env: Dict[str, str]) -> None:
        if False:
            i = 10
            return i + 15
        'Executes cmake with arguments and an environment.'
        command = [self._cmake_command] + args
        print(' '.join(command))
        try:
            check_call(command, cwd=self.build_dir, env=env)
        except (CalledProcessError, KeyboardInterrupt) as e:
            sys.exit(1)

    @staticmethod
    def defines(args: List[str], **kwargs: CMakeValue) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Adds definitions to a cmake argument list.'
        for (key, value) in sorted(kwargs.items()):
            if value is not None:
                args.append(f'-D{key}={value}')

    def get_cmake_cache_variables(self) -> Dict[str, CMakeValue]:
        if False:
            return 10
        'Gets values in CMakeCache.txt into a dictionary.\n        Returns:\n          dict: A ``dict`` containing the value of cached CMake variables.\n        '
        with open(self._cmake_cache_file) as f:
            return get_cmake_cache_variables_from_file(f)

    def generate(self, version: Optional[str], cmake_python_library: Optional[str], build_python: bool, build_test: bool, my_env: Dict[str, str], rerun: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Runs cmake to generate native build files.'
        if rerun and os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)
        ninja_build_file = os.path.join(self.build_dir, 'build.ninja')
        if os.path.exists(self._cmake_cache_file) and (not (USE_NINJA and (not os.path.exists(ninja_build_file)))):
            return
        args = []
        if USE_NINJA:
            os.environ['CMAKE_GENERATOR'] = 'Ninja'
            args.append('-GNinja')
        elif IS_WINDOWS:
            generator = os.getenv('CMAKE_GENERATOR', 'Visual Studio 16 2019')
            supported = ['Visual Studio 16 2019', 'Visual Studio 17 2022']
            if generator not in supported:
                print('Unsupported `CMAKE_GENERATOR`: ' + generator)
                print('Please set it to one of the following values: ')
                print('\n'.join(supported))
                sys.exit(1)
            args.append('-G' + generator)
            toolset_dict = {}
            toolset_version = os.getenv('CMAKE_GENERATOR_TOOLSET_VERSION')
            if toolset_version is not None:
                toolset_dict['version'] = toolset_version
                curr_toolset = os.getenv('VCToolsVersion')
                if curr_toolset is None:
                    print('When you specify `CMAKE_GENERATOR_TOOLSET_VERSION`, you must also activate the vs environment of this version. Please read the notes in the build steps carefully.')
                    sys.exit(1)
            if IS_64BIT:
                if platform.machine() == 'ARM64':
                    args.append('-A ARM64')
                else:
                    args.append('-Ax64')
                    toolset_dict['host'] = 'x64'
            if toolset_dict:
                toolset_expr = ','.join([f'{k}={v}' for (k, v) in toolset_dict.items()])
                args.append('-T' + toolset_expr)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        install_dir = os.path.join(base_dir, 'torch')
        _mkdir_p(install_dir)
        _mkdir_p(self.build_dir)
        build_options: Dict[str, CMakeValue] = {}
        additional_options = {'_GLIBCXX_USE_CXX11_ABI': 'GLIBCXX_USE_CXX11_ABI', 'CUDNN_LIB_DIR': 'CUDNN_LIBRARY', 'USE_CUDA_STATIC_LINK': 'CAFFE2_STATIC_LINK_CUDA'}
        additional_options.update({var: var for var in ('UBSAN_FLAGS', 'BLAS', 'WITH_BLAS', 'BUILDING_WITH_TORCH_LIBS', 'CUDA_HOST_COMPILER', 'CUDA_NVCC_EXECUTABLE', 'CUDA_SEPARABLE_COMPILATION', 'CUDNN_LIBRARY', 'CUDNN_INCLUDE_DIR', 'CUDNN_ROOT', 'EXPERIMENTAL_SINGLE_THREAD_POOL', 'INSTALL_TEST', 'JAVA_HOME', 'INTEL_MKL_DIR', 'INTEL_OMP_DIR', 'MKL_THREADING', 'MKLDNN_CPU_RUNTIME', 'MSVC_Z7_OVERRIDE', 'CAFFE2_USE_MSVC_STATIC_RUNTIME', 'Numa_INCLUDE_DIR', 'Numa_LIBRARIES', 'ONNX_ML', 'ONNX_NAMESPACE', 'ATEN_THREADING', 'WERROR', 'OPENSSL_ROOT_DIR', 'STATIC_DISPATCH_BACKEND', 'SELECTED_OP_LIST', 'TORCH_CUDA_ARCH_LIST', 'TRACING_BASED')})
        low_priority_aliases = {'CUDA_HOST_COMPILER': 'CMAKE_CUDA_HOST_COMPILER', 'CUDAHOSTCXX': 'CUDA_HOST_COMPILER', 'CMAKE_CUDA_HOST_COMPILER': 'CUDA_HOST_COMPILER', 'CMAKE_CUDA_COMPILER': 'CUDA_NVCC_EXECUTABLE', 'CUDACXX': 'CUDA_NVCC_EXECUTABLE'}
        for (var, val) in my_env.items():
            true_var = additional_options.get(var)
            if true_var is not None:
                build_options[true_var] = val
            elif var.startswith(('BUILD_', 'USE_', 'CMAKE_')) or var.endswith(('EXITCODE', 'EXITCODE__TRYRUN_OUTPUT')):
                build_options[var] = val
            if var in low_priority_aliases:
                key = low_priority_aliases[var]
                if key not in build_options:
                    build_options[key] = val
        py_lib_path = sysconfig.get_path('purelib')
        cmake_prefix_path = build_options.get('CMAKE_PREFIX_PATH', None)
        if cmake_prefix_path:
            build_options['CMAKE_PREFIX_PATH'] = py_lib_path + ';' + cast(str, cmake_prefix_path)
        else:
            build_options['CMAKE_PREFIX_PATH'] = py_lib_path
        build_options.update({'BUILD_PYTHON': build_python, 'BUILD_TEST': build_test, 'USE_NUMPY': USE_NUMPY})
        cmake__options = {'CMAKE_INSTALL_PREFIX': install_dir}
        specified_cmake__options = set(build_options).intersection(cmake__options)
        if len(specified_cmake__options) > 0:
            print(', '.join(specified_cmake__options) + ' should not be specified in the environment variable. They are directly set by PyTorch build script.')
            sys.exit(1)
        build_options.update(cmake__options)
        CMake.defines(args, PYTHON_EXECUTABLE=sys.executable, PYTHON_LIBRARY=cmake_python_library, PYTHON_INCLUDE_DIR=sysconfig.get_path('include'), TORCH_BUILD_VERSION=version, NUMPY_INCLUDE_DIR=NUMPY_INCLUDE_DIR, **build_options)
        expected_wrapper = '/usr/local/opt/ccache/libexec'
        if IS_DARWIN and os.path.exists(expected_wrapper):
            if 'CMAKE_C_COMPILER' not in build_options and 'CC' not in os.environ:
                CMake.defines(args, CMAKE_C_COMPILER=f'{expected_wrapper}/gcc')
            if 'CMAKE_CXX_COMPILER' not in build_options and 'CXX' not in os.environ:
                CMake.defines(args, CMAKE_CXX_COMPILER=f'{expected_wrapper}/g++')
        for env_var_name in my_env:
            if env_var_name.startswith('gh'):
                try:
                    my_env[env_var_name] = str(my_env[env_var_name].encode('utf-8'))
                except UnicodeDecodeError as e:
                    shex = ':'.join((f'{ord(c):02x}' for c in my_env[env_var_name]))
                    print(f'Invalid ENV[{env_var_name}] = {shex}', file=sys.stderr)
                    print(e, file=sys.stderr)
        args.append(base_dir)
        self.run(args, env=my_env)

    def build(self, my_env: Dict[str, str]) -> None:
        if False:
            return 10
        'Runs cmake to build binaries.'
        from .env import build_type
        build_args = ['--build', '.', '--target', 'install', '--config', build_type.build_type_string]
        max_jobs = os.getenv('MAX_JOBS')
        if max_jobs is not None or not USE_NINJA:
            max_jobs = max_jobs or str(multiprocessing.cpu_count())
            build_args += ['--']
            if IS_WINDOWS and (not USE_NINJA):
                build_args += [f'/p:CL_MPCount={max_jobs}']
            else:
                build_args += ['-j', max_jobs]
        self.run(build_args, my_env)