import copy
import glob
import importlib
import importlib.abc
import os
import re
import shlex
import shutil
import setuptools
import subprocess
import sys
import sysconfig
import warnings
import collections
from pathlib import Path
import errno
import torch
import torch._appdirs
from .file_baton import FileBaton
from ._cpp_extension_versioner import ExtensionVersioner
from .hipify import hipify_python
from .hipify.hipify_python import GeneratedFileCleaner
from typing import Dict, List, Optional, Union, Tuple
from torch.torch_version import TorchVersion, Version
from setuptools.command.build_ext import build_ext
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')
LIB_EXT = '.pyd' if IS_WINDOWS else '.so'
EXEC_EXT = '.exe' if IS_WINDOWS else ''
CLIB_PREFIX = '' if IS_WINDOWS else 'lib'
CLIB_EXT = '.dll' if IS_WINDOWS else '.so'
SHARED_FLAG = '/DLL' if IS_WINDOWS else '-shared'
_HERE = os.path.abspath(__file__)
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
TORCH_LIB_PATH = os.path.join(_TORCH_PATH, 'lib')
SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()
MINIMUM_GCC_VERSION = (5, 0, 0)
MINIMUM_MSVC_VERSION = (19, 0, 24215)
VersionRange = Tuple[Tuple[int, ...], Tuple[int, ...]]
VersionMap = Dict[str, VersionRange]
CUDA_GCC_VERSIONS: VersionMap = {'11.0': (MINIMUM_GCC_VERSION, (10, 0)), '11.1': (MINIMUM_GCC_VERSION, (11, 0)), '11.2': (MINIMUM_GCC_VERSION, (11, 0)), '11.3': (MINIMUM_GCC_VERSION, (11, 0)), '11.4': ((6, 0, 0), (12, 0)), '11.5': ((6, 0, 0), (12, 0)), '11.6': ((6, 0, 0), (12, 0)), '11.7': ((6, 0, 0), (12, 0))}
MINIMUM_CLANG_VERSION = (3, 3, 0)
CUDA_CLANG_VERSIONS: VersionMap = {'11.1': (MINIMUM_CLANG_VERSION, (11, 0)), '11.2': (MINIMUM_CLANG_VERSION, (12, 0)), '11.3': (MINIMUM_CLANG_VERSION, (12, 0)), '11.4': (MINIMUM_CLANG_VERSION, (13, 0)), '11.5': (MINIMUM_CLANG_VERSION, (13, 0)), '11.6': (MINIMUM_CLANG_VERSION, (14, 0)), '11.7': (MINIMUM_CLANG_VERSION, (14, 0))}
__all__ = ['get_default_build_root', 'check_compiler_ok_for_platform', 'get_compiler_abi_compatibility_and_version', 'BuildExtension', 'CppExtension', 'CUDAExtension', 'include_paths', 'library_paths', 'load', 'load_inline', 'is_ninja_available', 'verify_ninja_availability', 'remove_extension_h_precompiler_headers', 'get_cxx_compiler', 'check_compiler_is_gcc']

def _nt_quote_args(args: Optional[List[str]]) -> List[str]:
    if False:
        return 10
    'Quote command-line arguments for DOS/Windows conventions.\n\n    Just wraps every argument which contains blanks in double quotes, and\n    returns a new argument list.\n    '
    if not args:
        return []
    return [f'"{arg}"' if ' ' in arg else arg for arg in args]

def _find_cuda_home() -> Optional[str]:
    if False:
        return 10
    'Find the CUDA install path.'
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'], stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            if IS_WINDOWS:
                cuda_homes = glob.glob('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    if cuda_home and (not torch.cuda.is_available()):
        print(f"No CUDA runtime is found, using CUDA_HOME='{cuda_home}'", file=sys.stderr)
    return cuda_home

def _find_rocm_home() -> Optional[str]:
    if False:
        print('Hello World!')
    'Find the ROCm install path.'
    rocm_home = os.environ.get('ROCM_HOME') or os.environ.get('ROCM_PATH')
    if rocm_home is None:
        hipcc_path = shutil.which('hipcc')
        if hipcc_path is not None:
            rocm_home = os.path.dirname(os.path.dirname(os.path.realpath(hipcc_path)))
            if os.path.basename(rocm_home) == 'hip':
                rocm_home = os.path.dirname(rocm_home)
        else:
            fallback_path = '/opt/rocm'
            if os.path.exists(fallback_path):
                rocm_home = fallback_path
    if rocm_home and torch.version.hip is None:
        print(f"No ROCm runtime is found, using ROCM_HOME='{rocm_home}'", file=sys.stderr)
    return rocm_home

def _join_rocm_home(*paths) -> str:
    if False:
        return 10
    '\n    Join paths with ROCM_HOME, or raises an error if it ROCM_HOME is not set.\n\n    This is basically a lazy way of raising an error for missing $ROCM_HOME\n    only once we need to get any ROCm-specific path.\n    '
    if ROCM_HOME is None:
        raise OSError('ROCM_HOME environment variable is not set. Please set it to your ROCm install root.')
    elif IS_WINDOWS:
        raise OSError('Building PyTorch extensions using ROCm and Windows is not supported.')
    return os.path.join(ROCM_HOME, *paths)
ABI_INCOMPATIBILITY_WARNING = '\n\n                               !! WARNING !!\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nYour compiler ({}) may be ABI-incompatible with PyTorch!\nPlease use a compiler that is ABI-compatible with GCC 5.0 and above.\nSee https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html.\n\nSee https://gist.github.com/goldsborough/d466f43e8ffc948ff92de7486c5216d6\nfor instructions on how to install GCC 5 or higher.\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n                              !! WARNING !!\n'
WRONG_COMPILER_WARNING = '\n\n                               !! WARNING !!\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nYour compiler ({user_compiler}) is not compatible with the compiler Pytorch was\nbuilt with for this platform, which is {pytorch_compiler} on {platform}. Please\nuse {pytorch_compiler} to to compile your extension. Alternatively, you may\ncompile PyTorch from source using {user_compiler}, and then you can also use\n{user_compiler} to compile your extension.\n\nSee https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md for help\nwith compiling PyTorch from source.\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n                              !! WARNING !!\n'
CUDA_MISMATCH_MESSAGE = '\nThe detected CUDA version ({0}) mismatches the version that was used to compile\nPyTorch ({1}). Please make sure to use the same CUDA versions.\n'
CUDA_MISMATCH_WARN = "The detected CUDA version ({0}) has a minor version mismatch with the version that was used to compile PyTorch ({1}). Most likely this shouldn't be a problem."
CUDA_NOT_FOUND_MESSAGE = '\nCUDA was not found on the system, please set the CUDA_HOME or the CUDA_PATH\nenvironment variable or add NVCC to your system PATH. The extension compilation will fail.\n'
ROCM_HOME = _find_rocm_home()
HIP_HOME = _join_rocm_home('hip') if ROCM_HOME else None
IS_HIP_EXTENSION = True if ROCM_HOME is not None and torch.version.hip is not None else False
ROCM_VERSION = None
if torch.version.hip is not None:
    ROCM_VERSION = tuple((int(v) for v in torch.version.hip.split('.')[:2]))
CUDA_HOME = _find_cuda_home() if torch.cuda._is_compiled() else None
CUDNN_HOME = os.environ.get('CUDNN_HOME') or os.environ.get('CUDNN_PATH')
BUILT_FROM_SOURCE_VERSION_PATTERN = re.compile('\\d+\\.\\d+\\.\\d+\\w+\\+\\w+')
COMMON_MSVC_FLAGS = ['/MD', '/wd4819', '/wd4251', '/wd4244', '/wd4267', '/wd4275', '/wd4018', '/wd4190', '/wd4624', '/wd4067', '/wd4068', '/EHsc']
MSVC_IGNORE_CUDAFE_WARNINGS = ['base_class_has_different_dll_interface', 'field_without_dll_interface', 'dll_interface_conflict_none_assumed', 'dll_interface_conflict_dllexport_assumed']
COMMON_NVCC_FLAGS = ['-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__', '--expt-relaxed-constexpr']
COMMON_HIP_FLAGS = ['-fPIC', '-D__HIP_PLATFORM_AMD__=1', '-DUSE_ROCM=1']
COMMON_HIPCC_FLAGS = ['-DCUDA_HAS_FP16=1', '-D__HIP_NO_HALF_OPERATORS__=1', '-D__HIP_NO_HALF_CONVERSIONS__=1']
JIT_EXTENSION_VERSIONER = ExtensionVersioner()
PLAT_TO_VCVARS = {'win32': 'x86', 'win-amd64': 'x86_amd64'}

def get_cxx_compiler():
    if False:
        return 10
    if IS_WINDOWS:
        compiler = os.environ.get('CXX', 'cl')
    else:
        compiler = os.environ.get('CXX', 'c++')
    return compiler

def _is_binary_build() -> bool:
    if False:
        return 10
    return not BUILT_FROM_SOURCE_VERSION_PATTERN.match(torch.version.__version__)

def _accepted_compilers_for_platform() -> List[str]:
    if False:
        i = 10
        return i + 15
    return ['clang++', 'clang'] if IS_MACOS else ['g++', 'gcc', 'gnu-c++', 'gnu-cc', 'clang++', 'clang']

def _maybe_write(filename, new_content):
    if False:
        return 10
    '\n    Equivalent to writing the content into the file but will not touch the file\n    if it already had the right content (to avoid triggering recompile).\n    '
    if os.path.exists(filename):
        with open(filename) as f:
            content = f.read()
        if content == new_content:
            return
    with open(filename, 'w') as source_file:
        source_file.write(new_content)

def get_default_build_root() -> str:
    if False:
        return 10
    "\n    Return the path to the root folder under which extensions will built.\n\n    For each extension module built, there will be one folder underneath the\n    folder returned by this function. For example, if ``p`` is the path\n    returned by this function and ``ext`` the name of an extension, the build\n    folder for the extension will be ``p/ext``.\n\n    This directory is **user-specific** so that multiple users on the same\n    machine won't meet permission issues.\n    "
    return os.path.realpath(torch._appdirs.user_cache_dir(appname='torch_extensions'))

def check_compiler_ok_for_platform(compiler: str) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Verify that the compiler is the expected one for the current platform.\n\n    Args:\n        compiler (str): The compiler executable to check.\n\n    Returns:\n        True if the compiler is gcc/g++ on Linux or clang/clang++ on macOS,\n        and always True for Windows.\n    '
    if IS_WINDOWS:
        return True
    which = subprocess.check_output(['which', compiler], stderr=subprocess.STDOUT)
    compiler_path = os.path.realpath(which.decode(*SUBPROCESS_DECODE_ARGS).strip())
    if any((name in compiler_path for name in _accepted_compilers_for_platform())):
        return True
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    if IS_LINUX:
        pattern = re.compile('^COLLECT_GCC=(.*)$', re.MULTILINE)
        results = re.findall(pattern, version_string)
        if len(results) != 1:
            return 'clang version' in version_string
        compiler_path = os.path.realpath(results[0].strip())
        if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
            return True
        return any((name in compiler_path for name in _accepted_compilers_for_platform()))
    if IS_MACOS:
        return version_string.startswith('Apple clang')
    return False

def get_compiler_abi_compatibility_and_version(compiler) -> Tuple[bool, TorchVersion]:
    if False:
        print('Hello World!')
    '\n    Determine if the given compiler is ABI-compatible with PyTorch alongside its version.\n\n    Args:\n        compiler (str): The compiler executable name to check (e.g. ``g++``).\n            Must be executable in a shell process.\n\n    Returns:\n        A tuple that contains a boolean that defines if the compiler is (likely) ABI-incompatible with PyTorch,\n        followed by a `TorchVersion` string that contains the compiler version separated by dots.\n    '
    if not _is_binary_build():
        return (True, TorchVersion('0.0.0'))
    if os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') in ['ON', '1', 'YES', 'TRUE', 'Y']:
        return (True, TorchVersion('0.0.0'))
    if not check_compiler_ok_for_platform(compiler):
        warnings.warn(WRONG_COMPILER_WARNING.format(user_compiler=compiler, pytorch_compiler=_accepted_compilers_for_platform()[0], platform=sys.platform))
        return (False, TorchVersion('0.0.0'))
    if IS_MACOS:
        return (True, TorchVersion('0.0.0'))
    try:
        if IS_LINUX:
            minimum_required_version = MINIMUM_GCC_VERSION
            versionstr = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
            version = versionstr.decode(*SUBPROCESS_DECODE_ARGS).strip().split('.')
        else:
            minimum_required_version = MINIMUM_MSVC_VERSION
            compiler_info = subprocess.check_output(compiler, stderr=subprocess.STDOUT)
            match = re.search('(\\d+)\\.(\\d+)\\.(\\d+)', compiler_info.decode(*SUBPROCESS_DECODE_ARGS).strip())
            version = ['0', '0', '0'] if match is None else list(match.groups())
    except Exception:
        (_, error, _) = sys.exc_info()
        warnings.warn(f'Error checking compiler version for {compiler}: {error}')
        return (False, TorchVersion('0.0.0'))
    if tuple(map(int, version)) >= minimum_required_version:
        return (True, TorchVersion('.'.join(version)))
    compiler = f"{compiler} {'.'.join(version)}"
    warnings.warn(ABI_INCOMPATIBILITY_WARNING.format(compiler))
    return (False, TorchVersion('.'.join(version)))

def _check_cuda_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    if False:
        return 10
    if not CUDA_HOME:
        raise RuntimeError(CUDA_NOT_FOUND_MESSAGE)
    nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
    cuda_version_str = subprocess.check_output([nvcc, '--version']).strip().decode(*SUBPROCESS_DECODE_ARGS)
    cuda_version = re.search('release (\\d+[.]\\d+)', cuda_version_str)
    if cuda_version is None:
        return
    cuda_str_version = cuda_version.group(1)
    cuda_ver = Version(cuda_str_version)
    if torch.version.cuda is None:
        return
    torch_cuda_version = Version(torch.version.cuda)
    if cuda_ver != torch_cuda_version:
        if getattr(cuda_ver, 'major', None) is None:
            raise ValueError('setuptools>=49.4.0 is required')
        if cuda_ver.major != torch_cuda_version.major:
            raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
        warnings.warn(CUDA_MISMATCH_WARN.format(cuda_str_version, torch.version.cuda))
    if not (sys.platform.startswith('linux') and os.environ.get('TORCH_DONT_CHECK_COMPILER_ABI') not in ['ON', '1', 'YES', 'TRUE', 'Y'] and _is_binary_build()):
        return
    cuda_compiler_bounds: VersionMap = CUDA_CLANG_VERSIONS if compiler_name.startswith('clang') else CUDA_GCC_VERSIONS
    if cuda_str_version not in cuda_compiler_bounds:
        warnings.warn(f'There are no {compiler_name} version bounds defined for CUDA version {cuda_str_version}')
    else:
        (min_compiler_version, max_excl_compiler_version) = cuda_compiler_bounds[cuda_str_version]
        if 'V11.4.48' in cuda_version_str and cuda_compiler_bounds == CUDA_GCC_VERSIONS:
            max_excl_compiler_version = (11, 0)
        min_compiler_version_str = '.'.join(map(str, min_compiler_version))
        max_excl_compiler_version_str = '.'.join(map(str, max_excl_compiler_version))
        version_bound_str = f'>={min_compiler_version_str}, <{max_excl_compiler_version_str}'
        if compiler_version < TorchVersion(min_compiler_version_str):
            raise RuntimeError(f'The current installed version of {compiler_name} ({compiler_version}) is less than the minimum required version by CUDA {cuda_str_version} ({min_compiler_version_str}). Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).')
        if compiler_version >= TorchVersion(max_excl_compiler_version_str):
            raise RuntimeError(f'The current installed version of {compiler_name} ({compiler_version}) is greater than the maximum required version by CUDA {cuda_str_version}. Please make sure to use an adequate version of {compiler_name} ({version_bound_str}).')

class BuildExtension(build_ext):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
    C++/CUDA compilation (and support for CUDA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``nvcc``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and CUDA compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """

    @classmethod
    def with_options(cls, **options):
        if False:
            print('Hello World!')
        'Return a subclass with alternative constructor that extends any original keyword arguments to the original constructor with the given options.'

        class cls_with_options(cls):

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                kwargs.update(options)
                super().__init__(*args, **kwargs)
        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get('no_python_abi_suffix', False)
        self.use_ninja = kwargs.get('use_ninja', True)
        if self.use_ninja:
            msg = 'Attempted to use ninja as the BuildExtension backend but {}. Falling back to using the slow distutils backend.'
            if not is_ninja_available():
                warnings.warn(msg.format('we could not find ninja.'))
                self.use_ninja = False

    def finalize_options(self) -> None:
        if False:
            return 10
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (compiler_name, compiler_version) = self._check_abi()
        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                (_, ext) = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)
        if cuda_ext and (not IS_HIP_EXTENSION):
            _check_cuda_version(compiler_name, compiler_version)
        for extension in self.extensions:
            if isinstance(extension.extra_compile_args, dict):
                for ext in ['cxx', 'nvcc']:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []
            self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
            for name in ['COMPILER_TYPE', 'STDLIB', 'BUILD_ABI']:
                val = getattr(torch._C, f'_PYBIND11_{name}')
                if val is not None and (not IS_WINDOWS):
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)
            if 'nvcc_dlink' in extension.extra_compile_args:
                assert self.use_ninja, f'With dlink=True, ninja is required to build cuda extension {extension.name}.'
        self.compiler.src_extensions += ['.cu', '.cuh', '.hip']
        if torch.backends.mps.is_built():
            self.compiler.src_extensions += ['.mm']
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            if False:
                i = 10
                return i + 15
            cpp_format_prefix = '/{}:' if self.compiler.compiler_type == 'msvc' else '-{}='
            cpp_flag_prefix = cpp_format_prefix.format('std')
            cpp_flag = cpp_flag_prefix + 'c++17'
            if not any((flag.startswith(cpp_flag_prefix) for flag in cflags)):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            if False:
                return 10
            cflags = COMMON_NVCC_FLAGS + ['--compiler-options', "'-fPIC'"] + cflags + _get_cuda_arch_flags(cflags)
            _ccbin = os.getenv('CC')
            if _ccbin is not None and (not any((flag.startswith(('-ccbin', '--compiler-bindir')) for flag in cflags))):
                cflags.extend(['-ccbin', _ccbin])
            return cflags

        def convert_to_absolute_paths_inplace(paths):
            if False:
                for i in range(10):
                    print('nop')
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            if False:
                for i in range(10):
                    print('nop')
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = [_join_rocm_home('bin', 'hipcc') if IS_HIP_EXTENSION else _join_cuda_home('bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    if IS_HIP_EXTENSION:
                        cflags = COMMON_HIPCC_FLAGS + cflags + _get_rocm_arch_flags(cflags)
                    else:
                        cflags = unix_cuda_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                if IS_HIP_EXTENSION:
                    cflags = COMMON_HIP_FLAGS + cflags
                append_std17_if_no_std_present(cflags)
                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                self.compiler.set_executable('compiler_so', original_compiler)

        def unix_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
            if False:
                for i in range(10):
                    print('nop')
            'Compiles sources by outputting a ninja file and running it.'
            output_dir = os.path.abspath(output_dir)
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)
            (_, objects, extra_postargs, pp_opts, _) = self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(_is_cuda_file, sources))
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)
            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = cuda_post_cflags + _get_rocm_arch_flags(cuda_post_cflags)
                    cuda_post_cflags = COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                else:
                    cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std17_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]
            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = unix_cuda_flags(extra_postargs['nvcc_dlink'])
            else:
                cuda_dlink_post_cflags = None
            _write_ninja_file_and_compile_objects(sources=sources, objects=objects, cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags], post_cflags=[shlex.quote(f) for f in post_cflags], cuda_cflags=cuda_cflags, cuda_post_cflags=cuda_post_cflags, cuda_dlink_post_cflags=cuda_dlink_post_cflags, build_directory=output_dir, verbose=True, with_cuda=with_cuda)
            return objects

        def win_cuda_flags(cflags):
            if False:
                return 10
            return COMMON_NVCC_FLAGS + cflags + _get_cuda_arch_flags(cflags)

        def win_wrap_single_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
            if False:
                while True:
                    i = 10
            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                if False:
                    while True:
                        i = 10
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [m.group(2) for m in (src_regex.match(elem) for elem in cmd) if m]
                obj_regex = re.compile('/Fo(.*)')
                obj_list = [m.group(1) for m in (obj_regex.match(elem) for elem in cmd) if m]
                include_regex = re.compile('((\\-|\\/)I.*)')
                include_list = [m.group(1) for m in (include_regex.match(elem) for elem in cmd) if m]
                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []
                        cflags = win_cuda_flags(cflags) + ['-std=c++17', '--use-local-env']
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ['-Xcompiler', flag] + cflags
                        for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                            cflags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cflags
                        cmd = [nvcc, '-c', src, '-o', obj] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = COMMON_MSVC_FLAGS + self.cflags['cxx']
                        append_std17_if_no_std_present(cflags)
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = COMMON_MSVC_FLAGS + self.cflags
                        append_std17_if_no_std_present(cflags)
                        cmd += cflags
                return original_spawn(cmd)
            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros, include_dirs, debug, extra_preargs, extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        def win_wrap_ninja_compile(sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
            if False:
                for i in range(10):
                    print('nop')
            if not self.compiler.initialized:
                self.compiler.initialize()
            output_dir = os.path.abspath(output_dir)
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)
            (_, objects, extra_postargs, pp_opts, _) = self.compiler._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
            common_cflags = extra_preargs or []
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            common_cflags.extend(COMMON_MSVC_FLAGS)
            cflags = cflags + common_cflags + pp_opts
            with_cuda = any(map(_is_cuda_file, sources))
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs['cxx']
            else:
                post_cflags = list(extra_postargs)
            append_std17_if_no_std_present(post_cflags)
            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = ['-std=c++17', '--use-local-env']
                for common_cflag in common_cflags:
                    cuda_cflags.append('-Xcompiler')
                    cuda_cflags.append(common_cflag)
                for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                    cuda_cflags.append('-Xcudafe')
                    cuda_cflags.append('--diag_suppress=' + ignore_warning)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs['nvcc']
                else:
                    cuda_post_cflags = list(extra_postargs)
                cuda_post_cflags = win_cuda_flags(cuda_post_cflags)
            cflags = _nt_quote_args(cflags)
            post_cflags = _nt_quote_args(post_cflags)
            if with_cuda:
                cuda_cflags = _nt_quote_args(cuda_cflags)
                cuda_post_cflags = _nt_quote_args(cuda_post_cflags)
            if isinstance(extra_postargs, dict) and 'nvcc_dlink' in extra_postargs:
                cuda_dlink_post_cflags = win_cuda_flags(extra_postargs['nvcc_dlink'])
            else:
                cuda_dlink_post_cflags = None
            _write_ninja_file_and_compile_objects(sources=sources, objects=objects, cflags=cflags, post_cflags=post_cflags, cuda_cflags=cuda_cflags, cuda_post_cflags=cuda_post_cflags, cuda_dlink_post_cflags=cuda_dlink_post_cflags, build_directory=output_dir, verbose=True, with_cuda=with_cuda)
            return objects
        if self.compiler.compiler_type == 'msvc':
            if self.use_ninja:
                self.compiler.compile = win_wrap_ninja_compile
            else:
                self.compiler.compile = win_wrap_single_compile
        elif self.use_ninja:
            self.compiler.compile = unix_wrap_ninja_compile
        else:
            self.compiler._compile = unix_wrap_single_compile
        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        if False:
            return 10
        ext_filename = super().get_ext_filename(ext_name)
        if self.no_python_abi_suffix:
            ext_filename_parts = ext_filename.split('.')
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename

    def _check_abi(self) -> Tuple[str, TorchVersion]:
        if False:
            while True:
                i = 10
        if hasattr(self.compiler, 'compiler_cxx'):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()
        (_, version) = get_compiler_abi_compatibility_and_version(compiler)
        if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' in os.environ and ('DISTUTILS_USE_SDK' not in os.environ):
            msg = 'It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.This may lead to multiple activations of the VC env.Please set `DISTUTILS_USE_SDK=1` and try again.'
            raise UserWarning(msg)
        return (compiler, version)

    def _add_compile_flag(self, extension, flag):
        if False:
            for i in range(10):
                print('nop')
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        if False:
            i = 10
            return i + 15
        names = extension.name.split('.')
        name = names[-1]
        define = f'-DTORCH_EXTENSION_NAME={name}'
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        if False:
            print('Hello World!')
        self._add_compile_flag(extension, '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))

def CppExtension(name, sources, *args, **kwargs):
    if False:
        print('Hello World!')
    "\n    Create a :class:`setuptools.Extension` for C++.\n\n    Convenience method that creates a :class:`setuptools.Extension` with the\n    bare minimum (but often sufficient) arguments to build a C++ extension.\n\n    All arguments are forwarded to the :class:`setuptools.Extension`\n    constructor.\n\n    Example:\n        >>> # xdoctest: +SKIP\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)\n        >>> from setuptools import setup\n        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension\n        >>> setup(\n        ...     name='extension',\n        ...     ext_modules=[\n        ...         CppExtension(\n        ...             name='extension',\n        ...             sources=['extension.cpp'],\n        ...             extra_compile_args=['-g']),\n        ...     ],\n        ...     cmdclass={\n        ...         'build_ext': BuildExtension\n        ...     })\n    "
    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths()
    kwargs['library_dirs'] = library_dirs
    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)

def CUDAExtension(name, sources, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Create a :class:`setuptools.Extension` for CUDA/C++.\n\n    Convenience method that creates a :class:`setuptools.Extension` with the\n    bare minimum (but often sufficient) arguments to build a CUDA/C++\n    extension. This includes the CUDA include path, library path and runtime\n    library.\n\n    All arguments are forwarded to the :class:`setuptools.Extension`\n    constructor.\n\n    Example:\n        >>> # xdoctest: +SKIP\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)\n        >>> from setuptools import setup\n        >>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n        >>> setup(\n        ...     name=\'cuda_extension\',\n        ...     ext_modules=[\n        ...         CUDAExtension(\n        ...                 name=\'cuda_extension\',\n        ...                 sources=[\'extension.cpp\', \'extension_kernel.cu\'],\n        ...                 extra_compile_args={\'cxx\': [\'-g\'],\n        ...                                     \'nvcc\': [\'-O2\']})\n        ...     ],\n        ...     cmdclass={\n        ...         \'build_ext\': BuildExtension\n        ...     })\n\n    Compute capabilities:\n\n    By default the extension will be compiled to run on all archs of the cards visible during the\n    building process of the extension, plus PTX. If down the road a new card is installed the\n    extension may need to be recompiled. If a visible card has a compute capability (CC) that\'s\n    newer than the newest version for which your nvcc can build fully-compiled binaries, Pytorch\n    will make nvcc fall back to building kernels with the newest version of PTX your nvcc does\n    support (see below for details on PTX).\n\n    You can override the default behavior using `TORCH_CUDA_ARCH_LIST` to explicitly specify which\n    CCs you want the extension to support:\n\n    ``TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py``\n    ``TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py``\n\n    The +PTX option causes extension kernel binaries to include PTX instructions for the specified\n    CC. PTX is an intermediate representation that allows kernels to runtime-compile for any CC >=\n    the specified CC (for example, 8.6+PTX generates PTX that can runtime-compile for any GPU with\n    CC >= 8.6). This improves your binary\'s forward compatibility. However, relying on older PTX to\n    provide forward compat by runtime-compiling for newer CCs can modestly reduce performance on\n    those newer CCs. If you know exact CC(s) of the GPUs you want to target, you\'re always better\n    off specifying them individually. For example, if you want your extension to run on 8.0 and 8.6,\n    "8.0+PTX" would work functionally because it includes PTX that can runtime-compile for 8.6, but\n    "8.0 8.6" would be better.\n\n    Note that while it\'s possible to include all supported archs, the more archs get included the\n    slower the building process will be, as it will build a separate kernel image for each arch.\n\n    Note that CUDA-11.5 nvcc will hit internal compiler error while parsing torch/extension.h on Windows.\n    To workaround the issue, move python binding logic to pure C++ file.\n\n    Example use:\n        #include <ATen/ATen.h>\n        at::Tensor SigmoidAlphaBlendForwardCuda(....)\n\n    Instead of:\n        #include <torch/extension.h>\n        torch::Tensor SigmoidAlphaBlendForwardCuda(...)\n\n    Currently open issue for nvcc bug: https://github.com/pytorch/pytorch/issues/69460\n    Complete workaround code example: https://github.com/facebookresearch/pytorch3d/commit/cb170ac024a949f1f9614ffe6af1c38d972f7d48\n\n    Relocatable device code linking:\n\n    If you want to reference device symbols across compilation units (across object files),\n    the object files need to be built with `relocatable device code` (-rdc=true or -dc).\n    An exception to this rule is "dynamic parallelism" (nested kernel launches)  which is not used a lot anymore.\n    `Relocatable device code` is less optimized so it needs to be used only on object files that need it.\n    Using `-dlto` (Device Link Time Optimization) at the device code compilation step and `dlink` step\n    help reduce the protentional perf degradation of `-rdc`.\n    Note that it needs to be used at both steps to be useful.\n\n    If you have `rdc` objects you need to have an extra `-dlink` (device linking) step before the CPU symbol linking step.\n    There is also a case where `-dlink` is used without `-rdc`:\n    when an extension is linked against a static lib containing rdc-compiled objects\n    like the [NVSHMEM library](https://developer.nvidia.com/nvshmem).\n\n    Note: Ninja is required to build a CUDA Extension with RDC linking.\n\n    Example:\n        >>> # xdoctest: +SKIP\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)\n        >>> CUDAExtension(\n        ...        name=\'cuda_extension\',\n        ...        sources=[\'extension.cpp\', \'extension_kernel.cu\'],\n        ...        dlink=True,\n        ...        dlink_libraries=["dlink_lib"],\n        ...        extra_compile_args={\'cxx\': [\'-g\'],\n        ...                            \'nvcc\': [\'-O2\', \'-rdc=true\']})\n    '
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs += library_paths(cuda=True)
    kwargs['library_dirs'] = library_dirs
    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    if IS_HIP_EXTENSION:
        assert ROCM_VERSION is not None
        libraries.append('amdhip64' if ROCM_VERSION >= (3, 5) else 'hip_hcc')
        libraries.append('c10_hip')
        libraries.append('torch_hip')
    else:
        libraries.append('cudart')
        libraries.append('c10_cuda')
        libraries.append('torch_cuda')
    kwargs['libraries'] = libraries
    include_dirs = kwargs.get('include_dirs', [])
    if IS_HIP_EXTENSION:
        build_dir = os.getcwd()
        hipify_result = hipify_python.hipify(project_directory=build_dir, output_directory=build_dir, header_include_dirs=include_dirs, includes=[os.path.join(build_dir, '*')], extra_files=[os.path.abspath(s) for s in sources], show_detailed=True, is_pytorch_extension=True, hipify_extra_files_only=True)
        hipified_sources = set()
        for source in sources:
            s_abs = os.path.abspath(source)
            hipified_s_abs = hipify_result[s_abs].hipified_path if s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None else s_abs
            hipified_sources.add(os.path.relpath(hipified_s_abs, build_dir))
        sources = list(hipified_sources)
    include_dirs += include_paths(cuda=True)
    kwargs['include_dirs'] = include_dirs
    kwargs['language'] = 'c++'
    dlink_libraries = kwargs.get('dlink_libraries', [])
    dlink = kwargs.get('dlink', False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get('extra_compile_args', {})
        extra_compile_args_dlink = extra_compile_args.get('nvcc_dlink', [])
        extra_compile_args_dlink += ['-dlink']
        extra_compile_args_dlink += [f'-L{x}' for x in library_dirs]
        extra_compile_args_dlink += [f'-l{x}' for x in dlink_libraries]
        if torch.version.cuda is not None and TorchVersion(torch.version.cuda) >= '11.2':
            extra_compile_args_dlink += ['-dlto']
        extra_compile_args['nvcc_dlink'] = extra_compile_args_dlink
        kwargs['extra_compile_args'] = extra_compile_args
    return setuptools.Extension(name, sources, *args, **kwargs)

def include_paths(cuda: bool=False) -> List[str]:
    if False:
        print('Hello World!')
    '\n    Get the include paths required to build a C++ or CUDA extension.\n\n    Args:\n        cuda: If `True`, includes CUDA-specific include paths.\n\n    Returns:\n        A list of include path strings.\n    '
    lib_include = os.path.join(_TORCH_PATH, 'include')
    paths = [lib_include, os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'), os.path.join(lib_include, 'TH'), os.path.join(lib_include, 'THC')]
    if cuda and IS_HIP_EXTENSION:
        paths.append(os.path.join(lib_include, 'THH'))
        paths.append(_join_rocm_home('include'))
    elif cuda:
        cuda_home_include = _join_cuda_home('include')
        if cuda_home_include != '/usr/include':
            paths.append(cuda_home_include)
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, 'include'))
    return paths

def library_paths(cuda: bool=False) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the library paths required to build a C++ or CUDA extension.\n\n    Args:\n        cuda: If `True`, includes CUDA-specific library paths.\n\n    Returns:\n        A list of library path strings.\n    '
    paths = [TORCH_LIB_PATH]
    if cuda and IS_HIP_EXTENSION:
        lib_dir = 'lib'
        paths.append(_join_rocm_home(lib_dir))
        if HIP_HOME is not None:
            paths.append(os.path.join(HIP_HOME, 'lib'))
    elif cuda:
        if IS_WINDOWS:
            lib_dir = os.path.join('lib', 'x64')
        else:
            lib_dir = 'lib64'
            if not os.path.exists(_join_cuda_home(lib_dir)) and os.path.exists(_join_cuda_home('lib')):
                lib_dir = 'lib'
        paths.append(_join_cuda_home(lib_dir))
        if CUDNN_HOME is not None:
            paths.append(os.path.join(CUDNN_HOME, lib_dir))
    return paths

def load(name, sources: Union[str, List[str]], extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda: Optional[bool]=None, is_python_module=True, is_standalone=False, keep_intermediates=True):
    if False:
        while True:
            i = 10
    "\n    Load a PyTorch C++ extension just-in-time (JIT).\n\n    To load an extension, a Ninja build file is emitted, which is used to\n    compile the given sources into a dynamic library. This library is\n    subsequently loaded into the current Python process as a module and\n    returned from this function, ready for use.\n\n    By default, the directory to which the build file is emitted and the\n    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where\n    ``<tmp>`` is the temporary folder on the current platform and ``<name>``\n    the name of the extension. This location can be overridden in two ways.\n    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it\n    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled\n    into subfolders of this directory. Second, if the ``build_directory``\n    argument to this function is supplied, it overrides the entire path, i.e.\n    the library will be compiled into that folder directly.\n\n    To compile the sources, the default system compiler (``c++``) is used,\n    which can be overridden by setting the ``CXX`` environment variable. To pass\n    additional arguments to the compilation process, ``extra_cflags`` or\n    ``extra_ldflags`` can be provided. For example, to compile your extension\n    with optimizations, pass ``extra_cflags=['-O3']``. You can also use\n    ``extra_cflags`` to pass further include directories.\n\n    CUDA support with mixed compilation is provided. Simply pass CUDA source\n    files (``.cu`` or ``.cuh``) along with other sources. Such files will be\n    detected and compiled with nvcc rather than the C++ compiler. This includes\n    passing the CUDA lib64 directory as a library directory, and linking\n    ``cudart``. You can pass additional flags to nvcc via\n    ``extra_cuda_cflags``, just like with ``extra_cflags`` for C++. Various\n    heuristics for finding the CUDA install directory are used, which usually\n    work fine. If not, setting the ``CUDA_HOME`` environment variable is the\n    safest option.\n\n    Args:\n        name: The name of the extension to build. This MUST be the same as the\n            name of the pybind11 module!\n        sources: A list of relative or absolute paths to C++ source files.\n        extra_cflags: optional list of compiler flags to forward to the build.\n        extra_cuda_cflags: optional list of compiler flags to forward to nvcc\n            when building CUDA sources.\n        extra_ldflags: optional list of linker flags to forward to the build.\n        extra_include_paths: optional list of include directories to forward\n            to the build.\n        build_directory: optional path to use as build workspace.\n        verbose: If ``True``, turns on verbose logging of load steps.\n        with_cuda: Determines whether CUDA headers and libraries are added to\n            the build. If set to ``None`` (default), this value is\n            automatically determined based on the existence of ``.cu`` or\n            ``.cuh`` in ``sources``. Set it to `True`` to force CUDA headers\n            and libraries to be included.\n        is_python_module: If ``True`` (default), imports the produced shared\n            library as a Python module. If ``False``, behavior depends on\n            ``is_standalone``.\n        is_standalone: If ``False`` (default) loads the constructed extension\n            into the process as a plain dynamic library. If ``True``, build a\n            standalone executable.\n\n    Returns:\n        If ``is_python_module`` is ``True``:\n            Returns the loaded PyTorch extension as a Python module.\n\n        If ``is_python_module`` is ``False`` and ``is_standalone`` is ``False``:\n            Returns nothing. (The shared library is loaded into the process as\n            a side effect.)\n\n        If ``is_standalone`` is ``True``.\n            Return the path to the executable. (On Windows, TORCH_LIB_PATH is\n            added to the PATH environment variable as a side effect.)\n\n    Example:\n        >>> # xdoctest: +SKIP\n        >>> from torch.utils.cpp_extension import load\n        >>> module = load(\n        ...     name='extension',\n        ...     sources=['extension.cpp', 'extension_kernel.cu'],\n        ...     extra_cflags=['-O2'],\n        ...     verbose=True)\n    "
    return _jit_compile(name, [sources] if isinstance(sources, str) else sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory or _get_build_directory(name, verbose), verbose, with_cuda, is_python_module, is_standalone, keep_intermediates=keep_intermediates)

def _get_pybind11_abi_build_flags():
    if False:
        return 10
    abi_cflags = []
    for pname in ['COMPILER_TYPE', 'STDLIB', 'BUILD_ABI']:
        pval = getattr(torch._C, f'_PYBIND11_{pname}')
        if pval is not None and (not IS_WINDOWS):
            abi_cflags.append(f'-DPYBIND11_{pname}=\\"{pval}\\"')
    return abi_cflags

def _get_glibcxx_abi_build_flags():
    if False:
        i = 10
        return i + 15
    glibcxx_abi_cflags = ['-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    return glibcxx_abi_cflags

def check_compiler_is_gcc(compiler):
    if False:
        return 10
    if not IS_LINUX:
        return False
    env = os.environ.copy()
    env['LC_ALL'] = 'C'
    version_string = subprocess.check_output([compiler, '-v'], stderr=subprocess.STDOUT, env=env).decode(*SUBPROCESS_DECODE_ARGS)
    pattern = re.compile('^COLLECT_GCC=(.*)$', re.MULTILINE)
    results = re.findall(pattern, version_string)
    if len(results) != 1:
        return False
    compiler_path = os.path.realpath(results[0].strip())
    if os.path.basename(compiler_path) == 'c++' and 'gcc version' in version_string:
        return True
    return False

def _check_and_build_extension_h_precompiler_headers(extra_cflags, extra_include_paths, is_standalone=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Precompiled Headers(PCH) can pre-build the same headers and reduce build time for pytorch load_inline modules.\n    GCC offical manual: https://gcc.gnu.org/onlinedocs/gcc-4.0.4/gcc/Precompiled-Headers.html\n    PCH only works when built pch file(header.h.gch) and build target have the same build parameters. So, We need\n    add a signature file to record PCH file parameters. If the build parameters(signature) changed, it should rebuild\n    PCH file.\n\n    Note:\n    1. Windows and MacOS have different PCH mechanism. We only support Linux currently.\n    2. It only works on GCC/G++.\n    '
    if not IS_LINUX:
        return
    compiler = get_cxx_compiler()
    b_is_gcc = check_compiler_is_gcc(compiler)
    if b_is_gcc is False:
        return
    head_file = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h')
    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')

    def listToString(s):
        if False:
            while True:
                i = 10
        string = ''
        if s is None:
            return string
        for element in s:
            string += element + ' '
        return string

    def format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags, torch_include_dirs, extra_cflags, extra_include_paths):
        if False:
            return 10
        return re.sub('[ \\n]+', ' ', f'\n                {compiler} -x c++-header {head_file} -o {head_file_pch} {torch_include_dirs} {extra_include_paths} {extra_cflags} {common_cflags}\n            ').strip()

    def command_to_signature(cmd):
        if False:
            for i in range(10):
                print('nop')
        signature = cmd.replace(' ', '_')
        return signature

    def check_pch_signature_in_file(file_path, signature):
        if False:
            return 10
        b_exist = os.path.isfile(file_path)
        if b_exist is False:
            return False
        with open(file_path) as file:
            content = file.read()
            if signature == content:
                return True
            else:
                return False

    def _create_if_not_exist(path_dir):
        if False:
            return 10
        if not os.path.exists(path_dir):
            try:
                Path(path_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise RuntimeError(f'Fail to create path {path_dir}') from exc

    def write_pch_signature_to_file(file_path, pch_sign):
        if False:
            while True:
                i = 10
        _create_if_not_exist(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            f.write(pch_sign)
            f.close()

    def build_precompile_header(pch_cmd):
        if False:
            return 10
        try:
            subprocess.check_output(pch_cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Compile PreCompile Header fail, command: {pch_cmd}') from e
    extra_cflags_str = listToString(extra_cflags)
    extra_include_paths_str = ' '.join([f'-I{include}' for include in extra_include_paths] if extra_include_paths else [])
    lib_include = os.path.join(_TORCH_PATH, 'include')
    torch_include_dirs = [f'-I {lib_include}', '-I {}'.format(sysconfig.get_path('include')), '-I {}'.format(os.path.join(lib_include, 'torch', 'csrc', 'api', 'include'))]
    torch_include_dirs_str = listToString(torch_include_dirs)
    common_cflags = []
    if not is_standalone:
        common_cflags += ['-DTORCH_API_INCLUDE_EXTENSION_H']
    common_cflags += ['-std=c++17', '-fPIC']
    common_cflags += [f'{x}' for x in _get_pybind11_abi_build_flags()]
    common_cflags += [f'{x}' for x in _get_glibcxx_abi_build_flags()]
    common_cflags_str = listToString(common_cflags)
    pch_cmd = format_precompiler_header_cmd(compiler, head_file, head_file_pch, common_cflags_str, torch_include_dirs_str, extra_cflags_str, extra_include_paths_str)
    pch_sign = command_to_signature(pch_cmd)
    if os.path.isfile(head_file_pch) is not True:
        build_precompile_header(pch_cmd)
        write_pch_signature_to_file(head_file_signature, pch_sign)
    else:
        b_same_sign = check_pch_signature_in_file(head_file_signature, pch_sign)
        if b_same_sign is False:
            build_precompile_header(pch_cmd)
            write_pch_signature_to_file(head_file_signature, pch_sign)

def remove_extension_h_precompiler_headers():
    if False:
        while True:
            i = 10

    def _remove_if_file_exists(path_file):
        if False:
            return 10
        if os.path.exists(path_file):
            os.remove(path_file)
    head_file_pch = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.gch')
    head_file_signature = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h.sign')
    _remove_if_file_exists(head_file_pch)
    _remove_if_file_exists(head_file_signature)

def load_inline(name, cpp_sources, cuda_sources=None, functions=None, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None, build_directory=None, verbose=False, with_cuda=None, is_python_module=True, with_pytorch_error_handling=True, keep_intermediates=True, use_pch=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load a PyTorch C++ extension just-in-time (JIT) from string sources.\n\n    This function behaves exactly like :func:`load`, but takes its sources as\n    strings rather than filenames. These strings are stored to files in the\n    build directory, after which the behavior of :func:`load_inline` is\n    identical to :func:`load`.\n\n    See `the\n    tests <https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions_jit.py>`_\n    for good examples of using this function.\n\n    Sources may omit two required parts of a typical non-inline C++ extension:\n    the necessary header includes, as well as the (pybind11) binding code. More\n    precisely, strings passed to ``cpp_sources`` are first concatenated into a\n    single ``.cpp`` file. This file is then prepended with ``#include\n    <torch/extension.h>``.\n\n    Furthermore, if the ``functions`` argument is supplied, bindings will be\n    automatically generated for each function specified. ``functions`` can\n    either be a list of function names, or a dictionary mapping from function\n    names to docstrings. If a list is given, the name of each function is used\n    as its docstring.\n\n    The sources in ``cuda_sources`` are concatenated into a separate ``.cu``\n    file and  prepended with ``torch/types.h``, ``cuda.h`` and\n    ``cuda_runtime.h`` includes. The ``.cpp`` and ``.cu`` files are compiled\n    separately, but ultimately linked into a single library. Note that no\n    bindings are generated for functions in ``cuda_sources`` per  se. To bind\n    to a CUDA kernel, you must create a C++ function that calls it, and either\n    declare or define this C++ function in one of the ``cpp_sources`` (and\n    include its name in ``functions``).\n\n    See :func:`load` for a description of arguments omitted below.\n\n    Args:\n        cpp_sources: A string, or list of strings, containing C++ source code.\n        cuda_sources: A string, or list of strings, containing CUDA source code.\n        functions: A list of function names for which to generate function\n            bindings. If a dictionary is given, it should map function names to\n            docstrings (which are otherwise just the function names).\n        with_cuda: Determines whether CUDA headers and libraries are added to\n            the build. If set to ``None`` (default), this value is\n            automatically determined based on whether ``cuda_sources`` is\n            provided. Set it to ``True`` to force CUDA headers\n            and libraries to be included.\n        with_pytorch_error_handling: Determines whether pytorch error and\n            warning macros are handled by pytorch instead of pybind. To do\n            this, each function ``foo`` is called via an intermediary ``_safe_foo``\n            function. This redirection might cause issues in obscure cases\n            of cpp. This flag should be set to ``False`` when this redirect\n            causes issues.\n\n    Example:\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)\n        >>> from torch.utils.cpp_extension import load_inline\n        >>> source = """\n        at::Tensor sin_add(at::Tensor x, at::Tensor y) {\n          return x.sin() + y.sin();\n        }\n        """\n        >>> module = load_inline(name=\'inline_extension\',\n        ...                      cpp_sources=[source],\n        ...                      functions=[\'sin_add\'])\n\n    .. note::\n        By default, the Ninja backend uses #CPUS + 2 workers to build the\n        extension. This may use up too many resources on some systems. One\n        can control the number of workers by setting the `MAX_JOBS` environment\n        variable to a non-negative number.\n    '
    build_directory = build_directory or _get_build_directory(name, verbose)
    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]
    cuda_sources = cuda_sources or []
    if isinstance(cuda_sources, str):
        cuda_sources = [cuda_sources]
    cpp_sources.insert(0, '#include <torch/extension.h>')
    if use_pch is True:
        _check_and_build_extension_h_precompiler_headers(extra_cflags, extra_include_paths)
    else:
        remove_extension_h_precompiler_headers()
    if functions is not None:
        module_def = []
        module_def.append('PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {')
        if isinstance(functions, str):
            functions = [functions]
        if isinstance(functions, list):
            functions = {f: f for f in functions}
        elif not isinstance(functions, dict):
            raise ValueError(f"Expected 'functions' to be a list or dict, but was {type(functions)}")
        for (function_name, docstring) in functions.items():
            if with_pytorch_error_handling:
                module_def.append(f'm.def("{function_name}", torch::wrap_pybind_function({function_name}), "{docstring}");')
            else:
                module_def.append(f'm.def("{function_name}", {function_name}, "{docstring}");')
        module_def.append('}')
        cpp_sources += module_def
    cpp_source_path = os.path.join(build_directory, 'main.cpp')
    _maybe_write(cpp_source_path, '\n'.join(cpp_sources))
    sources = [cpp_source_path]
    if cuda_sources:
        cuda_sources.insert(0, '#include <torch/types.h>')
        cuda_sources.insert(1, '#include <cuda.h>')
        cuda_sources.insert(2, '#include <cuda_runtime.h>')
        cuda_source_path = os.path.join(build_directory, 'cuda.cu')
        _maybe_write(cuda_source_path, '\n'.join(cuda_sources))
        sources.append(cuda_source_path)
    return _jit_compile(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory, verbose, with_cuda, is_python_module, is_standalone=False, keep_intermediates=keep_intermediates)

def _jit_compile(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory: str, verbose: bool, with_cuda: Optional[bool], is_python_module, is_standalone, keep_intermediates=True) -> None:
    if False:
        i = 10
        return i + 15
    if is_python_module and is_standalone:
        raise ValueError('`is_python_module` and `is_standalone` are mutually exclusive.')
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    with_cudnn = any(('cudnn' in f for f in extra_ldflags or []))
    old_version = JIT_EXTENSION_VERSIONER.get_version(name)
    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(name, sources, build_arguments=[extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths], build_directory=build_directory, with_cuda=with_cuda, is_python_module=is_python_module, is_standalone=is_standalone)
    if version > 0:
        if version != old_version and verbose:
            print(f'The input conditions for extension module {name} have changed. ' + f'Bumping to version {version} and re-building as {name}_v{version}...', file=sys.stderr)
        name = f'{name}_v{version}'
    if version != old_version:
        baton = FileBaton(os.path.join(build_directory, 'lock'))
        if baton.try_acquire():
            try:
                with GeneratedFileCleaner(keep_intermediates=keep_intermediates) as clean_ctx:
                    if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                        hipify_result = hipify_python.hipify(project_directory=build_directory, output_directory=build_directory, header_include_dirs=extra_include_paths if extra_include_paths is not None else [], extra_files=[os.path.abspath(s) for s in sources], ignores=[_join_rocm_home('*'), os.path.join(_TORCH_PATH, '*')], show_detailed=verbose, show_progress=verbose, is_pytorch_extension=True, clean_ctx=clean_ctx)
                        hipified_sources = set()
                        for source in sources:
                            s_abs = os.path.abspath(source)
                            hipified_sources.add(hipify_result[s_abs].hipified_path if s_abs in hipify_result else s_abs)
                        sources = list(hipified_sources)
                    _write_ninja_file_and_build_library(name=name, sources=sources, extra_cflags=extra_cflags or [], extra_cuda_cflags=extra_cuda_cflags or [], extra_ldflags=extra_ldflags or [], extra_include_paths=extra_include_paths or [], build_directory=build_directory, verbose=verbose, with_cuda=with_cuda, is_standalone=is_standalone)
            finally:
                baton.release()
        else:
            baton.wait()
    elif verbose:
        print(f'No modifications detected for re-loaded extension module {name}, skipping build step...', file=sys.stderr)
    if verbose:
        print(f'Loading extension module {name}...', file=sys.stderr)
    if is_standalone:
        return _get_exec_path(name, build_directory)
    return _import_module_from_library(name, build_directory, is_python_module)

def _write_ninja_file_and_compile_objects(sources: List[str], objects, cflags, post_cflags, cuda_cflags, cuda_post_cflags, cuda_dlink_post_cflags, build_directory: str, verbose: bool, with_cuda: Optional[bool]) -> None:
    if False:
        return 10
    verify_ninja_availability()
    compiler = get_cxx_compiler()
    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...', file=sys.stderr)
    _write_ninja_file(path=build_file_path, cflags=cflags, post_cflags=post_cflags, cuda_cflags=cuda_cflags, cuda_post_cflags=cuda_post_cflags, cuda_dlink_post_cflags=cuda_dlink_post_cflags, sources=sources, objects=objects, ldflags=None, library_target=None, with_cuda=with_cuda)
    if verbose:
        print('Compiling objects...', file=sys.stderr)
    _run_ninja_build(build_directory, verbose, error_prefix='Error compiling objects for extension')

def _write_ninja_file_and_build_library(name, sources: List[str], extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, build_directory: str, verbose: bool, with_cuda: Optional[bool], is_standalone: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    verify_ninja_availability()
    compiler = get_cxx_compiler()
    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    extra_ldflags = _prepare_ldflags(extra_ldflags or [], with_cuda, verbose, is_standalone)
    build_file_path = os.path.join(build_directory, 'build.ninja')
    if verbose:
        print(f'Emitting ninja build file {build_file_path}...', file=sys.stderr)
    _write_ninja_file_to_build_library(path=build_file_path, name=name, sources=sources, extra_cflags=extra_cflags or [], extra_cuda_cflags=extra_cuda_cflags or [], extra_ldflags=extra_ldflags or [], extra_include_paths=extra_include_paths or [], with_cuda=with_cuda, is_standalone=is_standalone)
    if verbose:
        print(f'Building extension module {name}...', file=sys.stderr)
    _run_ninja_build(build_directory, verbose, error_prefix=f"Error building extension '{name}'")

def is_ninja_available():
    if False:
        for i in range(10):
            print('nop')
    'Return ``True`` if the `ninja <https://ninja-build.org/>`_ build system is available on the system, ``False`` otherwise.'
    try:
        subprocess.check_output('ninja --version'.split())
    except Exception:
        return False
    else:
        return True

def verify_ninja_availability():
    if False:
        return 10
    'Raise ``RuntimeError`` if `ninja <https://ninja-build.org/>`_ build system is not available on the system, does nothing otherwise.'
    if not is_ninja_available():
        raise RuntimeError('Ninja is required to load C++ extensions')

def _prepare_ldflags(extra_ldflags, with_cuda, verbose, is_standalone):
    if False:
        return 10
    if IS_WINDOWS:
        python_path = os.path.dirname(sys.executable)
        python_lib_path = os.path.join(python_path, 'libs')
        extra_ldflags.append('c10.lib')
        if with_cuda:
            extra_ldflags.append('c10_cuda.lib')
        extra_ldflags.append('torch_cpu.lib')
        if with_cuda:
            extra_ldflags.append('torch_cuda.lib')
            extra_ldflags.append('-INCLUDE:?warp_size@cuda@at@@YAHXZ')
        extra_ldflags.append('torch.lib')
        extra_ldflags.append(f'/LIBPATH:{TORCH_LIB_PATH}')
        if not is_standalone:
            extra_ldflags.append('torch_python.lib')
            extra_ldflags.append(f'/LIBPATH:{python_lib_path}')
    else:
        extra_ldflags.append(f'-L{TORCH_LIB_PATH}')
        extra_ldflags.append('-lc10')
        if with_cuda:
            extra_ldflags.append('-lc10_hip' if IS_HIP_EXTENSION else '-lc10_cuda')
        extra_ldflags.append('-ltorch_cpu')
        if with_cuda:
            extra_ldflags.append('-ltorch_hip' if IS_HIP_EXTENSION else '-ltorch_cuda')
        extra_ldflags.append('-ltorch')
        if not is_standalone:
            extra_ldflags.append('-ltorch_python')
        if is_standalone and 'TBB' in torch.__config__.parallel_info():
            extra_ldflags.append('-ltbb')
        if is_standalone:
            extra_ldflags.append(f'-Wl,-rpath,{TORCH_LIB_PATH}')
    if with_cuda:
        if verbose:
            print('Detected CUDA files, patching ldflags', file=sys.stderr)
        if IS_WINDOWS:
            extra_ldflags.append(f"/LIBPATH:{_join_cuda_home('lib', 'x64')}")
            extra_ldflags.append('cudart.lib')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f"/LIBPATH:{os.path.join(CUDNN_HOME, 'lib', 'x64')}")
        elif not IS_HIP_EXTENSION:
            extra_lib_dir = 'lib64'
            if not os.path.exists(_join_cuda_home(extra_lib_dir)) and os.path.exists(_join_cuda_home('lib')):
                extra_lib_dir = 'lib'
            extra_ldflags.append(f'-L{_join_cuda_home(extra_lib_dir)}')
            extra_ldflags.append('-lcudart')
            if CUDNN_HOME is not None:
                extra_ldflags.append(f"-L{os.path.join(CUDNN_HOME, 'lib64')}")
        elif IS_HIP_EXTENSION:
            assert ROCM_VERSION is not None
            extra_ldflags.append(f"-L{_join_rocm_home('lib')}")
            extra_ldflags.append('-lamdhip64' if ROCM_VERSION >= (3, 5) else '-lhip_hcc')
    return extra_ldflags

def _get_cuda_arch_flags(cflags: Optional[List[str]]=None) -> List[str]:
    if False:
        i = 10
        return i + 15
    '\n    Determine CUDA arch flags to use.\n\n    For an arch, say "6.1", the added compile flag will be\n    ``-gencode=arch=compute_61,code=sm_61``.\n    For an added "+PTX", an additional\n    ``-gencode=arch=compute_xx,code=compute_xx`` is added.\n\n    See select_compute_arch.cmake for corresponding named and supported arches\n    when building with CMake.\n    '
    if cflags is not None:
        for flag in cflags:
            if 'TORCH_EXTENSION_NAME' in flag:
                continue
            if 'arch' in flag:
                return []
    named_arches = collections.OrderedDict([('Kepler+Tesla', '3.7'), ('Kepler', '3.5+PTX'), ('Maxwell+Tegra', '5.3'), ('Maxwell', '5.0;5.2+PTX'), ('Pascal', '6.0;6.1+PTX'), ('Volta+Tegra', '7.2'), ('Volta', '7.0+PTX'), ('Turing', '7.5+PTX'), ('Ampere+Tegra', '8.7'), ('Ampere', '8.0;8.6+PTX'), ('Ada', '8.9+PTX'), ('Hopper', '9.0+PTX')])
    supported_arches = ['3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2', '7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0', '9.0a']
    valid_arch_strings = supported_arches + [s + '+PTX' for s in supported_arches]
    _arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
    if not _arch_list:
        arch_list = []
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            supported_sm = [int(arch.split('_')[1]) for arch in torch.cuda.get_arch_list() if 'sm_' in arch]
            max_supported_sm = max(((sm // 10, sm % 10) for sm in supported_sm))
            capability = min(max_supported_sm, capability)
            arch = f'{capability[0]}.{capability[1]}'
            if arch not in arch_list:
                arch_list.append(arch)
        arch_list = sorted(arch_list)
        arch_list[-1] += '+PTX'
    else:
        _arch_list = _arch_list.replace(' ', ';')
        for (named_arch, archval) in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archval)
        arch_list = _arch_list.split(';')
    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError(f'Unknown CUDA arch ({arch}) or GPU not supported')
        else:
            num = arch[0] + arch[2:].split('+')[0]
            flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
            if arch.endswith('+PTX'):
                flags.append(f'-gencode=arch=compute_{num},code=compute_{num}')
    return sorted(set(flags))

def _get_rocm_arch_flags(cflags: Optional[List[str]]=None) -> List[str]:
    if False:
        while True:
            i = 10
    if cflags is not None:
        for flag in cflags:
            if 'amdgpu-target' in flag or 'offload-arch' in flag:
                return ['-fno-gpu-rdc']
    _archs = os.environ.get('PYTORCH_ROCM_ARCH', None)
    if not _archs:
        archFlags = torch._C._cuda_getArchFlags()
        if archFlags:
            archs = archFlags.split()
        else:
            archs = []
    else:
        archs = _archs.replace(' ', ';').split(';')
    flags = [f'--offload-arch={arch}' for arch in archs]
    flags += ['-fno-gpu-rdc']
    return flags

def _get_build_directory(name: str, verbose: bool) -> str:
    if False:
        for i in range(10):
            print('nop')
    root_extensions_directory = os.environ.get('TORCH_EXTENSIONS_DIR')
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        cu_str = 'cpu' if torch.version.cuda is None else f"cu{torch.version.cuda.replace('.', '')}"
        python_version = f'py{sys.version_info.major}{sys.version_info.minor}'
        build_folder = f'{python_version}_{cu_str}'
        root_extensions_directory = os.path.join(root_extensions_directory, build_folder)
    if verbose:
        print(f'Using {root_extensions_directory} as PyTorch extensions root...', file=sys.stderr)
    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            print(f'Creating extension directory {build_directory}...', file=sys.stderr)
        os.makedirs(build_directory, exist_ok=True)
    return build_directory

def _get_num_workers(verbose: bool) -> Optional[int]:
    if False:
        while True:
            i = 10
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            print(f'Using envvar MAX_JOBS ({max_jobs}) as the number of workers...', file=sys.stderr)
        return int(max_jobs)
    if verbose:
        print('Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)', file=sys.stderr)
    return None

def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    if False:
        i = 10
        return i + 15
    command = ['ninja', '-v']
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(['-j', str(num_workers)])
    env = os.environ.copy()
    if IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
        from setuptools import distutils
        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]
        vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
        vc_env = {k.upper(): v for (k, v) in vc_env.items()}
        for (k, v) in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fileno = 1
        subprocess.run(command, stdout=stdout_fileno if verbose else subprocess.PIPE, stderr=subprocess.STDOUT, cwd=build_directory, check=True, env=env)
    except subprocess.CalledProcessError as e:
        (_, error, _) = sys.exc_info()
        message = error_prefix
        if hasattr(error, 'output') and error.output:
            message += f': {error.output.decode(*SUBPROCESS_DECODE_ARGS)}'
        raise RuntimeError(message) from e

def _get_exec_path(module_name, path):
    if False:
        for i in range(10):
            print('nop')
    if IS_WINDOWS and TORCH_LIB_PATH not in os.getenv('PATH', '').split(';'):
        torch_lib_in_path = any((os.path.exists(p) and os.path.samefile(p, TORCH_LIB_PATH) for p in os.getenv('PATH', '').split(';')))
        if not torch_lib_in_path:
            os.environ['PATH'] = f"{TORCH_LIB_PATH};{os.getenv('PATH', '')}"
    return os.path.join(path, f'{module_name}{EXEC_EXT}')

def _import_module_from_library(module_name, path, is_python_module):
    if False:
        return 10
    filepath = os.path.join(path, f'{module_name}{LIB_EXT}')
    if is_python_module:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert isinstance(spec.loader, importlib.abc.Loader)
        spec.loader.exec_module(module)
        return module
    else:
        torch.ops.load_library(filepath)

def _write_ninja_file_to_build_library(path, name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, extra_include_paths, with_cuda, is_standalone) -> None:
    if False:
        return 10
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_cuda_cflags = [flag.strip() for flag in extra_cuda_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]
    user_includes = [os.path.abspath(file) for file in extra_include_paths]
    system_includes = include_paths(with_cuda)
    python_include_path = sysconfig.get_path('include', scheme='nt' if IS_WINDOWS else 'posix_prefix')
    if python_include_path is not None:
        system_includes.append(python_include_path)
    if IS_WINDOWS:
        user_includes += system_includes
        system_includes.clear()
    common_cflags = []
    if not is_standalone:
        common_cflags.append(f'-DTORCH_EXTENSION_NAME={name}')
        common_cflags.append('-DTORCH_API_INCLUDE_EXTENSION_H')
    common_cflags += [f'{x}' for x in _get_pybind11_abi_build_flags()]
    common_cflags += [f'-I{include}' for include in user_includes]
    common_cflags += [f'-isystem {include}' for include in system_includes]
    common_cflags += [f'{x}' for x in _get_glibcxx_abi_build_flags()]
    if IS_WINDOWS:
        cflags = common_cflags + COMMON_MSVC_FLAGS + ['/std:c++17'] + extra_cflags
        cflags = _nt_quote_args(cflags)
    else:
        cflags = common_cflags + ['-fPIC', '-std=c++17'] + extra_cflags
    if with_cuda and IS_HIP_EXTENSION:
        cuda_flags = ['-DWITH_HIP'] + cflags + COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS
        cuda_flags += extra_cuda_cflags
        cuda_flags += _get_rocm_arch_flags(cuda_flags)
    elif with_cuda:
        cuda_flags = common_cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags()
        if IS_WINDOWS:
            for flag in COMMON_MSVC_FLAGS:
                cuda_flags = ['-Xcompiler', flag] + cuda_flags
            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                cuda_flags = ['-Xcudafe', '--diag_suppress=' + ignore_warning] + cuda_flags
            cuda_flags = cuda_flags + ['-std=c++17']
            cuda_flags = _nt_quote_args(cuda_flags)
            cuda_flags += _nt_quote_args(extra_cuda_cflags)
        else:
            cuda_flags += ['--compiler-options', "'-fPIC'"]
            cuda_flags += extra_cuda_cflags
            if not any((flag.startswith('-std=') for flag in cuda_flags)):
                cuda_flags.append('-std=c++17')
            cc_env = os.getenv('CC')
            if cc_env is not None:
                cuda_flags = ['-ccbin', cc_env] + cuda_flags
    else:
        cuda_flags = None

    def object_file_path(source_file: str) -> str:
        if False:
            return 10
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_cuda_file(source_file) and with_cuda:
            target = f'{file_name}.cuda.o'
        else:
            target = f'{file_name}.o'
        return target
    objects = [object_file_path(src) for src in sources]
    ldflags = ([] if is_standalone else [SHARED_FLAG]) + extra_ldflags
    if IS_MACOS:
        ldflags.append('-undefined dynamic_lookup')
    elif IS_WINDOWS:
        ldflags = _nt_quote_args(ldflags)
    ext = EXEC_EXT if is_standalone else LIB_EXT
    library_target = f'{name}{ext}'
    _write_ninja_file(path=path, cflags=cflags, post_cflags=None, cuda_cflags=cuda_flags, cuda_post_cflags=None, cuda_dlink_post_cflags=None, sources=sources, objects=objects, ldflags=ldflags, library_target=library_target, with_cuda=with_cuda)

def _write_ninja_file(path, cflags, post_cflags, cuda_cflags, cuda_post_cflags, cuda_dlink_post_cflags, sources, objects, ldflags, library_target, with_cuda) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Write a ninja file that does the desired compiling and linking.\n\n    `path`: Where to write this file\n    `cflags`: list of flags to pass to $cxx. Can be None.\n    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.\n    `cuda_cflags`: list of flags to pass to $nvcc. Can be None.\n    `cuda_postflags`: list of flags to append to the $nvcc invocation. Can be None.\n    `sources`: list of paths to source files\n    `objects`: list of desired paths to objects, one per source.\n    `ldflags`: list of flags to pass to linker. Can be None.\n    `library_target`: Name of the output library. Can be None; in that case,\n                      we do no linking.\n    `with_cuda`: If we should be compiling with CUDA.\n    '

    def sanitize_flags(flags):
        if False:
            while True:
                i = 10
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]
    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)
    assert len(sources) == len(objects)
    assert len(sources) > 0
    compiler = get_cxx_compiler()
    config = ['ninja_required_version = 1.3']
    config.append(f'cxx = {compiler}')
    if with_cuda or cuda_dlink_post_cflags:
        if 'PYTORCH_NVCC' in os.environ:
            nvcc = os.getenv('PYTORCH_NVCC')
        elif IS_HIP_EXTENSION:
            nvcc = _join_rocm_home('bin', 'hipcc')
        else:
            nvcc = _join_cuda_home('bin', 'nvcc')
        config.append(f'nvcc = {nvcc}')
    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f"cflags = {' '.join(cflags)}"]
    flags.append(f"post_cflags = {' '.join(post_cflags)}")
    if with_cuda:
        flags.append(f"cuda_cflags = {' '.join(cuda_cflags)}")
        flags.append(f"cuda_post_cflags = {' '.join(cuda_post_cflags)}")
    flags.append(f"cuda_dlink_post_cflags = {' '.join(cuda_dlink_post_cflags)}")
    flags.append(f"ldflags = {' '.join(ldflags)}")
    sources = [os.path.abspath(file) for file in sources]
    compile_rule = ['rule compile']
    if IS_WINDOWS:
        compile_rule.append('  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags')
        compile_rule.append('  deps = msvc')
    else:
        compile_rule.append('  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags')
        compile_rule.append('  depfile = $out.d')
        compile_rule.append('  deps = gcc')
    if with_cuda:
        cuda_compile_rule = ['rule cuda_compile']
        nvcc_gendeps = ''
        if torch.version.cuda is not None:
            cuda_compile_rule.append('  depfile = $out.d')
            cuda_compile_rule.append('  deps = gcc')
            nvcc_gendeps = '--generate-dependencies-with-compile --dependency-output $out.d'
        cuda_compile_rule.append(f'  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags')
    build = []
    for (source_file, object_file) in zip(sources, objects):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        rule = 'cuda_compile' if is_cuda_source else 'compile'
        if IS_WINDOWS:
            source_file = source_file.replace(':', '$:')
            object_file = object_file.replace(':', '$:')
        source_file = source_file.replace(' ', '$ ')
        object_file = object_file.replace(' ', '$ ')
        build.append(f'build {object_file}: {rule} {source_file}')
    if cuda_dlink_post_cflags:
        devlink_out = os.path.join(os.path.dirname(objects[0]), 'dlink.o')
        devlink_rule = ['rule cuda_devlink']
        devlink_rule.append('  command = $nvcc $in -o $out $cuda_dlink_post_cflags')
        devlink = [f"build {devlink_out}: cuda_devlink {' '.join(objects)}"]
        objects += [devlink_out]
    else:
        (devlink_rule, devlink) = ([], [])
    if library_target is not None:
        link_rule = ['rule link']
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(['where', 'cl']).decode(*SUBPROCESS_DECODE_ARGS).split('\r\n')
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(':', '$:')
            else:
                raise RuntimeError('MSVC is required to load C++ extensions')
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append('  command = $cxx $in $ldflags -o $out')
        link = [f"build {library_target}: link {' '.join(objects)}"]
        default = [f'default {library_target}']
    else:
        (link_rule, link, default) = ([], [], [])
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = '\n\n'.join(('\n'.join(b) for b in blocks))
    content += '\n'
    _maybe_write(path, content)

def _join_cuda_home(*paths) -> str:
    if False:
        return 10
    '\n    Join paths with CUDA_HOME, or raises an error if it CUDA_HOME is not set.\n\n    This is basically a lazy way of raising an error for missing $CUDA_HOME\n    only once we need to get any CUDA-specific path.\n    '
    if CUDA_HOME is None:
        raise OSError('CUDA_HOME environment variable is not set. Please set it to your CUDA install root.')
    return os.path.join(CUDA_HOME, *paths)

def _is_cuda_file(path: str) -> bool:
    if False:
        print('Hello World!')
    valid_ext = ['.cu', '.cuh']
    if IS_HIP_EXTENSION:
        valid_ext.append('.hip')
    return os.path.splitext(path)[1] in valid_ext