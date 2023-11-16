"""
This file must not depend on any other CuPy modules.
"""
import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
_cuda_path = ''
_nvcc_path = ''
_rocm_path = ''
_hipcc_path = ''
_cub_path = ''
"\nLibrary Preloading\n------------------\n\nWheel packages are built against specific versions of CUDA libraries\n(cuTENSOR/NCCL/cuDNN).\nTo avoid loading wrong version, these shared libraries are manually\npreloaded.\n\n# TODO(kmaehashi): Support NCCL\n\nExample of `_preload_config` is as follows:\n\n{\n    # installation source\n    'packaging': 'pip',\n\n    # CUDA version string\n    'cuda': '11.0',\n\n    'cudnn': {\n        # cuDNN version string\n        'version': '8.0.0',\n\n        # names of the shared library\n        'filenames': ['libcudnn.so.X.Y.Z']  # or `cudnn64_X.dll` for Windows\n    }\n}\n\nThe configuration file is intended solely for internal purposes and\nnot expected to be parsed by end-users.\n"
_preload_config = None
_preload_libs = {'cudnn': None, 'nccl': None, 'cutensor': None}
_debug = os.environ.get('CUPY_DEBUG_LIBRARY_LOAD', '0') == '1'

def _log(msg: str) -> None:
    if False:
        return 10
    if _debug:
        sys.stderr.write(f'[CUPY_DEBUG_LIBRARY_LOAD] {msg}\n')
        sys.stderr.flush()

def get_cuda_path():
    if False:
        i = 10
        return i + 15
    global _cuda_path
    if _cuda_path == '':
        _cuda_path = _get_cuda_path()
    return _cuda_path

def get_nvcc_path():
    if False:
        while True:
            i = 10
    global _nvcc_path
    if _nvcc_path == '':
        _nvcc_path = _get_nvcc_path()
    return _nvcc_path

def get_rocm_path():
    if False:
        for i in range(10):
            print('nop')
    global _rocm_path
    if _rocm_path == '':
        _rocm_path = _get_rocm_path()
    return _rocm_path

def get_hipcc_path():
    if False:
        return 10
    global _hipcc_path
    if _hipcc_path == '':
        _hipcc_path = _get_hipcc_path()
    return _hipcc_path

def get_cub_path():
    if False:
        print('Hello World!')
    global _cub_path
    if _cub_path == '':
        _cub_path = _get_cub_path()
    return _cub_path

def _get_cuda_path():
    if False:
        i = 10
        return i + 15
    cuda_path = os.environ.get('CUDA_PATH', '')
    if os.path.exists(cuda_path):
        return cuda_path
    nvcc_path = shutil.which('nvcc')
    if nvcc_path is not None:
        return os.path.dirname(os.path.dirname(nvcc_path))
    if os.path.exists('/usr/local/cuda'):
        return '/usr/local/cuda'
    return None

def _get_nvcc_path():
    if False:
        i = 10
        return i + 15
    nvcc_path = os.environ.get('NVCC', None)
    if nvcc_path is not None:
        return nvcc_path
    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None
    return shutil.which('nvcc', path=os.path.join(cuda_path, 'bin'))

def _get_rocm_path():
    if False:
        while True:
            i = 10
    rocm_path = os.environ.get('ROCM_HOME', '')
    if os.path.exists(rocm_path):
        return rocm_path
    hipcc_path = shutil.which('hipcc')
    if hipcc_path is not None:
        return os.path.dirname(os.path.dirname(hipcc_path))
    if os.path.exists('/opt/rocm'):
        return '/opt/rocm'
    return None

def _get_hipcc_path():
    if False:
        for i in range(10):
            print('nop')
    rocm_path = get_rocm_path()
    if rocm_path is None:
        return None
    return shutil.which('hipcc', path=os.path.join(rocm_path, 'bin'))

def _get_cub_path():
    if False:
        for i in range(10):
            print('nop')
    from cupy_backends.cuda.api import runtime
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not runtime.is_hip:
        if os.path.isdir(os.path.join(current_dir, '_core/include/cupy/_cccl/cub')):
            _cub_path = '<bundle>'
        else:
            _cub_path = None
    else:
        rocm_path = get_rocm_path()
        if rocm_path is not None and os.path.isdir(os.path.join(rocm_path, 'include/hipcub')):
            _cub_path = '<ROCm>'
        else:
            _cub_path = None
    return _cub_path

def _setup_win32_dll_directory():
    if False:
        return 10
    if sys.platform.startswith('win32'):
        config = get_preload_config()
        is_conda = config is not None and config['packaging'] == 'conda'
        cuda_path = get_cuda_path()
        if cuda_path is not None:
            if is_conda:
                cuda_bin_path = cuda_path
            else:
                cuda_bin_path = os.path.join(cuda_path, 'bin')
        else:
            cuda_bin_path = None
            warnings.warn('CUDA path could not be detected. Set CUDA_PATH environment variable if CuPy fails to load.')
        _log('CUDA_PATH: {}'.format(cuda_path))
        wheel_libdir = os.path.join(get_cupy_install_path(), 'cupy', '.data', 'lib')
        if os.path.isdir(wheel_libdir):
            _log('Wheel shared libraries: {}'.format(wheel_libdir))
        else:
            _log('Not wheel distribution ({} not found)'.format(wheel_libdir))
            wheel_libdir = None
        if (3, 8) <= sys.version_info:
            if cuda_bin_path is not None:
                _log('Adding DLL search path: {}'.format(cuda_bin_path))
                os.add_dll_directory(cuda_bin_path)
            if wheel_libdir is not None:
                _log('Adding DLL search path: {}'.format(wheel_libdir))
                os.add_dll_directory(wheel_libdir)
        elif wheel_libdir is not None:
            _log('Adding to PATH: {}'.format(wheel_libdir))
            path = os.environ.get('PATH', '')
            os.environ['PATH'] = wheel_libdir + os.pathsep + path

def get_cupy_install_path():
    if False:
        while True:
            i = 10
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_cupy_cuda_lib_path():
    if False:
        while True:
            i = 10
    'Returns the directory where CUDA external libraries are installed.\n\n    This environment variable only affects wheel installations.\n\n    Shared libraries are looked up from\n    `$CUPY_CUDA_LIB_PATH/$CUDA_VER/$LIB_NAME/$LIB_VER/{lib,lib64,bin}`,\n    e.g., `~/.cupy/cuda_lib/11.2/cudnn/8.1.1/lib64/libcudnn.so.8.1.1`.\n\n    The default $CUPY_CUDA_LIB_PATH is `~/.cupy/cuda_lib`.\n    '
    cupy_cuda_lib_path = os.environ.get('CUPY_CUDA_LIB_PATH', None)
    if cupy_cuda_lib_path is None:
        return os.path.expanduser('~/.cupy/cuda_lib')
    return os.path.abspath(cupy_cuda_lib_path)

def get_preload_config() -> Optional[Dict[str, Any]]:
    if False:
        return 10
    global _preload_config
    if _preload_config is None:
        _preload_config = _get_json_data('_wheel.json')
    return _preload_config

def _get_json_data(name: str) -> Optional[Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    config_path = os.path.join(get_cupy_install_path(), 'cupy', '.data', name)
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        return json.load(f)

def _can_attempt_preload(lib: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Returns if the preload can be attempted.'
    config = get_preload_config()
    if config is None or config['packaging'] == 'conda':
        _log(f'Not preloading {lib} as this is not a pip wheel installation')
        return False
    if lib not in _preload_libs:
        raise AssertionError(f'Unknown preload library: {lib}')
    if lib not in config:
        _log(f'Preload {lib} not configured in wheel')
        return False
    if _preload_libs[lib] is not None:
        _log(f'Preload already attempted: {lib}')
        return False
    return True

def _preload_library(lib):
    if False:
        for i in range(10):
            print('nop')
    'Preload dependent shared libraries.\n\n    The preload configuration file (cupy/.data/_wheel.json) will be added\n    during the wheel build process.\n    '
    _log(f'Preloading triggered for library: {lib}')
    if not _can_attempt_preload(lib):
        return
    _preload_libs[lib] = {}
    config = get_preload_config()
    cuda_version = config['cuda']
    _log('CuPy wheel package built for CUDA {}'.format(cuda_version))
    cupy_cuda_lib_path = get_cupy_cuda_lib_path()
    _log('CuPy CUDA library directory: {}'.format(cupy_cuda_lib_path))
    version = config[lib]['version']
    filenames = config[lib]['filenames']
    for filename in filenames:
        _log(f'Looking for {lib} version {version} ({filename})')
        libpath_cands = [os.path.join(cupy_cuda_lib_path, config['cuda'], lib, version, x, filename) for x in ['lib', 'lib64', 'bin']]
        for libpath in libpath_cands:
            if not os.path.exists(libpath):
                _log('Rejected candidate (not found): {}'.format(libpath))
                continue
            try:
                _log(f'Trying to load {libpath}')
                _preload_libs[lib][libpath] = ctypes.CDLL(libpath)
                _log('Loaded')
                break
            except Exception as e:
                e_type = type(e).__name__
                msg = f'CuPy failed to preload library ({libpath}): {e_type} ({e})'
                _log(msg)
                warnings.warn(msg)
        else:
            _log('File {} could not be found'.format(filename))
            _log(f'Trying to load {filename} from default search path')
            try:
                _preload_libs[lib][filename] = ctypes.CDLL(filename)
                _log('Loaded')
            except Exception as e:
                _log(f'Library {lib} could not be preloaded: {e}')

def _preload_warning(lib, exc):
    if False:
        return 10
    config = get_preload_config()
    if config is not None and lib in config:
        msg = '\n{lib} library could not be loaded.\n\nReason: {exc_type} ({exc})\n\nYou can install the library by:\n'
        if config['packaging'] == 'pip':
            msg += '\n  $ python -m cupyx.tools.install_library --library {lib} --cuda {cuda}\n'
        elif config['packaging'] == 'conda':
            msg += '\n  $ conda install -c conda-forge {lib}\n'
        else:
            raise AssertionError
        msg = msg.format(lib=lib, exc_type=type(exc).__name__, exc=str(exc), cuda=config['cuda'])
        warnings.warn(msg)

def _detect_duplicate_installation():
    if False:
        for i in range(10):
            print('nop')
    if sys.version_info < (3, 8):
        return
    import importlib.metadata
    known = ['cupy', 'cupy-cuda80', 'cupy-cuda90', 'cupy-cuda91', 'cupy-cuda92', 'cupy-cuda100', 'cupy-cuda101', 'cupy-cuda102', 'cupy-cuda110', 'cupy-cuda111', 'cupy-cuda112', 'cupy-cuda113', 'cupy-cuda114', 'cupy-cuda115', 'cupy-cuda116', 'cupy-cuda117', 'cupy-cuda118', 'cupy-cuda11x', 'cupy-cuda12x', 'cupy-rocm-4-0', 'cupy-rocm-4-1', 'cupy-rocm-4-2', 'cupy-rocm-4-3', 'cupy-rocm-5-0']
    cupy_installed = [name for name in known if list(importlib.metadata.distributions(name=name))]
    if 1 < len(cupy_installed):
        cupy_packages_list = ', '.join(sorted(cupy_installed))
        warnings.warn(f'\n--------------------------------------------------------------------------------\n\n  CuPy may not function correctly because multiple CuPy packages are installed\n  in your environment:\n\n    {cupy_packages_list}\n\n  Follow these steps to resolve this issue:\n\n    1. For all packages listed above, run the following command to remove all\n       existing CuPy installations:\n\n         $ pip uninstall <package_name>\n\n      If you previously installed CuPy via conda, also run the following:\n\n         $ conda uninstall cupy\n\n    2. Install the appropriate CuPy package.\n       Refer to the Installation Guide for detailed instructions.\n\n         https://docs.cupy.dev/en/stable/install.html\n\n--------------------------------------------------------------------------------\n')

def _diagnose_import_error() -> str:
    if False:
        print('Hello World!')
    msg = 'Failed to import CuPy.\n\nIf you installed CuPy via wheels (cupy-cudaXXX or cupy-rocm-X-X), make sure that the package matches with the version of CUDA or ROCm installed.\n\nOn Linux, you may need to set LD_LIBRARY_PATH environment variable depending on how you installed CUDA/ROCm.\nOn Windows, try setting CUDA_PATH environment variable.\n\nCheck the Installation Guide for details:\n  https://docs.cupy.dev/en/latest/install.html'
    if sys.platform == 'win32':
        try:
            msg += _diagnose_win32_dll_load()
        except Exception as e:
            msg += f'\n\nThe cause could not be identified: {type(e).__name__}: {e}'
    return msg

def _diagnose_win32_dll_load() -> str:
    if False:
        print('Hello World!')
    depends = _get_json_data('_depends.json')
    if depends is None:
        return ''
    from ctypes import wintypes
    kernel32 = ctypes.CDLL('kernel32')
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    lines = ['', '', f'CUDA Path: {get_cuda_path()}', 'DLL dependencies:']
    filepath = ctypes.create_unicode_buffer(2 ** 15)
    for name in depends['depends']:
        try:
            dll = ctypes.CDLL(name)
            kernel32.GetModuleFileNameW(dll._handle, filepath, len(filepath))
            lines.append(f'  {name} -> {filepath.value}')
        except FileNotFoundError:
            lines.append(f'  {name} -> not found')
        except Exception as e:
            lines.append(f'  {name} -> error ({type(e).__name__}: {e})')
    return '\n'.join(lines)