import contextlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import List, Set
import cupy_builder
import cupy_builder.install_utils as utils
from cupy_builder import _environment
from cupy_builder._context import Context
PLATFORM_LINUX = sys.platform.startswith('linux')
PLATFORM_WIN32 = sys.platform.startswith('win32')
minimum_cudnn_version = 7600
minimum_hip_version = 305
_cuda_path = 'NOT_INITIALIZED'
_rocm_path = 'NOT_INITIALIZED'
_compiler_base_options = None

@contextlib.contextmanager
def _tempdir():
    if False:
        return 10
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def get_rocm_path():
    if False:
        return 10
    global _rocm_path
    if _rocm_path != 'NOT_INITIALIZED':
        return _rocm_path
    _rocm_path = os.environ.get('ROCM_HOME', '')
    return _rocm_path

def get_cuda_path():
    if False:
        print('Hello World!')
    global _cuda_path
    if _cuda_path != 'NOT_INITIALIZED':
        return _cuda_path
    nvcc_path = utils.search_on_path(('nvcc', 'nvcc.exe'))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(os.path.join(os.path.dirname(nvcc_path), '..'))
    cuda_path = os.environ.get('CUDA_PATH', '')
    if len(cuda_path) > 0 and cuda_path != cuda_path_default:
        utils.print_warning('nvcc path != CUDA_PATH', 'nvcc path: %s' % cuda_path_default, 'CUDA_PATH: %s' % cuda_path)
    if os.path.exists(cuda_path):
        _cuda_path = cuda_path
    elif cuda_path_default is not None:
        _cuda_path = cuda_path_default
    elif os.path.exists('/usr/local/cuda'):
        _cuda_path = '/usr/local/cuda'
    else:
        _cuda_path = None
    return _cuda_path

def get_nvcc_path() -> List[str]:
    if False:
        i = 10
        return i + 15
    nvcc = os.environ.get('NVCC', None)
    if nvcc:
        return shlex.split(nvcc)
    cuda_path = get_cuda_path()
    if cuda_path is None:
        return None
    if PLATFORM_WIN32:
        nvcc_bin = 'bin/nvcc.exe'
    else:
        nvcc_bin = 'bin/nvcc'
    nvcc_path = os.path.join(cuda_path, nvcc_bin)
    if os.path.exists(nvcc_path):
        return [nvcc_path]
    else:
        return None

def get_hipcc_path() -> List[str]:
    if False:
        while True:
            i = 10
    hipcc = os.environ.get('HIPCC', None)
    if hipcc:
        return shlex.split(hipcc)
    rocm_path = get_rocm_path()
    if rocm_path is None:
        return None
    if PLATFORM_WIN32:
        hipcc_bin = 'bin/hipcc.exe'
    else:
        hipcc_bin = 'bin/hipcc'
    hipcc_path = os.path.join(rocm_path, hipcc_bin)
    if os.path.exists(hipcc_path):
        return [hipcc_path]
    else:
        return None

def get_compiler_setting(ctx: Context, use_hip):
    if False:
        while True:
            i = 10
    cuda_path = None
    rocm_path = None
    if use_hip:
        rocm_path = get_rocm_path()
    else:
        cuda_path = get_cuda_path()
    include_dirs = ctx.include_dirs.copy()
    library_dirs = ctx.library_dirs.copy()
    define_macros = []
    extra_compile_args = []
    if cuda_path:
        include_dirs.append(os.path.join(cuda_path, 'include'))
        if PLATFORM_WIN32:
            library_dirs.append(os.path.join(cuda_path, 'bin'))
            library_dirs.append(os.path.join(cuda_path, 'lib', 'x64'))
        else:
            library_dirs.append(os.path.join(cuda_path, 'lib64'))
            library_dirs.append(os.path.join(cuda_path, 'lib'))
    if rocm_path:
        include_dirs.append(os.path.join(rocm_path, 'include'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'hip'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'rocrand'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'hiprand'))
        include_dirs.append(os.path.join(rocm_path, 'include', 'roctracer'))
        library_dirs.append(os.path.join(rocm_path, 'lib'))
    if use_hip:
        extra_compile_args.append('-std=c++11')
    if PLATFORM_WIN32:
        nvtx_path = _environment.get_nvtx_path()
        if nvtx_path is not None and os.path.exists(nvtx_path):
            include_dirs.append(os.path.join(nvtx_path, 'include'))
        else:
            define_macros.append(('CUPY_NO_NVTX', '1'))
    cupy_header = os.path.join(cupy_builder.get_context().source_root, 'cupy/_core/include')
    global _jitify_path
    _jitify_path = os.path.join(cupy_header, 'cupy/_jitify')
    global _cub_path
    if rocm_path:
        _cub_path = os.path.join(rocm_path, 'include', 'hipcub')
        if not os.path.exists(_cub_path):
            raise Exception('Please install hipCUB and retry')
        _thrust_path = None
        _libcudacxx_path = None
    else:
        _cub_path = os.path.join(cupy_header, 'cupy/_cccl/cub')
        _thrust_path = os.path.join(cupy_header, 'cupy/_cccl/thrust')
        _libcudacxx_path = os.path.join(cupy_header, 'cupy/_cccl/libcudacxx')
    include_dirs.insert(0, cupy_header)
    include_dirs.insert(0, _cub_path)
    if _thrust_path and _libcudacxx_path:
        include_dirs.insert(0, _thrust_path)
        include_dirs.insert(0, _libcudacxx_path)
    return {'include_dirs': include_dirs, 'library_dirs': library_dirs, 'define_macros': define_macros, 'language': 'c++', 'extra_compile_args': extra_compile_args}

def _match_output_lines(output_lines, regexs):
    if False:
        return 10
    if len(output_lines) < len(regexs):
        return None
    matches = [None] * len(regexs)
    for i in range(len(output_lines) - len(regexs)):
        for j in range(len(regexs)):
            m = re.match(regexs[j], output_lines[i + j])
            if not m:
                break
            matches[j] = m
        else:
            return matches
    return None

def get_compiler_base_options(compiler_path: List[str]) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Returns base options for nvcc compiler.\n\n    '
    global _compiler_base_options
    if _compiler_base_options is None:
        _compiler_base_options = _get_compiler_base_options(compiler_path)
    return _compiler_base_options

def _get_compiler_base_options(compiler_path):
    if False:
        while True:
            i = 10
    with _tempdir() as temp_dir:
        test_cu_path = os.path.join(temp_dir, 'test.cu')
        test_out_path = os.path.join(temp_dir, 'test.out')
        with open(test_cu_path, 'w') as f:
            f.write('int main() { return 0; }')
        proc = subprocess.Popen(compiler_path + ['-o', test_out_path, test_cu_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdoutdata, stderrdata) = proc.communicate()
        stderrlines = stderrdata.split(b'\n')
        if proc.returncode != 0:
            matches = _match_output_lines(stderrlines, [b'^ERROR: No supported gcc/g\\+\\+ host compiler found, but .* is available.$', b"^ *Use 'nvcc (.*)' to use that instead.$"])
            if matches is not None:
                base_opts = matches[1].group(1)
                base_opts = base_opts.decode('utf8').split(' ')
                return base_opts
            raise RuntimeError('Encountered unknown error while testing nvcc:\n' + stderrdata.decode('utf8'))
    return []
_hip_version = None
_thrust_version = None
_cudnn_version = None
_nccl_version = None
_cutensor_version = None
_cub_path = None
_cub_version = None
_jitify_path = None
_jitify_version = None
_compute_capabilities = None
_cusparselt_version = None

def check_hip_version(compiler, settings):
    if False:
        i = 10
        return i + 15
    global _hip_version
    try:
        out = build_and_run(compiler, '\n        #include <hip/hip_version.h>\n        #include <stdio.h>\n        int main() {\n          printf("%d", HIP_VERSION);\n          return 0;\n        }\n        ', include_dirs=settings['include_dirs'])
    except Exception as e:
        utils.print_warning('Cannot check HIP version', str(e))
        return False
    _hip_version = int(out)
    if _hip_version < minimum_hip_version:
        utils.print_warning('ROCm/HIP version is too old: %d' % _hip_version, 'ROCm 3.5.0 or newer is required')
        return False
    return True

def get_hip_version(formatted: bool=False) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Return ROCm version cached in check_hip_version().'
    global _hip_version
    if _hip_version is None:
        msg = 'check_hip_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return str(_hip_version)
    return _hip_version

def check_compute_capabilities(compiler, settings):
    if False:
        while True:
            i = 10
    'Return compute capabilities of the installed devices.'
    global _compute_capabilities
    try:
        src = '\n        #include <cuda_runtime_api.h>\n        #include <stdio.h>\n        #define CHECK_CUDART(x) { if ((x) != cudaSuccess) return 1; }\n\n        int main() {\n          int device_count;\n          CHECK_CUDART(cudaGetDeviceCount(&device_count));\n          for (int i = 0; i < device_count; i++) {\n              cudaDeviceProp prop;\n              CHECK_CUDART(cudaGetDeviceProperties(&prop, i));\n              printf("%d%d ", prop.major, prop.minor);\n          }\n          return 0;\n        }\n        '
        out = build_and_run(compiler, src, include_dirs=settings['include_dirs'], libraries=('cudart',), library_dirs=settings['library_dirs'])
        _compute_capabilities = set([int(o) for o in out.split()])
    except Exception as e:
        utils.print_warning('Cannot check compute capability\n{0}'.format(e))
        return False
    return True

def get_compute_capabilities(formatted: bool=False) -> Set[int]:
    if False:
        return 10
    return _compute_capabilities

def check_thrust_version(compiler, settings):
    if False:
        print('Hello World!')
    global _thrust_version
    try:
        out = build_and_run(compiler, '\n        #include <thrust/version.h>\n        #include <stdio.h>\n\n        int main() {\n          printf("%d", THRUST_VERSION);\n          return 0;\n        }\n        ', include_dirs=settings['include_dirs'])
    except Exception as e:
        utils.print_warning('Cannot check Thrust version\n{0}'.format(e))
        return False
    _thrust_version = int(out)
    return True

def get_thrust_version(formatted=False):
    if False:
        i = 10
        return i + 15
    'Return Thrust version cached in check_thrust_version().'
    global _thrust_version
    if _thrust_version is None:
        msg = 'check_thrust_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return str(_thrust_version)
    return _thrust_version

def check_cudnn_version(compiler, settings):
    if False:
        i = 10
        return i + 15
    global _cudnn_version
    try:
        out = build_and_run(compiler, '\n        #include <cudnn.h>\n        #include <stdio.h>\n        int main() {\n          printf("%d", CUDNN_VERSION);\n          return 0;\n        }\n        ', include_dirs=settings['include_dirs'])
    except Exception as e:
        utils.print_warning('Cannot check cuDNN version\n{0}'.format(e))
        return False
    _cudnn_version = int(out)
    if not minimum_cudnn_version <= _cudnn_version:
        min_major = str(minimum_cudnn_version)
        utils.print_warning('Unsupported cuDNN version: {}'.format(str(_cudnn_version)), 'cuDNN >=v{} is required'.format(min_major))
        return False
    return True

def get_cudnn_version(formatted=False):
    if False:
        return 10
    'Return cuDNN version cached in check_cudnn_version().'
    global _cudnn_version
    if _cudnn_version is None:
        msg = 'check_cudnn_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        return str(_cudnn_version)
    return _cudnn_version

def check_nccl_version(compiler, settings):
    if False:
        print('Hello World!')
    global _nccl_version
    try:
        out = build_and_run(compiler, '\n                            #ifndef CUPY_USE_HIP\n                            #include <nccl.h>\n                            #else\n                            #include <rccl.h>\n                            #endif\n                            #include <stdio.h>\n                            #ifdef NCCL_MAJOR\n                            #ifndef NCCL_VERSION_CODE\n                            #  define NCCL_VERSION_CODE                             (NCCL_MAJOR * 1000 + NCCL_MINOR * 100 + NCCL_PATCH)\n                            #endif\n                            #else\n                            #  define NCCL_VERSION_CODE 0\n                            #endif\n                            int main() {\n                              printf("%d", NCCL_VERSION_CODE);\n                              return 0;\n                            }\n                            ', include_dirs=settings['include_dirs'], define_macros=settings['define_macros'])
    except Exception as e:
        utils.print_warning('Cannot include NCCL\n{0}'.format(e))
        return False
    _nccl_version = int(out)
    return True

def get_nccl_version(formatted=False):
    if False:
        return 10
    'Return NCCL version cached in check_nccl_version().'
    global _nccl_version
    if _nccl_version is None:
        msg = 'check_nccl_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        if _nccl_version == 0:
            return '1.x'
        return str(_nccl_version)
    return _nccl_version

def check_nvtx(compiler, settings):
    if False:
        while True:
            i = 10
    if PLATFORM_WIN32:
        if _environment.get_nvtx_path() is None:
            utils.print_warning('NVTX unavailable')
            return False
    return True

def check_cub_version(compiler, settings):
    if False:
        return 10
    global _cub_version
    global _cub_path
    try:
        out = build_and_run(compiler, '\n                            #ifndef CUPY_USE_HIP\n                            #include <cub/version.cuh>\n                            #else\n                            #include <hipcub/hipcub_version.hpp>\n                            #endif\n                            #include <stdio.h>\n\n                            int main() {\n                              #ifndef CUPY_USE_HIP\n                              printf("%d", CUB_VERSION);\n                              #else\n                              printf("%d", HIPCUB_VERSION);\n                              #endif\n                              return 0;\n                            }', include_dirs=settings['include_dirs'], define_macros=settings['define_macros'])
    except Exception as e:
        try:
            cupy_cub_include = _cub_path
            a = subprocess.run(' '.join(['git', 'describe', '--tags']), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=cupy_cub_include)
            if a.returncode == 0:
                tag = a.stdout.decode()[:-1]
                if tag.startswith('v'):
                    tag = tag[1:]
                tag = tag.split('.')
                out = int(tag[0]) * 100000 + int(tag[1]) * 100
                try:
                    out += int(tag[2])
                except ValueError:
                    local_patch = tag[2].split('-')
                    out += int(local_patch[0]) + int(local_patch[1])
            else:
                raise RuntimeError('Cannot determine CUB version from git tag\n{0}'.format(e))
        except Exception as e:
            utils.print_warning('Cannot determine CUB version\n{0}'.format(e))
            out = -1
    _cub_version = int(out)
    settings['define_macros'].append(('CUPY_CUB_VERSION_CODE', _cub_version))
    return True

def get_cub_version(formatted=False):
    if False:
        while True:
            i = 10
    'Return CUB version cached in check_cub_version().'
    global _cub_version
    if _cub_version is None:
        msg = 'check_cub_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        if _cub_version == -1:
            return '<unknown>'
        return str(_cub_version)
    return _cub_version

def check_jitify_version(compiler, settings):
    if False:
        return 10
    global _jitify_version
    try:
        cupy_jitify_include = _jitify_path
        a = subprocess.run(' '.join(['git', 'rev-parse', '--short', 'HEAD']), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=cupy_jitify_include)
        if a.returncode == 0:
            out = a.stdout.decode()[:-1]
        else:
            raise RuntimeError('Cannot determine Jitify version from git')
    except Exception as e:
        utils.print_warning('Cannot determine Jitify version\n{}'.format(e))
        out = -1
    _jitify_version = out
    settings['define_macros'].append(('CUPY_JITIFY_VERSION_CODE', _jitify_version))
    return True

def get_jitify_version(formatted=True):
    if False:
        for i in range(10):
            print('nop')
    'Return Jitify version cached in check_jitify_version().'
    global _jitify_version
    if _jitify_version is None:
        msg = 'check_jitify_version() must be called first.'
        raise RuntimeError(msg)
    if formatted:
        if _jitify_version == -1:
            return '<unknown>'
        return _jitify_version
    raise RuntimeError('Jitify version is a commit string')

def check_cutensor_version(compiler, settings):
    if False:
        i = 10
        return i + 15
    global _cutensor_version
    try:
        out = build_and_run(compiler, '\n        #include <cutensor.h>\n        #include <stdio.h>\n        #ifdef CUTENSOR_MAJOR\n        #ifndef CUTENSOR_VERSION\n        #define CUTENSOR_VERSION                 (CUTENSOR_MAJOR * 1000 + CUTENSOR_MINOR * 100 + CUTENSOR_PATCH)\n        #endif\n        #else\n        #  define CUTENSOR_VERSION 0\n        #endif\n        int main(int argc, char* argv[]) {\n          printf("%d", CUTENSOR_VERSION);\n          return 0;\n        }\n        ', include_dirs=settings['include_dirs'])
    except Exception as e:
        utils.print_warning('Cannot check cuTENSOR version\n{0}'.format(e))
        return False
    _cutensor_version = int(out)
    if _cutensor_version < 1000:
        utils.print_warning('Unsupported cuTENSOR version: {}'.format(_cutensor_version))
        return False
    return True

def get_cutensor_version(formatted=False):
    if False:
        return 10
    'Return cuTENSOR version cached in check_cutensor_version().'
    global _cutensor_version
    if _cutensor_version is None:
        msg = 'check_cutensor_version() must be called first.'
        raise RuntimeError(msg)
    return _cutensor_version

def check_cusparselt_version(compiler, settings):
    if False:
        for i in range(10):
            print('nop')
    global _cusparselt_version
    try:
        out = build_and_run(compiler, '\n        #include <cusparseLt.h>\n        #include <stdio.h>\n        #ifndef CUSPARSELT_VERSION\n        #define CUSPARSELT_VERSION 0\n        #endif\n        int main(int argc, char* argv[]) {\n          printf("%d", CUSPARSELT_VERSION);\n          return 0;\n        }\n        ', include_dirs=settings['include_dirs'])
    except Exception as e:
        utils.print_warning('Cannot check cuSPARSELt version\n{0}'.format(e))
        return False
    _cusparselt_version = int(out)
    return True

def get_cusparselt_version(formatted=False):
    if False:
        i = 10
        return i + 15
    'Return cuSPARSELt version cached in check_cusparselt_version().'
    global _cusparselt_version
    if _cusparselt_version is None:
        msg = 'check_cusparselt_version() must be called first.'
        raise RuntimeError(msg)
    return _cusparselt_version

def build_shlib(compiler, source, libraries=(), include_dirs=(), library_dirs=(), define_macros=None, extra_compile_args=()):
    if False:
        for i in range(10):
            print('nop')
    with _tempdir() as temp_dir:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)
        objects = compiler.compile([fname], output_dir=temp_dir, include_dirs=include_dirs, macros=define_macros, extra_postargs=list(extra_compile_args))
        try:
            postargs = ['/MANIFEST'] if PLATFORM_WIN32 else []
            compiler.link_shared_lib(objects, os.path.join(temp_dir, 'a'), libraries=libraries, library_dirs=library_dirs, extra_postargs=postargs, target_lang='c++')
        except Exception as e:
            msg = 'Cannot build a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)

def build_and_run(compiler, source, libraries=(), include_dirs=(), library_dirs=(), define_macros=None, extra_compile_args=()):
    if False:
        i = 10
        return i + 15
    with _tempdir() as temp_dir:
        fname = os.path.join(temp_dir, 'a.cpp')
        with open(fname, 'w') as f:
            f.write(source)
        objects = compiler.compile([fname], output_dir=temp_dir, include_dirs=include_dirs, macros=define_macros, extra_postargs=list(extra_compile_args))
        try:
            postargs = ['/MANIFEST'] if PLATFORM_WIN32 else []
            compiler.link_executable(objects, os.path.join(temp_dir, 'a'), libraries=libraries, library_dirs=library_dirs, extra_postargs=postargs, target_lang='c++')
        except Exception as e:
            msg = 'Cannot build a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)
        try:
            out = subprocess.check_output(os.path.join(temp_dir, 'a'))
            return out
        except Exception as e:
            msg = 'Cannot execute a stub file.\nOriginal error: {0}'.format(e)
            raise Exception(msg)