import contextlib
import io
import os
import pickle
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import pytest
import cupy
from cupy import testing
from cupy import _util
from cupy._core import _accelerator
from cupy.cuda import compiler
from cupy.cuda import memory
from cupy_backends.cuda.libs import nvrtc
_test_source1 = '\nextern "C" __global__\nvoid test_sum(const float* x1, const float* x2, float* y, unsigned int N) {\n    int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N)\n        y[tid] = x1[tid] + x2[tid];\n}\n'
_test_compile_src = '\nextern "C" __global__\nvoid test_op(const float* x1, const float* x2, float* y, unsigned int N) {\n    int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    int j;  // To generate a warning to appear in the log stream\n    if (tid < N)\n        y[tid] = x1[tid] OP x2[tid];\n}\n'
_test_source2 = '\nextern "C"{\n\n__global__ void test_sum(const float* x1, const float* x2, float* y, \\\n                         unsigned int N)\n{\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N)\n    {\n        y[tid] = x1[tid] + x2[tid];\n    }\n}\n\n__global__ void test_multiply(const float* x1, const float* x2, float* y, \\\n                              unsigned int N)\n{\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N)\n    {\n        y[tid] = x1[tid] * x2[tid];\n    }\n}\n\n}\n'
_test_source3 = '\n#ifndef PRECISION\n    #define PRECISION 2\n#endif\n\n#if PRECISION == 2\n    #define TYPE double\n#elif PRECISION == 1\n    #define TYPE float\n#else\n    #error precision not supported\n#endif\n\nextern "C"{\n\n__global__ void test_sum(const TYPE* x1, const TYPE* x2, TYPE* y, \\\n                         unsigned int N)\n{\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N)\n    {\n        y[tid] = x1[tid] + x2[tid];\n    }\n}\n\n__global__ void test_multiply(const TYPE* x1, const TYPE* x2, TYPE* y, \\\n                              unsigned int N)\n{\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N)\n    {\n        y[tid] = x1[tid] * x2[tid];\n    }\n}\n\n}\n'
_test_source4 = '\nextern "C"{\n\n__global__ void test_kernel_inner(float *arr, int N)\n{\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n\n    if (tid < N)\n        arr[tid] = 1.0;\n}\n\n__global__ void test_kernel(float *arr, int N, int inner_blk)\n{\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n\n    if (tid < N/inner_blk)\n        test_kernel_inner<<<1, inner_blk>>>(arr+tid*inner_blk, inner_blk);\n}\n\n}\n'
_test_source5 = '\nextern "C" __global__\nvoid test_div(const float* x1, const float* x2, float* y, unsigned int N) {\n    int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N)\n        y[tid] = x1[tid] / (x2[tid] + 1.0);\n}\n'
_test_cuComplex = '\n#include <cuComplex.h>\n#define N 100\n\nextern "C"{\n/* ------------------- double complex ------------------- */\n\n__global__ void test_add(cuDoubleComplex* arr1, cuDoubleComplex* arr2,\n                         cuDoubleComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCadd(arr1[tid], arr2[tid]);\n    }\n}\n\n__global__ void test_sub(cuDoubleComplex* arr1, cuDoubleComplex* arr2,\n                         cuDoubleComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCsub(arr1[tid], arr2[tid]);\n    }\n}\n\n__global__ void test_mul(cuDoubleComplex* arr1, cuDoubleComplex* arr2,\n                         cuDoubleComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCmul(arr1[tid], arr2[tid]);\n    }\n}\n\n__global__ void test_div(cuDoubleComplex* arr1, cuDoubleComplex* arr2,\n                         cuDoubleComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCdiv(arr1[tid], arr2[tid]);\n    }\n}\n\n__global__ void test_conj(cuDoubleComplex* arr, cuDoubleComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuConj(arr[tid]);\n    }\n}\n\n__global__ void test_abs(cuDoubleComplex* arr, double* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCabs(arr[tid]);\n    }\n}\n\n__global__ void test_fma(cuDoubleComplex* A, cuDoubleComplex* B,\n                         cuDoubleComplex* C, cuDoubleComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCfma(A[tid], B[tid], C[tid]);\n    }\n}\n\n__global__ void test_make(cuDoubleComplex* arr) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    cuDoubleComplex out = make_cuDoubleComplex(1.8, 2.9);\n    if (tid < N) {\n        arr[tid] = make_cuDoubleComplex(cuCreal(out), -3.*cuCimag(out));\n    }\n}\n\n__global__ void test_downcast(cuDoubleComplex* arr, cuComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuComplexDoubleToFloat(arr[tid]);\n    }\n}\n\n__global__ void test_add_scalar(cuDoubleComplex* arr, cuDoubleComplex scalar,\n                                cuDoubleComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCadd(arr[tid], scalar);\n    }\n}\n\n/* ------------------- single complex ------------------- */\n\n__global__ void test_addf(cuComplex* arr1, cuComplex* arr2,\n                          cuComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCaddf(arr1[tid], arr2[tid]);\n    }\n}\n\n__global__ void test_subf(cuComplex* arr1, cuComplex* arr2,\n                          cuComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCsubf(arr1[tid], arr2[tid]);\n    }\n}\n\n__global__ void test_mulf(cuComplex* arr1, cuComplex* arr2,\n                          cuComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCmulf(arr1[tid], arr2[tid]);\n    }\n}\n\n__global__ void test_divf(cuComplex* arr1, cuComplex* arr2,\n                          cuComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCdivf(arr1[tid], arr2[tid]);\n    }\n}\n\n__global__ void test_conjf(cuComplex* arr, cuComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuConjf(arr[tid]);\n    }\n}\n\n__global__ void test_absf(cuFloatComplex* arr, float* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCabsf(arr[tid]);\n    }\n}\n\n__global__ void test_fmaf(cuFloatComplex* A, cuFloatComplex* B,\n                          cuFloatComplex* C, cuFloatComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCfmaf(A[tid], B[tid], C[tid]);\n    }\n}\n\n__global__ void test_makef(cuComplex* arr) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    cuComplex out = make_cuFloatComplex(1.8, 2.9);\n    if (tid < N) {\n        arr[tid] = make_cuFloatComplex(cuCrealf(out), -3.*cuCimagf(out));\n    }\n}\n\n__global__ void test_upcast(cuComplex* arr, cuDoubleComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuComplexFloatToDouble(arr[tid]);\n    }\n}\n\n__global__ void test_addf_scalar(cuComplex* arr, cuComplex scalar,\n                                 cuComplex* out) {\n    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    if (tid < N) {\n        out[tid] = cuCadd(arr[tid], scalar);\n    }\n}\n\n}\n'
test_const_mem = '\nextern "C"{\n__constant__ float some_array[100];\n\n__global__ void multiply_by_const(float* x, int N) {\n    int id = threadIdx.x + blockIdx.x * blockDim.x;\n\n    if (id < N) {\n        x[id] *= some_array[id];\n    }\n}\n}\n'
test_cxx_template = '\n#include <cupy/complex.cuh>\n\ntemplate<typename T>\n__global__ void my_sqrt(T* input, int N) {\n  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;\n  if (x < N) {\n    input[x] *= input[x];\n  }\n}\n\n__global__ void my_func(double* input, int N) {\n  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;\n  if (x < N) {\n    input[x] *= input[x];\n  }\n}\n'
test_cast = '\nextern "C" __global__ void my_func(void* input, int N) {\n  double* arr = (double*)(input);\n  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;\n  if (x < N) {\n    arr[x] = 3.0 * arr[x] - 8.0;\n  }\n}\n'

@contextlib.contextmanager
def use_temporary_cache_dir():
    if False:
        while True:
            i = 10
    target1 = 'cupy.cuda.compiler.get_cache_dir'
    target2 = 'cupy.cuda.compiler._empty_file_preprocess_cache'
    temp_cache = {}
    with tempfile.TemporaryDirectory() as path:
        with mock.patch(target1, lambda : path):
            with mock.patch(target2, temp_cache):
                yield path

@contextlib.contextmanager
def compile_in_memory(in_memory):
    if False:
        print('Hello World!')
    target = 'cupy.cuda.compiler._get_bool_env_variable'

    def new_target(name, default):
        if False:
            i = 10
            return i + 15
        if name == 'CUPY_CACHE_IN_MEMORY':
            return in_memory
        else:
            val = os.environ.get(name)
            if val is None or len(val) == 0:
                return default
            try:
                return int(val) == 1
            except ValueError:
                return False
    with mock.patch(target, new_target) as m:
        yield m

@testing.parameterize({'backend': 'nvrtc', 'in_memory': False}, {'backend': 'nvrtc', 'in_memory': True}, {'backend': 'nvrtc', 'in_memory': True, 'clean_up': True}, {'backend': 'nvrtc', 'in_memory': False, 'jitify': True}, {'backend': 'nvrtc', 'in_memory': True, 'jitify': True}, {'backend': 'nvrtc', 'in_memory': True, 'clean_up': True, 'jitify': True}, {'backend': 'nvcc', 'in_memory': False})
class TestRaw(unittest.TestCase):
    _nvcc_ver = None
    _nvrtc_ver = None

    def setUp(self):
        if False:
            while True:
                i = 10
        if hasattr(self, 'clean_up'):
            if cupy.cuda.runtime.is_hip:
                self.skipTest('Clearing memo hits a nvrtc bug in other tests')
            _util.clear_memo()
        self.dev = cupy.cuda.runtime.getDevice()
        assert self.dev != 1
        if not hasattr(self, 'jitify'):
            self.jitify = False
        if cupy.cuda.runtime.is_hip and self.jitify:
            self.skipTest('Jitify does not support ROCm/HIP')
        self.temporary_cache_dir_context = use_temporary_cache_dir()
        self.in_memory_context = compile_in_memory(self.in_memory)
        self.cache_dir = self.temporary_cache_dir_context.__enter__()
        self.in_memory_context.__enter__()
        self.kern = cupy.RawKernel(_test_source1, 'test_sum', backend=self.backend, jitify=self.jitify)
        self.mod2 = cupy.RawModule(code=_test_source2, backend=self.backend, jitify=self.jitify)
        self.mod3 = cupy.RawModule(code=_test_source3, options=('-DPRECISION=2',), backend=self.backend, jitify=self.jitify)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if self.in_memory and _accelerator.ACCELERATOR_CUB not in _accelerator.get_reduction_accelerators():
            files = os.listdir(self.cache_dir)
            for f in files:
                if f == 'test_load_cubin.cu':
                    count = 1
                    break
            else:
                count = 0
            assert len(files) == count
        self.in_memory_context.__exit__(*sys.exc_info())
        self.temporary_cache_dir_context.__exit__(*sys.exc_info())

    def _helper(self, kernel, dtype):
        if False:
            while True:
                i = 10
        N = 10
        x1 = cupy.arange(N ** 2, dtype=dtype).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=dtype)
        y = cupy.zeros((N, N), dtype=dtype)
        kernel((N,), (N,), (x1, x2, y, N ** 2))
        return (x1, x2, y)

    def test_basic(self):
        if False:
            while True:
                i = 10
        (x1, x2, y) = self._helper(self.kern, cupy.float32)
        assert cupy.allclose(y, x1 + x2)

    def test_kernel_attributes(self):
        if False:
            print('Hello World!')
        attrs = self.kern.attributes
        for attribute in ['binary_version', 'cache_mode_ca', 'const_size_bytes', 'local_size_bytes', 'max_dynamic_shared_size_bytes', 'max_threads_per_block', 'num_regs', 'preferred_shared_memory_carveout', 'ptx_version', 'shared_size_bytes']:
            assert attribute in attrs
        if not cupy.cuda.runtime.is_hip:
            assert self.kern.num_regs > 0
        assert self.kern.max_threads_per_block > 0
        assert self.kern.shared_size_bytes == 0

    def test_module(self):
        if False:
            print('Hello World!')
        module = self.mod2
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')
        (x1, x2, y) = self._helper(ker_sum, cupy.float32)
        assert cupy.allclose(y, x1 + x2)
        (x1, x2, y) = self._helper(ker_times, cupy.float32)
        assert cupy.allclose(y, x1 * x2)

    def test_compiler_flag(self):
        if False:
            while True:
                i = 10
        module = self.mod3
        ker_sum = module.get_function('test_sum')
        ker_times = module.get_function('test_multiply')
        (x1, x2, y) = self._helper(ker_sum, cupy.float64)
        assert cupy.allclose(y, x1 + x2)
        (x1, x2, y) = self._helper(ker_times, cupy.float64)
        assert cupy.allclose(y, x1 * x2)

    def test_invalid_compiler_flag(self):
        if False:
            for i in range(10):
                print('nop')
        if cupy.cuda.runtime.is_hip and self.backend == 'nvrtc':
            self.skipTest('hiprtc does not handle #error macro properly')
        if self.jitify:
            ex_type = cupy.cuda.compiler.JitifyException
        else:
            ex_type = cupy.cuda.compiler.CompileException
        with pytest.raises(ex_type) as ex:
            mod = cupy.RawModule(code=_test_source3, options=('-DPRECISION=3',), backend=self.backend, jitify=self.jitify)
            mod.get_function('test_sum')
        if not self.jitify:
            assert 'precision not supported' in str(ex.value)

    def _find_nvcc_ver(self):
        if False:
            while True:
                i = 10
        if self._nvcc_ver:
            return self._nvcc_ver
        nvcc_ver_pattern = 'release (\\d+\\.\\d+)'
        cmd = cupy.cuda.get_nvcc_path().split()
        cmd += ['--version']
        output = compiler._run_cc(cmd, self.cache_dir, 'nvcc')
        match = re.search(nvcc_ver_pattern, output)
        assert match
        (major, minor) = match.group(1).split('.')
        self._nvcc_ver = int(major) * 1000 + int(minor) * 10
        return self._nvcc_ver

    def _find_nvrtc_ver(self):
        if False:
            while True:
                i = 10
        if self._nvrtc_ver:
            return self._nvrtc_ver
        (major, minor) = nvrtc.getVersion()
        self._nvrtc_ver = int(major) * 1000 + int(minor) * 10
        return self._nvrtc_ver

    def _check_ptx_loadable(self, compiler: str):
        if False:
            for i in range(10):
                print('nop')
        if compiler == 'nvrtc':
            compiler_ver = self._find_nvrtc_ver()
        elif compiler == 'nvcc':
            compiler_ver = self._find_nvcc_ver()
        driver_ver = cupy.cuda.runtime.driverGetVersion()
        if driver_ver < compiler_ver:
            raise pytest.skip()

    def _generate_file(self, ext: str):
        if False:
            for i in range(10):
                print('nop')
        if not cupy.cuda.runtime.is_hip:
            cc = cupy.cuda.get_nvcc_path()
            arch = '-gencode=arch=compute_{CC},code=sm_{CC}'.format(CC=compiler._get_arch())
            code = _test_source5
        else:
            cc = cupy._environment.get_hipcc_path()
            arch = '-v'
            code = compiler._convert_to_hip_source(_test_source5, None, False)
        cmd = cc.split()
        source = '{}/test_load_cubin.cu'.format(self.cache_dir)
        file_path = self.cache_dir + 'test_load_cubin'
        with open(source, 'w') as f:
            f.write(code)
        if not cupy.cuda.runtime.is_hip:
            if ext == 'cubin':
                file_path += '.cubin'
                flag = '-cubin'
            elif ext == 'ptx':
                file_path += '.ptx'
                flag = '-ptx'
            else:
                raise ValueError
        else:
            file_path += '.hsaco'
            flag = '--genco'
        cmd += [arch, flag, source, '-o', file_path]
        cc = 'nvcc' if not cupy.cuda.runtime.is_hip else 'hipcc'
        compiler._run_cc(cmd, self.cache_dir, cc)
        return file_path

    @unittest.skipIf(cupy.cuda.runtime.is_hip, 'HIP uses hsaco, not cubin')
    def test_load_cubin(self):
        if False:
            print('Hello World!')
        file_path = self._generate_file('cubin')
        mod = cupy.RawModule(path=file_path, backend=self.backend)
        ker = mod.get_function('test_div')
        (x1, x2, y) = self._helper(ker, cupy.float32)
        assert cupy.allclose(y, x1 / (x2 + 1.0))

    @unittest.skipIf(cupy.cuda.runtime.is_hip, 'HIP uses hsaco, not ptx')
    def test_load_ptx(self):
        if False:
            print('Hello World!')
        self._check_ptx_loadable('nvcc')
        file_path = self._generate_file('ptx')
        mod = cupy.RawModule(path=file_path, backend=self.backend)
        ker = mod.get_function('test_div')
        (x1, x2, y) = self._helper(ker, cupy.float32)
        assert cupy.allclose(y, x1 / (x2 + 1.0))

    @unittest.skipIf(not cupy.cuda.runtime.is_hip, 'CUDA uses cubin/ptx, not hsaco')
    def test_load_hsaco(self):
        if False:
            i = 10
            return i + 15
        file_path = self._generate_file('hsaco')
        mod = cupy.RawModule(path=file_path, backend=self.backend)
        ker = mod.get_function('test_div')
        (x1, x2, y) = self._helper(ker, cupy.float32)
        assert cupy.allclose(y, x1 / (x2 + 1.0))

    def test_module_load_failure(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(cupy.cuda.driver.CUDADriverError) as ex:
            mod = cupy.RawModule(path=os.path.expanduser('~/this_does_not_exist.cubin'), backend=self.backend)
            mod.get_function('nonexisting_kernel')
        assert 'CUDA_ERROR_FILE_NOT_FOUND' in str(ex.value) or 'hipErrorFileNotFound' in str(ex.value)

    def test_module_neither_code_nor_path(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError):
            cupy.RawModule()

    def test_module_both_code_and_path(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError):
            cupy.RawModule(code=_test_source1, path='test.cubin')

    def test_get_function_failure(self):
        if False:
            print('Hello World!')
        with pytest.raises(cupy.cuda.driver.CUDADriverError) as ex:
            self.mod2.get_function('no_such_kernel')
        assert 'CUDA_ERROR_NOT_FOUND' in str(ex.value) or 'hipErrorNotFound' in str(ex.value)

    @unittest.skipIf(cupy.cuda.runtime.is_hip, 'ROCm/HIP does not support dynamic parallelism')
    def test_dynamical_parallelism(self):
        if False:
            i = 10
            return i + 15
        self._check_ptx_loadable('nvrtc')
        ker = cupy.RawKernel(_test_source4, 'test_kernel', options=('-dc',), backend=self.backend, jitify=self.jitify)
        N = 169
        inner_chunk = 13
        x = cupy.zeros((N,), dtype=cupy.float32)
        ker((1,), (N // inner_chunk,), (x, N, inner_chunk))
        assert (x == 1.0).all()

    def test_dynamical_parallelism_compile_failure(self):
        if False:
            i = 10
            return i + 15
        ker = cupy.RawKernel(_test_source4, 'test_kernel', backend=self.backend, jitify=self.jitify)
        N = 10
        inner_chunk = 2
        x = cupy.zeros((N,), dtype=cupy.float32)
        use_ptx = os.environ.get('CUPY_COMPILE_WITH_PTX', False)
        if self.backend == 'nvrtc' and (use_ptx or (cupy.cuda.driver._is_cuda_python() and cupy.cuda.runtime.runtimeGetVersion() < 11010) or (not cupy.cuda.driver._is_cuda_python() and (not cupy.cuda.runtime.is_hip) and (cupy.cuda.driver.get_build_version() < 11010))):
            error = cupy.cuda.driver.CUDADriverError
        else:
            error = cupy.cuda.compiler.CompileException
        with pytest.raises(error):
            ker((1,), (N // inner_chunk,), (x, N, inner_chunk))

    @unittest.skipIf(cupy.cuda.runtime.is_hip, 'HIP code should not use cuFloatComplex')
    def test_cuFloatComplex(self):
        if False:
            for i in range(10):
                print('nop')
        N = 100
        block = 32
        grid = (N + block - 1) // block
        dtype = cupy.complex64
        mod = cupy.RawModule(code=_test_cuComplex, translate_cucomplex=True, jitify=self.jitify)
        a = cupy.random.random((N,)) + 1j * cupy.random.random((N,))
        a = a.astype(dtype)
        b = cupy.random.random((N,)) + 1j * cupy.random.random((N,))
        b = b.astype(dtype)
        c = cupy.random.random((N,)) + 1j * cupy.random.random((N,))
        c = c.astype(dtype)
        out = cupy.zeros((N,), dtype=dtype)
        out_float = cupy.zeros((N,), dtype=cupy.float32)
        out_up = cupy.zeros((N,), dtype=cupy.complex128)
        ker = mod.get_function('test_addf')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()
        ker = mod.get_function('test_subf')
        ker((grid,), (block,), (a, b, out))
        assert (out == a - b).all()
        ker = mod.get_function('test_mulf')
        ker((grid,), (block,), (a, b, out))
        assert cupy.allclose(out, a * b)
        ker = mod.get_function('test_divf')
        ker((grid,), (block,), (a, b, out))
        assert (out == a / b).all()
        ker = mod.get_function('test_conjf')
        ker((grid,), (block,), (a, out))
        assert (out == cupy.conj(a)).all()
        ker = mod.get_function('test_absf')
        ker((grid,), (block,), (a, out_float))
        assert (out_float == cupy.abs(a)).all()
        ker = mod.get_function('test_fmaf')
        ker((grid,), (block,), (a, b, c, out))
        assert cupy.allclose(out, a * b + c)
        ker = mod.get_function('test_makef')
        ker((grid,), (block,), (out,))
        assert cupy.allclose(out, 1.8 - 1j * 8.7)
        ker = mod.get_function('test_upcast')
        ker((grid,), (block,), (a, out_up))
        assert (out_up == a.astype(cupy.complex128)).all()
        b = cupy.complex64(2 + 3j)
        ker = mod.get_function('test_addf_scalar')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()

    @unittest.skipIf(cupy.cuda.runtime.is_hip, 'HIP code should not use cuDoubleComplex')
    def test_cuDoubleComplex(self):
        if False:
            print('Hello World!')
        N = 100
        block = 32
        grid = (N + block - 1) // block
        dtype = cupy.complex128
        mod = cupy.RawModule(code=_test_cuComplex, translate_cucomplex=True, jitify=self.jitify)
        a = cupy.random.random((N,)) + 1j * cupy.random.random((N,))
        a = a.astype(dtype)
        b = cupy.random.random((N,)) + 1j * cupy.random.random((N,))
        b = b.astype(dtype)
        c = cupy.random.random((N,)) + 1j * cupy.random.random((N,))
        c = c.astype(dtype)
        out = cupy.zeros((N,), dtype=dtype)
        out_float = cupy.zeros((N,), dtype=cupy.float64)
        out_down = cupy.zeros((N,), dtype=cupy.complex64)
        ker = mod.get_function('test_add')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()
        ker = mod.get_function('test_sub')
        ker((grid,), (block,), (a, b, out))
        assert (out == a - b).all()
        ker = mod.get_function('test_mul')
        ker((grid,), (block,), (a, b, out))
        assert cupy.allclose(out, a * b)
        ker = mod.get_function('test_div')
        ker((grid,), (block,), (a, b, out))
        assert (out == a / b).all()
        ker = mod.get_function('test_conj')
        ker((grid,), (block,), (a, out))
        assert (out == cupy.conj(a)).all()
        ker = mod.get_function('test_abs')
        ker((grid,), (block,), (a, out_float))
        assert (out_float == cupy.abs(a)).all()
        ker = mod.get_function('test_fma')
        ker((grid,), (block,), (a, b, c, out))
        assert cupy.allclose(out, a * b + c)
        ker = mod.get_function('test_make')
        ker((grid,), (block,), (out,))
        assert (out == 1.8 - 1j * 8.7).all()
        ker = mod.get_function('test_downcast')
        ker((grid,), (block,), (a, out_down))
        assert (out_down == a.astype(cupy.complex64)).all()
        b = cupy.complex128(2 + 3j)
        ker = mod.get_function('test_add_scalar')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()
        b = 2 + 3j
        ker = mod.get_function('test_add_scalar')
        ker((grid,), (block,), (a, b, out))
        assert (out == a + b).all()

    def test_const_memory(self):
        if False:
            print('Hello World!')
        mod = cupy.RawModule(code=test_const_mem, backend=self.backend, jitify=self.jitify)
        ker = mod.get_function('multiply_by_const')
        mem_ptr = mod.get_global('some_array')
        const_arr = cupy.ndarray((100,), cupy.float32, mem_ptr)
        data = cupy.arange(100, dtype=cupy.float32)
        const_arr[...] = data
        output_arr = cupy.ones(100, dtype=cupy.float32)
        ker((1,), (100,), (output_arr, cupy.int32(100)))
        assert (data == output_arr).all()

    def test_template_specialization(self):
        if False:
            while True:
                i = 10
        if self.backend == 'nvcc':
            self.skipTest('nvcc does not support template specialization')
        if cupy.cuda.runtime.is_hip and hasattr(self, 'clean_up'):
            self.skipTest('skip a potential hiprtc bug')
        if cupy.cuda.runtime.is_hip:
            name_expressions = ['my_sqrt<int>', 'my_sqrt<float>', 'my_sqrt<thrust::complex<double>>', 'my_func']
        else:
            name_expressions = ['my_sqrt<int>', 'my_sqrt<float>', 'my_sqrt<complex<double>>', 'my_func']
        mod = cupy.RawModule(code=test_cxx_template, name_expressions=name_expressions, jitify=self.jitify)
        dtypes = (cupy.int32, cupy.float32, cupy.complex128, cupy.float64)
        for (ker_T, dtype) in zip(name_expressions, dtypes):
            if cupy.cuda.runtime.is_hip:
                mangled_name = mod.module.mapping.get(ker_T)
                if mangled_name == '':
                    continue
            ker = mod.get_function(ker_T)
            in_arr = cupy.testing.shaped_random((10,), dtype=dtype)
            out_arr = in_arr ** 2
            ker((1,), (10,), (in_arr, 10))
            assert cupy.allclose(in_arr, out_arr)

    def test_template_failure(self):
        if False:
            i = 10
            return i + 15
        name_expressions = ['my_sqrt<int>']
        if self.backend == 'nvcc':
            with pytest.raises(ValueError) as e:
                cupy.RawModule(code=test_cxx_template, backend=self.backend, name_expressions=name_expressions)
            assert 'nvrtc' in str(e.value)
            return
        mod = cupy.RawModule(code=test_cxx_template, jitify=self.jitify)
        match = 'named symbol not found' if not cupy.cuda.runtime.is_hip else 'hipErrorNotFound'
        with pytest.raises(cupy.cuda.driver.CUDADriverError, match=match):
            mod.get_function('my_sqrt<int>')
        mod = cupy.RawModule(code=test_cxx_template, name_expressions=name_expressions, jitify=self.jitify)
        if cupy.cuda.runtime.is_hip:
            msg = 'hipErrorNotFound'
        else:
            msg = 'named symbol not found'
        with pytest.raises(cupy.cuda.driver.CUDADriverError, match=msg):
            mod.get_function('my_sqrt<double>')

    def test_raw_pointer(self):
        if False:
            for i in range(10):
                print('nop')
        mod = cupy.RawModule(code=test_cast, backend=self.backend, jitify=self.jitify)
        ker = mod.get_function('my_func')
        a = cupy.ones((100,), dtype=cupy.float64)
        memptr = memory.alloc(100 * a.dtype.itemsize)
        memptr.copy_from(a.data, 100 * a.dtype.itemsize)
        b = cupy.ndarray((100,), cupy.float64, memptr=memptr)
        ker((1,), (100,), (memptr, 100))
        a = 3.0 * a - 8.0
        assert (a == b).all()

    @testing.multi_gpu(2)
    def test_context_switch_RawKernel(self):
        if False:
            print('Hello World!')
        (x1, x2, y) = self._helper(self.kern, cupy.float32)
        with cupy.cuda.Device(1):
            (x1, x2, y) = self._helper(self.kern, cupy.float32)
            assert cupy.allclose(y, x1 + x2)

    @testing.multi_gpu(2)
    def test_context_switch_RawModule1(self):
        if False:
            for i in range(10):
                print('nop')
        module = self.mod2
        with cupy.cuda.Device(0):
            module.get_function('test_sum')
        with cupy.cuda.Device(1):
            ker_sum = module.get_function('test_sum')
            (x1, x2, y) = self._helper(ker_sum, cupy.float32)
            assert cupy.allclose(y, x1 + x2)

    @testing.multi_gpu(2)
    def test_context_switch_RawModule2(self):
        if False:
            return 10
        module = self.mod2
        with cupy.cuda.Device(0):
            ker_sum = module.get_function('test_sum')
        with cupy.cuda.Device(1):
            (x1, x2, y) = self._helper(ker_sum, cupy.float32)
            assert cupy.allclose(y, x1 + x2)

    @testing.multi_gpu(2)
    def test_context_switch_RawModule3(self):
        if False:
            while True:
                i = 10
        device0 = cupy.cuda.Device(0)
        device1 = cupy.cuda.Device(1)
        if device0.compute_capability != device1.compute_capability:
            raise pytest.skip()
        with device0:
            file_path = self._generate_file('cubin')
            mod = cupy.RawModule(path=file_path, backend=self.backend)
            mod.get_function('test_div')
        with device1:
            ker = mod.get_function('test_div')
            (x1, x2, y) = self._helper(ker, cupy.float32)
            assert cupy.allclose(y, x1 / (x2 + 1.0))

    @testing.multi_gpu(2)
    def test_context_switch_RawModule4(self):
        if False:
            return 10
        device0 = cupy.cuda.Device(0)
        device1 = cupy.cuda.Device(1)
        if device0.compute_capability != device1.compute_capability:
            raise pytest.skip()
        with device0:
            file_path = self._generate_file('cubin')
            mod = cupy.RawModule(path=file_path, backend=self.backend)
            ker = mod.get_function('test_div')
        with device1:
            (x1, x2, y) = self._helper(ker, cupy.float32)
            assert cupy.allclose(y, x1 / (x2 + 1.0))

    @testing.multi_gpu(2)
    def test_context_switch_RawModule5(self):
        if False:
            return 10
        if self.backend == 'nvcc':
            self.skipTest('nvcc does not support template specialization')
        name_expressions = ['my_sqrt<unsigned int>']
        name = name_expressions[0]
        with cupy.cuda.Device(0):
            mod = cupy.RawModule(code=test_cxx_template, name_expressions=name_expressions, jitify=self.jitify)
            mod.get_function(name)
        with cupy.cuda.Device(1):
            ker = mod.get_function(name)
            in_arr = cupy.testing.shaped_random((10,), dtype=cupy.uint32)
            out_arr = in_arr ** 2
            ker((1,), (10,), (in_arr, 10))
            assert cupy.allclose(in_arr, out_arr)

    @testing.multi_gpu(2)
    def test_context_switch_RawModule6(self):
        if False:
            print('Hello World!')
        if self.backend == 'nvcc':
            self.skipTest('nvcc does not support template specialization')
        name_expressions = ['my_sqrt<unsigned int>']
        name = name_expressions[0]
        with cupy.cuda.Device(0):
            mod = cupy.RawModule(code=test_cxx_template, name_expressions=name_expressions, jitify=self.jitify)
            ker = mod.get_function(name)
        with cupy.cuda.Device(1):
            in_arr = cupy.testing.shaped_random((10,), dtype=cupy.uint32)
            out_arr = in_arr ** 2
            ker((1,), (10,), (in_arr, 10))
            assert cupy.allclose(in_arr, out_arr)

    @unittest.skipUnless(not cupy.cuda.runtime.is_hip, 'only CUDA raises warning')
    def test_compile_kernel(self):
        if False:
            print('Hello World!')
        kern = cupy.RawKernel(_test_compile_src, 'test_op', options=('-DOP=+',), backend=self.backend, jitify=self.jitify)
        log = io.StringIO()
        with use_temporary_cache_dir():
            kern.compile(log_stream=log)
        assert 'warning' in log.getvalue()
        (x1, x2, y) = self._helper(kern, cupy.float32)
        assert cupy.allclose(y, x1 + x2)

    @unittest.skipUnless(not cupy.cuda.runtime.is_hip, 'only CUDA raises warning')
    def test_compile_module(self):
        if False:
            return 10
        module = cupy.RawModule(code=_test_compile_src, backend=self.backend, options=('-DOP=+',), jitify=self.jitify)
        log = io.StringIO()
        with use_temporary_cache_dir():
            module.compile(log_stream=log)
        assert 'warning' in log.getvalue()
        kern = module.get_function('test_op')
        (x1, x2, y) = self._helper(kern, cupy.float32)
        assert cupy.allclose(y, x1 + x2)
_test_grid_sync = '\n#include <cooperative_groups.h>\n\nextern "C" __global__\nvoid test_grid_sync(const float* x1, const float* x2, float* y, int n) {\n    namespace cg = cooperative_groups;\n    cg::grid_group grid = cg::this_grid();\n    int size = gridDim.x * blockDim.x;\n    int tid = blockDim.x * blockIdx.x + threadIdx.x;\n    for (int i = tid; i < n; i += size) {\n        y[i] = x1[i];\n    }\n    cg::sync(grid);\n    for (int i = n - 1 - tid; i >= 0; i -= size) {\n        y[i] += x2[i];\n    }\n}\n'

@testing.parameterize(*testing.product({'n': [10, 100, 1000], 'block': [64, 256]}))
@unittest.skipUnless(9000 <= cupy.cuda.runtime.runtimeGetVersion(), 'Requires CUDA 9.x or later')
@unittest.skipUnless(60 <= int(cupy.cuda.device.get_compute_capability()), 'Requires compute capability 6.0 or later')
@unittest.skipIf(cupy.cuda.runtime.is_hip, 'Skip on HIP')
class TestRawGridSync(unittest.TestCase):

    def test_grid_sync_rawkernel(self):
        if False:
            while True:
                i = 10
        n = self.n
        with use_temporary_cache_dir():
            kern_grid_sync = cupy.RawKernel(_test_grid_sync, 'test_grid_sync', backend='nvcc', enable_cooperative_groups=True)
            x1 = cupy.arange(n ** 2, dtype='float32').reshape(n, n)
            x2 = cupy.ones((n, n), dtype='float32')
            y = cupy.zeros((n, n), dtype='float32')
            block = self.block
            grid = (n * n + block - 1) // block
            kern_grid_sync((grid,), (block,), (x1, x2, y, n ** 2))
            assert cupy.allclose(y, x1 + x2)

    def test_grid_sync_rawmodule(self):
        if False:
            while True:
                i = 10
        n = self.n
        with use_temporary_cache_dir():
            mod_grid_sync = cupy.RawModule(code=_test_grid_sync, backend='nvcc', enable_cooperative_groups=True)
            x1 = cupy.arange(n ** 2, dtype='float32').reshape(n, n)
            x2 = cupy.ones((n, n), dtype='float32')
            y = cupy.zeros((n, n), dtype='float32')
            kern = mod_grid_sync.get_function('test_grid_sync')
            block = self.block
            grid = (n * n + block - 1) // block
            kern((grid,), (block,), (x1, x2, y, n ** 2))
            assert cupy.allclose(y, x1 + x2)
_test_script = "\nimport pickle\nimport sys\n\nimport cupy as cp\n\n\nN = 100\na = cp.random.random(N, dtype=cp.float32)\nb = cp.random.random(N, dtype=cp.float32)\nc = cp.empty_like(a)\nwith open('raw.pkl', 'rb') as f:\n    ker = pickle.load(f)\n\nif len(sys.argv) == 2:\n    ker = ker.get_function(sys.argv[1])\n\nker((1,), (100,), (a, b, c, N))\nassert cp.allclose(a + b, c)\nassert ker.enable_cooperative_groups\n"

@testing.parameterize(*testing.product({'compile': (False, True), 'raw': ('ker', 'mod', 'mod_ker')}))
@unittest.skipUnless(60 <= int(cupy.cuda.device.get_compute_capability()), 'Requires compute capability 6.0 or later')
@unittest.skipIf(cupy.cuda.runtime.is_hip, 'HIP does not support enable_cooperative_groups')
class TestRawPicklable(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.temporary_dir_context = use_temporary_cache_dir()
        self.temp_dir = self.temporary_dir_context.__enter__()
        if self.raw == 'ker':
            self.ker = cupy.RawKernel(_test_source1, 'test_sum', backend='nvcc', enable_cooperative_groups=True)
        else:
            self.mod = cupy.RawModule(code=_test_source1, backend='nvcc', enable_cooperative_groups=True)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.temporary_dir_context.__exit__(*sys.exc_info())

    def _helper(self):
        if False:
            i = 10
            return i + 15
        N = 10
        x1 = cupy.arange(N ** 2, dtype=cupy.float32).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=cupy.float32)
        y = cupy.zeros((N, N), dtype=cupy.float32)
        if self.raw == 'ker':
            ker = self.ker
        else:
            ker = self.mod.get_function('test_sum')
        ker((N,), (N,), (x1, x2, y, N ** 2))
        assert cupy.allclose(x1 + x2, y)

    def test_raw_picklable(self):
        if False:
            for i in range(10):
                print('nop')
        if self.compile:
            self._helper()
        if self.raw == 'ker':
            obj = self.ker
        elif self.raw == 'mod':
            obj = self.mod
        elif self.raw == 'mod_ker':
            obj = self.mod.get_function('test_sum')
        with open(self.temp_dir + '/raw.pkl', 'wb') as f:
            pickle.dump(obj, f)
        with open(self.temp_dir + '/TestRawPicklable.py', 'w') as f:
            f.write(_test_script)
        test_args = ['test_sum'] if self.raw == 'mod' else []
        s = subprocess.run([sys.executable, 'TestRawPicklable.py'] + test_args, cwd=self.temp_dir)
        s.check_returncode()
std_code = '\n#include <type_traits>\n\ntemplate<typename T,\n         typename = typename std::enable_if<std::is_integral<T>::value>::type>\n__global__ void shift (T* a, int N) {\n    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;\n    if (tid < N) {\n        a[tid] += 100;\n    }\n}\n'

@testing.parameterize(*testing.product({'jitify': (False, True)}))
@unittest.skipIf(cupy.cuda.runtime.is_hip, 'Jitify does not support ROCm/HIP')
class TestRawJitify(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.temporary_dir_context = use_temporary_cache_dir()
        self.temp_dir = self.temporary_dir_context.__enter__()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.temporary_dir_context.__exit__(*sys.exc_info())

    def _helper(self, header, options=()):
        if False:
            print('Hello World!')
        code = header
        code += _test_source1
        mod1 = cupy.RawModule(code=code, backend='nvrtc', options=options, jitify=self.jitify)
        N = 10
        x1 = cupy.arange(N ** 2, dtype=cupy.float32).reshape(N, N)
        x2 = cupy.ones((N, N), dtype=cupy.float32)
        y = cupy.zeros((N, N), dtype=cupy.float32)
        ker = mod1.get_function('test_sum')
        ker((N,), (N,), (x1, x2, y, N ** 2))
        assert cupy.allclose(x1 + x2, y)

    def _helper2(self, type_str):
        if False:
            return 10
        mod2 = cupy.RawModule(code=std_code, jitify=self.jitify, name_expressions=('shift<%s>' % type_str,))
        ker = mod2.get_function('shift<%s>' % type_str)
        N = 256
        a = cupy.random.random_integers(0, 7, N).astype(cupy.int32)
        b = a.copy()
        ker((1,), (N,), (a, N))
        assert cupy.allclose(a, b + 100)

    def test_jitify1(self):
        if False:
            print('Hello World!')
        hdr = '#include <cub/block/block_reduce.cuh>\n'
        options = ('-DCUB_DISABLE_BF16_SUPPORT',)
        if self.jitify:
            self._helper(hdr, options)
        else:
            with pytest.raises(cupy.cuda.compiler.CompileException) as ex:
                self._helper(hdr, options)
            assert 'cannot open source file' in str(ex.value)

    def test_jitify2(self):
        if False:
            return 10
        if self.jitify:
            self._helper2('int')
        else:
            with pytest.raises(cupy.cuda.compiler.CompileException) as ex:
                self._helper2('int')
            assert 'cannot open source file' in str(ex.value)

    def test_jitify3(self):
        if False:
            while True:
                i = 10
        ex_type = cupy.cuda.compiler.CompileException
        with pytest.raises(ex_type) as ex:
            self._helper2('float')
        if self.jitify:
            assert 'Error in parsing name expression' in str(ex.value)
        else:
            assert 'cannot open source file' in str(ex.value)

    def test_jitify4(self):
        if False:
            print('Hello World!')
        code = '\n        __global__ void i_am_broken() {\n        '
        if self.jitify:
            ex_type = cupy.cuda.compiler.JitifyException
        else:
            ex_type = cupy.cuda.compiler.CompileException
        with pytest.raises(ex_type):
            mod = cupy.RawModule(code=code, jitify=self.jitify)
            ker = mod.get_function('i_am_broken')

    def test_jitify5(self):
        if False:
            while True:
                i = 10
        hdr = 'I_INCLUDE_SOMETHING.h'
        with open(self.temp_dir + '/' + hdr, 'w') as f:
            dummy = '#include <cupy/I_DO_NOT_EXIST_WAH_HA_HA.h>\n'
            f.write(dummy)
        hdr = '#include "' + hdr + '"\n'
        if self.jitify:
            self._helper(hdr, options=('-I' + self.temp_dir,))
        else:
            with pytest.raises(cupy.cuda.compiler.CompileException) as ex:
                self._helper(hdr, options=('-I' + self.temp_dir,))
            assert 'cannot open source file' in str(ex.value)