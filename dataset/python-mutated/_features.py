import sys
from typing import Any, Dict, List
import cupy_builder.install_build as build
import cupy_builder.install_utils as utils
from cupy_builder import Context

class Feature:
    NOT_AVAILABLE = -1
    _UNDETERMINED = -100

    def __init__(self, ctx: Context):
        if False:
            while True:
                i = 10
        self.name = ''
        self.required = False
        self.modules: List[str] = []
        self.includes: List[str] = []
        self.libraries: List[str] = []
        self.static_libraries: List[str] = []
        self._version: Any = self._UNDETERMINED

    def configure(self, compiler: Any, settings: Any) -> bool:
        if False:
            return 10
        self._version = None
        return True

    def get_version(self) -> Any:
        if False:
            while True:
                i = 10
        assert self._version != self._UNDETERMINED, 'not configured yet'
        return self._version

    def __contains__(self, key: Any) -> bool:
        if False:
            return 10
        if not isinstance(key, str):
            return False
        try:
            self.__getitem__(key)
        except AttributeError:
            return False
        return True

    def __getitem__(self, key: str) -> Any:
        if False:
            return 10
        if key == 'file':
            return self.modules
        elif key == 'include':
            return self.includes
        return getattr(self, key)

def _from_dict(d: Dict[str, Any], ctx: Context) -> Feature:
    if False:
        for i in range(10):
            print('nop')
    f = Feature(ctx)
    f.name = d['name']
    f.required = d.get('required', False)
    f.libraries = d['libraries']
    f.static_libraries = d.get('static_libraries', [])
    f.modules = d['file']
    f.includes = d['include']
    if 'check_method' in d:
        f.configure = d['check_method']
        f._version = None
        if 'version_method' in d:
            f.get_version = d['version_method']
    return f
_cuda_files = ['cupy_backends.cuda.api.driver', 'cupy_backends.cuda.api._driver_enum', 'cupy_backends.cuda.api.runtime', 'cupy_backends.cuda.api._runtime_enum', 'cupy_backends.cuda.libs.cublas', 'cupy_backends.cuda.libs.curand', 'cupy_backends.cuda.libs.cusparse', 'cupy_backends.cuda.libs.nvrtc', 'cupy_backends.cuda.libs.profiler', 'cupy_backends.cuda.stream', 'cupy_backends.cuda._softlink', 'cupy._core._accelerator', 'cupy._core._carray', 'cupy._core._cub_reduction', 'cupy._core._dtype', 'cupy._core._fusion_kernel', 'cupy._core._fusion_thread_local', 'cupy._core._fusion_trace', 'cupy._core._fusion_variable', 'cupy._core._kernel', 'cupy._core._memory_range', 'cupy._core._optimize_config', 'cupy._core._reduction', 'cupy._core._routines_binary', 'cupy._core._routines_indexing', 'cupy._core._routines_linalg', 'cupy._core._routines_logic', 'cupy._core._routines_manipulation', 'cupy._core._routines_math', 'cupy._core._routines_sorting', 'cupy._core._routines_statistics', 'cupy._core._scalar', 'cupy._core.core', 'cupy._core.flags', 'cupy._core.internal', 'cupy._core.fusion', 'cupy._core.new_fusion', 'cupy._core.raw', 'cupy.cuda.common', 'cupy.cuda.cufft', 'cupy.cuda.device', 'cupy.cuda.memory', 'cupy.cuda.memory_hook', 'cupy.cuda.pinned_memory', 'cupy.cuda.function', 'cupy.cuda.stream', 'cupy.cuda.graph', 'cupy.cuda.texture', 'cupy.fft._cache', 'cupy.fft._callback', 'cupy.lib._polynomial', 'cupy._util']

def get_features(ctx: Context) -> Dict[str, Feature]:
    if False:
        for i in range(10):
            print('nop')
    HIP_cuda_nvtx_cusolver = {'name': 'cuda', 'required': True, 'file': _cuda_files + ['cupy_backends.cuda.libs.nvtx', 'cupy_backends.cuda.libs.cusolver', 'cupyx.cusolver'], 'include': ['hip/hip_runtime_api.h', 'hip/hiprtc.h', 'hipblas.h', 'hiprand/hiprand.h', 'hipsparse.h', 'hipfft.h', 'roctx.h', 'rocsolver.h'], 'libraries': ['amdhip64', 'hipblas', ('hipfft', lambda hip_version: hip_version >= 401), 'hiprand', 'hipsparse', 'rocfft', 'roctx64', 'rocblas', 'rocsolver', 'rocsparse'], 'check_method': build.check_hip_version, 'version_method': build.get_hip_version}
    CUDA_cusolver = {'name': 'cusolver', 'required': True, 'file': ['cupy_backends.cuda.libs.cusolver', 'cupyx.cusolver'], 'include': ['cusolverDn.h'], 'libraries': ['cusolver']}
    CUDA_cudnn = {'name': 'cudnn', 'file': ['cupy_backends.cuda.libs.cudnn', 'cupyx.cudnn'], 'include': ['cudnn.h'], 'libraries': ['cudnn'], 'check_method': build.check_cudnn_version, 'version_method': build.get_cudnn_version}
    CUDA_nccl = {'name': 'nccl', 'file': ['cupy_backends.cuda.libs.nccl'], 'include': ['nccl.h'], 'libraries': ['nccl'], 'check_method': build.check_nccl_version, 'version_method': build.get_nccl_version}
    CUDA_nvtx = {'name': 'nvtx', 'file': ['cupy_backends.cuda.libs.nvtx'], 'include': ['nvtx3/nvToolsExt.h'], 'libraries': [], 'check_method': build.check_nvtx}
    CUDA_cutensor = {'name': 'cutensor', 'file': ['cupy_backends.cuda.libs.cutensor', 'cupyx.cutensor'], 'include': ['cutensor.h'], 'libraries': ['cutensor', 'cublas'], 'check_method': build.check_cutensor_version, 'version_method': build.get_cutensor_version}
    CUDA_cub = {'name': 'cub', 'required': True, 'file': [('cupy.cuda.cub', ['cupy/cuda/cupy_cub.cu'])], 'include': ['cub/util_namespace.cuh'], 'libraries': ['pthread', 'rt', 'dl'] if sys.platform == 'linux' else [], 'static_libraries': ['cudart_static'], 'check_method': build.check_cub_version, 'version_method': build.get_cub_version}
    CUDA_jitify = {'name': 'jitify', 'required': True, 'file': ['cupy.cuda.jitify'], 'include': ['cuda.h', 'cuda_runtime.h', 'nvrtc.h'], 'libraries': ['cuda', 'cudart', 'nvrtc'], 'check_method': build.check_jitify_version, 'version_method': build.get_jitify_version}
    CUDA_random = {'name': 'random', 'required': True, 'file': ['cupy.random._bit_generator', ('cupy.random._generator_api', ['cupy/random/cupy_distributions.cu'])], 'include': [], 'libraries': ['cudart', 'curand']}
    HIP_random = {'name': 'random', 'required': True, 'file': ['cupy.random._bit_generator', ('cupy.random._generator_api', ['cupy/random/cupy_distributions.cu'])], 'include': ['hiprand/hiprand.h'], 'libraries': ['amdhip64', 'hiprand'], 'check_method': build.check_hip_version, 'version_method': build.get_hip_version}
    CUDA_cusparselt = {'name': 'cusparselt', 'file': ['cupy_backends.cuda.libs.cusparselt'], 'include': ['cusparseLt.h'], 'libraries': ['cusparseLt'], 'check_method': build.check_cusparselt_version, 'version_method': build.get_cusparselt_version}
    HIP_cub = {'name': 'cub', 'required': True, 'file': [('cupy.cuda.cub', ['cupy/cuda/cupy_cub.cu'])], 'include': ['hipcub/hipcub_version.hpp'], 'libraries': ['amdhip64'], 'check_method': build.check_cub_version, 'version_method': build.get_cub_version}
    HIP_nccl = {'name': 'nccl', 'file': ['cupy_backends.cuda.libs.nccl'], 'include': ['rccl.h'], 'libraries': ['rccl'], 'check_method': build.check_nccl_version, 'version_method': build.get_nccl_version}
    HIP_thrust = {'name': 'thrust', 'required': True, 'file': [('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu'])], 'include': ['thrust/version.h'], 'libraries': ['amdhip64']}
    CUDA_thrust = {'name': 'thrust', 'required': True, 'file': [('cupy.cuda.thrust', ['cupy/cuda/cupy_thrust.cu'])], 'include': ['thrust/device_ptr.h', 'thrust/sequence.h', 'thrust/sort.h'], 'libraries': ['cudart'], 'check_method': build.check_thrust_version, 'version_method': build.get_thrust_version}
    COMMON_dlpack = {'name': 'dlpack', 'required': True, 'file': ['cupy._core.dlpack'], 'include': ['cupy/_dlpack/dlpack.h'], 'libraries': []}
    if ctx.use_hip:
        features = [_from_dict(HIP_cuda_nvtx_cusolver, ctx), _from_dict(HIP_cub, ctx), _from_dict(HIP_nccl, ctx), _from_dict(HIP_random, ctx), _from_dict(HIP_thrust, ctx), _from_dict(COMMON_dlpack, ctx)]
    else:
        features = [CUDA_cuda(ctx), _from_dict(CUDA_cusolver, ctx), _from_dict(CUDA_cudnn, ctx), _from_dict(CUDA_nccl, ctx), _from_dict(CUDA_nvtx, ctx), _from_dict(CUDA_cutensor, ctx), _from_dict(CUDA_cub, ctx), _from_dict(CUDA_jitify, ctx), _from_dict(CUDA_random, ctx), _from_dict(CUDA_thrust, ctx), _from_dict(CUDA_cusparselt, ctx), _from_dict(COMMON_dlpack, ctx)]
    return {f.name: f for f in features}

class CUDA_cuda(Feature):
    minimum_cuda_version = 10020

    def __init__(self, ctx: Context):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(ctx)
        self.name = 'cuda'
        self.required = True
        self.modules = _cuda_files
        self.includes = ['cublas_v2.h', 'cuda.h', 'cuda_profiler_api.h', 'cuda_runtime.h', 'cufft.h', 'curand.h', 'cusparse.h']
        self.libraries = (['cudart'] if ctx.use_cuda_python else ['cuda', 'cudart']) + ['cublas', 'cufft', 'curand', 'cusparse']
        self._version = self._UNDETERMINED

    def configure(self, compiler: Any, settings: Any) -> bool:
        if False:
            print('Hello World!')
        try:
            out = build.build_and_run(compiler, '\n            #include <cuda.h>\n            #include <stdio.h>\n            int main() {\n              printf("%d", CUDA_VERSION);\n              return 0;\n            }\n            ', include_dirs=settings['include_dirs'])
        except Exception as e:
            utils.print_warning('Cannot check CUDA version', str(e))
            return False
        self._version = int(out)
        if self._version < self.minimum_cuda_version:
            utils.print_warning('CUDA version is too old: %d' % self._version, 'CUDA 10.2 or newer is required')
            return False
        return True