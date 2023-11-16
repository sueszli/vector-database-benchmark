"""This file is allowed to initialize CUDA context when imported."""
import functools
import torch
import torch.cuda
from torch.testing._internal.common_utils import LazyVal, TEST_NUMBA, TEST_WITH_ROCM, TEST_CUDA, IS_WINDOWS
import inspect
import contextlib
CUDA_ALREADY_INITIALIZED_ON_IMPORT = torch.cuda.is_initialized()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
CUDA_DEVICE = torch.device('cuda:0') if TEST_CUDA else None
if TEST_WITH_ROCM:
    TEST_CUDNN = LazyVal(lambda : TEST_CUDA)
else:
    TEST_CUDNN = LazyVal(lambda : TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.tensor(1.0, device=CUDA_DEVICE)))
TEST_CUDNN_VERSION = LazyVal(lambda : torch.backends.cudnn.version() if TEST_CUDNN else 0)
SM53OrLater = LazyVal(lambda : torch.cuda.is_available() and torch.cuda.get_device_capability() >= (5, 3))
SM60OrLater = LazyVal(lambda : torch.cuda.is_available() and torch.cuda.get_device_capability() >= (6, 0))
SM70OrLater = LazyVal(lambda : torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 0))
SM75OrLater = LazyVal(lambda : torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7, 5))
SM80OrLater = LazyVal(lambda : torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0))
SM90OrLater = LazyVal(lambda : torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0))
PLATFORM_SUPPORTS_FLASH_ATTENTION: bool = LazyVal(lambda : TEST_CUDA and (not TEST_WITH_ROCM) and (not IS_WINDOWS) and SM80OrLater)
PLATFORM_SUPPORTS_MEM_EFF_ATTENTION: bool = LazyVal(lambda : TEST_CUDA and (not TEST_WITH_ROCM))
PLATFORM_SUPPORTS_FUSED_ATTENTION: bool = LazyVal(lambda : PLATFORM_SUPPORTS_FLASH_ATTENTION or PLATFORM_SUPPORTS_MEM_EFF_ATTENTION)
PLATFORM_SUPPORTS_FUSED_SDPA: bool = TEST_CUDA and (not TEST_WITH_ROCM)
if TEST_NUMBA:
    try:
        import numba.cuda
        TEST_NUMBA_CUDA = numba.cuda.is_available()
    except Exception as e:
        TEST_NUMBA_CUDA = False
        TEST_NUMBA = False
else:
    TEST_NUMBA_CUDA = False
__cuda_ctx_rng_initialized = False

def initialize_cuda_context_rng():
    if False:
        for i in range(10):
            print('nop')
    global __cuda_ctx_rng_initialized
    assert TEST_CUDA, 'CUDA must be available when calling initialize_cuda_context_rng'
    if not __cuda_ctx_rng_initialized:
        for i in range(torch.cuda.device_count()):
            torch.randn(1, device=f'cuda:{i}')
        __cuda_ctx_rng_initialized = True

def tf32_is_not_fp32():
    if False:
        print('Hello World!')
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:
        return False
    if int(torch.version.cuda.split('.')[0]) < 11:
        return False
    return True

@contextlib.contextmanager
def tf32_off():
    if False:
        return 10
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul

@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-05):
    if False:
        for i in range(10):
            print('nop')
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_precision = self.precision
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        self.precision = tf32_precision
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, deterministic=None, allow_tf32=True):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul
        self.precision = old_precision

def tf32_on_and_off(tf32_precision=1e-05):
    if False:
        return 10

    def with_tf32_disabled(self, function_call):
        if False:
            return 10
        with tf32_off():
            function_call()

    def with_tf32_enabled(self, function_call):
        if False:
            while True:
                i = 10
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        if False:
            return 10
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if False:
                while True:
                    i = 10
            for (k, v) in zip(arg_names, args):
                kwargs[k] = v
            cond = tf32_is_not_fp32()
            if 'device' in kwargs:
                cond = cond and torch.device(kwargs['device']).type == 'cuda'
            if 'dtype' in kwargs:
                cond = cond and kwargs['dtype'] in {torch.float32, torch.complex64}
            if cond:
                with_tf32_disabled(kwargs['self'], lambda : f(**kwargs))
                with_tf32_enabled(kwargs['self'], lambda : f(**kwargs))
            else:
                f(**kwargs)
        return wrapped
    return wrapper

def with_tf32_off(f):
    if False:
        for i in range(10):
            print('nop')

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        with tf32_off():
            return f(*args, **kwargs)
    return wrapped

def _get_magma_version():
    if False:
        i = 10
        return i + 15
    if 'Magma' not in torch.__config__.show():
        return (0, 0)
    position = torch.__config__.show().find('Magma ')
    version_str = torch.__config__.show()[position + len('Magma '):].split('\n')[0]
    return tuple((int(x) for x in version_str.split('.')))

def _get_torch_cuda_version():
    if False:
        i = 10
        return i + 15
    if torch.version.cuda is None:
        return (0, 0)
    cuda_version = str(torch.version.cuda)
    return tuple((int(x) for x in cuda_version.split('.')))

def _get_torch_rocm_version():
    if False:
        i = 10
        return i + 15
    if not TEST_WITH_ROCM:
        return (0, 0)
    rocm_version = str(torch.version.hip)
    rocm_version = rocm_version.split('-')[0]
    return tuple((int(x) for x in rocm_version.split('.')))

def _check_cusparse_generic_available():
    if False:
        print('Hello World!')
    return not TEST_WITH_ROCM

def _check_hipsparse_generic_available():
    if False:
        i = 10
        return i + 15
    if not TEST_WITH_ROCM:
        return False
    rocm_version = str(torch.version.hip)
    rocm_version = rocm_version.split('-')[0]
    rocm_version_tuple = tuple((int(x) for x in rocm_version.split('.')))
    return not (rocm_version_tuple is None or rocm_version_tuple < (5, 1))
TEST_CUSPARSE_GENERIC = _check_cusparse_generic_available()
TEST_HIPSPARSE_GENERIC = _check_hipsparse_generic_available()

def _create_scaling_models_optimizers(device='cuda', optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    if False:
        while True:
            i = 10
    mod_control = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    mod_scaling = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(device=device)
    with torch.no_grad():
        for (c, s) in zip(mod_control.parameters(), mod_scaling.parameters()):
            s.copy_(c)
    kwargs = {'lr': 1.0}
    if optimizer_kwargs is not None:
        kwargs.update(optimizer_kwargs)
    opt_control = optimizer_ctor(mod_control.parameters(), **kwargs)
    opt_scaling = optimizer_ctor(mod_scaling.parameters(), **kwargs)
    return (mod_control, mod_scaling, opt_control, opt_scaling)

def _create_scaling_case(device='cuda', dtype=torch.float, optimizer_ctor=torch.optim.SGD, optimizer_kwargs=None):
    if False:
        print('Hello World!')
    data = [(torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)), (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)), (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device)), (torch.randn((8, 8), dtype=dtype, device=device), torch.randn((8, 8), dtype=dtype, device=device))]
    loss_fn = torch.nn.MSELoss().cuda()
    skip_iter = 2
    return _create_scaling_models_optimizers(device=device, optimizer_ctor=optimizer_ctor, optimizer_kwargs=optimizer_kwargs) + (data, loss_fn, skip_iter)
if not CUDA_ALREADY_INITIALIZED_ON_IMPORT:
    assert not torch.cuda.is_initialized()