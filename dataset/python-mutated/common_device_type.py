import copy
import gc
import inspect
import runpy
import sys
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps, partial
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, Dict, Set
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, skipCUDANonDefaultStreamIf, TEST_WITH_ASAN, TEST_WITH_UBSAN, TEST_WITH_TSAN, IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, IS_WINDOWS, TEST_MPS, _TestParametrizer, compose_parametrize_fns, dtype_name, TEST_WITH_MIOPEN_SUGGEST_NHWC, NATIVE_DEVICES, skipIfTorchDynamo
from torch.testing._internal.common_cuda import _get_torch_cuda_version, TEST_CUSPARSE_GENERIC, TEST_HIPSPARSE_GENERIC, _get_torch_rocm_version
from torch.testing._internal.common_dtype import get_all_dtypes
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def _dtype_test_suffix(dtypes):
    if False:
        return 10
    ' Returns the test suffix for a dtype, sequence of dtypes, or None. '
    if isinstance(dtypes, (list, tuple)):
        if len(dtypes) == 0:
            return ''
        return '_' + '_'.join((dtype_name(d) for d in dtypes))
    elif dtypes:
        return f'_{dtype_name(dtypes)}'
    else:
        return ''

def _update_param_kwargs(param_kwargs, name, value):
    if False:
        print('Hello World!')
    ' Adds a kwarg with the specified name and value to the param_kwargs dict. '
    plural_name = f'{name}s'
    if name in param_kwargs:
        del param_kwargs[name]
    if plural_name in param_kwargs:
        del param_kwargs[plural_name]
    if isinstance(value, (list, tuple)):
        param_kwargs[plural_name] = value
    elif value is not None:
        param_kwargs[name] = value

class DeviceTypeTestBase(TestCase):
    device_type: str = 'generic_device_type'
    _stop_test_suite = False
    _tls = threading.local()
    _tls.precision = TestCase._precision
    _tls.rel_tol = TestCase._rel_tol

    @property
    def precision(self):
        if False:
            while True:
                i = 10
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        if False:
            print('Hello World!')
        self._tls.precision = prec

    @property
    def rel_tol(self):
        if False:
            print('Hello World!')
        return self._tls.rel_tol

    @rel_tol.setter
    def rel_tol(self, prec):
        if False:
            return 10
        self._tls.rel_tol = prec

    @classmethod
    def get_primary_device(cls):
        if False:
            return 10
        return cls.device_type

    @classmethod
    def _init_and_get_primary_device(cls):
        if False:
            return 10
        try:
            return cls.get_primary_device()
        except Exception:
            if hasattr(cls, 'setUpClass'):
                cls.setUpClass()
            return cls.get_primary_device()

    @classmethod
    def get_all_devices(cls):
        if False:
            while True:
                i = 10
        return [cls.get_primary_device()]

    @classmethod
    def _get_dtypes(cls, test):
        if False:
            while True:
                i = 10
        if not hasattr(test, 'dtypes'):
            return None
        default_dtypes = test.dtypes.get('all')
        msg = f"@dtypes is mandatory when using @dtypesIf however '{test.__name__}' didn't specify it"
        assert default_dtypes is not None, msg
        return test.dtypes.get(cls.device_type, default_dtypes)

    def _get_precision_override(self, test, dtype):
        if False:
            i = 10
            return i + 15
        if not hasattr(test, 'precision_overrides'):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    def _get_tolerance_override(self, test, dtype):
        if False:
            while True:
                i = 10
        if not hasattr(test, 'tolerance_overrides'):
            return (self.precision, self.rel_tol)
        return test.tolerance_overrides.get(dtype, tol(self.precision, self.rel_tol))

    def _apply_precision_override_for_test(self, test, param_kwargs):
        if False:
            print('Hello World!')
        dtype = param_kwargs['dtype'] if 'dtype' in param_kwargs else None
        dtype = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else dtype
        if dtype:
            self.precision = self._get_precision_override(test, dtype)
            (self.precision, self.rel_tol) = self._get_tolerance_override(test, dtype)

    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        if False:
            return 10

        def instantiate_test_helper(cls, name, *, test, param_kwargs=None, decorator_fn=lambda _: []):
            if False:
                print('Hello World!')
            param_kwargs = {} if param_kwargs is None else param_kwargs
            test_sig_params = inspect.signature(test).parameters
            if 'device' in test_sig_params or 'devices' in test_sig_params:
                device_arg: str = cls._init_and_get_primary_device()
                if hasattr(test, 'num_required_devices'):
                    device_arg = cls.get_all_devices()
                _update_param_kwargs(param_kwargs, 'device', device_arg)
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                guard_precision = self.precision
                guard_rel_tol = self.rel_tol
                try:
                    self._apply_precision_override_for_test(test, param_kwargs)
                    result = test(self, **param_kwargs)
                except RuntimeError as rte:
                    self._stop_test_suite = self._should_stop_test_suite()
                    if getattr(test, '__unittest_expecting_failure__', False) and self._stop_test_suite:
                        import sys
                        print('Suppressing fatal exception to trigger unexpected success', file=sys.stderr)
                        return
                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol
                return result
            assert not hasattr(cls, name), f'Redefinition of test {name}'
            setattr(cls, name, instantiated_test)

        def default_parametrize_fn(test, generic_cls, device_cls):
            if False:
                print('Hello World!')
            yield (test, '', {}, lambda _: [])
        parametrize_fn = getattr(test, 'parametrize_fn', default_parametrize_fn)
        dtypes = cls._get_dtypes(test)
        if dtypes is not None:

            def dtype_parametrize_fn(test, generic_cls, device_cls, dtypes=dtypes):
                if False:
                    return 10
                for dtype in dtypes:
                    param_kwargs: Dict[str, Any] = {}
                    _update_param_kwargs(param_kwargs, 'dtype', dtype)
                    yield (test, '', param_kwargs, lambda _: [])
            parametrize_fn = compose_parametrize_fns(dtype_parametrize_fn, parametrize_fn)
        for (test, test_suffix, param_kwargs, decorator_fn) in parametrize_fn(test, generic_cls, cls):
            test_suffix = '' if test_suffix == '' else '_' + test_suffix
            device_suffix = '_' + cls.device_type
            dtype_kwarg = None
            if 'dtype' in param_kwargs or 'dtypes' in param_kwargs:
                dtype_kwarg = param_kwargs['dtypes'] if 'dtypes' in param_kwargs else param_kwargs['dtype']
            test_name = f'{name}{test_suffix}{device_suffix}{_dtype_test_suffix(dtype_kwarg)}'
            instantiate_test_helper(cls=cls, name=test_name, test=test, param_kwargs=param_kwargs, decorator_fn=decorator_fn)

    def run(self, result=None):
        if False:
            return 10
        super().run(result=result)
        if self._stop_test_suite:
            result.stop()

class CPUTestBase(DeviceTypeTestBase):
    device_type = 'cpu'

    def _should_stop_test_suite(self):
        if False:
            for i in range(10):
                print('nop')
        return False

class CUDATestBase(DeviceTypeTestBase):
    device_type = 'cuda'
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]

    def has_cudnn(self):
        if False:
            print('Hello World!')
        return not self.no_cudnn

    @classmethod
    def get_primary_device(cls):
        if False:
            while True:
                i = 10
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        if False:
            i = 10
            return i + 15
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.cuda.device_count()
        prim_device = cls.get_primary_device()
        cuda_str = 'cuda:{0}'
        non_primary_devices = [cuda_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma
        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()
        cls.primary_device = f'cuda:{torch.cuda.current_device()}'
lazy_ts_backend_init = False

class LazyTestBase(DeviceTypeTestBase):
    device_type = 'lazy'

    def _should_stop_test_suite(self):
        if False:
            print('Hello World!')
        return False

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        import torch._lazy
        import torch._lazy.metrics
        import torch._lazy.ts_backend
        global lazy_ts_backend_init
        if not lazy_ts_backend_init:
            torch._lazy.ts_backend.init()
            lazy_ts_backend_init = True

class MPSTestBase(DeviceTypeTestBase):
    device_type = 'mps'
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        if False:
            print('Hello World!')
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        if False:
            return 10
        prim_device = cls.get_primary_device()
        return [prim_device]

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.primary_device = 'mps:0'

    def _should_stop_test_suite(self):
        if False:
            i = 10
            return i + 15
        return False

class PrivateUse1TestBase(DeviceTypeTestBase):
    primary_device: ClassVar[str]
    device_mod = None
    device_type = 'privateuse1'

    @classmethod
    def get_primary_device(cls):
        if False:
            while True:
                i = 10
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        if False:
            for i in range(10):
                print('nop')
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = cls.device_mod.device_count()
        prim_device = cls.get_primary_device()
        device_str = f'{cls.device_type}:{{0}}'
        non_primary_devices = [device_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.device_type = torch._C._get_privateuse1_backend_name()
        cls.device_mod = getattr(torch, cls.device_type, None)
        assert cls.device_mod is not None, f'torch has no module of `{cls.device_type}`, you should register\n                                            a module by `torch._register_device_module`.'
        cls.primary_device = f'{cls.device_type}:{cls.device_mod.current_device()}'

def get_device_type_test_bases():
    if False:
        print('Hello World!')
    test_bases: List[Any] = list()
    if IS_SANDCASTLE or IS_FBCODE:
        if IS_REMOTE_GPU:
            if not TEST_WITH_ASAN and (not TEST_WITH_TSAN) and (not TEST_WITH_UBSAN):
                test_bases.append(CUDATestBase)
        else:
            test_bases.append(CPUTestBase)
    else:
        test_bases.append(CPUTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)
        device_type = torch._C._get_privateuse1_backend_name()
        device_mod = getattr(torch, device_type, None)
        if hasattr(device_mod, 'is_available') and device_mod.is_available():
            test_bases.append(PrivateUse1TestBase)
    return test_bases
device_type_test_bases = get_device_type_test_bases()

def filter_desired_device_types(device_type_test_bases, except_for=None, only_for=None):
    if False:
        print('Hello World!')
    intersect = set(except_for if except_for else []) & set(only_for if only_for else [])
    assert not intersect, f'device ({intersect}) appeared in both except_for and only_for'
    if except_for:
        device_type_test_bases = filter(lambda x: x.device_type not in except_for, device_type_test_bases)
    if only_for:
        device_type_test_bases = filter(lambda x: x.device_type in only_for, device_type_test_bases)
    return list(device_type_test_bases)
_TORCH_TEST_DEVICES = os.environ.get('TORCH_TEST_DEVICES', None)
if _TORCH_TEST_DEVICES:
    for path in _TORCH_TEST_DEVICES.split(':'):
        mod = runpy.run_path(path, init_globals=globals())
        device_type_test_bases.append(mod['TEST_CLASS'])
PYTORCH_CUDA_MEMCHECK = os.getenv('PYTORCH_CUDA_MEMCHECK', '0') == '1'
PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY = 'PYTORCH_TESTING_DEVICE_ONLY_FOR'
PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY = 'PYTORCH_TESTING_DEVICE_EXCEPT_FOR'

def instantiate_device_type_tests(generic_test_class, scope, except_for=None, only_for=None, include_lazy=False, allow_mps=False):
    if False:
        while True:
            i = 10
    del scope[generic_test_class.__name__]
    empty_name = generic_test_class.__name__ + '_base'
    empty_class = type(empty_name, generic_test_class.__bases__, {})
    generic_members = set(generic_test_class.__dict__.keys()) - set(empty_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith('test')]
    test_bases = device_type_test_bases.copy()
    if allow_mps and TEST_MPS and (MPSTestBase not in test_bases):
        test_bases.append(MPSTestBase)
    desired_device_type_test_bases = filter_desired_device_types(test_bases, except_for, only_for)
    if include_lazy:
        if IS_FBCODE:
            print('TorchScript backend not yet supported in FBCODE/OVRSOURCE builds', file=sys.stderr)
        else:
            desired_device_type_test_bases.append(LazyTestBase)

    def split_if_not_empty(x: str):
        if False:
            return 10
        return x.split(',') if len(x) != 0 else []
    env_only_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, ''))
    env_except_for = split_if_not_empty(os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, ''))
    desired_device_type_test_bases = filter_desired_device_types(desired_device_type_test_bases, env_except_for, env_only_for)
    for base in desired_device_type_test_bases:
        class_name = generic_test_class.__name__ + base.device_type.upper()
        device_type_test_class: Any = type(class_name, (base, empty_class), {})
        for name in generic_members:
            if name in generic_tests:
                test = getattr(generic_test_class, name)
                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test), generic_cls=generic_test_class)
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            else:
                assert name not in device_type_test_class.__dict__, f'Redefinition of directly defined member {name}'
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class

class OpDTypes(Enum):
    supported = 0
    unsupported = 1
    supported_backward = 2
    unsupported_backward = 3
    any_one = 4
    none = 5
    any_common_cpu_cuda_one = 6
ANY_DTYPE_ORDER = (torch.float32, torch.float64, torch.complex64, torch.complex128, torch.float16, torch.bfloat16, torch.long, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)

class ops(_TestParametrizer):

    def __init__(self, op_list, *, dtypes: Union[OpDTypes, Sequence[torch.dtype]]=OpDTypes.supported, allowed_dtypes: Optional[Sequence[torch.dtype]]=None):
        if False:
            while True:
                i = 10
        self.op_list = list(op_list)
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None

    def _parametrize_test(self, test, generic_cls, device_cls):
        if False:
            for i in range(10):
                print('nop')
        ' Parameterizes the given test function across each op and its associated dtypes. '
        if device_cls is None:
            raise RuntimeError('The @ops decorator is only intended to be used in a device-specific context; use it with instantiate_device_type_tests() instead of instantiate_parametrized_tests()')
        op = check_exhausted_iterator = object()
        for op in self.op_list:
            dtypes: Union[Set[torch.dtype], Set[None]]
            if isinstance(self.opinfo_dtypes, Sequence):
                dtypes = set(self.opinfo_dtypes)
            elif self.opinfo_dtypes == OpDTypes.unsupported_backward:
                dtypes = set(get_all_dtypes()).difference(op.supported_backward_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.supported_backward:
                dtypes = op.supported_backward_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.unsupported:
                dtypes = set(get_all_dtypes()).difference(op.supported_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.supported:
                dtypes = op.supported_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.any_one:
                supported = op.supported_dtypes(device_cls.device_type)
                supported_backward = op.supported_backward_dtypes(device_cls.device_type)
                supported_both = supported.intersection(supported_backward)
                dtype_set = supported_both if len(supported_both) > 0 else supported
                for dtype in ANY_DTYPE_ORDER:
                    if dtype in dtype_set:
                        dtypes = {dtype}
                        break
                else:
                    dtypes = {}
            elif self.opinfo_dtypes == OpDTypes.any_common_cpu_cuda_one:
                supported = op.dtypes.intersection(op.dtypesIfCUDA)
                if supported:
                    dtypes = {next((dtype for dtype in ANY_DTYPE_ORDER if dtype in supported))}
                else:
                    dtypes = {}
            elif self.opinfo_dtypes == OpDTypes.none:
                dtypes = {None}
            else:
                raise RuntimeError(f'Unknown OpDType: {self.opinfo_dtypes}')
            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)
            test_name = op.formatted_name
            for dtype in dtypes:
                param_kwargs = {'op': op}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)
                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        if False:
                            for i in range(10):
                                print('nop')
                        return test(*args, **kwargs)
                    decorator_fn = partial(op.get_decorators, generic_cls.__name__, test.__name__, device_cls.device_type, dtype)
                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    print(f'Failed to instantiate {test_name} for op {op.name}!')
                    raise ex
        if op is check_exhausted_iterator:
            raise ValueError('An empty op_list was passed to @ops. Note that this may result from reuse of a generator.')

class skipIf:

    def __init__(self, dep, reason, device_type=None):
        if False:
            i = 10
            return i + 15
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):
        if False:
            i = 10
            return i + 15

        @wraps(fn)
        def dep_fn(slf, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if self.device_type is None or self.device_type == slf.device_type:
                if isinstance(self.dep, str) and getattr(slf, self.dep, True) or (isinstance(self.dep, bool) and self.dep):
                    raise unittest.SkipTest(self.reason)
            return fn(slf, *args, **kwargs)
        return dep_fn

class skipCPUIf(skipIf):

    def __init__(self, dep, reason):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(dep, reason, device_type='cpu')

class skipCUDAIf(skipIf):

    def __init__(self, dep, reason):
        if False:
            return 10
        super().__init__(dep, reason, device_type='cuda')

class skipLazyIf(skipIf):

    def __init__(self, dep, reason):
        if False:
            return 10
        super().__init__(dep, reason, device_type='lazy')

class skipMetaIf(skipIf):

    def __init__(self, dep, reason):
        if False:
            print('Hello World!')
        super().__init__(dep, reason, device_type='meta')

class skipMPSIf(skipIf):

    def __init__(self, dep, reason):
        if False:
            return 10
        super().__init__(dep, reason, device_type='mps')

class skipXLAIf(skipIf):

    def __init__(self, dep, reason):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(dep, reason, device_type='xla')

class skipPRIVATEUSE1If(skipIf):

    def __init__(self, dep, reason):
        if False:
            i = 10
            return i + 15
        device_type = torch._C._get_privateuse1_backend_name()
        super().__init__(dep, reason, device_type=device_type)

def _has_sufficient_memory(device, size):
    if False:
        while True:
            i = 10
    if torch.device(device).type == 'cuda':
        if not torch.cuda.is_available():
            return False
        gc.collect()
        torch.cuda.empty_cache()
        if device == 'cuda':
            device = 'cuda:0'
        return torch.cuda.memory.mem_get_info(device)[0] >= size
    if device == 'xla':
        raise unittest.SkipTest('TODO: Memory availability checks for XLA?')
    if device != 'cpu':
        raise unittest.SkipTest('Unknown device type')
    if not HAS_PSUTIL:
        raise unittest.SkipTest('Need psutil to determine if memory is sufficient')
    if TEST_WITH_ASAN or TEST_WITH_TSAN or TEST_WITH_UBSAN:
        effective_size = size * 10
    else:
        effective_size = size
    if psutil.virtual_memory().available < effective_size:
        gc.collect()
    return psutil.virtual_memory().available >= effective_size

def largeTensorTest(size, device=None):
    if False:
        for i in range(10):
            print('nop')
    'Skip test if the device has insufficient memory to run the test\n\n    size may be a number of bytes, a string of the form "N GB", or a callable\n\n    If the test is a device generic test, available memory on the primary device will be checked.\n    It can also be overriden by the optional `device=` argument.\n    In other tests, the `device=` argument needs to be specified.\n    '
    if isinstance(size, str):
        assert size.endswith(('GB', 'gb')), 'only bytes or GB supported'
        size = 1024 ** 3 * int(size[:-2])

    def inner(fn):
        if False:
            while True:
                i = 10

        @wraps(fn)
        def dep_fn(self, *args, **kwargs):
            if False:
                return 10
            size_bytes = size(self, *args, **kwargs) if callable(size) else size
            _device = device if device is not None else self.get_primary_device()
            if not _has_sufficient_memory(_device, size_bytes):
                raise unittest.SkipTest(f'Insufficient {_device} memory')
            return fn(self, *args, **kwargs)
        return dep_fn
    return inner

class expectedFailure:

    def __init__(self, device_type):
        if False:
            for i in range(10):
                print('nop')
        self.device_type = device_type

    def __call__(self, fn):
        if False:
            print('Hello World!')

        @wraps(fn)
        def efail_fn(slf, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if self.device_type is None or self.device_type == slf.device_type:
                try:
                    fn(slf, *args, **kwargs)
                except Exception:
                    return
                else:
                    slf.fail('expected test to fail, but it passed')
            return fn(slf, *args, **kwargs)
        return efail_fn

class onlyOn:

    def __init__(self, device_type):
        if False:
            while True:
                i = 10
        self.device_type = device_type

    def __call__(self, fn):
        if False:
            for i in range(10):
                print('nop')

        @wraps(fn)
        def only_fn(slf, *args, **kwargs):
            if False:
                print('Hello World!')
            if self.device_type != slf.device_type:
                reason = f'Only runs on {self.device_type}'
                raise unittest.SkipTest(reason)
            return fn(slf, *args, **kwargs)
        return only_fn

class deviceCountAtLeast:

    def __init__(self, num_required_devices):
        if False:
            i = 10
            return i + 15
        self.num_required_devices = num_required_devices

    def __call__(self, fn):
        if False:
            return 10
        assert not hasattr(fn, 'num_required_devices'), f'deviceCountAtLeast redefinition for {fn.__name__}'
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if len(devices) < self.num_required_devices:
                reason = f'fewer than {self.num_required_devices} devices detected'
                raise unittest.SkipTest(reason)
            return fn(slf, devices, *args, **kwargs)
        return multi_fn

def onlyNativeDeviceTypes(fn):
    if False:
        for i in range(10):
            print('nop')

    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.device_type not in NATIVE_DEVICES:
            reason = f"onlyNativeDeviceTypes: doesn't run on {self.device_type}"
            raise unittest.SkipTest(reason)
        return fn(self, *args, **kwargs)
    return only_fn

class precisionOverride:

    def __init__(self, d):
        if False:
            return 10
        assert isinstance(d, dict), 'precisionOverride not given a dtype : precision dict!'
        for dtype in d.keys():
            assert isinstance(dtype, torch.dtype), f'precisionOverride given unknown dtype {dtype}'
        self.d = d

    def __call__(self, fn):
        if False:
            print('Hello World!')
        fn.precision_overrides = self.d
        return fn
tol = namedtuple('tol', ['atol', 'rtol'])

class toleranceOverride:

    def __init__(self, d):
        if False:
            while True:
                i = 10
        assert isinstance(d, dict), 'toleranceOverride not given a dtype : tol dict!'
        for (dtype, prec) in d.items():
            assert isinstance(dtype, torch.dtype), f'toleranceOverride given unknown dtype {dtype}'
            assert isinstance(prec, tol), 'toleranceOverride not given a dtype : tol dict!'
        self.d = d

    def __call__(self, fn):
        if False:
            while True:
                i = 10
        fn.tolerance_overrides = self.d
        return fn

class dtypes:

    def __init__(self, *args, device_type='all'):
        if False:
            return 10
        if len(args) > 0 and isinstance(args[0], (list, tuple)):
            for arg in args:
                assert isinstance(arg, (list, tuple)), f'When one dtype variant is a tuple or list, all dtype variants must be. Received non-list non-tuple dtype {str(arg)}'
                assert all((isinstance(dtype, torch.dtype) for dtype in arg)), f'Unknown dtype in {str(arg)}'
        else:
            assert all((isinstance(arg, torch.dtype) for arg in args)), f'Unknown dtype in {str(args)}'
        self.args = args
        self.device_type = device_type

    def __call__(self, fn):
        if False:
            for i in range(10):
                print('nop')
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, f'dtypes redefinition for {self.device_type}'
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn

class dtypesIfCPU(dtypes):

    def __init__(self, *args):
        if False:
            return 10
        super().__init__(*args, device_type='cpu')

class dtypesIfCUDA(dtypes):

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, device_type='cuda')

class dtypesIfMPS(dtypes):

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        super().__init__(*args, device_type='mps')

class dtypesIfPRIVATEUSE1(dtypes):

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, device_type=torch._C._get_privateuse1_backend_name())

def onlyCPU(fn):
    if False:
        i = 10
        return i + 15
    return onlyOn('cpu')(fn)

def onlyCUDA(fn):
    if False:
        for i in range(10):
            print('nop')
    return onlyOn('cuda')(fn)

def onlyMPS(fn):
    if False:
        return 10
    return onlyOn('mps')(fn)

def onlyPRIVATEUSE1(fn):
    if False:
        print('Hello World!')
    device_type = torch._C._get_privateuse1_backend_name()
    device_mod = getattr(torch, device_type, None)
    if device_mod is None:
        reason = f'Skip as torch has no module of {device_type}'
        return unittest.skip(reason)(fn)
    return onlyOn(device_type)(fn)

def onlyCUDAAndPRIVATEUSE1(fn):
    if False:
        for i in range(10):
            print('nop')

    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.device_type not in ('cuda', torch._C._get_privateuse1_backend_name()):
            reason = f"onlyCUDAAndPRIVATEUSE1: doesn't run on {self.device_type}"
            raise unittest.SkipTest(reason)
        return fn(self, *args, **kwargs)
    return only_fn

def disablecuDNN(fn):
    if False:
        while True:
            i = 10

    @wraps(fn)
    def disable_cudnn(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        return fn(self, *args, **kwargs)
    return disable_cudnn

def disableMkldnn(fn):
    if False:
        return 10

    @wraps(fn)
    def disable_mkldnn(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if torch.backends.mkldnn.is_available():
            with torch.backends.mkldnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        return fn(self, *args, **kwargs)
    return disable_mkldnn

def expectedFailureCUDA(fn):
    if False:
        return 10
    return expectedFailure('cuda')(fn)

def expectedFailureMeta(fn):
    if False:
        print('Hello World!')
    return skipIfTorchDynamo()(expectedFailure('meta')(fn))

def expectedFailureXLA(fn):
    if False:
        return 10
    return expectedFailure('xla')(fn)

def skipCPUIfNoLapack(fn):
    if False:
        return 10
    return skipCPUIf(not torch._C.has_lapack, 'PyTorch compiled without Lapack')(fn)

def skipCPUIfNoFFT(fn):
    if False:
        for i in range(10):
            print('nop')
    return skipCPUIf(not torch._C.has_spectral, 'PyTorch is built without FFT support')(fn)

def skipCPUIfNoMkl(fn):
    if False:
        i = 10
        return i + 15
    return skipCPUIf(not TEST_MKL, 'PyTorch is built without MKL support')(fn)

def skipCPUIfNoMklSparse(fn):
    if False:
        for i in range(10):
            print('nop')
    return skipCPUIf(IS_WINDOWS or not TEST_MKL, 'PyTorch is built without MKL support')(fn)

def skipCPUIfNoMkldnn(fn):
    if False:
        return 10
    return skipCPUIf(not torch.backends.mkldnn.is_available(), 'PyTorch is built without mkldnn support')(fn)

def skipCUDAIfNoMagma(fn):
    if False:
        i = 10
        return i + 15
    return skipCUDAIf('no_magma', 'no MAGMA library detected')(skipCUDANonDefaultStreamIf(True)(fn))

def has_cusolver():
    if False:
        print('Hello World!')
    return not TEST_WITH_ROCM

def has_hipsolver():
    if False:
        while True:
            i = 10
    rocm_version = _get_torch_rocm_version()
    return rocm_version >= (5, 3)

def skipCUDAIfNoCusolver(fn):
    if False:
        return 10
    return skipCUDAIf(not has_cusolver() and (not has_hipsolver()), 'cuSOLVER not available')(fn)

def skipCUDAIfNoMagmaAndNoCusolver(fn):
    if False:
        print('Hello World!')
    if has_cusolver():
        return fn
    else:
        return skipCUDAIfNoMagma(fn)

def skipCUDAIfNoMagmaAndNoLinalgsolver(fn):
    if False:
        print('Hello World!')
    if has_cusolver() or has_hipsolver():
        return fn
    else:
        return skipCUDAIfNoMagma(fn)

def skipCUDAIfRocm(func=None, *, msg="test doesn't currently work on the ROCm stack"):
    if False:
        i = 10
        return i + 15

    def dec_fn(fn):
        if False:
            i = 10
            return i + 15
        reason = f'skipCUDAIfRocm: {msg}'
        return skipCUDAIf(TEST_WITH_ROCM, reason=reason)(fn)
    if func:
        return dec_fn(func)
    return dec_fn

def skipCUDAIfNotRocm(fn):
    if False:
        while True:
            i = 10
    return skipCUDAIf(not TEST_WITH_ROCM, "test doesn't currently work on the CUDA stack")(fn)

def skipCUDAIfRocmVersionLessThan(version=None):
    if False:
        for i in range(10):
            print('nop')

    def dec_fn(fn):
        if False:
            for i in range(10):
                print('nop')

        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if self.device_type == 'cuda':
                if not TEST_WITH_ROCM:
                    reason = 'ROCm not available'
                    raise unittest.SkipTest(reason)
                rocm_version_tuple = _get_torch_rocm_version()
                if rocm_version_tuple is None or version is None or rocm_version_tuple < tuple(version):
                    reason = f'ROCm {rocm_version_tuple} is available but {version} required'
                    raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn

def skipCUDAIfNotMiopenSuggestNHWC(fn):
    if False:
        print('Hello World!')
    return skipCUDAIf(not TEST_WITH_MIOPEN_SUGGEST_NHWC, "test doesn't currently work without MIOpen NHWC activation")(fn)

def skipCUDAVersionIn(versions: List[Tuple[int, int]]=None):
    if False:
        print('Hello World!')

    def dec_fn(fn):
        if False:
            while True:
                i = 10

        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            version = _get_torch_cuda_version()
            if version == (0, 0):
                return fn(self, *args, **kwargs)
            if version in (versions or []):
                reason = f'test skipped for CUDA version {version}'
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn

def skipCUDAIfVersionLessThan(versions: Tuple[int, int]=None):
    if False:
        print('Hello World!')

    def dec_fn(fn):
        if False:
            i = 10
            return i + 15

        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if False:
                print('Hello World!')
            version = _get_torch_cuda_version()
            if version == (0, 0):
                return fn(self, *args, **kwargs)
            if version < versions:
                reason = f'test skipped for CUDA versions < {version}'
                raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn

def skipCUDAIfCudnnVersionLessThan(version=0):
    if False:
        while True:
            i = 10

    def dec_fn(fn):
        if False:
            for i in range(10):
                print('nop')

        @wraps(fn)
        def wrap_fn(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            if self.device_type == 'cuda':
                if self.no_cudnn:
                    reason = 'cuDNN not available'
                    raise unittest.SkipTest(reason)
                if self.cudnn_version is None or self.cudnn_version < version:
                    reason = f'cuDNN version {self.cudnn_version} is available but {version} required'
                    raise unittest.SkipTest(reason)
            return fn(self, *args, **kwargs)
        return wrap_fn
    return dec_fn

def skipCUDAIfNoCusparseGeneric(fn):
    if False:
        i = 10
        return i + 15
    return skipCUDAIf(not TEST_CUSPARSE_GENERIC, 'cuSparse Generic API not available')(fn)

def skipCUDAIfNoHipsparseGeneric(fn):
    if False:
        print('Hello World!')
    return skipCUDAIf(not TEST_HIPSPARSE_GENERIC, 'hipSparse Generic API not available')(fn)

def skipCUDAIfNoSparseGeneric(fn):
    if False:
        return 10
    return skipCUDAIf(not (TEST_CUSPARSE_GENERIC or TEST_HIPSPARSE_GENERIC), 'Sparse Generic API not available')(fn)

def skipCUDAIfNoCudnn(fn):
    if False:
        for i in range(10):
            print('nop')
    return skipCUDAIfCudnnVersionLessThan(0)(fn)

def skipCUDAIfMiopen(fn):
    if False:
        return 10
    return skipCUDAIf(torch.version.hip is not None, 'Marked as skipped for MIOpen')(fn)

def skipCUDAIfNoMiopen(fn):
    if False:
        print('Hello World!')
    return skipCUDAIf(torch.version.hip is None, 'MIOpen is not available')(skipCUDAIfNoCudnn(fn))

def skipLazy(fn):
    if False:
        print('Hello World!')
    return skipLazyIf(True, "test doesn't work with lazy tensors")(fn)

def skipMeta(fn):
    if False:
        for i in range(10):
            print('nop')
    return skipMetaIf(True, "test doesn't work with meta tensors")(fn)

def skipXLA(fn):
    if False:
        print('Hello World!')
    return skipXLAIf(True, 'Marked as skipped for XLA')(fn)

def skipMPS(fn):
    if False:
        print('Hello World!')
    return skipMPSIf(True, "test doesn't work on MPS backend")(fn)

def skipPRIVATEUSE1(fn):
    if False:
        while True:
            i = 10
    return skipPRIVATEUSE1If(True, "test doesn't work on privateuse1 backend")(fn)

def get_all_device_types() -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    return ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']