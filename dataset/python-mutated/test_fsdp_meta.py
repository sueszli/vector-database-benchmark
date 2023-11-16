import itertools
import sys
from typing import Union
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import always_wrap_policy as always_wrap, enable_wrap, ModuleWrapPolicy, wrap
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, skip_but_pass_in_sandcastle_if, TEST_WITH_DEV_DBG_ASAN
_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init
except ImportError:
    _TORCHDISTX_AVAIL = False
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

def _reset_params_if_meta(is_meta: bool, model: nn.Module):
    if False:
        for i in range(10):
            print('nop')
    if is_meta:
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

class MyLinear(nn.Linear):
    """
    Linear layer with deterministic reset_parameters for testing.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)

    def reset_parameters(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        torch.manual_seed(42)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.weight, 1.0)

class MyBuffer(nn.Module):

    def __init__(self, device: torch.device):
        if False:
            while True:
                i = 10
        super().__init__()
        self.register_buffer('buf', torch.empty((3, 3), device=device))

    def reset_parameters(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        torch.manual_seed(42)
        torch.nn.init.xavier_uniform_(self.buf, 0.5)

class MyModel(nn.Module):

    def __init__(self, device: torch.device):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.lin1 = MyLinear(2, 2, bias=False, device=device)
        self.lin2 = MyLinear(2, 2, bias=False, device=device)
        self.buf_mod = MyBuffer(device)

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.lin2(self.lin1(x))

class NestedModel(nn.Module):

    def __init__(self, device):
        if False:
            return 10
        super().__init__()
        self.lin1 = MyLinear(2, 2, bias=False, device=device)
        self.lin1 = wrap(self.lin1)
        self.lin2 = MyLinear(2, 2, bias=False, device=device)
        self.l3 = MyModel(device=device)
        self.l3 = wrap(self.l3)

    def forward(self, x):
        if False:
            return 10
        return self.l3(self.lin2(self.lin1(x)))

def _init_with_reset_params(module: nn.Module):
    if False:
        return 10
    '\n    to_empty + reset_parameters() init function example for modules\n    initialized with device="meta"\n    '
    has_meta_states = any((t.is_meta for t in itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))))
    if has_meta_states:
        device = torch.device('cuda', torch.cuda.current_device())
        module.to_empty(device=device, recurse=False)
        module.reset_parameters()

def _init_with_torchdistX(module: nn.Module):
    if False:
        i = 10
        return i + 15
    '\n    torchdistX-based deferred module initialization function example\n    using ``materialize_module``.\n    '
    assert _TORCHDISTX_AVAIL

    def check_fn(k):
        if False:
            return 10
        return not isinstance(k, FSDP)
    deferred_init.materialize_module(module, check_fn=check_fn)

class TestFSDPWithMetaDevice(FSDPTest):

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        return 2

    @property
    def process_group(self):
        if False:
            print('Hello World!')
        return dist.distributed_c10d._get_default_group()

    def _compare_fsdp(self, fsdp1, fsdp2):
        if False:
            for i in range(10):
                print('nop')
        with FSDP.summon_full_params(fsdp1):
            with FSDP.summon_full_params(fsdp2):
                for (p1, p2) in zip(fsdp1.parameters(), fsdp2.parameters()):
                    self.assertTrue(torch.allclose(p1, p2), f'{p1} vs {p2}')

    def _test_simple_model_with_meta_device(self, meta_module_fn, init_fn=None):
        if False:
            while True:
                i = 10
        model = meta_module_fn()
        is_meta = next(model.parameters()).is_meta
        fsdp_meta = FSDP(model, auto_wrap_policy=always_wrap, param_init_fn=init_fn)
        meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=0.001)
        regular = MyModel(device='cuda')
        _reset_params_if_meta(is_meta, regular)
        fsdp_regular = FSDP(regular, auto_wrap_policy=always_wrap)
        regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=0.001)
        self._compare_fsdp(fsdp_meta, fsdp_regular)
        inp = torch.randn(10, 2, device='cuda')
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        meta_opt.step()
        regular_opt.step()
        self._compare_fsdp(fsdp_meta, fsdp_regular)
        model = meta_module_fn()
        fsdp_meta = FSDP(model, param_init_fn=init_fn)
        meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=0.001)
        regular = MyModel(device='cuda')
        _reset_params_if_meta(is_meta, regular)
        fsdp_regular = FSDP(regular, auto_wrap_policy=always_wrap)
        regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=0.001)
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        meta_opt.step()
        regular_opt.step()
        self._compare_fsdp(fsdp_meta, fsdp_regular)

    @skip_if_lt_x_gpu(2)
    def test_simple_model_with_meta_device_reset_params(self):
        if False:
            print('Hello World!')

        def meta_module_fn():
            if False:
                while True:
                    i = 10
            return MyModel(device='meta')
        self._test_simple_model_with_meta_device(meta_module_fn, _init_with_reset_params)

    @skip_if_lt_x_gpu(2)
    def test_simple_model_with_meta_device_default_init(self):
        if False:
            for i in range(10):
                print('nop')

        def meta_module_fn():
            if False:
                while True:
                    i = 10
            return MyModel(device='meta')
        self._test_simple_model_with_meta_device(meta_module_fn)

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(not _TORCHDISTX_AVAIL, 'Test requires torchdistX: https://github.com/pytorch/torchdistX')
    def test_simple_model_with_torchdistX_default_init(self):
        if False:
            return 10

        def meta_module_fn():
            if False:
                return 10
            return deferred_init.deferred_init(MyModel, device='cuda')
        self._test_simple_model_with_meta_device(meta_module_fn)

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(not _TORCHDISTX_AVAIL, 'Test requires torchdistX: https://github.com/pytorch/torchdistX')
    def test_simple_model_with_torchdistX_init_fn(self):
        if False:
            i = 10
            return i + 15

        def meta_module_fn():
            if False:
                print('Hello World!')
            return deferred_init.deferred_init(MyModel, device='cuda')
        self._test_simple_model_with_meta_device(meta_module_fn, init_fn=_init_with_torchdistX)

    def _test_nested_model_with_meta_device(self, auto_wrap, meta_module_fn, init_fn=None):
        if False:
            print('Hello World!')
        if auto_wrap:
            module = meta_module_fn()
            is_meta = next(module.parameters()).is_meta or next(module.buffers()).is_meta
            fsdp_meta = FSDP(module, auto_wrap_policy=always_wrap, param_init_fn=init_fn)
            meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=0.001)
            module_regular = NestedModel(device='cuda')
            _reset_params_if_meta(is_meta, module_regular)
            fsdp_regular = FSDP(module_regular, auto_wrap_policy=always_wrap)
            regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=0.001)
        else:
            with enable_wrap(wrapper_cls=FSDP, param_init_fn=init_fn):
                module = meta_module_fn()
                is_meta = next(module.parameters()).is_meta
                fsdp_meta = wrap(module)
                meta_opt = torch.optim.SGD(fsdp_meta.parameters(), lr=0.001)
            module_regular = NestedModel(device='cuda')
            _reset_params_if_meta(is_meta, module_regular)
            with enable_wrap(wrapper_cls=FSDP):
                module_regular.lin1 = wrap(module_regular.lin1)
                module_regular.l3 = wrap(module_regular.l3)
                fsdp_regular = wrap(module_regular)
                regular_opt = torch.optim.SGD(fsdp_regular.parameters(), lr=0.001)
        self._compare_fsdp(fsdp_meta, fsdp_regular)
        inp = torch.randn(10, 2, device='cuda')
        fsdp_meta(inp).sum().backward()
        fsdp_regular(inp).sum().backward()
        meta_opt.step()
        regular_opt.step()
        self._compare_fsdp(fsdp_meta, fsdp_regular)

    @skip_if_lt_x_gpu(2)
    @parametrize('auto_wrap', [True, False])
    def test_nested_model_with_meta_device_reset_params(self, auto_wrap):
        if False:
            print('Hello World!')

        def meta_module_fn():
            if False:
                for i in range(10):
                    print('nop')
            return NestedModel(device='meta')
        self._test_nested_model_with_meta_device(auto_wrap=auto_wrap, meta_module_fn=meta_module_fn, init_fn=_init_with_reset_params)

    @skip_if_lt_x_gpu(2)
    @parametrize('auto_wrap', [True, False])
    def test_nested_model_with_meta_device_default_init(self, auto_wrap):
        if False:
            return 10

        def meta_module_fn():
            if False:
                return 10
            return NestedModel(device='meta')
        self._test_nested_model_with_meta_device(auto_wrap=auto_wrap, meta_module_fn=meta_module_fn)

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(not _TORCHDISTX_AVAIL, 'Test requires torchdistX: https://github.com/pytorch/torchdistX')
    @parametrize('auto_wrap', [True, False])
    def test_nested_model_with_torchdistX_default_init(self, auto_wrap):
        if False:
            i = 10
            return i + 15

        def meta_module_fn():
            if False:
                for i in range(10):
                    print('nop')
            return deferred_init.deferred_init(NestedModel, device='cuda')
        self._test_nested_model_with_meta_device(auto_wrap=auto_wrap, meta_module_fn=meta_module_fn)

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(not _TORCHDISTX_AVAIL, 'Test requires torchdistX: https://github.com/pytorch/torchdistX')
    @parametrize('auto_wrap', [True, False])
    def test_nested_model_with_torchdistX_init_fn(self, auto_wrap):
        if False:
            i = 10
            return i + 15

        def meta_module_fn():
            if False:
                for i in range(10):
                    print('nop')
            return deferred_init.deferred_init(NestedModel, device='cuda')
        self._test_nested_model_with_meta_device(auto_wrap=auto_wrap, meta_module_fn=meta_module_fn, init_fn=_init_with_torchdistX)

    def _test_bad_arg(self, meta_module_fn):
        if False:
            while True:
                i = 10
        mod = meta_module_fn()
        with self.assertRaisesRegex(ValueError, 'to be callable'):
            FSDP(mod, param_init_fn=42)

    @skip_if_lt_x_gpu(2)
    @skip_but_pass_in_sandcastle_if(not _TORCHDISTX_AVAIL, 'Test requires torchdistX: https://github.com/pytorch/torchdistX')
    def test_bad_arg_torchdistx(self):
        if False:
            print('Hello World!')

        def meta_module_fn():
            if False:
                return 10
            return deferred_init.deferred_init(NestedModel, 'cuda')
        self._test_bad_arg(meta_module_fn)

    @skip_if_lt_x_gpu(2)
    def test_bad_arg_meta(self):
        if False:
            i = 10
            return i + 15

        def meta_module_fn():
            if False:
                return 10
            return NestedModel(device='meta')
        self._test_bad_arg(meta_module_fn)

    @skip_if_lt_x_gpu(2)
    def test_meta_device_with_mixed_precision(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests meta device initialization with a ``param_init_fn`` when\n        specifying mixed precision with ``param_dtype=torch.float32``.\n        '

        class FakeLinear(nn.Module):

            def __init__(self, in_dim: int, out_dim: int, device: Union[torch.device, str]) -> None:
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.weight = nn.Parameter(torch.randn((in_dim, out_dim), device=device))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    print('Hello World!')
                return x @ self.weight

        class Model(nn.Module):

            def __init__(self) -> None:
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.lin1 = nn.Linear(5, 5, device='meta')
                self.lin2 = FakeLinear(5, 5, device='meta')
                self.relu = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                return self.lin2(self.relu(self.lin1(x)))

            def _module_init_fn(self, module: nn.Module):
                if False:
                    while True:
                        i = 10
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

        def _param_init_fn(module: nn.Module) -> None:
            if False:
                print('Hello World!')
            module.to_empty(device=torch.device('cuda'))
            module.apply(model._module_init_fn)
        model = Model()
        FSDP(model, auto_wrap_policy=ModuleWrapPolicy({nn.Linear}), mixed_precision=MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float16), param_init_fn=_param_init_fn, device_id=torch.cuda.current_device())
instantiate_parametrized_tests(TestFSDPWithMetaDevice)
if __name__ == '__main__':
    run_tests()