import functools
import os
import sys
import warnings
from collections import namedtuple
from contextlib import nullcontext
from copy import deepcopy
from itertools import chain
from typing import Any, Tuple
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp import CPUOffload, FlatParameter, FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp._flat_param import _FSDP_USE_UNSAFE_SETATTR
from torch.distributed.fsdp._runtime_utils import HOMOGENEOUS_ATTR_NAMES
from torch.distributed.fsdp.wrap import always_wrap_policy, ModuleWrapPolicy, transformer_auto_wrap_policy
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import _assert_module_states, CUDAInitMode, FSDPInitMode, FSDPTest, FSDPTestMultiThread, NestedWrappedModule, TransformerWithSharedParams
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

class MyModel(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.a = nn.Linear(2, 2)
        self.b = nn.Linear(2, 2)

    def forward(self, x, y):
        if False:
            print('Hello World!')
        return self.b(self.a(x + y))

class TestFSDPMiscMultiProcess(FSDPTest):

    @property
    def world_size(self):
        if False:
            while True:
                i = 10
        return 2

    @property
    def process_group(self):
        if False:
            while True:
                i = 10
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    @parametrize('use_index', [True, False])
    def test_fsdp_device_id(self, use_index):
        if False:
            print('Hello World!')
        '\n        Tests the FSDP ``device_id`` argument:\n          - Wrapping a CPU module should move the module to the GPU matching\n          ``device_id``\n          - Wrapping a GPU module already on the GPU matching ``device_id``\n          should not raise an error\n          - Wrapping a GPU module already on GPU and passing a GPU device\n          without specifying a device ID (i.e. ``torch.device("cuda")``) warns\n        '
        dev_id = torch.cuda.current_device() if use_index else torch.device('cuda', torch.cuda.current_device())

        def _check_device_matches(module, device_id):
            if False:
                for i in range(10):
                    print('nop')
            'Checks that the ``FlatParameter``s in ``module`` have device\n            matching ``device_id``.'
            devices = {p.device for p in module.parameters() if isinstance(p, FlatParameter)}
            assert len(devices) > 0
            self.assertEqual(1, len(devices))
            found_device = devices.pop()
            if use_index and (not isinstance(device_id, torch.device)):
                device = torch.device('cuda', device_id)
            else:
                device = device_id
            self.assertEqual(found_device, device)
        nested_wrapped_module = NestedWrappedModule.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_NEVER, fsdp_kwargs={'device_id': dev_id})
        _check_device_matches(nested_wrapped_module, dev_id)
        nested_wrapped_module = NestedWrappedModule.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs={'device_id': dev_id})
        _check_device_matches(nested_wrapped_module, dev_id)
        regex = 'does not have an explicit index'
        context = self.assertWarnsRegex(expected_warning=UserWarning, expected_regex=regex)
        with context:
            nested_wrapped_module = NestedWrappedModule.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs={'device_id': torch.device('cuda')})
        _check_device_matches(nested_wrapped_module, torch.device('cuda', torch.cuda.current_device()))

    @skip_if_lt_x_gpu(2)
    def test_fsdp_zero2_eval_with_prefetch(self):
        if False:
            return 10

        class Mnist(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)
                self.ln = nn.LayerNorm(9216)

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                x = self.conv1(x)
                x = torch.nn.functional.relu(x)
                x = self.conv2(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.ln(x)
                x = self.fc1(x)
                x = torch.nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = torch.nn.functional.log_softmax(x, dim=1)
                loss = torch.nn.functional.cross_entropy(output, y)
                return loss
        model = Mnist().cuda()
        model1 = Mnist().cuda()
        model1.load_state_dict(model.state_dict())
        fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, forward_prefetch=True, use_orig_params=True, auto_wrap_policy=ModuleWrapPolicy([nn.Linear, nn.Conv2d]))
        ddp_model = torch.nn.parallel.DistributedDataParallel(model1)
        fsdp_opt = torch.optim.SGD(fsdp_model.parameters(), lr=0.0001)
        ddp_opt = torch.optim.SGD(ddp_model.parameters(), lr=0.0001)
        seed = self.rank + 20231010
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        losses = []
        grads = []
        for i in range(5):
            x = torch.randn(8, 1, 28, 28, device='cuda').requires_grad_()
            y = torch.randint(low=0, high=9, size=(8,), device='cuda')
            for (model, opt) in ((fsdp_model, fsdp_opt), (ddp_model, ddp_opt)):
                seed = self.rank + i
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                loss = model(x, y).sum()
                losses.append(loss)
                loss.backward()
                opt.step()
                grads.append(x.grad)
                opt.zero_grad()
            assert torch.allclose(losses[0], losses[1])
            assert torch.allclose(grads[0], grads[1])
            losses.clear()
            grads.clear()
        with torch.no_grad():
            fsdp_model.eval()
            ddp_model.eval()
            for _ in range(5):
                x = torch.randn(8, 1, 28, 28, device='cuda').requires_grad_()
                y = torch.randint(low=0, high=9, size=(8,), device='cuda')
                fsdp_loss = fsdp_model(x, y)
                ddp_loss = ddp_model(x, y)
                assert torch.allclose(fsdp_loss, ddp_loss)
        fsdp_model.train()
        ddp_model.train()
        for i in range(5):
            x = torch.randn(8, 1, 28, 28, device='cuda').requires_grad_()
            y = torch.randint(low=0, high=9, size=(8,), device='cuda')
            for (model, opt) in ((fsdp_model, fsdp_opt), (ddp_model, ddp_opt)):
                seed = self.rank + i
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                loss = model(x, y).sum()
                losses.append(loss)
                loss.backward()
                opt.step()
                grads.append(x.grad)
                opt.zero_grad()
            assert torch.allclose(losses[0], losses[1])
            assert torch.allclose(grads[0], grads[1])
            losses.clear()
            grads.clear()

    @skip_if_lt_x_gpu(2)
    @parametrize('use_second_layer', [True, False])
    @parametrize('sharding_strategy', [ShardingStrategy.NO_SHARD, None])
    def test_fsdp_module_no_compute_grad(self, use_second_layer, sharding_strategy):
        if False:
            print('Hello World!')

        class MyModel(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.a = nn.Linear(10, 10)
                self.b = nn.Linear(10, 10)

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                out1 = self.a(x)
                if use_second_layer:
                    out2 = self.b(y)
                    return (out1, out2)
                else:
                    return out1
        fsdp = FSDP(MyModel().cuda(), sharding_strategy=sharding_strategy, auto_wrap_policy=always_wrap_policy)
        x = torch.randn(10, 10, device='cuda')
        y = torch.randn(10, 10, device='cuda')
        for i in range(4):
            if use_second_layer:
                (a, b) = fsdp(x, y)
            else:
                a = fsdp(x, y)
            loss = a.sum()
            loss.backward()
            a_grad = fsdp.module.a._handle.flat_param.grad
            b_grad = fsdp.module.b._handle.flat_param.grad
            self.assertIsNotNone(a_grad)
            self.assertIsNone(b_grad)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_not_all_outputs_used_in_loss(self):
        if False:
            i = 10
            return i + 15
        self.run_subtests({'sharding_strategy': [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.NO_SHARD]}, self._test_fsdp_not_all_outputs_used_in_loss)

    def _test_fsdp_not_all_outputs_used_in_loss(self, sharding_strategy: ShardingStrategy):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.lin1 = nn.Linear(4, 4)
                self.lin2 = nn.Linear(4, 4)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                a = self.lin1(x)
                b = self.lin2(x)
                return (a, b)

        def _check_resharded(fsdp_module):
            if False:
                i = 10
                return i + 15
            handle = fsdp_module._handle
            if not handle:
                return
            param = handle.flat_param
            if handle.uses_sharded_strategy:
                full_param = param._full_param_padded
                self.assertEqual(full_param.storage().size(), 0)
            self.assertEqual(param.data_ptr(), param._local_shard.data_ptr())

        def _check_equal(local, fsdp):
            if False:
                return 10
            with FSDP.summon_full_params(fsdp):
                for (p1, p2) in zip(fsdp.parameters(), local.parameters()):
                    torch.testing.assert_close(p1, p2)
        fsdp_ctor = functools.partial(FSDP, sharding_strategy=sharding_strategy)
        m = MyModule().cuda()
        m_local = deepcopy(m)
        local_m = m_local
        prev_params = [p.clone() for p in m_local.parameters()]
        m.lin1 = fsdp_ctor(m.lin1)
        m = fsdp_ctor(m)
        _check_equal(m_local, m)
        opt = torch.optim.SGD(m.parameters(), lr=0.001)
        opt_local = torch.optim.SGD(local_m.parameters(), lr=0.001)
        for i in range(6):
            t = torch.ones(4, device='cuda')
            (a, b) = m(t)
            (local_a, local_b) = local_m(t)
            if i < 2:
                loss = (a @ b).sum()
                loss_local = (local_a @ local_b).sum()
            else:
                loss = a.sum()
                loss_local = local_a.sum()
            loss.backward()
            loss_local.backward()
            _check_resharded(m)
            opt.step()
            opt_local.step()
            _check_equal(m_local, m)
            self.assertTrue(any((not torch.equal(p1, p2) for (p1, p2) in zip(prev_params, m_local.parameters()))))
            prev_params = [p.clone() for p in local_m.parameters()]
            opt.zero_grad()
            opt_local.zero_grad()
        dist.barrier()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_optim_overlap_no_use_orig_params_error(self):
        if False:
            print('Hello World!')
        fsdp_overlap = FSDP(MyModel().cuda(), auto_wrap_policy=always_wrap_policy, use_orig_params=False)
        optim_cls = torch.optim.SGD
        optim_kwargs = {'lr': 0.03}
        _apply_optimizer_in_backward(optimizer_class=optim_cls, params=fsdp_overlap.parameters(), optimizer_kwargs=optim_kwargs, register_hook=False)
        inp = torch.randn(10, 10, device='cuda')
        with self.assertRaisesRegex(RuntimeError, 'only supported with use_orig_params=True'):
            fsdp_overlap(inp, inp)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_optimizer_overlap(self):
        if False:
            i = 10
            return i + 15
        torch.manual_seed(0)
        for cpu_offload in [True, False]:
            offload = CPUOffload(offload_params=cpu_offload)
            model = MyModel().cuda()
            model_overlap = deepcopy(model)
            fsdp = FSDP(model.cuda(), auto_wrap_policy=always_wrap_policy, use_orig_params=True, cpu_offload=offload)
            fsdp_overlap = FSDP(model_overlap.cuda(), auto_wrap_policy=always_wrap_policy, use_orig_params=True, cpu_offload=offload)
            optim_cls = torch.optim.SGD
            optim_kwargs = {'lr': 0.03}
            _apply_optimizer_in_backward(optimizer_class=optim_cls, params=fsdp_overlap.parameters(), optimizer_kwargs=optim_kwargs, register_hook=False)
            for p in fsdp_overlap.parameters():
                assert hasattr(p, '_in_backward_optimizers')
            optim = optim_cls(fsdp.parameters(), **optim_kwargs)
            for (p1, p2) in zip(fsdp.parameters(), fsdp_overlap.parameters()):
                self.assertEqual(p1, p2)
            with FSDP.summon_full_params(fsdp_overlap):
                fsdp_overlap_prev_params = [(n, p.clone()) for (n, p) in fsdp_overlap.named_parameters()]
            for i in range(6):
                inp = torch.randn(2, 2, device='cuda')
                with torch.no_grad():
                    inp_clone = inp.clone()
                fsdp(inp, inp).sum().backward()
                fsdp_overlap(inp_clone, inp_clone).sum().backward()
                optim.step()
                optim.zero_grad()
                for fsdp_unit in FSDP.fsdp_modules(fsdp_overlap):
                    handle = fsdp_unit._handle
                    if handle:
                        handle_grad = handle.sharded_grad
                        self.assertEqual(None, handle_grad, 'Overlapped FSDP sharded_grad is not None!')
                with FSDP.summon_full_params(fsdp_overlap, with_grads=True):
                    for ((n, p), (n_prev, p_prev)) in zip(fsdp_overlap.named_parameters(), fsdp_overlap_prev_params):
                        self.assertNotEqual(p, p_prev, f'{n_prev} Params at iter {i} same as previous iter!')
                with FSDP.summon_full_params(fsdp_overlap):
                    with FSDP.summon_full_params(fsdp):
                        for ((n_overlap, p_overlap), (n, p)) in zip(fsdp_overlap.named_parameters(), fsdp.named_parameters()):
                            self.assertEqual(n_overlap, n)
                            self.assertEqual(p, p_overlap, f'Rank {self.rank}: Params not equal at iteration {i}: {n_overlap} - {p} vs {p_overlap}')
                            self.assertEqual(None, p.grad, f'Expected param {n} grad to be None')
                            self.assertEqual(None, p_overlap.grad, f'Expected param {n_overlap} grad to be None')
                    fsdp_overlap_prev_params = [(n, p.clone()) for (n, p) in fsdp_overlap.named_parameters()]

    @skip_if_lt_x_gpu(2)
    def test_fsdp_cpu_training(self):
        if False:
            i = 10
            return i + 15
        'Tests FSDP training on CPU.'
        torch.manual_seed(0)
        gloo_pg = dist.new_group(backend='gloo')
        for ss in [ShardingStrategy.NO_SHARD, ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2]:
            model = MyModel()
            fsdp = FSDP(model, auto_wrap_policy=always_wrap_policy, process_group=gloo_pg, device_id=torch.device('cpu'))
            inp = torch.randn(2, 2)
            fsdp(inp, inp).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_cpu_init_stays_on_cpu(self):
        if False:
            print('Hello World!')
        'Tests that passing a CPU module to FSDP preserves that the wrapped\n        module is on CPU after FSDP initialization, albeit after logging a\n        warning, and that FSDP moves CPU input to GPU before the forward.'
        torch.cuda.set_device(self.rank)
        regex = 'passed-in `module` is on CPU'
        context = self.assertWarnsRegex(expected_warning=UserWarning, expected_regex=regex)
        with context:
            nested_wrapped_module = NestedWrappedModule.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_NEVER)
            fsdp_model = FSDP(nested_wrapped_module, self.process_group)
        devices = {p.device for p in fsdp_model.parameters()}
        self.assertEqual(1, len(devices))
        self.assertEqual(torch.device('cpu'), devices.pop())
        fsdp_model = fsdp_model.cuda()
        inp = fsdp_model.module.get_input(device=torch.device('cpu'))
        fsdp_model(*inp).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_cpu_init_with_sync_module_states(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that passing ``sync_module_states=True`` raises an error for\n        a CPU module since the synchronization requires GPU communication,\n        while additionally passing ``device_id`` does not raise an error, even\n        when the model has CPU buffers.\n        '

        def init_nested_wrapped_module():
            if False:
                i = 10
                return i + 15
            return NestedWrappedModule.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_NEVER)
        with self.assertRaisesRegex(ValueError, 'The module has CPU parameters or buffers when `sync_module_states=True`'):
            FSDP(init_nested_wrapped_module(), self.process_group, sync_module_states=True)
        nested_wrapped_module = init_nested_wrapped_module()
        nested_wrapped_module.register_buffer('buf', torch.ones((2, 2), device='cpu') * self.rank)
        nested_wrapped_module.module[0].register_buffer('buf', torch.ones((3, 2), device='cpu') * self.rank)
        nested_wrapped_module = FSDP(nested_wrapped_module, self.process_group, auto_wrap_policy=ModuleWrapPolicy({nn.Linear}), device_id=torch.cuda.current_device(), sync_module_states=True)
        self.assertEqual(nested_wrapped_module.buf.device, torch.device('cuda', torch.cuda.current_device()))
        self.assertEqual(nested_wrapped_module.buf, torch.zeros((2, 2)))
        self.assertEqual(nested_wrapped_module.module.module[0].buf.device, torch.device('cuda', torch.cuda.current_device()))
        self.assertEqual(nested_wrapped_module.module.module[0].buf, torch.zeros((3, 2)))

class TestFSDPMiscMultiThread(FSDPTestMultiThread):

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        return 2

    @property
    def process_group(self):
        if False:
            return 10
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_namedtuple(self):
        if False:
            return 10

        class MyModule(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.lin = nn.Linear(100, 100)

            def forward(self, x):
                if False:
                    return 10
                return x
        m = MyModule().cuda()
        m = FSDP(m)
        t = torch.ones(1, device='cuda', requires_grad=True)
        MyOutputType = namedtuple('MyOutputType', ['a', 'b', 'c', 'd'], defaults=(t, t, t, t))
        inp = MyOutputType()
        out = m(inp)
        for x in out:
            self.assertNotEqual([], list(x._backward_hooks.values()))

    @skip_if_lt_x_gpu(2)
    def test_device_id_auto_wrap(self):
        if False:
            return 10
        'Tests that ``auto_wrap_policy`` propagates ``device_id`` to all\n        nested FSDP instances.'
        self.run_subtests({'use_callable': [False, True]}, self._test_device_id_auto_wrap)

    def _test_device_id_auto_wrap(self, use_callable: bool):
        if False:
            for i in range(10):
                print('nop')
        module_classes = {TransformerEncoderLayer, TransformerDecoderLayer}
        if use_callable:
            auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=module_classes)
        else:
            auto_wrap_policy = ModuleWrapPolicy(module_classes)
        fsdp_kwargs = {'auto_wrap_policy': auto_wrap_policy, 'device_id': torch.cuda.current_device()}
        fsdp_model = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.RECURSIVE, CUDAInitMode.CUDA_BEFORE, fsdp_kwargs)
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            self.assertEqual(fsdp_module.compute_device, torch.device('cuda', torch.cuda.current_device()))

    @skip_if_lt_x_gpu(2)
    def test_fsdp_device_id_cpu_offload(self):
        if False:
            return 10
        '\n        Tests FSDP when specifying both ``device_id`` and parameter CPU\n        offloading.\n        '
        self.run_subtests({'use_orig_params': [False, True]}, self._test_fsdp_device_id_cpu_offload)

    def _test_fsdp_device_id_cpu_offload(self, use_orig_params: bool):
        if False:
            return 10

        class MyModel(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.seq = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
                self.lin = nn.Linear(10, 10)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.lin(self.seq(x))
        model = MyModel()
        auto_wrap_policy = ModuleWrapPolicy({nn.Sequential})
        fsdp_model = FSDP(model, auto_wrap_policy=auto_wrap_policy, cpu_offload=CPUOffload(offload_params=True), device_id=torch.cuda.current_device(), use_orig_params=use_orig_params)
        cpu_device = torch.device('cpu')
        for handle in traversal_utils._get_fsdp_handles(fsdp_model):
            self.assertEqual(handle.flat_param.device, cpu_device)

    @skip_if_lt_x_gpu(2)
    def test_module_device_mismatches_device_id(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that specifying a ``device_id`` argument to FSDP for a GPU\n        module that does not match the GPU device ID raises an error.'
        torch.cuda.set_device(self.rank)
        context = self.assertRaisesRegex(ValueError, f'cuda:{self.rank} vs cuda:0') if self.rank != 0 else nullcontext()
        with context:
            NestedWrappedModule.init(self.process_group, FSDPInitMode.RECURSIVE, cuda_init_mode=CUDAInitMode.CUDA_BEFORE, fsdp_kwargs={'device_id': 0})

    @skip_if_lt_x_gpu(2)
    def test_cpu_gpu_module(self):
        if False:
            return 10
        'Tests a CPU + GPU module supported if device_id is passed\n        in, errors if device_id is not.\n        '
        torch.cuda.set_device(self.rank)

        class CPUGPUModule(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = nn.Linear(1, 1).cuda()
                self.b = nn.Linear(1, 1)
        cpu_gpu = CPUGPUModule()
        fsdp = FSDP(cpu_gpu, device_id=torch.cuda.current_device())
        for param in fsdp.parameters():
            self.assertEqual(param.device, torch.device(torch.cuda.current_device()))
        with self.assertRaisesRegex(RuntimeError, 'please pass in device_id'):
            FSDP(CPUGPUModule())

    @skip_if_lt_x_gpu(2)
    def test_fsdp_ignored_module_meta(self):
        if False:
            print('Hello World!')
        torch.cuda.set_device(self.rank)

        class CPUGPUModule(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = nn.Linear(1, 1)
                self.b = nn.Linear(1, 1)
        with torch.device('meta'):
            m = CPUGPUModule()
        m = FSDP(m, device_id=self.rank, ignored_modules=[m.a], use_orig_params=True)
        meta_device = torch.device('meta')
        self.assertEqual(meta_device, next(m.a.parameters()).device)
        with torch.device('meta'):
            m = CPUGPUModule()
        m = FSDP(m, device_id=torch.cuda.current_device(), ignored_modules=[m.a], use_orig_params=True, param_init_fn=lambda m: m.to_empty(device=torch.cuda.current_device(), recurse=False))
        self.assertEqual(meta_device, next(m.a.parameters()).device)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_device_id_no_move_ignored_params_and_bufs(self):
        if False:
            i = 10
            return i + 15

        class CPUGPUModule(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = nn.Linear(1, 1)
                self.b = nn.Linear(1, 1)
                self.a.register_buffer('buf', torch.ones(1))
        m = CPUGPUModule()
        m = FSDP(m, device_id=self.rank, ignored_modules=[m.a], use_orig_params=True)
        ignored_params = m.a.parameters()
        ignored_bufs = m.a.buffers()
        for t in chain(ignored_params, ignored_bufs):
            self.assertEqual(torch.device('cpu'), t.device)

    @skip_if_lt_x_gpu(2)
    def test_multigpu_module(self):
        if False:
            print('Hello World!')
        '\n        Module on multiple GPUs wrapped in FSDP should raise an error.\n        '

        class MultiGPUModule(nn.Module):

            def __init__(self, rank):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.rank = rank
                self.a = nn.Linear(1, 1).cuda(self.rank)
                self.b = nn.Linear(1, 1).cuda((self.rank + 1) % dist.get_world_size())
        with self.assertRaisesRegex(RuntimeError, 'FSDP only supports single device modules'):
            FSDP(MultiGPUModule(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_no_params(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that device_id and cpu init work if module has no params\n        (they are effective noops, but ensure FSDP does not assume module\n        has parameters during init)\n        '
        torch.cuda.set_device(self.rank)
        no_params = nn.ReLU()
        module = FSDP(no_params)
        no_params = nn.ReLU().cuda()
        module = FSDP(no_params)
        no_params = nn.ReLU()
        module = FSDP(no_params, device_id=torch.cuda.current_device())
        no_params = nn.ReLU().cuda()
        context = self.assertRaisesRegex(ValueError, f'Inconsistent.*cuda:{self.rank} vs cuda:0') if self.rank != 0 else nullcontext()
        with context:
            FSDP(no_params, device_id=0)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_same_model_across_ranks(self):
        if False:
            i = 10
            return i + 15
        '\n        FSDP broadcasts model from rank 0 to ensure it starts off with the same\n        values.\n        '

        class MyModel(nn.Module):

            def __init__(self, rank):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                torch.manual_seed(rank)
                torch.cuda.manual_seed(rank)
                self.lin = nn.Linear(10, 10, bias=False)
                self.register_buffer('buffer', torch.ones(1) * rank)
        m = MyModel(self.rank).cuda()
        _assert_module_states(m, process_group=self.process_group, assert_fn=self.assertNotEqual)
        fsdp = FSDP(m, sync_module_states=True)
        with fsdp.summon_full_params(fsdp):
            _assert_module_states(fsdp, process_group=self.process_group, assert_fn=self.assertEqual)
        m = MyModel(self.rank)
        _assert_module_states(m, process_group=self.process_group, assert_fn=self.assertNotEqual)
        fsdp = FSDP(m, device_id=torch.cuda.current_device(), sync_module_states=True)
        with fsdp.summon_full_params(fsdp):
            _assert_module_states(fsdp, process_group=self.process_group, assert_fn=self.assertEqual)

    @skip_if_lt_x_gpu(2)
    def test_homogeneous_attributes(self):
        if False:
            return 10
        '\n        Tests that passing heterogeneous values for attributes designated as\n        homogeneous raises an error.\n        '
        all_attr_name_and_values = [('_use_orig_params', False, True), ('limit_all_gathers', False, True), ('_use_full_prec_in_eval', False, True)]
        self.assertEqual([attr_name_and_values[0] for attr_name_and_values in all_attr_name_and_values], HOMOGENEOUS_ATTR_NAMES)
        self.run_subtests({'attr_name_and_values': all_attr_name_and_values}, self._test_homogeneous_attributes)

    def _test_homogeneous_attributes(self, attr_name_and_values: Tuple[str, Any, Any]):
        if False:
            print('Hello World!')
        model = NestedWrappedModule.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE, {})
        attr_name = attr_name_and_values[0]
        if '_use_full_prec_in_eval' == attr_name:
            model.module[1] = FSDP(model.module[1])
            os.environ['FSDP_USE_FULL_PREC_IN_EVAL'] = '1'
            fsdp_model = FSDP(model)
        else:
            fsdp_kwargs_inner = {attr_name.lstrip('_'): attr_name_and_values[1]}
            fsdp_kwargs_outer = {attr_name.lstrip('_'): attr_name_and_values[2]}
            model.module[1] = FSDP(model.module[1], **fsdp_kwargs_inner)
            fsdp_model = FSDP(model, **fsdp_kwargs_outer)
        with self.assertRaisesRegex(ValueError, f'Expects one homogeneous value for {attr_name}'):
            inp = fsdp_model.module.get_input(torch.device('cuda'))
            fsdp_model(*inp)

class TestFSDPMiscWorldSize1(FSDPTestMultiThread):

    @property
    def world_size(self) -> int:
        if False:
            print('Hello World!')
        return 1

    @skip_if_lt_x_gpu(1)
    def test_world_size_1_sharding_strategy_warning(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests that FSDP issues a warning when it switches to using ``NO_SHARD``\n        when the world size is 1.\n        '
        warning_prefix = 'FSDP is switching to use `NO_SHARD` instead of'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            FSDP(nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.NO_SHARD)
            for warning in w:
                self.assertTrue(warning.category != UserWarning or not str(warning.message).startswith(warning_prefix))
        warning_suffix = ' since the world size is 1.'
        expected_regex_full_shard = warning_prefix + ' ' + str(ShardingStrategy.FULL_SHARD) + warning_suffix
        with self.assertWarnsRegex(UserWarning, expected_regex_full_shard):
            FSDP(nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.FULL_SHARD)
        with self.assertWarnsRegex(UserWarning, expected_regex_full_shard):
            FSDP(nn.Linear(3, 3).cuda())
        expected_regex_shard_grad_op = warning_prefix + ' ' + str(ShardingStrategy.SHARD_GRAD_OP) + warning_suffix
        with self.assertWarnsRegex(UserWarning, expected_regex_shard_grad_op):
            FSDP(nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.SHARD_GRAD_OP)

    @skip_if_lt_x_gpu(1)
    def test_training_device_mismatch_errors(self):
        if False:
            while True:
                i = 10
        '\n        Tests that, when training starts, if FSDP parameters are not on the\n        expected device, then an informative error is raised. This applies for\n        both no parameter CPU offloading and parameter CPU offloading.\n        '
        model = torch.nn.Linear(10, 10)
        fsdp_model = FSDP(model)
        inp = torch.randn((2, 10))
        with self.assertRaisesRegex(RuntimeError, 'An FSDP-managed module unexpectedly has parameters on cpu. Make sure to move the module to cuda:0 before training.'):
            fsdp_model(inp)
        model = torch.nn.Linear(10, 10)
        fsdp_model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
        fsdp_model.to(torch.device('cuda'))
        inp = torch.randn((2, 10))
        with self.assertRaisesRegex(RuntimeError, 'An FSDP-managed module with parameter CPU offloading enabled has parameters on cuda:0. Make sure to not move the module from CPU when offloading parameters.'):
            fsdp_model(inp)

    @skip_if_lt_x_gpu(2)
    def test_unsafe_setattr(self):
        if False:
            print('Hello World!')
        '\n        Tests that the environment variable for using unsafe setattr gates as\n        expected.\n        '
        self.run_subtests({'use_orig_params': [False, True]}, self._test_unsafe_setattr)

    def _test_unsafe_setattr(self, use_orig_params: bool):
        if False:
            print('Hello World!')
        called_setattr_override = False

        class SetattrLinear(nn.Module):

            def __init__(self, in_dim: int, out_dim: int, device: torch.device) -> None:
                if False:
                    print('Hello World!')
                super().__init__()
                self.weight = nn.Parameter(torch.randn((in_dim, out_dim), device=device))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    print('Hello World!')
                return x @ self.weight

            def __setattr__(self, name: str, value: Any) -> None:
                if False:
                    print('Hello World!')
                nonlocal called_setattr_override
                called_setattr_override = True
                return super().__setattr__(name, value)
        module = SetattrLinear(5, 5, torch.device('cuda'))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        inp = torch.randn((8, 5), device=torch.device('cuda'))
        called_setattr_override = False
        fsdp_module(inp)
        self.assertTrue(called_setattr_override)
        os.environ[_FSDP_USE_UNSAFE_SETATTR] = '1'
        module = SetattrLinear(5, 5, torch.device('cuda'))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        called_setattr_override = False
        fsdp_module(inp)
        self.assertFalse(called_setattr_override)
        os.environ[_FSDP_USE_UNSAFE_SETATTR] = '0'
        module = SetattrLinear(5, 5, torch.device('cuda'))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        called_setattr_override = False
        fsdp_module(inp)
        self.assertTrue(called_setattr_override)
instantiate_parametrized_tests(TestFSDPMiscMultiThread)
instantiate_parametrized_tests(TestFSDPMiscMultiProcess)
if __name__ == '__main__':
    run_tests()