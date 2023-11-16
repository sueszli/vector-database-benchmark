import sys
import torch
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch import distributed as dist
from torch.distributed._composable import fully_shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import CUDAInitMode, FSDPInitMode, FSDPTest, TransformerWithSharedParams
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

class Model(torch.nn.Module):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.layer0 = torch.nn.Linear(3, 5)
        layer1_modules = [torch.nn.Linear(5, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]
        self.layer1 = torch.nn.Sequential(*layer1_modules)
        self.layer2 = torch.nn.Linear(4, 2)
        self.layer3 = torch.nn.Linear(2, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        if False:
            return 10
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer1(z))
        z = self.relu(self.layer2(z))
        z = self.relu(self.layer3(z))
        return z

    def get_input(self, device):
        if False:
            i = 10
            return i + 15
        return (torch.randn((8, 3)).to(device),)

    def get_loss(self, input, output):
        if False:
            while True:
                i = 10
        return output.sum()

    def run_backward(self, loss):
        if False:
            for i in range(10):
                print('nop')
        loss.backward()

class IgnoredModule(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        return x @ self.weight

class ModelWithIgnoredModules(Model):
    """Adds a variable number of :class:`IgnoredModule` to ``self.layer1``."""

    def __init__(self, num_ignored: int) -> None:
        if False:
            print('Hello World!')
        assert num_ignored >= 0
        super().__init__()
        layer1_modules = [torch.nn.Linear(5, 4), torch.nn.Linear(4, 4)] + [IgnoredModule(4, 4) for _ in range(num_ignored)] + [torch.nn.Linear(4, 4)]
        self.layer1 = torch.nn.Sequential(*layer1_modules)

class TestFSDPIgnoredModules(FSDPTest):

    def _train_model(self, model, optim, num_iters, device=torch.device('cuda')):
        if False:
            i = 10
            return i + 15
        for _ in range(num_iters):
            module = model.module if isinstance(model, FSDP) else model
            inp = module.get_input(device)
            output = model(*inp)
            loss = module.get_loss(inp, output).to(device)
            module.run_backward(loss)
            optim.step()

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_transformer(self):
        if False:
            i = 10
            return i + 15
        "Tests that ignored modules' parameters are not flattened for a\n        transformer model with shared parameters."
        self.run_subtests({'use_orig_params': [False, True], 'ignore_modules': [True, False], 'use_auto_wrap': [False, True], 'composable': [False]}, self._test_ignored_modules_transformer)

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_transformer_composable(self):
        if False:
            print('Hello World!')
        "Tests that ignored modules' parameters are not flattened for a\n        transformer model with shared parameters."
        self.run_subtests({'use_orig_params': [True], 'ignore_modules': [True, False], 'use_auto_wrap': [False, True], 'composable': [True]}, self._test_ignored_modules_transformer)

    def _test_ignored_modules_transformer(self, use_orig_params: bool, ignore_modules: bool, use_auto_wrap: bool, composable: bool):
        if False:
            while True:
                i = 10
        model: nn.Module = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE, deterministic=True)
        fsdp_kwargs = {'process_group': self.process_group}
        if use_auto_wrap:
            model.output_proj.weight = nn.Parameter(model.output_proj.weight.clone())
            fsdp_kwargs['policy' if composable else 'auto_wrap_policy'] = ModuleWrapPolicy({nn.Linear})
        if ignore_modules:
            fsdp_kwargs['ignored_modules'] = [model.transformer]
        else:
            fsdp_kwargs['ignored_states'] = list(model.transformer.parameters())
        wrapper_cls = fully_shard if composable else FSDP
        wrapped_model = wrapper_cls(model, **fsdp_kwargs)
        nonwrapped_model: nn.Module = TransformerWithSharedParams.init(self.process_group, FSDPInitMode.NO_FSDP, CUDAInitMode.CUDA_BEFORE, deterministic=True)
        if use_auto_wrap:
            nonwrapped_model.output_proj.weight = nn.Parameter(nonwrapped_model.output_proj.weight.clone())
        total_numel = sum((p.numel() for p in nonwrapped_model.parameters()))
        ignored_numel = sum((p.numel() for p in nonwrapped_model.transformer.parameters()))
        nonignored_numel = total_numel - ignored_numel
        fsdp_managed_numel = 0
        with FSDP.summon_full_params(wrapped_model):
            for handle in traversal_utils._get_fsdp_handles(wrapped_model):
                flat_param = handle.flat_param
                flat_param_numel = flat_param.numel()
                if composable or use_orig_params:
                    padding_numel = sum((numel for (numel, is_padding) in zip(flat_param._numels_with_padding, flat_param._is_padding_mask) if is_padding))
                    flat_param_numel -= padding_numel
                fsdp_managed_numel += flat_param_numel
        self.assertEqual(fsdp_managed_numel, nonignored_numel)
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_nested(self):
        if False:
            print('Hello World!')
        "Tests that passing a module with nested FSDP modules does not\n        error and still ignores non-FSDP modules' parameters."
        self.run_subtests({'use_orig_params': [False, True], 'ignore_modules': [True, False], 'composable': [False]}, self._test_ignored_modules_nested)

    @skip_if_lt_x_gpu(2)
    def test_ignored_modules_nested_composable(self):
        if False:
            return 10
        "Tests that passing a module with nested FSDP modules does not\n        error and still ignores non-FSDP modules' parameters."
        self.run_subtests({'use_orig_params': [True], 'ignore_modules': [True, False], 'composable': [True]}, self._test_ignored_modules_nested)

    def _test_ignored_modules_nested(self, use_orig_params: bool, ignore_modules: bool, composable: bool):
        if False:
            print('Hello World!')
        model = Model().cuda()
        model.layer1[1] = FSDP(model.layer1[1], use_orig_params=use_orig_params) if not composable else fully_shard(model.layer1[1])
        if ignore_modules:
            wrapped_model = FSDP(model, ignored_modules=[model.layer1], use_orig_params=use_orig_params) if not composable else fully_shard(model, ignored_modules=[model.layer1])
        else:
            wrapped_model = FSDP(model, ignored_states=[model.layer1], use_orig_params=use_orig_params) if not composable else fully_shard(model, ignored_states=[model.layer1])
        nonwrapped_model = Model()
        total_numel = sum((p.numel() for p in nonwrapped_model.parameters()))
        ignored_numel = sum((p.numel() for p in nonwrapped_model.layer1.parameters()))
        nonignored_numel = total_numel - ignored_numel
        with FSDP.summon_full_params(wrapped_model):
            flat_param = wrapped_model.params[0] if not composable else _get_module_fsdp_state(wrapped_model).params[0]
            flat_param_numel = flat_param.numel()
            if composable or use_orig_params:
                padding_numel = sum((numel for (numel, is_padding) in zip(flat_param._numels_with_padding, flat_param._is_padding_mask) if is_padding))
                flat_param_numel -= padding_numel
                self.assertEqual(flat_param_numel, nonignored_numel)
            self.assertEqual(flat_param_numel, nonignored_numel)
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    @parametrize('composable', [True, False])
    def test_ignored_modules_invalid(self, composable):
        if False:
            return 10
        'Tests that passing an FSDP module as an ignored module or the\n        top-level module itself errors.'
        model = Model().cuda()
        wrap_cls = FSDP if composable else fully_shard
        model.layer1 = wrap_cls(model.layer1)
        with self.assertRaises(ValueError, msg='`ignored_modules` should not include FSDP modules'):
            wrap_cls(model, ignored_modules=[model.layer1])
        with self.assertWarnsRegex(expected_warning=UserWarning, expected_regex='Trying to ignore the top-level module passed into the FSDP constructor itself will result in all parameters being ignored'):
            new_model = Model().cuda()
            wrap_cls(new_model, ignored_modules=[new_model])

    @skip_if_lt_x_gpu(2)
    def test_diff_ignored_modules_across_ranks(self):
        if False:
            return 10
        "\n        Tests ignoring different modules across ranks.\n\n        Args:\n            pass_ignored_modules_to_root (bool): If ``False``, does not pass\n                any ignored modules (including those already ignored in child\n                FSDP instances) to the root FSDP instance; if ``True``, passes\n                all ignored modules (representing a superset of the children's\n                ignored modules) to the root FSDP instance.\n        "
        self.run_subtests({'pass_ignored_modules_to_root': [False, True], 'ignore_modules': [True, False], 'composable': [True, False]}, self._test_diff_ignored_modules_across_ranks)

    def _test_diff_ignored_modules_across_ranks(self, pass_ignored_modules_to_root: bool, ignore_modules: bool, composable: bool):
        if False:
            return 10
        wrap_cls = FSDP if composable else fully_shard
        model = ModelWithIgnoredModules(num_ignored=self.rank + 1).cuda()
        layer1_ignored_modules = [m for m in model.layer1.modules() if isinstance(m, IgnoredModule)]
        ignore_kwargs = {'ignored_modules': layer1_ignored_modules} if ignore_modules else {'ignored_states': (p for m in layer1_ignored_modules for p in m.parameters())}
        model.layer1 = wrap_cls(model.layer1, **ignore_kwargs)
        model.layer3 = wrap_cls(model.layer3)
        model_ignored_modules = [m for m in model.modules() if isinstance(m, IgnoredModule)] if pass_ignored_modules_to_root else []
        ignore_kwargs_top = {'ignored_modules': model_ignored_modules} if ignore_modules else {'ignored_states': {p for m in model_ignored_modules for p in m.parameters()}}
        wrapped_model = wrap_cls(model, **ignore_kwargs_top)
        optim = torch.optim.Adam(wrapped_model.parameters(), lr=0.001)
        self._train_model(wrapped_model, optim, 3)

    @skip_if_lt_x_gpu(2)
    @parametrize('ignore_modules', [True, False])
    @parametrize('composable', [True, False])
    def test_ignored_modules_not_under_wrapped_root(self, ignore_modules: bool, composable: bool):
        if False:
            i = 10
            return i + 15
        model = Model().cuda()
        ignored_modules = list(model.layer1.children())[1:]
        ignore_kwargs = {'ignored_modules': ignored_modules} if ignore_modules else {'ignored_states': {p for m in ignored_modules for p in m.parameters()}}
        wrap_cls = FSDP if composable else fully_shard
        model.layer1 = wrap_cls(model.layer1, **ignore_kwargs)
        model.layer3 = wrap_cls(model.layer3, **ignore_kwargs)
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        self._train_model(model, optim, 3)

    @skip_if_lt_x_gpu(1)
    def test_ignored_states_check(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests that passing invalid ``ignored_modules`` or ``ignored_states``\n        raises an appropriate error.\n        '
        self.run_subtests({'ignore_modules': [True, False]}, self._test_ignored_states_check)

    def _test_ignored_states_check(self, ignore_modules: bool):
        if False:
            i = 10
            return i + 15
        model = Model().cuda()
        ignored_modules = list(model.layer1.children())[1:]
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        ignored_states = ignored_params.union(set(ignored_modules))
        if ignore_modules:
            with self.assertRaisesRegex(ValueError, "ignored_modules expects nn.Module list elements but got types \\[<class 'torch.nn.parameter.Parameter'>\\]"):
                FSDP(model, ignored_modules=ignored_params)
            with self.assertRaisesRegex(ValueError, 'Cannot pass both ignored_modules and ignored_states at the same time'):
                FSDP(model, ignored_modules=ignored_modules, ignored_states=ignored_params)
        else:
            with self.assertRaisesRegex(ValueError, "ignored_states expects all nn.Parameter or all nn.Module list elements but got types \\[<class 'torch.nn.modules.linear.Linear'>, <class 'torch.nn.parameter.Parameter'>\\]"):
                FSDP(model, ignored_states=ignored_states)
instantiate_parametrized_tests(TestFSDPIgnoredModules)
if __name__ == '__main__':
    run_tests()