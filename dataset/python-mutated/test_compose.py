import copy
import sys
from typing import Dict
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import checkpoint, fully_shard, replicate
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_dist_composable import CompositeModel, CompositeParamModel, UnitModule
from torch.testing._internal.common_distributed import SaveForwardInputsModel, skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import instantiate_parametrized_tests, run_tests, TEST_WITH_DEV_DBG_ASAN
if not dist.is_available():
    print('Distributed not available, skipping tests', file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print('Skip dev-asan as torch + multiprocessing spawn have known issues', file=sys.stderr)
    sys.exit(0)

class TestFSDPCheckpoint(FSDPTest):

    @property
    def world_size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 2

    def _test_parity(self, base_model: nn.Module, test_model: nn.Module, inp_size: torch.Size, inp_device: torch.device, grad_to_none: bool, use_same_inputs_across_ranks: bool):
        if False:
            for i in range(10):
                print('nop')
        LR = 0.01
        base_optim = torch.optim.Adam(base_model.parameters(), lr=LR)
        test_optim = torch.optim.Adam(test_model.parameters(), lr=LR)
        for _ in range(5):
            if use_same_inputs_across_ranks:
                torch.manual_seed(0)
            x = torch.randn(inp_size, device=inp_device)
            test_loss = test_model(x).sum()
            base_loss = base_model(x).sum()
            self.assertEqual(test_loss, base_loss)
            test_loss.backward()
            test_optim.step()
            test_optim.zero_grad(set_to_none=grad_to_none)
            base_loss.backward()
            base_optim.step()
            base_optim.zero_grad(set_to_none=grad_to_none)

    @skip_if_lt_x_gpu(2)
    def test_wrap_same_submodule(self):
        if False:
            return 10
        model = UnitModule(device=torch.device('cuda'))
        base_model = copy.deepcopy(model)
        test_model = copy.deepcopy(model)
        test_model.seq = checkpoint(test_model.seq)
        test_model.seq = fully_shard(test_model.seq, policy=ModuleWrapPolicy({nn.Linear}))
        self.run_subtests({'base_model': [base_model], 'test_model': [test_model], 'inp_size': [torch.Size((2, 100))], 'inp_device': [torch.device('cuda')], 'grad_to_none': [True, False], 'use_same_inputs_across_ranks': [True]}, self._test_parity)

    def _test_checkpoint_fsdp_submodules(self):
        if False:
            return 10
        model = CompositeModel(device=torch.device('cuda'))
        base_model = copy.deepcopy(model)
        test_model = copy.deepcopy(model)
        test_model.u1 = fully_shard(test_model.u1, policy=None)
        test_model.u2 = fully_shard(test_model.u2)
        test_model.u1.seq = checkpoint(test_model.u1.seq)
        test_model.u2.seq = checkpoint(test_model.u2.seq)
        self.run_subtests({'base_model': [base_model], 'test_model': [test_model], 'inp_size': [torch.Size((2, 100))], 'inp_device': [torch.device('cuda')], 'grad_to_none': [True, False], 'use_same_inputs_across_ranks': [True]}, self._test_parity)

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_non_reentrant(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_checkpoint_fsdp_submodules()

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fully_shard_cast_forward_inputs(self):
        if False:
            i = 10
            return i + 15
        self.run_subtests({'checkpoint_strict_submodule': [False, True]}, self._test_checkpoint_fully_shard_cast_forward_inputs)

    def _test_checkpoint_fully_shard_cast_forward_inputs(self, checkpoint_strict_submodule: bool):
        if False:
            while True:
                i = 10
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        fp16_mp = MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True)
        fp32_mp = MixedPrecision(param_dtype=torch.float32, cast_forward_inputs=True)
        model = SaveForwardInputsModel(forward_inputs=forward_inputs, cast_forward_inputs=False).cuda()
        x = torch.zeros(2, 100, device='cuda')
        fully_shard(model.c2, mixed_precision=fp16_mp)
        if checkpoint_strict_submodule:
            checkpoint(model.c2.l)
        else:
            checkpoint(model.c2)
        fully_shard(model, mixed_precision=fp32_mp)
        loss = model(x).sum()
        loss.backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float16)

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_replicate_correct_replicate_params(self):
        if False:
            i = 10
            return i + 15
        model = CompositeParamModel(device=torch.device('cuda'))
        fully_shard(model.u1, policy=ModuleWrapPolicy({nn.Linear}))
        fully_shard(model.u2, policy=ModuleWrapPolicy({nn.Linear}))
        replicate(model)
        inp = torch.randn(2, 100, device='cuda')
        model(inp).sum().backward()
        param_names = replicate.state(model)._replicate_param_names
        replicated_modules = [(name, mod) for (name, mod) in model.named_children() if mod not in [model.u1, model.u2]]
        replicated_param_names = [f'{module_name}.{n}' for (module_name, mod) in replicated_modules for (n, _) in mod.named_parameters()]
        replicated_param_names.extend([n for (n, _) in model.named_parameters(recurse=False)])
        self.assertEqual(set(param_names), set(replicated_param_names))

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_with_param(self):
        if False:
            i = 10
            return i + 15
        model = CompositeParamModel(device=torch.device('cuda'))
        base_model = copy.deepcopy(model)
        test_model = copy.deepcopy(model)
        test_model.u1.seq = checkpoint(test_model.u1.seq)
        test_model.u2.seq = checkpoint(test_model.u2.seq)
        test_model = fully_shard(test_model)
        self.run_subtests({'base_model': [base_model], 'test_model': [test_model], 'inp_size': [torch.Size((2, 100))], 'inp_device': [torch.device('cuda')], 'grad_to_none': [True, False], 'use_same_inputs_across_ranks': [True]}, self._test_parity)

    @skip_if_lt_x_gpu(2)
    def test_checkpoint_fsdp_submodules_with_param_no_shard(self):
        if False:
            return 10
        model = CompositeParamModel(device=torch.device('cuda'))
        base_model = copy.deepcopy(model)
        test_model = copy.deepcopy(model)
        test_model.u1.seq = checkpoint(test_model.u1.seq)
        test_model.u2.seq = checkpoint(test_model.u2.seq)
        test_model = fully_shard(test_model, strategy=ShardingStrategy.NO_SHARD)
        self.run_subtests({'base_model': [base_model], 'test_model': [test_model], 'inp_size': [torch.Size((2, 100))], 'inp_device': [torch.device('cuda')], 'grad_to_none': [True, False], 'use_same_inputs_across_ranks': [True]}, self._test_parity)

    @skip_if_lt_x_gpu(2)
    def test_composable_fsdp_replicate(self):
        if False:
            for i in range(10):
                print('nop')
        model = CompositeModel(device=torch.device('cuda'))
        fully_shard(model.l1)
        with self.assertRaisesRegex(AssertionError, 'Cannot apply .*replicate'):
            replicate(model.l1)
        replicate(model.l2)

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_replicate_composability(self):
        if False:
            print('Hello World!')
        '\n        Tests composing ``fully_shard`` and ``replicate``. To save unit test\n        time, we run the different configs in subtests.\n        '
        self.run_subtests({'config': ['1fm,1r', '1r,1fm', '1r,1fa', '1r1fm,1fm', '1r1fa,1fm', '1fm1fm,1r1r,1fm']}, self._test_replicate_in_fully_shard)

    def _test_replicate_in_fully_shard(self, config: str):
        if False:
            return 10
        "\n        To interpret the config, each comma delineates a level in the module\n        tree ordered bottom-up; 'r' means ``replicate``; 'f' means\n        ``fully_shard``; 'a' means auto wrap; and 'm' means manual wrap.\n        "
        torch.manual_seed(0)
        if config == '1fm,1r':
            base_model = CompositeModel(device=torch.device('cuda'))
            test_model = copy.deepcopy(base_model)
            fully_shard(test_model.l1)
            replicate(test_model)
        elif config == '1r,1fm':
            base_model = CompositeParamModel(torch.device('cuda'))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model)
        elif config == '1r,1fa':
            base_model = CompositeParamModel(torch.device('cuda'))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model, policy=ModuleWrapPolicy({UnitModule}))
        elif config == '1r1fm,1fm':
            base_model = CompositeParamModel(torch.device('cuda'))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model.u2)
            fully_shard(test_model)
        elif config == '1r1fa,1fm':
            base_model = CompositeParamModel(torch.device('cuda'))
            test_model = copy.deepcopy(base_model)
            replicate(test_model.u1)
            fully_shard(test_model.u2, policy=ModuleWrapPolicy({UnitModule}))
            fully_shard(test_model)
        elif config == '1fm1fm,1r1r,1fm':
            base_model = CompositeParamModel(torch.device('cuda'))
            test_model = copy.deepcopy(base_model)
            fully_shard(test_model.u1.seq)
            fully_shard(test_model.u2.seq)
            replicate(test_model.u1)
            replicate(test_model.u2)
            fully_shard(test_model)
        else:
            raise ValueError(f'Unknown config: {config}')
        replicate(base_model)
        torch.manual_seed(self.rank + 1)
        self._test_parity(base_model, test_model, torch.Size((2, 100)), torch.device('cuda'), True, False)

    @skip_if_lt_x_gpu(2)
    def test_state_dict_fsdp_submodules(self):
        if False:
            for i in range(10):
                print('nop')
        model = CompositeModel(device=torch.device('cuda'))
        full_shard_args = {'strategy': ShardingStrategy.FULL_SHARD}
        no_shard_args = {'strategy': ShardingStrategy.NO_SHARD}
        model.u1 = fully_shard(model.u1, **full_shard_args)
        model.u2 = fully_shard(model.u2, **no_shard_args)
        FSDP.set_state_dict_type(model, StateDictType.SHARDED_STATE_DICT)
        state_dict = model.state_dict()
        for (fqn, tensor) in state_dict.items():
            if 'u1' in fqn:
                self.assertIsInstance(tensor, ShardedTensor)
            elif 'u2' in fqn:
                self.assertIsInstance(tensor, torch.Tensor)
        _ = FSDP.get_state_dict_type(model)
instantiate_parametrized_tests(TestFSDPCheckpoint)
if __name__ == '__main__':
    run_tests()