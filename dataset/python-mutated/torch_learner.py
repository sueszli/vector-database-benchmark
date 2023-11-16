import logging
import pathlib
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence, Union, Tuple
from ray.rllib.core.learner.learner import FrameworkHyperparameters, Learner, LearnerHyperparameters, TorchCompileWhatToCompile
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule
from ray.rllib.core.rl_module.rl_module import RLModule, ModuleID, SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchDDPRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override, OverrideToImplementCustomLogic
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import convert_to_torch_tensor, copy_torch_tensors
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
(torch, nn) = try_import_torch()
if torch:
    from ray.air._internal.torch_utils import get_device
logger = logging.getLogger(__name__)

class TorchLearner(Learner):
    framework: str = 'torch'

    def __init__(self, *, framework_hyperparameters: Optional[FrameworkHyperparameters]=None, **kwargs):
        if False:
            return 10
        super().__init__(framework_hyperparameters=framework_hyperparameters or FrameworkHyperparameters(), **kwargs)
        self._device = None
        self._torch_compile_forward_train = False
        self._torch_compile_complete_update = False
        if self._framework_hyperparameters.torch_compile:
            if self._framework_hyperparameters.what_to_compile == TorchCompileWhatToCompile.COMPLETE_UPDATE:
                self._torch_compile_complete_update = True
                self._compiled_update_initialized = False
            else:
                self._torch_compile_forward_train = True

    @OverrideToImplementCustomLogic
    @override(Learner)
    def configure_optimizers_for_module(self, module_id: ModuleID, hps: LearnerHyperparameters) -> None:
        if False:
            print('Hello World!')
        module = self._module[module_id]
        optimizer = torch.optim.Adam(self.get_parameters(module))
        params = self.get_parameters(module)
        self.register_optimizer(module_id=module_id, optimizer=optimizer, params=params, lr_or_lr_schedule=hps.learning_rate)

    def _uncompiled_update(self, batch: NestedDict, **kwargs):
        if False:
            print('Hello World!')
        'Performs a single update given a batch of data.'
        fwd_out = self.module.forward_train(batch)
        loss_per_module = self.compute_loss(fwd_out=fwd_out, batch=batch)
        gradients = self.compute_gradients(loss_per_module)
        postprocessed_gradients = self.postprocess_gradients(gradients)
        self.apply_gradients(postprocessed_gradients)
        return (fwd_out, loss_per_module, self._metrics)

    @override(Learner)
    def compute_gradients(self, loss_per_module: Mapping[str, TensorType], **kwargs) -> ParamDict:
        if False:
            print('Hello World!')
        for optim in self._optimizer_parameters:
            optim.zero_grad(set_to_none=True)
        loss_per_module[ALL_MODULES].backward()
        grads = {pid: p.grad for (pid, p) in self._params.items()}
        return grads

    @override(Learner)
    def apply_gradients(self, gradients_dict: ParamDict) -> None:
        if False:
            while True:
                i = 10
        for optim in self._optimizer_parameters:
            optim.zero_grad(set_to_none=True)
        for (pid, grad) in gradients_dict.items():
            self._params[pid].grad = grad
        for optim in self._optimizer_parameters:
            optim.step()

    @override(Learner)
    def set_module_state(self, state: Mapping[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        'Sets the weights of the underlying MultiAgentRLModule'
        state = convert_to_torch_tensor(state, device=self._device)
        return self._module.set_state(state)

    @override(Learner)
    def _save_optimizers(self, path: Union[str, pathlib.Path]) -> None:
        if False:
            i = 10
            return i + 15
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        optim_state = self.get_optimizer_state()
        for (name, state) in optim_state.items():
            torch.save(state, path / f'{name}.pt')

    @override(Learner)
    def _load_optimizers(self, path: Union[str, pathlib.Path]) -> None:
        if False:
            for i in range(10):
                print('nop')
        path = pathlib.Path(path)
        if not path.exists():
            raise ValueError(f'Directory {path} does not exist.')
        state = {}
        for name in self._named_optimizers.keys():
            state[name] = torch.load(path / f'{name}.pt')
        self.set_optimizer_state(state)

    @override(Learner)
    def get_optimizer_state(self) -> Mapping[str, Any]:
        if False:
            while True:
                i = 10
        optimizer_name_state = {}
        for (name, optim) in self._named_optimizers.items():
            optim_state_dict = optim.state_dict()
            optim_state_dict_cpu = copy_torch_tensors(optim_state_dict, device='cpu')
            optimizer_name_state[name] = optim_state_dict_cpu
        return optimizer_name_state

    @override(Learner)
    def set_optimizer_state(self, state: Mapping[str, Any]) -> None:
        if False:
            print('Hello World!')
        for (name, state_dict) in state.items():
            if name not in self._named_optimizers:
                raise ValueError(f'Optimizer {name} in `state` is not known.Known optimizers are {self._named_optimizers.keys()}')
            optim = self._named_optimizers[name]
            state_dict_correct_device = copy_torch_tensors(state_dict, device=self._device)
            optim.load_state_dict(state_dict_correct_device)

    @override(Learner)
    def get_param_ref(self, param: Param) -> Hashable:
        if False:
            return 10
        return param

    @override(Learner)
    def get_parameters(self, module: RLModule) -> Sequence[Param]:
        if False:
            for i in range(10):
                print('nop')
        return list(module.parameters())

    @override(Learner)
    def _convert_batch_type(self, batch: MultiAgentBatch) -> MultiAgentBatch:
        if False:
            while True:
                i = 10
        batch = convert_to_torch_tensor(batch.policy_batches, device=self._device)
        length = max((len(b) for b in batch.values()))
        batch = MultiAgentBatch(batch, env_steps=length)
        return batch

    @override(Learner)
    def add_module(self, *, module_id: ModuleID, module_spec: SingleAgentRLModuleSpec) -> None:
        if False:
            return 10
        super().add_module(module_id=module_id, module_spec=module_spec)
        module = self._module[module_id]
        if self._torch_compile_forward_train:
            module.compile(self._framework_hyperparameters.torch_compile_cfg)
        elif self._torch_compile_complete_update:
            torch._dynamo.reset()
            self._compiled_update_initialized = False
            torch_compile_cfg = self._framework_hyperparameters.torch_compile_cfg
            self._possibly_compiled_update = torch.compile(self._uncompiled_update, backend=torch_compile_cfg.torch_dynamo_backend, mode=torch_compile_cfg.torch_dynamo_mode, **torch_compile_cfg.kwargs)
        if isinstance(module, TorchRLModule):
            self._module[module_id].to(self._device)
            if self.distributed:
                if self._torch_compile_complete_update or self._torch_compile_forward_train:
                    raise ValueError('Using torch distributed and torch compile together tested for now. Please disable torch compile.')
                self._module.add_module(module_id, TorchDDPRLModule(module), override=True)

    @override(Learner)
    def remove_module(self, module_id: ModuleID) -> None:
        if False:
            return 10
        super().remove_module(module_id)
        if self._torch_compile_complete_update:
            torch._dynamo.reset()
            self._compiled_update_initialized = False
            torch_compile_cfg = self._framework_hyperparameters.torch_compile_cfg
            self._possibly_compiled_update = torch.compile(self._uncompiled_update, backend=torch_compile_cfg.torch_dynamo_backend, mode=torch_compile_cfg.torch_dynamo_mode, **torch_compile_cfg.kwargs)

    @override(Learner)
    def build(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Builds the TorchLearner.\n\n        This method is specific to TorchLearner. Before running super() it will\n        initialze the device properly based on the `_use_gpu` and `_distributed`\n        flags, so that `_make_module()` can place the created module on the correct\n        device. After running super() it will wrap the module in a TorchDDPRLModule\n        if `_distributed` is True.\n        '
        if self._use_gpu:
            if self._distributed:
                self._device = get_device()
            else:
                assert self._local_gpu_idx < torch.cuda.device_count(), f'local_gpu_idx {self._local_gpu_idx} is not a valid GPU id or is  not available.'
                self._device = torch.device(self._local_gpu_idx)
        else:
            self._device = torch.device('cpu')
        super().build()
        if self._torch_compile_complete_update:
            torch._dynamo.reset()
            self._compiled_update_initialized = False
            torch_compile_cfg = self._framework_hyperparameters.torch_compile_cfg
            self._possibly_compiled_update = torch.compile(self._uncompiled_update, backend=torch_compile_cfg.torch_dynamo_backend, mode=torch_compile_cfg.torch_dynamo_mode, **torch_compile_cfg.kwargs)
        else:
            if self._torch_compile_forward_train:
                if isinstance(self._module, TorchRLModule):
                    self._module.compile(self._framework_hyperparameters.torch_compile_cfg)
                elif isinstance(self._module, MultiAgentRLModule):
                    for module in self._module._rl_modules.values():
                        if isinstance(self._module, TorchRLModule):
                            module.compile(self._framework_hyperparameters.torch_compile_cfg)
                else:
                    raise ValueError('Torch compile is only supported for TorchRLModule and MultiAgentRLModule.')
            self._possibly_compiled_update = self._uncompiled_update
        self._make_modules_ddp_if_necessary()

    @override(Learner)
    def _update(self, batch: NestedDict) -> Tuple[Any, Any, Any]:
        if False:
            return 10
        if self._torch_compile_complete_update and (not self._compiled_update_initialized):
            self._compiled_update_initialized = True
            return self._uncompiled_update(batch)
        else:
            return self._possibly_compiled_update(batch)

    @OverrideToImplementCustomLogic
    def _make_modules_ddp_if_necessary(self) -> None:
        if False:
            return 10
        'Default logic for (maybe) making all Modules within self._module DDP.'
        if self._distributed:
            if isinstance(self._module, TorchRLModule):
                self._module = TorchDDPRLModule(self._module)
            else:
                assert isinstance(self._module, MultiAgentRLModule)
                for key in self._module.keys():
                    sub_module = self._module[key]
                    if isinstance(sub_module, TorchRLModule):
                        self._module.add_module(key, TorchDDPRLModule(sub_module), override=True)

    def _is_module_compatible_with_learner(self, module: RLModule) -> bool:
        if False:
            return 10
        return isinstance(module, nn.Module)

    @override(Learner)
    def _check_registered_optimizer(self, optimizer: Optimizer, params: Sequence[Param]) -> None:
        if False:
            return 10
        super()._check_registered_optimizer(optimizer, params)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(f'The optimizer ({optimizer}) is not a torch.optim.Optimizer! Only use torch.optim.Optimizer subclasses for TorchLearner.')
        for param in params:
            if not isinstance(param, torch.Tensor):
                raise ValueError(f'One of the parameters ({param}) in the registered optimizer is not a torch.Tensor!')

    @override(Learner)
    def _make_module(self) -> MultiAgentRLModule:
        if False:
            for i in range(10):
                print('nop')
        module = super()._make_module()
        self._map_module_to_device(module)
        return module

    def _map_module_to_device(self, module: MultiAgentRLModule) -> None:
        if False:
            while True:
                i = 10
        'Moves the module to the correct device.'
        if isinstance(module, torch.nn.Module):
            module.to(self._device)
        else:
            for key in module.keys():
                if isinstance(module[key], torch.nn.Module):
                    module[key].to(self._device)

    @override(Learner)
    def _get_tensor_variable(self, value, dtype=None, trainable=False) -> 'torch.Tensor':
        if False:
            while True:
                i = 10
        return torch.tensor(value, requires_grad=trainable, device=self._device, dtype=dtype or (torch.float32 if isinstance(value, float) else torch.int32 if isinstance(value, int) else None))

    @staticmethod
    @override(Learner)
    def _get_optimizer_lr(optimizer: 'torch.optim.Optimizer') -> float:
        if False:
            return 10
        for g in optimizer.param_groups:
            return g['lr']

    @staticmethod
    @override(Learner)
    def _set_optimizer_lr(optimizer: 'torch.optim.Optimizer', lr: float) -> None:
        if False:
            i = 10
            return i + 15
        for g in optimizer.param_groups:
            g['lr'] = lr

    @staticmethod
    @override(Learner)
    def _get_clip_function() -> Callable:
        if False:
            i = 10
            return i + 15
        from ray.rllib.utils.torch_utils import clip_gradients
        return clip_gradients