from __future__ import annotations
import logging
from typing import List, Dict, overload
import torch
from torch import Tensor
from ..base.compressor import Compressor, Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING
from ..base.target_space import TargetType
_logger = logging.getLogger(__name__)

class LsqQuantizer(Quantizer):
    __doc__ = "\n    LsqQuantizer, as defined in: `LEARNED STEP SIZE QUANTIZATION <https://arxiv.org/pdf/1902.08153.pdf>`__,\n    authors Steven K. Esser and Jeffrey L. McKinstry provide an algorithm to train the scales with gradients.\n\n    ..\n\n        The authors introduce a novel means to estimate and scale the task loss gradient at each weight and activation\n        layer's quantizer step size, such that it can be learned in conjunction with other network parameters.\n\n    Parameters\n    ----------\n    model\n        Model to be quantized.\n    config_list\n        A list of dict, each dict configure which module need to be quantized, and how to quantize.\n        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.\n    evaluator\n        {evaluator_docstring}\n\n    Examples\n    --------\n        >>> from nni.compression.quantization import LsqQuantizer\n        >>> from nni.compression.utils import TorchEvaluator\n        >>> model = ...\n        >>> optimizer = ...\n        >>> max_steps, max_epochs = ..., ...\n        >>> evaluator = TorchEvaluator(train, optimizer, training_step)\n        >>> quantizer = LsqQuantizer(model, configure_list, evaluator)\n        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)\n    ".format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator):
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            i = 10
            return i + 15
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            i = 10
            return i + 15
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.is_init = False
        self.check_validation()
        self.register_scale()
        self.register_lsq_apply_method()
        self.register_track_func()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], evaluator: Evaluator | None=None):
        if False:
            for i in range(10):
                print('nop')
        return super().from_compressor(compressor, new_config_list, evaluator=evaluator)

    def check_validation(self) -> None:
        if False:
            return 10
        for ts in self._target_spaces.values():
            for target_space in ts.values():
                if target_space.quant_scheme != 'symmetric':
                    warn_msg = f'LsqQuantizer only supports symmetric mode, but got {target_space.quant_scheme}'
                    _logger.warning(warn_msg)
                if target_space.quant_dtype.startswith('uint') and target_space.type is TargetType.PARAMETER:
                    warn_msg = f'In the LsqQuantizer, quantization of parameters only supports int type'
                    _logger.warning(warn_msg)

    def register_track_func(self):
        if False:
            return 10
        for (module_name, _) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.init_scale)

    def init_scale(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if False:
            print('Hello World!')

        def mean_reduce_func(converted_target: Tensor) -> torch.Tensor:
            if False:
                return 10
            return converted_target.detach().mean(dim=-1)
        if self.is_init or not self.check_target(wrapper, target_name):
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        init_target = torch.tensor([0.01]).to(target.device)
        if not target_space._scaler:
            target_space.scale.data = init_target.view(1)
            target_space.zero_point = torch.tensor(0.0).to(target.device)
        else:
            new_target = init_target.expand(target.shape).to(target.device)
            new_target_scale = target_space._scaler.shrink(new_target, mean_reduce_func, keepdim=True)
            target_space.scale.data = new_target_scale
            target_space.zero_point = torch.zeros_like(new_target_scale)

    def register_lsq_apply_method(self):
        if False:
            for i in range(10):
                print('nop')
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                target_space.apply_method = 'lsq_clamp_round'

    def register_scale(self):
        if False:
            while True:
                i = 10
        for (module_name, ts) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            for (target_name, _) in ts.items():
                if hasattr(wrapper, f'{target_name}_scale'):
                    delattr(wrapper, f'{target_name}_scale')
                try:
                    device = next(wrapper.parameters()).device
                except StopIteration:
                    try:
                        device = next(wrapper.buffers()).device
                    except StopIteration:
                        device = next(self.bound_model.parameters()).device
                param = torch.nn.Parameter(torch.Tensor([0.01]).to(device))
                wrapper.register_parameter(f'{target_name}_scale', param)

    def patch_optimizer_param_group(self):
        if False:
            return 10
        module_name_param_dict = super().patch_optimizer_param_group()
        for (module_name, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                if module_name not in module_name_param_dict:
                    module_name_param_dict[module_name] = []
                module_name_param_dict[module_name].append(target_space.scale)
        return module_name_param_dict

    def register_trigger(self, evaluator: Evaluator):
        if False:
            for i in range(10):
                print('nop')

        def optimizer_task():
            if False:
                for i in range(10):
                    print('nop')
            self.is_init = True
        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            print('Hello World!')
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        if False:
            i = 10
            return i + 15
        module_name_param_dict = self.patch_optimizer_param_group()
        if len(module_name_param_dict) > 0:
            evaluator.patch_optim_param_group(module_name_param_dict)
        self.register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        if False:
            while True:
                i = 10
        pass