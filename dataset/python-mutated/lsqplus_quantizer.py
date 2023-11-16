from __future__ import annotations
import logging
from typing import List, Dict, overload
import torch
from torch import Tensor
from ..base.compressor import Compressor, Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING
_logger = logging.getLogger(__name__)

class LsqPlusQuantizer(Quantizer):
    __doc__ = '\n    LsqPlusQuantizer, as defined in: `LSQ+: Improving low-bit quantization through learnable offsets and better\ninitialization <https://arxiv.org/pdf/2004.09576.pdf>`__,\n    authors ybhalgat, jinwonl, markusn, tijmen provide an algorithm to train the scale and zero_point with gradients.\n\n    ..\n        The proposed LSQ+ (Learnable Step Size Quantization Plus) method introduces learnable offsets and an improved initialization strategy to address limitations of traditional \n        low-bit quantization techniques, such as gradient mismatch and limited representational capacity. \n\n    Parameters\n    ----------\n    model\n        Model to be quantized.\n    config_list\n        A list of dict, each dict configure which module need to be quantized, and how to quantize.\n        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.\n    evaluator\n        {evaluator_docstring}\n\n    Examples\n    --------\n        >>> from nni.compression.quantization import LsqQuantizer\n        >>> from nni.compression.utils import TorchEvaluator\n        >>> model = ...\n        >>> optimizer = ...\n        >>> max_steps, max_epochs = ..., ...\n        >>> evaluator = TorchEvaluator(train, optimizer, training_step)\n        >>> quantizer = LsqQuantizer(model, configure_list, evaluator)\n        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)\n    '.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator):
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            for i in range(10):
                print('nop')
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.is_init = False
        self.check_validation()
        self.register_scale_zp()
        self.register_lsq_plus_apply_method()
        self.register_track_func()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], evaluator: Evaluator | None=None):
        if False:
            for i in range(10):
                print('nop')
        return super().from_compressor(compressor, new_config_list, evaluator=evaluator)

    def check_validation(self) -> None:
        if False:
            print('Hello World!')
        for ts in self._target_spaces.values():
            for target_space in ts.values():
                if target_space.quant_scheme != 'affine':
                    warn_msg = f'LsqPlusQuantizer only supports affine mode, but got {target_space.quant_scheme}'
                    _logger.warning(warn_msg)

    def register_track_func(self):
        if False:
            while True:
                i = 10
        for (module_name, _) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.init_scale_zp)

    def init_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if False:
            for i in range(10):
                print('nop')

        def mean_reduce_func(converted_target: Tensor) -> torch.Tensor:
            if False:
                return 10
            return converted_target.detach().mean(dim=-1)
        if self.is_init or not self.check_target(wrapper, target_name):
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        init_scale_target = torch.tensor([0.01]).to(target.device)
        init_zp_target = torch.tensor([(target_space.qmax - target_space.qmin) / 2]).to(target.device)
        if not target_space._scaler:
            target_space.scale.data = init_scale_target
            target_space.zero_point.data = init_zp_target
        else:
            new_target_scale = init_scale_target.expand(target.shape).to(target.device)
            new_target_scale = target_space._scaler.shrink(new_target_scale, mean_reduce_func, keepdim=True)
            target_space.scale.data = new_target_scale
            new_target_zp = init_zp_target.expand(target.shape).to(target.device)
            new_target_zp = target_space._scaler.shrink(new_target_zp, mean_reduce_func, keepdim=True)
            target_space.zero_point.data = new_target_zp

    def register_lsq_plus_apply_method(self):
        if False:
            while True:
                i = 10
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                target_space.apply_method = 'lsq_plus_clamp_round'

    def register_scale_zp(self):
        if False:
            while True:
                i = 10
        for (module_name, ts) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            for (target_name, _) in ts.items():
                if hasattr(wrapper, f'{target_name}_scale'):
                    delattr(wrapper, f'{target_name}_scale')
                scale_param = torch.nn.Parameter()
                wrapper.register_parameter(f'{target_name}_scale', scale_param)
                if hasattr(wrapper, f'{target_name}_zero_point'):
                    delattr(wrapper, f'{target_name}_zero_point')
                zp_param = torch.nn.Parameter()
                wrapper.register_parameter(f'{target_name}_zero_point', zp_param)

    def patch_optimizer_param_group(self):
        if False:
            while True:
                i = 10
        module_name_param_dict = super().patch_optimizer_param_group()
        for (module_name, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                if module_name not in module_name_param_dict:
                    module_name_param_dict[module_name] = []
                module_name_param_dict[module_name].append(target_space.scale)
                module_name_param_dict[module_name].append(target_space.zero_point)
        return module_name_param_dict

    def register_trigger(self, evaluator: Evaluator):
        if False:
            i = 10
            return i + 15

        def optimizer_task():
            if False:
                return 10
            self.is_init = True
        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            for i in range(10):
                print('nop')
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