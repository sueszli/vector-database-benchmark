from __future__ import annotations
from typing import List, Dict, Union, overload
import torch
from torch import Tensor
from ..base.compressor import Compressor, Quantizer
from ..base.wrapper import ModuleWrapper
from ..base.target_space import QuantizationTargetSpace
from ..utils import Evaluator, _EVALUATOR_DOCSTRING

class PtqQuantizer(Quantizer):
    __doc__ = '\n    Post Training Quantization\n\n    Parameters\n    ----------\n    model\n        Model to be quantized.\n    config_list\n        A list of dict, each dict configure which module need to be quantized, and how to quantize.\n        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.\n    evaluator\n        {evaluator_docstring}\n\n    Examples\n    --------\n        >>> from nni.compression.quantization import PtqQuantizer\n        >>> from nni.compression.utils import TorchEvaluator\n        >>> model = ...\n        >>> optimizer = ...\n        >>> max_steps, max_epochs = ..., ...\n        >>> evaluator = TorchEvaluator(train, optimizer, training_step)\n        >>> quantizer = PtqQuantizer(model, configure_list, evaluator)\n        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)\n    '.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator):
        if False:
            print('Hello World!')
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            i = 10
            return i + 15
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, existed_wrappers: Dict[str, ModuleWrapper] | None=None, is_bias_correction: bool=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(model, config_list, evaluator, existed_wrappers)
        self.evaluator: Evaluator
        self.is_compressed = False
        self.is_bias_correction = is_bias_correction
        self.register_ptq_apply_method()
        self.register_track_func()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], evaluator: Evaluator | None=None):
        if False:
            return 10
        return super().from_compressor(compressor, new_config_list, evaluator=evaluator)

    def register_ptq_apply_method(self):
        if False:
            while True:
                i = 10
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                target_space.apply_method = 'clamp_round' if self.is_compressed else 'bypass'

    def register_track_func(self):
        if False:
            for i in range(10):
                print('nop')
        for (module_name, _) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.track_min_max_val)

    def track_min_max_val(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if False:
            i = 10
            return i + 15

        def amin_reduce_func(converted_target: Tensor):
            if False:
                while True:
                    i = 10
            return converted_target.detach().amin(dim=-1)

        def amax_reduce_func(converted_target: Tensor):
            if False:
                print('Hello World!')
            return converted_target.detach().amax(dim=-1)
        if target_name not in wrapper.quantization_target_spaces:
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space._scaler:
            current_amin = target_space._scaler.shrink(target, amin_reduce_func, keepdim=True)
            current_amax = target_space._scaler.shrink(target, amax_reduce_func, keepdim=True)
        else:
            current_amin = target.detach().reshape(-1).amin(-1)
            current_amax = target.detach().reshape(-1).amax(-1)
        target_space.tracked_max = update_tracked_value(target_space.tracked_max, current_amax, 'max')
        target_space.tracked_min = update_tracked_value(target_space.tracked_min, current_amin, 'min')

    def update_scale_zp(self):
        if False:
            while True:
                i = 10
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                (scale, zero_point) = compute_scale_zp(target_space)
                (target_space.scale, target_space.zero_point) = (scale, zero_point)

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            for i in range(10):
                print('nop')
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        if False:
            print('Hello World!')
        module_name_param_dict = self.patch_optimizer_param_group()
        if len(module_name_param_dict) > 0:
            evaluator.patch_optim_param_group(module_name_param_dict)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        if False:
            i = 10
            return i + 15
        self.evaluator.evaluate()
        self.update_scale_zp()
        self.is_compressed = True
        self.register_ptq_apply_method()
        if self.is_bias_correction:
            self.bias_correction()

    def bias_correction(self):
        if False:
            i = 10
            return i + 15
        assert self.is_bias_correction, f'is_bias_correction should be True, but got {self.is_bias_correction}'
        for (module_name, _) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            setattr(wrapper, 'is_bias_correction', self.is_bias_correction)
        self.evaluator.evaluate()
        for (module_name, _) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.update_bias()
            delattr(wrapper, 'is_bias_correction')
            delattr(wrapper, 'bias_correction')
            delattr(wrapper, 'bias_element_num')
        self.evaluator.evaluate()
        self.update_scale_zp()

def compute_scale_zp(target_space: QuantizationTargetSpace):
    if False:
        print('Hello World!')
    if target_space.tracked_max is None or target_space.tracked_min is None:
        return
    tracked_min = torch.min(target_space.tracked_min, torch.zeros_like(target_space.tracked_min))
    tracked_max = torch.max(target_space.tracked_max, torch.zeros_like(target_space.tracked_max))
    zero_point = torch.zeros_like(tracked_min)
    if target_space.quant_scheme in ['symmetric', None]:
        abs_max = torch.max(torch.abs(tracked_min), torch.abs(tracked_max))
        scale = abs_max / (float(target_space.qmax - target_space.qmin) / 2)
        scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
        zero_point_val = (target_space.qmax + target_space.qmin + 1) // 2
        zero_point = torch.full_like(zero_point, zero_point_val)
    elif target_space.quant_scheme == 'affine':
        scale = (tracked_max - tracked_min) / float(target_space.qmax - target_space.qmin)
        scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
        zero_point = target_space.qmin - torch.round(tracked_min / scale)
    else:
        raise RuntimeError(f'Unknown quant_scheme {target_space.quant_scheme}')
    zero_point = torch.clamp(zero_point, target_space.qmin, target_space.qmax)
    return (scale, zero_point)

def update_tracked_value(original_val: Union[Tensor, None], current_val: Tensor, mode: str='max'):
    if False:
        return 10
    if original_val is None:
        return current_val
    assert current_val is not None
    assert original_val.shape == current_val.shape
    if mode == 'max':
        return torch.max(original_val, current_val)
    elif mode == 'min':
        return torch.min(original_val, current_val)
    else:
        raise TypeError(f'Type:{mode} is not supported')