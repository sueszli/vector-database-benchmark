from __future__ import annotations
from typing import Dict, List, overload
import torch
from torch import Tensor
from ..base.compressor import Compressor, Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING

class QATQuantizer(Quantizer):
    __doc__ = "\n    Quantizer defined in:\n    `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference\n    <http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf>`__\n\n    Authors Benoit Jacob and Skirmantas Kligys provide an algorithm to quantize the model with training.\n\n    ..\n\n        We propose an approach that simulates quantization effects in the forward pass of training.\n        Backpropagation still happens as usual, and all weights and biases are stored in floating point\n        so that they can be easily nudged by small amounts.\n        The forward propagation pass however simulates quantized inference as it will happen in the inference engine,\n        by implementing in floating-point arithmetic the rounding behavior of the quantization scheme:\n\n        * Weights are quantized before they are convolved with the input. If batch normalization (see [17]) is used for the layer,\n          the batch normalization parameters are “folded into” the weights before quantization.\n\n        * Activations are quantized at points where they would be during inference,\n          e.g. after the activation function is applied to a convolutional or fully connected layer's output,\n          or after a bypass connection adds or concatenates the outputs of several layers together such as in ResNets.\n\n    Parameters\n    ----------\n    model\n        Model to be quantized.\n    config_list\n        A list of dict, each dict configure which module need to be quantized, and how to quantize.\n        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.\n    evaluator\n        {evaluator_docstring}\n    quant_start_step\n        The steps for warmup training before QAT begin.\n\n    Examples\n    --------\n        >>> from nni.compression.quantization import QATQuantizer\n        >>> from nni.compression.utils import TorchEvaluator\n        >>> model = ...\n        >>> optimizer = ...\n        >>> max_steps, max_epochs = ..., ...\n        >>> evaluator = TorchEvaluator(train, optimizer, training_step)\n        >>> quantizer = QATQuantizer(model, configure_list, evaluator)\n        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)\n    ".format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, quant_start_step: int=0):
        if False:
            while True:
                i = 10
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, quant_start_step: int=0, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            return 10
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, quant_start_step: int=0, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            print('Hello World!')
        super().__init__(model, config_list, evaluator, existed_wrappers)
        self.evaluator: Evaluator
        self.quant_start_step = max(quant_start_step, 0)
        self.current_step = 0
        self.register_qat_apply_method()
        self.register_track_func()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], quant_start_step: int=0, evaluator: Evaluator | None=None):
        if False:
            i = 10
            return i + 15
        return super().from_compressor(compressor, new_config_list, quant_start_step=quant_start_step, evaluator=evaluator)

    def register_qat_apply_method(self):
        if False:
            print('Hello World!')
        if self.current_step < self.quant_start_step:
            for (_, ts) in self._target_spaces.items():
                for (_, target_space) in ts.items():
                    target_space.apply_method = 'bypass'
        elif self.current_step == self.quant_start_step:
            for (_, ts) in self._target_spaces.items():
                for (_, target_space) in ts.items():
                    target_space.apply_method = 'qat_clamp_round'
        else:
            pass

    def register_track_func(self):
        if False:
            print('Hello World!')
        for (module_name, _) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.track_min_max_val)
            wrapper.register_track_func(self.update_scale_zp)

    def track_min_max_val(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if False:
            for i in range(10):
                print('nop')
        if not wrapper.training or not self.check_target(wrapper, target_name):
            return
        return track_min_max_val(wrapper, target_name, target)

    def update_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if False:
            for i in range(10):
                print('nop')
        if not wrapper.training or self.current_step < self.quant_start_step or (not self.check_target(wrapper, target_name)):
            return
        if target_name in wrapper.quantization_target_spaces:
            target_space = wrapper.quantization_target_spaces[target_name]
            if target_space.tracked_max is None or target_space.tracked_min is None:
                return
            tracked_min = torch.min(target_space.tracked_min, torch.zeros_like(target_space.tracked_min))
            tracked_max = torch.max(target_space.tracked_max, torch.zeros_like(target_space.tracked_max))
            zero_point = torch.zeros_like(tracked_min)
            (qmin, qmax) = (target_space.qmin, target_space.qmax)
            assert isinstance(qmin, int) and isinstance(qmax, int)
            if target_space.quant_scheme in ['symmetric', None]:
                abs_max = torch.max(torch.abs(tracked_min), torch.abs(tracked_max))
                scale = abs_max / (float(qmax - qmin) / 2)
                scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
                zero_point_val = (qmax + qmin + 1) // 2
                zero_point = torch.full_like(zero_point, zero_point_val)
            elif target_space.quant_scheme == 'affine':
                scale = (tracked_max - tracked_min) / float(qmax - qmin)
                scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
                zero_point = qmin - torch.round(tracked_min / scale)
            else:
                raise RuntimeError(f'Unknown quant_scheme {target_space.quant_scheme}')
            zero_point = torch.clamp(zero_point, qmin, qmax)
            (target_space.scale, target_space.zero_point) = (scale, zero_point)

    def track_forward(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().track_forward(*args, **kwargs)
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                if target_space.scale is None:
                    assert target_space.tracked_max is not None
                    target_space.scale = torch.empty_like(target_space.tracked_max)
                if target_space.zero_point is None:
                    assert target_space.tracked_max is not None
                    target_space.zero_point = torch.empty_like(target_space.tracked_max)

    def register_trigger(self, evaluator: Evaluator):
        if False:
            while True:
                i = 10

        def optimizer_task():
            if False:
                for i in range(10):
                    print('nop')
            self.current_step += 1
            if self.current_step == self.quant_start_step:
                self.register_qat_apply_method()
        evaluator.patch_optimizer_step(before_step_tasks=[], after_step_tasks=[optimizer_task])

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            print('Hello World!')
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator) -> None:
        if False:
            while True:
                i = 10
        module_name_param_dict = self.patch_optimizer_param_group()
        if len(module_name_param_dict) > 0:
            evaluator.patch_optim_param_group(module_name_param_dict)
        self.register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

def track_min_max_val(wrapper: ModuleWrapper, target_name: str, target: Tensor):
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
            while True:
                i = 10
        return converted_target.detach().amax(dim=-1)
    if target_name in wrapper.quantization_target_spaces:
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space._scaler:
            current_amin = target_space._scaler.shrink(target, amin_reduce_func, keepdim=True)
            current_amax = target_space._scaler.shrink(target, amax_reduce_func, keepdim=True)
        else:
            current_amin = target.detach().reshape(-1).amin(-1)
            current_amax = target.detach().reshape(-1).amax(-1)
        target_space.tracked_max = current_amax if target_space.tracked_max is None else update_ema(target_space.tracked_max, current_amax, 0.99)
        target_space.tracked_min = current_amin if target_space.tracked_min is None else update_ema(target_space.tracked_min, current_amin, 0.99)

def update_ema(biased_ema: Tensor, current_val: Tensor, decay: float):
    if False:
        print('Hello World!')
    '\n    Exponential moving average method.\n    '
    return biased_ema * decay + (1 - decay) * current_val