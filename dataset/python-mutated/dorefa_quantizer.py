from __future__ import annotations
import logging
from typing import List, Dict, Union, overload
import torch
import torch.nn as nn
from torch import Tensor
from nni.common.version import torch_version_is_2
from ..base.compressor import Compressor, Quantizer
from ..base.wrapper import ModuleWrapper
from ..utils import Evaluator, _EVALUATOR_DOCSTRING
from ..base.target_space import TargetType, QuantizationTargetSpace
ACTIVATION_LIST = [nn.ReLU, nn.RReLU, nn.LeakyReLU, nn.PReLU, nn.Softplus, nn.ELU, nn.CELU, nn.SELU, nn.GELU, nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.Softsign, nn.Hardtanh, nn.Threshold, nn.Tanhshrink, nn.Softshrink, nn.Hardshrink, nn.LogSigmoid, nn.Softmin, nn.Softmax, nn.LogSoftmax, nn.Hardswish]
_logger = logging.getLogger(__name__)
is_proper_torch_version = torch_version_is_2()

class DoReFaQuantizer(Quantizer):
    __doc__ = '\n    Dorefa-Quantizer, as defined in:\n    `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`__,\n    authors Shuchang Zhou and Yuxin Wu provide an algorithm named DoReFa to quantize the weight, activation and gradients with training.\n\n    Parameters\n    ----------\n    model\n        Model to be quantized.\n    config_list\n        A list of dict, each dict configure which module need to be quantized, and how to quantize.\n        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.\n    evaluator\n        {evaluator_docstring}\n\n    Examples\n    --------\n        >>> from nni.compression.quantization import DoReFaQuantizer\n        >>> from nni.compression.utils import TorchEvaluator\n        >>> model = ...\n        >>> optimizer = ...\n        >>> max_steps, max_epochs = ..., ...\n        >>> evaluator = TorchEvaluator(train, optimizer, training_step)\n        >>> quantizer = DoReFaQuantizer(model, configure_list, evaluator)\n        >>> _, calibration_config = quantizer.compress(max_steps, max_epochs)\n    '.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator):
        if False:
            print('Hello World!')
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            print('Hello World!')
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            return 10
        super().__init__(model, config_list, evaluator, existed_wrappers=existed_wrappers)
        self.evaluator: Evaluator
        self.is_init = False
        self.check_validation()
        self.register_dorefa_apply_method()
        self.register_track_func()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], evaluator: Evaluator | None=None):
        if False:
            for i in range(10):
                print('nop')
        return super().from_compressor(compressor, new_config_list, evaluator=evaluator)

    def check_validation(self) -> None:
        if False:
            i = 10
            return i + 15
        for ts in self._target_spaces.values():
            for target_space in ts.values():
                assert target_space.quant_scheme != None
                if target_space.type is TargetType.PARAMETER and target_space.quant_scheme != 'affine':
                    warn_msg = f'Only supports affine mode for weight quantization, bug got {target_space.quant_scheme}'
                    _logger.warning(warn_msg)
                elif target_space.type is TargetType.OUTPUT:
                    module = target_space._wrapper.module
                    fused_modules = target_space._wrapper.fused_modules
                    if not isinstance(module, tuple(ACTIVATION_LIST)) and (not (fused_modules and any([isinstance(item, tuple(ACTIVATION_LIST)) for item in fused_modules[1:]]))):
                        raise ValueError('Output quantization is only supported for activation function or' + f'activation module fusion, but got {type(module)}')
                    if target_space.quant_scheme != 'affine':
                        warn_msg = f'Only supports affine mode for output quantization, bug got {target_space.quant_scheme}'
                        _logger.warning(warn_msg)
                if target_space._scaler is not None:
                    raise ValueError("DoRefa Qauntizer doesn't support for granularity, please set it to False")

    def _quant_dequant_gradient_hook(self, target_space: QuantizationTargetSpace) -> None:
        if False:
            while True:
                i = 10

        def quant_dequant_gradient(module: nn.Module, grad_output):
            if False:
                print('Hello World!')
            tracked_max = torch.tensor(1.0 + 0.5 / (2 ** target_space.quant_bits - 1)).to(grad_output[0].device)
            tracked_min = torch.tensor(0 - 0.5 / (2 ** target_space.quant_bits - 1)).to(grad_output[0].device)
            (scale, zero_point) = init_scale_zp(tracked_max, tracked_min, target_space.qmax, target_space.qmin, 'affine')
            new_grad_output = []
            for g_o in grad_output:
                grad_o = torch.abs(g_o.clone().detach())
                dim_lis = list(range(len(grad_o.shape)))
                dim_lis.pop(0)
                max_grad = torch.amax(grad_o, dim=dim_lis, keepdim=True)
                uniform_k = torch.zeros_like(max_grad).to(g_o.device)
                N_k = uniform_k.uniform_(-0.5, 0.5) / (2 ** target_space.quant_bits - 1)
                q_grad_o = g_o / (2 * max_grad) + 0.5 + N_k
                quantized_grad = zero_point + q_grad_o / scale
                quantized_grad = torch.round(torch.clamp(quantized_grad, target_space.qmin, target_space.qmax))
                dequantized_grad = (quantized_grad - zero_point) * scale
                new_grad_output.append((dequantized_grad - 0.5) * 2 * max_grad)
            return tuple(new_grad_output)
        target_space._wrapper.module.register_full_backward_pre_hook(quant_dequant_gradient)

    def register_output_backward_hook(self):
        if False:
            print('Hello World!')
        for ts in self._target_spaces.values():
            is_output = any([target_space.type is TargetType.OUTPUT for target_space in ts.values()])
            is_param = any([target_space.type is TargetType.PARAMETER for target_space in ts.values()])
            if is_param and (not is_output):
                if is_proper_torch_version:
                    for target_space in ts.values():
                        if target_space.type is TargetType.PARAMETER:
                            self._quant_dequant_gradient_hook(target_space)
                            break
                else:
                    warn_msg = f'Gradient quantization is only supported for torch version >= 2.0.0'
                    _logger.warning(warn_msg)

    def register_dorefa_apply_method(self):
        if False:
            while True:
                i = 10
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                if target_space.type is TargetType.PARAMETER:
                    target_space.apply_method = 'dorefa_clamp_round_weight'
                elif target_space.type is TargetType.INPUT:
                    target_space.apply_method = 'clamp_round'
                elif target_space.type is TargetType.OUTPUT:
                    target_space.apply_method = 'dorefa_clamp_round_output'

    def register_track_func(self):
        if False:
            while True:
                i = 10
        for (module_name, _) in self._target_spaces.items():
            wrapper = self._module_wrappers[module_name]
            wrapper.register_track_func(self.initialize_scale_zp)
            wrapper.register_track_func(self.update_scale_zp)

    def update_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor) -> None:
        if False:
            print('Hello World!')
        if not self.check_target(wrapper, target_name):
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space.type is not TargetType.INPUT:
            return
        current_amin = target.detach().reshape(-1).amin(-1)
        current_amax = target.detach().reshape(-1).amax(-1)
        tracked_min = torch.min(current_amin, torch.zeros_like(current_amin))
        tracked_max = torch.max(current_amax, torch.zeros_like(current_amax))
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

    def initialize_scale_zp(self, wrapper: ModuleWrapper, target_name: str, target: Tensor):
        if False:
            while True:
                i = 10
        if self.is_init or not self.check_target(wrapper, target_name):
            return
        target_space = wrapper.quantization_target_spaces[target_name]
        if target_space.type is TargetType.INPUT:
            return
        elif target_space.type in [TargetType.OUTPUT, TargetType.PARAMETER]:
            tracked_max = torch.tensor(1.0).to(target.device)
            tracked_min = torch.tensor(0.0).to(target.device)
            (scale, zero_point) = init_scale_zp(tracked_max, tracked_min, target_space.qmax, target_space.qmin, 'affine')
        else:
            raise RuntimeError(f'Unknown target_name {target_name}')
        (target_space.scale, target_space.zero_point) = (scale, zero_point)

    def register_trigger(self, evaluator: Evaluator):
        if False:
            return 10

        def optimizer_task():
            if False:
                for i in range(10):
                    print('nop')
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
        self.register_output_backward_hook()
        module_name_param_dict = self.patch_optimizer_param_group()
        if len(module_name_param_dict) > 0:
            evaluator.patch_optim_param_group(module_name_param_dict)
        self.register_trigger(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator) -> None:
        if False:
            while True:
                i = 10
        pass

def init_scale_zp(tracked_max: Tensor, tracked_min: Tensor, qmax: int, qmin: int, quant_scheme: Union[str, None]=None):
    if False:
        i = 10
        return i + 15
    tracked_min = torch.min(tracked_min, torch.zeros_like(tracked_min))
    tracked_max = torch.max(tracked_max, torch.zeros_like(tracked_max))
    zero_point = torch.zeros_like(tracked_min)
    if quant_scheme == 'affine':
        scale = (tracked_max - tracked_min) / float(qmax - qmin)
        scale = torch.max(scale, torch.full_like(scale, torch.finfo(torch.float32).eps))
        zero_point = qmin - torch.round(tracked_min / scale)
    elif quant_scheme in ['symmetric', None]:
        raise ValueError(f'Unsupported quant_scheme {quant_scheme}')
    else:
        raise RuntimeError(f'Unknown quant_scheme {quant_scheme}')
    zero_point = torch.clamp(zero_point, qmin, qmax)
    return (scale, zero_point)