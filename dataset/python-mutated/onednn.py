import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.nn.functional as F
import torch.ao.nn.quantized.reference as nnqr
from ._common_operator_config_utils import _get_conv_configs, _get_linear_configs, _get_binary_op_configs, _get_bn_configs, _get_cat_config, _get_default_op_configs, _get_embedding_op_configs, _get_fixed_qparams_op_configs, _get_ln_configs, _get_rnn_op_configs, _get_share_qparams_op_configs
from .backend_config import BackendPatternConfig, BackendConfig, DTypeConfig, ObservationType
from ..fuser_method_mappings import _sequential_wrapper2
import operator
from torch.ao.quantization.utils import MatchAllNode
import itertools
onednn_weighted_op_int8_dtype_config = DTypeConfig(input_dtype=torch.quint8, output_dtype=torch.quint8, weight_dtype=torch.qint8, bias_dtype=torch.float)
onednn_op_quint8_dtype_config = DTypeConfig(input_dtype=torch.quint8, output_dtype=torch.quint8)
onednn_dynamic_int8_dtype_config = DTypeConfig(input_dtype=torch.quint8, output_dtype=torch.float, weight_dtype=torch.qint8, bias_dtype=torch.float, is_dynamic=True)
onednn_weight_only_qint8_dtype_config = DTypeConfig(input_dtype=torch.float, output_dtype=torch.float, weight_dtype=torch.qint8)
onednn_input_output_only_quint8_dtype_config = DTypeConfig(input_dtype=torch.quint8, output_dtype=torch.quint8, weight_dtype=torch.float, bias_dtype=torch.float)

def _fuse_linear_bn_leaky_relu(is_qat, linear, bn, leaky_relu):
    if False:
        while True:
            i = 10
    'Given the linear, bn and leaky_relu modules, fuses them and returns the fused module\n    Args:\n        is_qat: a flag for whether we are using quantization aware training fusion\n                or post training quantization fusion\n        linear: Module instance of type Linear\n        bn: BatchNorm1d instance that needs to be fused with the linear layer\n        leaky_relu: LeakyReLU instance that needs to be fused with the linear layer\n    Examples::\n        >>> # xdoctest: +SKIP(failing)\n        >>> m1 = nn.Linear(20, 10)\n        >>> b1 = nn.BatchNorm1d(10)\n        >>> lr = nn.LeakyReLU(0.01)\n        >>> m2 = _fuse_linear_bn_leaky_relu(m1, b1, lr)\n    '
    assert linear.training == bn.training and bn.training == leaky_relu.training, 'Linear, BN and LeakyReLU all must be in the same mode (train or eval).'
    if is_qat:
        raise NotImplementedError(f'Cannot fuse train modules: {(linear, bn, leaky_relu)}')
    else:
        map_to_fused_module_eval = {nn.Linear: nni.LinearLeakyReLU}
        fused_module = map_to_fused_module_eval.get(type(linear), None)
        if fused_module is not None:
            fused_linear = nn.utils.fusion.fuse_linear_bn_eval(linear, bn)
            fm = fused_module(fused_linear, leaky_relu)
            return fm
        else:
            raise NotImplementedError(f'Cannot fuse eval modules: {(linear, bn, leaky_relu)}')
observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
conv_dtype_configs = [onednn_weighted_op_int8_dtype_config]
conv_configs = _get_conv_configs(conv_dtype_configs)

def _fuse_conv_add_left(is_qat, add, conv, _):
    if False:
        for i in range(10):
            print('nop')
    return nni.ConvAdd2d(conv, add)

def _conv_add_root_node_getter_left(pattern):
    if False:
        print('Hello World!')
    (_, conv, _) = pattern
    return conv

def _conv_add_extra_inputs_getter_left(pattern):
    if False:
        while True:
            i = 10
    ' get inputs pattern for extra inputs, inputs for root node\n    are assumed to be copied over from root node to the fused node\n    '
    (_, conv, extra_input) = pattern
    return [extra_input]

def _fuse_conv_bn_add_left(is_qat, add, bn_conv, _):
    if False:
        for i in range(10):
            print('nop')
    (bn, conv) = bn_conv
    if is_qat:
        raise NotImplementedError(f'Cannot fuse train modules: {(conv, bn, add)}')
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAdd2d(fused_conv, add)

def _conv_bn_add_root_node_getter_left(add_pattern):
    if False:
        print('Hello World!')
    (_, bn_conv, _) = add_pattern
    (bn, conv) = bn_conv
    return conv

def _conv_bn_add_extra_inputs_getter_left(add_pattern):
    if False:
        i = 10
        return i + 15
    ' get inputs pattern for extra inputs, inputs for root node\n    are assumed to be copied over from root node to the fused node\n    '
    (_, bn_conv, extra_input) = add_pattern
    (bn, conv) = bn_conv
    return [extra_input]
conv_add_left_optioins = itertools.product([True, False], [torch.add, operator.add])
for (with_bn, add_op) in conv_add_left_optioins:
    if with_bn:
        conv_configs.append(BackendPatternConfig()._set_pattern_complex_format((add_op, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode)).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_fuser_method(_fuse_conv_bn_add_left)._set_root_node_getter(_conv_bn_add_root_node_getter_left)._set_extra_inputs_getter(_conv_bn_add_extra_inputs_getter_left).set_fused_module(nni.ConvAdd2d))
    else:
        conv_configs.append(BackendPatternConfig()._set_pattern_complex_format((add_op, nn.Conv2d, MatchAllNode)).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_fuser_method(_fuse_conv_add_left)._set_root_node_getter(_conv_add_root_node_getter_left)._set_extra_inputs_getter(_conv_add_extra_inputs_getter_left).set_fused_module(nni.ConvAdd2d))

def _fuse_conv_add_right(is_qat, add, _, conv):
    if False:
        print('Hello World!')
    return nni.ConvAdd2d(conv, add)

def _conv_add_root_node_getter_right(pattern):
    if False:
        while True:
            i = 10
    (add, _, conv) = pattern
    return conv

def _conv_add_extra_inputs_getter_right(pattern):
    if False:
        i = 10
        return i + 15
    ' get inputs pattern for extra inputs, inputs for root node\n    are assumed to be copied over from root node to the fused node\n    '
    (_, extra_input, conv) = pattern
    return [extra_input]

def _fuse_conv_bn_add_right(is_qat, add, _, bn_conv):
    if False:
        return 10
    (bn, conv) = bn_conv
    if is_qat:
        raise NotImplementedError(f'Cannot fuse train modules: {(conv, bn, add)}')
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAdd2d(fused_conv, add)

def _conv_bn_add_root_node_getter_right(pattern):
    if False:
        for i in range(10):
            print('nop')
    (add, _, bn_conv) = pattern
    (bn, conv) = bn_conv
    return conv

def _conv_bn_add_extra_inputs_getter_right(pattern):
    if False:
        i = 10
        return i + 15
    ' get inputs pattern for extra inputs, inputs for root node\n    are assumed to be copied over from root node to the fused node\n    '
    (_, extra_input, bn_conv) = pattern
    (bn, conv) = bn_conv
    return [extra_input]
conv_add_optioins = itertools.product([True, False], [torch.add, operator.add])
for (with_bn, add_op) in conv_add_optioins:
    if with_bn:
        conv_configs.append(BackendPatternConfig()._set_pattern_complex_format((add_op, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d))).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_fuser_method(_fuse_conv_bn_add_right)._set_root_node_getter(_conv_bn_add_root_node_getter_right)._set_extra_inputs_getter(_conv_bn_add_extra_inputs_getter_right).set_fused_module(nni.ConvAdd2d))
    else:
        conv_configs.append(BackendPatternConfig()._set_pattern_complex_format((add_op, MatchAllNode, nn.Conv2d)).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_fuser_method(_fuse_conv_add_right)._set_root_node_getter(_conv_add_root_node_getter_right)._set_extra_inputs_getter(_conv_add_extra_inputs_getter_right).set_fused_module(nni.ConvAdd2d))
conv_configs.append(BackendPatternConfig(nni.ConvAdd2d).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_root_module(nn.Conv2d).set_reference_quantized_module(nnqr.Conv2d))

def _fuse_conv_add_relu_left(is_qat, relu, add_pattern):
    if False:
        i = 10
        return i + 15
    (add, conv, _) = add_pattern
    return nni.ConvAddReLU2d(conv, add, relu)

def _conv_add_relu_root_node_getter_left(pattern):
    if False:
        i = 10
        return i + 15
    (relu, add_pattern) = pattern
    (_, conv, _) = add_pattern
    return conv

def _conv_add_relu_extra_inputs_getter_left(pattern):
    if False:
        for i in range(10):
            print('nop')
    ' get inputs pattern for extra inputs, inputs for root node\n    are assumed to be copied over from root node to the fused node\n    '
    (relu, add_pattern) = pattern
    (_, conv, extra_input) = add_pattern
    return [extra_input]

def _fuse_conv_bn_add_relu_left(is_qat, relu, add_pattern):
    if False:
        while True:
            i = 10
    (add, bn_conv, _) = add_pattern
    (bn, conv) = bn_conv
    if is_qat:
        raise NotImplementedError(f'Cannot fuse train modules: {(conv, bn, add, relu)}')
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAddReLU2d(fused_conv, add, relu)

def _conv_bn_add_relu_root_node_getter_left(pattern):
    if False:
        return 10
    (relu, add_pattern) = pattern
    (_, bn_conv, _) = add_pattern
    (bn, conv) = bn_conv
    return conv

def _conv_bn_add_relu_extra_inputs_getter_left(pattern):
    if False:
        print('Hello World!')
    ' get inputs pattern for extra inputs, inputs for root node\n    are assumed to be copied over from root node to the fused node\n    '
    (relu, add_pattern) = pattern
    (_, bn_conv, extra_input) = add_pattern
    (bn, conv) = bn_conv
    return [extra_input]
conv_add_relu_left_optioins = itertools.product([True, False], [torch.add, operator.add])
for (with_bn, add_op) in conv_add_relu_left_optioins:
    if with_bn:
        conv_configs.append(BackendPatternConfig()._set_pattern_complex_format((nn.ReLU, (add_op, (nn.BatchNorm2d, nn.Conv2d), MatchAllNode))).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_fuser_method(_fuse_conv_bn_add_relu_left)._set_root_node_getter(_conv_bn_add_relu_root_node_getter_left)._set_extra_inputs_getter(_conv_bn_add_relu_extra_inputs_getter_left).set_fused_module(nni.ConvAddReLU2d))
    else:
        conv_configs.append(BackendPatternConfig()._set_pattern_complex_format((nn.ReLU, (add_op, nn.Conv2d, MatchAllNode))).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_fuser_method(_fuse_conv_add_relu_left)._set_root_node_getter(_conv_add_relu_root_node_getter_left)._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_left).set_fused_module(nni.ConvAddReLU2d))

def _fuse_conv_add_relu_right(is_qat, relu, add_pattern):
    if False:
        while True:
            i = 10
    (add, _, conv) = add_pattern
    return nni.ConvAddReLU2d(conv, add, relu)

def _conv_add_relu_root_node_getter_right(pattern):
    if False:
        for i in range(10):
            print('nop')
    (relu, add_pattern) = pattern
    (_, _, conv) = add_pattern
    return conv

def _conv_add_relu_extra_inputs_getter_right(pattern):
    if False:
        while True:
            i = 10
    ' get inputs pattern for extra inputs, inputs for root node\n    are assumed to be copied over from root node to the fused node\n    '
    (relu, add_pattern) = pattern
    (_, extra_input, conv) = add_pattern
    return [extra_input]

def _fuse_conv_bn_add_relu_right(is_qat, relu, add_pattern):
    if False:
        i = 10
        return i + 15
    (add, _, bn_conv) = add_pattern
    (bn, conv) = bn_conv
    if is_qat:
        raise NotImplementedError(f'Cannot fuse train modules: {(conv, bn, add, relu)}')
    else:
        fused_conv = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        return nni.ConvAddReLU2d(fused_conv, add, relu)

def _conv_bn_add_relu_root_node_getter_right(pattern):
    if False:
        i = 10
        return i + 15
    (relu, add_pattern) = pattern
    (_, _, bn_conv) = add_pattern
    (bn, conv) = bn_conv
    return conv

def _conv_bn_add_relu_extra_inputs_getter_right(pattern):
    if False:
        return 10
    ' get inputs pattern for extra inputs, inputs for root node\n    are assumed to be copied over from root node to the fused node\n    '
    (relu, add_pattern) = pattern
    (_, extra_input, bn_conv) = add_pattern
    (bn, conv) = bn_conv
    return [extra_input]
conv_add_relu_optioins = itertools.product([True, False], [torch.add, operator.add])
for (with_bn, add_op) in conv_add_relu_optioins:
    if with_bn:
        conv_configs.append(BackendPatternConfig()._set_pattern_complex_format((nn.ReLU, (add_op, MatchAllNode, (nn.BatchNorm2d, nn.Conv2d)))).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_fuser_method(_fuse_conv_bn_add_relu_right)._set_root_node_getter(_conv_bn_add_relu_root_node_getter_right)._set_extra_inputs_getter(_conv_bn_add_relu_extra_inputs_getter_right).set_fused_module(nni.ConvAddReLU2d))
    else:
        conv_configs.append(BackendPatternConfig()._set_pattern_complex_format((nn.ReLU, (add_op, MatchAllNode, nn.Conv2d))).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_fuser_method(_fuse_conv_add_relu_right)._set_root_node_getter(_conv_add_relu_root_node_getter_right)._set_extra_inputs_getter(_conv_add_relu_extra_inputs_getter_right).set_fused_module(nni.ConvAddReLU2d))
conv_configs.append(BackendPatternConfig(nni.ConvAddReLU2d).set_observation_type(observation_type).set_dtype_configs(conv_dtype_configs).set_root_module(nn.Conv2d).set_reference_quantized_module(nnqr.Conv2d))
linear_dtype_configs = [onednn_weighted_op_int8_dtype_config, onednn_dynamic_int8_dtype_config]
linear_configs = _get_linear_configs(linear_dtype_configs)

def _add_eltwise_fusion_configs(configs, root_module, root_op, post_module, post_op, dtype_configs, fuser_method, fused_module, observation_type, ref_quant_module):
    if False:
        i = 10
        return i + 15
    configs.append(BackendPatternConfig((root_module, post_module)).set_dtype_configs(dtype_configs).set_fuser_method(fuser_method).set_fused_module(fused_module))
    configs.append(BackendPatternConfig((root_module, post_op)).set_dtype_configs(dtype_configs).set_fuser_method(fuser_method).set_fused_module(fused_module))
    configs.append(BackendPatternConfig(fused_module).set_observation_type(observation_type).set_dtype_configs(dtype_configs).set_root_module(root_module).set_reference_quantized_module(ref_quant_module))
    configs.append(BackendPatternConfig((root_op, post_module)).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
    configs.append(BackendPatternConfig((root_op, post_op)).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
_add_eltwise_fusion_configs(linear_configs, nn.Linear, F.linear, nn.LeakyReLU, F.leaky_relu, linear_dtype_configs, _sequential_wrapper2(nni.LinearLeakyReLU), nni.LinearLeakyReLU, observation_type, nnqr.Linear)
linear_configs.append(BackendPatternConfig((nn.Linear, nn.BatchNorm1d, nn.LeakyReLU)).set_dtype_configs(linear_dtype_configs).set_fuser_method(_fuse_linear_bn_leaky_relu).set_fused_module(nni.LinearLeakyReLU))
_add_eltwise_fusion_configs(linear_configs, nn.Linear, F.linear, nn.Tanh, torch.tanh, linear_dtype_configs, _sequential_wrapper2(nni.LinearTanh), nni.LinearTanh, observation_type, nnqr.Linear)
binary_op_dtype_configs = [onednn_op_quint8_dtype_config]
default_op_dtype_configs = [onednn_op_quint8_dtype_config]
fixed_qparams_op_dtype_configs = [onednn_op_quint8_dtype_config]
share_qparams_op_dtype_configs = [onednn_op_quint8_dtype_config]
rnn_op_dtype_configs = [onednn_dynamic_int8_dtype_config]
embedding_op_dtype_configs = [onednn_weight_only_qint8_dtype_config]
layer_norm_op_dtype_configs = [onednn_input_output_only_quint8_dtype_config]

def get_onednn_backend_config() -> BackendConfig:
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the `BackendConfig` for PyTorch's native ONEDNN backend.\n    "
    return BackendConfig('onednn').set_backend_pattern_configs(conv_configs).set_backend_pattern_configs(linear_configs).set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)).set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)).set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)).set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)).set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)).set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)).set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs)).set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)).set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))
__all__ = ['get_onednn_backend_config']