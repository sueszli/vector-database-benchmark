import operator
import torch
from torch.ao.quantization.backend_config import BackendConfig, DTypeConfig, ObservationType, BackendPatternConfig
weighted_op_quint8_dtype_config = DTypeConfig(input_dtype=torch.quint8, output_dtype=torch.quint8, weight_dtype=torch.qint8, bias_dtype=torch.float)
from typing import List

def get_linear_configs():
    if False:
        return 10
    linear_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    linear_configs.append(BackendPatternConfig(torch.ops.aten.addmm.default).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 2, 'bias': 0}))
    linear_configs.append(BackendPatternConfig(torch.ops.aten.mm.default).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 1}))
    return linear_configs

def get_conv_configs():
    if False:
        return 10
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    conv_configs.append(BackendPatternConfig(torch.ops.aten.convolution.default).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 1, 'bias': 2}))
    conv_configs.append(BackendPatternConfig((torch.ops.aten.convolution.default, torch.ops.aten.relu.default)).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 1, 'bias': 2}))
    conv_configs.append(BackendPatternConfig((torch.ops.aten.convolution.default, torch.ops.aten.relu_.default)).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 1, 'bias': 2}))
    return conv_configs

def get_pooling_configs():
    if False:
        for i in range(10):
            print('nop')
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    def root_node_getter(node_pattern):
        if False:
            print('Hello World!')
        (getitem, maxpool, index) = node_pattern
        return maxpool
    backend_pattern_configs.append(BackendPatternConfig()._set_pattern_complex_format((operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0)).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_root_node_getter(root_node_getter))
    return backend_pattern_configs

def get_relu_configs():
    if False:
        print('Hello World!')
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    backend_pattern_configs.append(BackendPatternConfig(torch.ops.aten.relu.default).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
    return backend_pattern_configs

def get_binary_op_configs():
    if False:
        while True:
            i = 10
    binary_op_configs: List[BackendPatternConfig] = []
    dtype_configs = [weighted_op_quint8_dtype_config]
    num_tensor_args_to_observation_type_mapping = {0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT, 1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT, 2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT}
    for op_with_quantized_bop_scalar_variant in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
        bop_patterns = [(op_with_quantized_bop_scalar_variant, torch.ops.aten.relu.default), op_with_quantized_bop_scalar_variant, (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu_.default)]
        for bop_pattern in bop_patterns:
            binary_op_configs.append(BackendPatternConfig(bop_pattern).set_dtype_configs(dtype_configs)._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))
    return binary_op_configs

def get_qnnpack_pt2e_backend_config():
    if False:
        i = 10
        return i + 15
    return BackendConfig('qnnpack_pytorch_2.0_export').set_backend_pattern_configs(get_linear_configs()).set_backend_pattern_configs(get_binary_op_configs()).set_backend_pattern_configs(get_conv_configs()).set_backend_pattern_configs(get_pooling_configs()).set_backend_pattern_configs(get_relu_configs())