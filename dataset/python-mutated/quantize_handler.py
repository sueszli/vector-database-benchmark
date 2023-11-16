from abc import ABC
from typing import Callable, Dict, List, Optional, Type
import torch
from torch.ao.quantization.backend_config import BackendConfig, DTypeConfig, ObservationType
from torch.ao.quantization.utils import NodePattern, Pattern, QuantizerCls
from torch.fx.graph import Node
from .utils import all_node_args_have_no_tensors
__all__ = ['QuantizeHandler', 'BinaryOpQuantizeHandler', 'CatQuantizeHandler', 'ConvReluQuantizeHandler', 'LinearReLUQuantizeHandler', 'BatchNormQuantizeHandler', 'EmbeddingQuantizeHandler', 'RNNDynamicQuantizeHandler', 'DefaultNodeQuantizeHandler', 'FixedQParamsOpQuantizeHandler', 'CopyNodeQuantizeHandler', 'GeneralTensorShapeOpQuantizeHandler', 'CustomModuleQuantizeHandler', 'StandaloneModuleQuantizeHandler']

def _default_root_node_getter(node_pattern):
    if False:
        i = 10
        return i + 15
    if node_pattern is None:
        return node_pattern
    while not isinstance(node_pattern, Node):
        node_pattern = node_pattern[-1]
    return node_pattern

class QuantizeHandler(ABC):
    """ Base handler class for the quantizer patterns
    """

    def __init__(self, node_pattern: NodePattern, modules: Dict[str, torch.nn.Module], root_node_getter: Optional[Callable]=None, is_custom_module=False, is_standalone_module=False):
        if False:
            i = 10
            return i + 15
        ' Records pattern information in __init__, which will be used\n        in convert\n        '
        self.node_pattern = node_pattern
        self.modules = modules
        if root_node_getter is None:
            root_node_getter = _default_root_node_getter
        self.root_node = root_node_getter(node_pattern)
        self.is_custom_module_ = is_custom_module
        self.is_standalone_module_ = is_standalone_module
        self.num_tensor_args = 0
        if isinstance(self.root_node, Node):
            cache_for_no_tensor_check: Dict[Node, bool] = {}
            for arg_idx in range(len(self.root_node.args)):
                arg = self.root_node.args[arg_idx]
                if isinstance(arg, Node) and (not all_node_args_have_no_tensors(arg, self.modules, cache_for_no_tensor_check)):
                    self.num_tensor_args += 1

    def is_general_tensor_value_op(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns True if the operator works for both floating point and\n        quantized input, and does some computation based on the input Tensor,\n        or the ops that only re-arranges the Tensor values or query some metadata\n        about the Tensor\n        so we need to insert observer/fake_quant for the output of the\n        operator (same observer instance as input)\n        since the distribution of values is different for input and output\n        Tensors (for HistogramObserver) while they share the same quantization\n        parameters\n        Example operator: avgpool2d, reshape, transpose, maxpool2d\n        Example observed operator:\n        observer_0 - avgpool2d - observer_0 (same observer instance as input)\n        '
        return False

    def is_custom_module(self):
        if False:
            return 10
        return self.is_custom_module_

    def is_standalone_module(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_standalone_module_

def _get_quantize_handler_cls(observation_type: ObservationType, dtype_configs: List[DTypeConfig], num_tensor_args_to_observation_type: Dict[int, ObservationType]) -> Type[QuantizeHandler]:
    if False:
        while True:
            i = 10
    '\n    Return a configurable QuantizeHandler that matches the given specifications from the backend.\n    '

    class ConfigurableQuantizeHandler(QuantizeHandler):

        def __init__(self, node_pattern: NodePattern, modules: Dict[str, torch.nn.Module], root_node_getter: Optional[Callable]=None):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(node_pattern, modules, root_node_getter)
            if num_tensor_args_to_observation_type:
                assert self.num_tensor_args in num_tensor_args_to_observation_type, f'Must provide observation_type config for tensor number {self.num_tensor_args} in num_tensor_args_to_observation_type for {node_pattern}'
                self.observation_type = num_tensor_args_to_observation_type[self.num_tensor_args]
            else:
                self.observation_type = observation_type
            self.dtype_configs = dtype_configs

        def is_general_tensor_value_op(self) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    return ConfigurableQuantizeHandler

def _get_pattern_to_quantize_handlers(backend_config: BackendConfig) -> Dict[Pattern, QuantizerCls]:
    if False:
        print('Hello World!')
    '\n    Note: Quantize handler is just a holder for some check methods like\n    (should_insert_observer_for_output), maybe this can be a enum as well,\n    we can refactor this after we convert the path for fbgemm/qnnpack fully to the\n    new path, this is not exposed to backend developers\n    '
    pattern_to_quantize_handlers = {}
    for (pattern, config) in backend_config._pattern_complex_format_to_config.items():
        observation_type = config.observation_type
        dtype_configs = config.dtype_configs
        num_tensor_args_to_observation_type = config._num_tensor_args_to_observation_type
        pattern_to_quantize_handlers[pattern] = _get_quantize_handler_cls(observation_type, dtype_configs, num_tensor_args_to_observation_type)
    return pattern_to_quantize_handlers

class BinaryOpQuantizeHandler(QuantizeHandler):
    pass

class CatQuantizeHandler(QuantizeHandler):
    pass

class ConvReluQuantizeHandler(QuantizeHandler):
    pass

class LinearReLUQuantizeHandler(QuantizeHandler):
    pass

class BatchNormQuantizeHandler(QuantizeHandler):
    pass

class EmbeddingQuantizeHandler(QuantizeHandler):
    pass

class RNNDynamicQuantizeHandler(QuantizeHandler):
    pass

class DefaultNodeQuantizeHandler(QuantizeHandler):
    """ Common quantized op, first input and first output will be quantized
    """
    pass

class FixedQParamsOpQuantizeHandler(QuantizeHandler):
    pass

class CopyNodeQuantizeHandler(QuantizeHandler):
    pass

class GeneralTensorShapeOpQuantizeHandler(QuantizeHandler):
    pass

class CustomModuleQuantizeHandler(QuantizeHandler):
    pass

class StandaloneModuleQuantizeHandler(QuantizeHandler):
    pass