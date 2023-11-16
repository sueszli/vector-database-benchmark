from itertools import chain
from operator import getitem
import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import symbolic_trace
from torch.nn.utils import parametrize
from typing import Type, Set, Dict, Callable, Tuple, Optional, Union
from torch.ao.pruning import BaseSparsifier
from .parametrization import FakeStructuredSparsity, BiasHook, module_contains_param
from .match_utils import apply_match, MatchAllNode
from .prune_functions import prune_linear, prune_linear_linear, prune_linear_activation_linear, prune_conv2d, prune_conv2d_conv2d, prune_conv2d_activation_conv2d, prune_conv2d_activation_pool_conv2d, prune_conv2d_pool_activation_conv2d, prune_conv2d_pool_flatten_linear, prune_lstm_output_linear, prune_lstm_output_layernorm_linear

def _get_supported_structured_pruning_modules():
    if False:
        i = 10
        return i + 15
    SUPPORTED_STRUCTURED_PRUNING_MODULES = {nn.Linear, nn.Conv2d, nn.LSTM}
    return SUPPORTED_STRUCTURED_PRUNING_MODULES

def _get_supported_activation_functions():
    if False:
        print('Hello World!')
    SUPPORTED_ACTIVATION_FUNCTIONS = {F.relu, F.rrelu, F.hardtanh, F.relu6, F.sigmoid, F.hardsigmoid, F.tanh, F.silu, F.mish, F.hardswish, F.elu, F.celu, F.selu, F.hardshrink, F.leaky_relu, F.logsigmoid, F.softplus, F.prelu, F.softsign, F.tanhshrink, F.gelu}
    return SUPPORTED_ACTIVATION_FUNCTIONS

def _get_supported_activation_modules():
    if False:
        while True:
            i = 10
    SUPPORTED_ACTIVATION_MODULES = {nn.ReLU, nn.RReLU, nn.Hardtanh, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid, nn.Tanh, nn.SiLU, nn.Mish, nn.Hardswish, nn.ELU, nn.CELU, nn.SELU, nn.Hardshrink, nn.LeakyReLU, nn.LogSigmoid, nn.Softplus, nn.PReLU, nn.Softsign, nn.Tanhshrink, nn.GELU}
    return SUPPORTED_ACTIVATION_MODULES

def _get_default_structured_pruning_patterns() -> Dict[Tuple[Union[Type[nn.Module], Callable, MatchAllNode, str], ...], Callable[..., None]]:
    if False:
        while True:
            i = 10
    '\n    Returns the patterns for conv2d / linear conversion for each element in the activation functions/modules defined above.\n    '
    patterns: Dict[Tuple[Union[Type[nn.Module], Callable, MatchAllNode, str], ...], Callable[..., None]] = {(nn.Linear, 'output'): prune_linear, (nn.Linear, nn.Linear): prune_linear_linear, (nn.Conv2d, 'output'): prune_conv2d, (nn.Conv2d, nn.Conv2d): prune_conv2d_conv2d, (nn.LSTM, getitem, nn.Linear): prune_lstm_output_linear, (nn.LSTM, getitem, nn.LayerNorm, nn.Linear): prune_lstm_output_layernorm_linear}
    for activation in chain(_get_supported_activation_functions(), _get_supported_activation_modules()):
        patterns.update({(nn.Linear, activation, nn.Linear): prune_linear_activation_linear, (nn.Conv2d, activation, nn.Conv2d): prune_conv2d_activation_conv2d, (nn.Conv2d, activation, nn.AvgPool2d, nn.Conv2d): prune_conv2d_activation_pool_conv2d, (nn.Conv2d, activation, F.avg_pool2d, nn.Conv2d): prune_conv2d_activation_pool_conv2d, (nn.Conv2d, activation, nn.MaxPool2d, nn.Conv2d): prune_conv2d_activation_pool_conv2d, (nn.Conv2d, activation, F.max_pool2d, nn.Conv2d): prune_conv2d_activation_pool_conv2d, (nn.Conv2d, nn.AvgPool2d, activation, nn.Conv2d): prune_conv2d_pool_activation_conv2d, (nn.Conv2d, F.avg_pool2d, activation, nn.Conv2d): prune_conv2d_pool_activation_conv2d, (nn.Conv2d, nn.MaxPool2d, activation, nn.Conv2d): prune_conv2d_pool_activation_conv2d, (nn.Conv2d, F.max_pool2d, activation, nn.Conv2d): prune_conv2d_pool_activation_conv2d, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear): prune_conv2d_pool_flatten_linear, (nn.Conv2d, nn.AdaptiveAvgPool2d, torch.flatten, nn.Linear): prune_conv2d_pool_flatten_linear, (nn.Conv2d, nn.AdaptiveMaxPool2d, nn.Flatten, nn.Linear): prune_conv2d_pool_flatten_linear, (nn.Conv2d, nn.AdaptiveMaxPool2d, torch.flatten, nn.Linear): prune_conv2d_pool_flatten_linear})
    return patterns

class BaseStructuredSparsifier(BaseSparsifier):
    """Base class for structured pruning.

    Abstract methods that need to be implemented:
        - update_mask: Function to compute a new mask for all keys in the
            `groups` attribute.

    Args:
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    """

    def __init__(self, defaults, patterns=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(defaults)
        if patterns is None:
            patterns = _get_default_structured_pruning_patterns()
        self.patterns = patterns

    def make_config_from_model(self, model: nn.Module, SUPPORTED_MODULES: Optional[Set[Type]]=None) -> None:
        if False:
            while True:
                i = 10
        if SUPPORTED_MODULES is None:
            SUPPORTED_MODULES = _get_supported_structured_pruning_modules()
        super().make_config_from_model(model, SUPPORTED_MODULES=SUPPORTED_MODULES)

    def _prepare(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        'This function will attach the FakeStructuredSparsity parameterizations\n        and BiasHooks at the appropriate points in the model.\n        '
        for config in self.groups:
            module = config['module']
            tensor_name = config['tensor_name']
            parametrization = config.get('parametrization', FakeStructuredSparsity)
            tensor = getattr(module, tensor_name)
            mask = config.get('mask', torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device))
            self.state[config['tensor_fqn']]['mask'] = mask
            parametrize.register_parametrization(module, tensor_name, parametrization(mask))
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune_bias = config.get('prune_bias', True)
                if module.bias is not None:
                    module.register_parameter('_bias', nn.Parameter(module.bias.detach()))
                    module.bias = None
                    module.prune_bias = prune_bias
                module.register_forward_hook(BiasHook(module.parametrizations.weight[0], prune_bias))

    def prune(self) -> None:
        if False:
            print('Hello World!')
        '\n        This function will FX symbolically trace the model and then find instances of the patterns\n        defined in self.patterns (by default SUPPORTED_STRUCTURED_PRUNING_PATTERNS ).\n\n        For each pattern, it will apply to corresponding conversion function, which will modify the output\n        and input size expected by the modules within the pattern\n        '
        self.traced = symbolic_trace(self.model)
        modules = dict(self.traced.named_modules())
        for node in self.traced.graph.nodes:
            for (pattern, convert_fn) in self.patterns.items():
                matched = apply_match(modules, pattern, node, [])
                if matched is None:
                    continue
                first_module = modules.get(node.target)
                if first_module is not None and parametrize.is_parametrized(first_module) and module_contains_param(first_module, FakeStructuredSparsity):
                    convert_block = []
                    for node in matched:
                        if node.op == 'call_module':
                            convert_block.append(modules.get(node.target))
                        elif node.op == 'call_function':
                            convert_block.append(node.target)
                    convert_fn(*convert_block)
        for module in self.traced.modules():
            if module_contains_param(module, FakeStructuredSparsity):
                raise Exception(f'Error: {module} still contains FakeStructuredSparsity parametrizations!')
        self.traced.graph.lint()
        self.traced.recompile()
        return self.traced