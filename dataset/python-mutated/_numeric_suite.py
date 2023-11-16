import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set
from torch.ao.quantization.quantization_mappings import get_default_compare_output_module_list
NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST = {nnqd.Linear, nnq.Linear, nnqd.LSTM, nn.LSTM}

def _find_match(str_list: Union[Dict[str, Any], List[str]], key_str: str, postfix: str) -> Optional[str]:
    if False:
        print('Hello World!')
    split_str = key_str.split('.')
    if split_str[-1] == postfix:
        match_string = ''.join(key_str.split('.')[0:-1])
        for s2 in str_list:
            pattern1 = ''.join(s2.split('.')[0:-1])
            pattern2 = ''.join(s2.split('.')[0:-2])
            if match_string == pattern1:
                return s2
            if match_string == pattern2:
                return s2
        if postfix == '_packed_params':
            match_string = ''.join(key_str.split('.')[0:-2])
            if len(match_string) == 0:
                return None
            for s2 in str_list:
                pattern1 = ''.join(s2.split('.')[0:-1])
                pattern2 = ''.join(s2.split('.')[0:-2])
                if match_string == pattern1:
                    return s2
                if match_string == pattern2:
                    return s2
        return None
    else:
        return None

def compare_weights(float_dict: Dict[str, Any], quantized_dict: Dict[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
    if False:
        i = 10
        return i + 15
    "Compare the weights of the float module with its corresponding quantized\n    module. Return a dict with key corresponding to module names and each entry being\n    a dictionary with two keys 'float' and 'quantized', containing the float and\n    quantized weights. This dict can be used to compare and compute the quantization\n    error of the weights of float and quantized models.\n\n    Example usage::\n\n        wt_compare_dict = compare_weights(\n            float_model.state_dict(), qmodel.state_dict())\n        for key in wt_compare_dict:\n            print(\n                key,\n                compute_error(\n                    wt_compare_dict[key]['float'],\n                    wt_compare_dict[key]['quantized'].dequantize()\n                )\n            )\n\n    Args:\n        float_dict: state dict of the float model\n        quantized_dict: state dict of the quantized model\n\n    Return:\n        weight_dict: dict with key corresponding to module names and each entry being\n        a dictionary with two keys 'float' and 'quantized', containing the float and\n        quantized weights\n    "
    torch._C._log_api_usage_once('quantization_api._numeric_suite.compare_weights')
    weight_dict: Dict[str, Dict] = {}
    for key in quantized_dict:
        match_key = _find_match(float_dict, key, 'weight')
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]['float'] = float_dict[match_key]
            weight_dict[key]['quantized'] = quantized_dict[key]
            continue
        match_key = _find_match(float_dict, key, '_packed_params')
        if match_key is not None:
            weight_dict[key] = {}
            weight_dict[key]['float'] = float_dict[match_key]
            weight_dict[key]['quantized'] = quantized_dict[key][0]
        split_str = key.split('.')
        if split_str[-1] == 'param' and split_str[-3] == '_all_weight_values':
            layer = split_str[-2]
            module_name = '.'.join(split_str[:-3])
            float_weight_ih_key = module_name + '.weight_ih_l' + layer
            float_weight_hh_key = module_name + '.weight_hh_l' + layer
            if float_weight_ih_key in float_dict and float_weight_hh_key in float_dict:
                weight_dict[key] = {}
                weight_dict[key]['float'] = float_dict[float_weight_ih_key]
                weight_dict[key]['quantized'] = quantized_dict[key].__getstate__()[0][4][0].__getstate__()[0][0]
                weight_dict[key]['float'] = float_dict[float_weight_hh_key]
                weight_dict[key]['quantized'] = quantized_dict[key].__getstate__()[0][4][1].__getstate__()[0][0]
    return weight_dict

def _get_logger_dict_helper(mod: nn.Module, target_dict: Dict[str, Any], prefix: str='') -> None:
    if False:
        while True:
            i = 10
    'This is the helper function for get_logger_dict\n\n    Args:\n        mod: module we want to save all logger stats\n        prefix: prefix for the current module\n        target_dict: the dictionary used to save all logger stats\n    '

    def get_prefix(prefix):
        if False:
            i = 10
            return i + 15
        return prefix if prefix == '' else prefix + '.'
    for (name, child) in mod.named_children():
        if isinstance(child, Logger):
            target_dict[get_prefix(prefix) + 'stats'] = child.stats
            break
    for (name, child) in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_logger_dict_helper(child, target_dict, module_prefix)

def get_logger_dict(mod: nn.Module, prefix: str='') -> Dict[str, Dict]:
    if False:
        i = 10
        return i + 15
    'Traverse the modules and save all logger stats into target dict.\n    This is mainly used for quantization accuracy debug.\n\n    Type of loggers supported:\n        ShadowLogger: used to log the outputs of the quantized module and its matching float shadow module,\n        OutputLogger: used to log the outputs of the modules\n\n    Args:\n        mod: module we want to save all logger stats\n        prefix: prefix for the current module\n\n    Return:\n        target_dict: the dictionary used to save all logger stats\n\n    '
    torch._C._log_api_usage_once('quantization_api._numeric_suite.get_logger_dict')
    target_dict: Dict[str, Dict] = {}
    _get_logger_dict_helper(mod, target_dict, prefix)
    return target_dict

class Logger(nn.Module):
    """Base class for stats logging
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.stats = {}
        self.dtype = torch.quint8

    def forward(self, x):
        if False:
            print('Hello World!')
        '\n        '
        pass

class ShadowLogger(Logger):
    """Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.stats['float'] = []
        self.stats['quantized'] = []

    def forward(self, x, y):
        if False:
            i = 10
            return i + 15
        '\n        '
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        self.stats['quantized'].append(x.detach())
        self.stats['float'].append(y.detach())

class OutputLogger(Logger):
    """Class used to log the outputs of the module
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.stats['tensor_val'] = []

    def forward(self, x):
        if False:
            print('Hello World!')
        '\n        '
        self.stats['tensor_val'].append(x)
        return x

def _convert_tuple_to_list(t: Any) -> Any:
    if False:
        print('Hello World!')
    return [_convert_tuple_to_list(x) for x in t] if type(t) is tuple else t

def _dequantize_tensor_list(t: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    return [_dequantize_tensor_list(x) for x in t] if type(t) is list else t.dequantize() if t.is_quantized else t

class Shadow(nn.Module):
    """Shadow module attaches the float module to its matching quantized module
    as the shadow. Then it uses Logger module to process the outputs of both
    modules.

    Args:
        q_module: module quantized from float_module that we want to shadow
        float_module: float module used to shadow q_module
        logger_cls: type of logger used to process the outputs of q_module and
            float_module. ShadowLogger or custom loggers can be used.
    """

    def __init__(self, q_module, float_module, logger_cls):
        if False:
            return 10
        super().__init__()
        self.orig_module = q_module
        self.shadow_module = float_module
        self.dequant = nnq.DeQuantize()
        self.logger = logger_cls()

    def forward(self, *x) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        '
        xl = _convert_tuple_to_list(x)
        output = self.orig_module(*xl)
        xl_float = _dequantize_tensor_list(xl)
        shadow_output = self.shadow_module(*xl_float)
        self.logger(output, shadow_output)
        return output

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        '
        output = self.orig_module.add(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add(x, y)
        self.logger(output, shadow_output)
        return output

    def add_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        '
        output = self.orig_module.add_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.add_scalar(x, y)
        self.logger(output, shadow_output)
        return output

    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        '
        output = self.orig_module.mul(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.mul(x, y)
        self.logger(output, shadow_output)
        return output

    def mul_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        '
        output = self.orig_module.mul_scalar(x, y)
        x = x.dequantize()
        shadow_output = self.shadow_module.mul_scalar(x, y)
        self.logger(output, shadow_output)
        return output

    def cat(self, x: List[torch.Tensor], dim: int=0) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        '
        output = self.orig_module.cat(x, dim)
        x = [y.dequantize() for y in x]
        shadow_output = self.shadow_module.cat(x, dim)
        self.logger(output, shadow_output)
        return output

    def add_relu(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        '
        output = self.orig_module.add_relu(x, y)
        x = x.dequantize()
        y = y.dequantize()
        shadow_output = self.shadow_module.add_relu(x, y)
        self.logger(output, shadow_output)
        return output

def prepare_model_with_stubs(float_module: nn.Module, q_module: nn.Module, module_swap_list: Set[type], logger_cls: Callable) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Prepare the model by attaching the float module to its matching quantized\n    module as the shadow if the float module type is in module_swap_list.\n\n    Example usage::\n\n        prepare_model_with_stubs(float_model, q_model, module_swap_list, Logger)\n        q_model(data)\n        ob_dict = get_logger_dict(q_model)\n\n    Args:\n        float_module: float module used to generate the q_module\n        q_module: module quantized from float_module\n        module_swap_list: list of float module types to attach the shadow\n        logger_cls: type of logger to be used in shadow module to process the outputs of\n            quantized module and its float shadow module\n    '
    torch._C._log_api_usage_once('quantization_api._numeric_suite.prepare_model_with_stubs')
    float_module_children = {}
    for (name, mod) in float_module.named_children():
        float_module_children[name] = mod
    reassign = {}
    for (name, mod) in q_module.named_children():
        if name not in float_module_children:
            continue
        float_mod = float_module_children[name]
        if type(float_mod) not in module_swap_list:
            prepare_model_with_stubs(float_mod, mod, module_swap_list, logger_cls)
        if type(float_mod) in module_swap_list and (not _is_identical_module_type(mod, float_mod)):
            reassign[name] = Shadow(mod, float_mod, logger_cls)
    for (key, value) in reassign.items():
        q_module._modules[key] = value

def _is_identical_module_type(mod1, mod2):
    if False:
        while True:
            i = 10
    mod1_module_types = [type(mod) for mod in mod1.modules()]
    mod2_module_types = [type(mod) for mod in mod2.modules()]
    return mod1_module_types == mod2_module_types

def compare_model_stub(float_model: nn.Module, q_model: nn.Module, module_swap_list: Set[type], *data, logger_cls=ShadowLogger) -> Dict[str, Dict]:
    if False:
        i = 10
        return i + 15
    "Compare quantized module in a model with its floating point counterpart,\n    feeding both of them the same input. Return a dict with key corresponding to\n    module names and each entry being a dictionary with two keys 'float' and\n    'quantized', containing the output tensors of quantized and its matching\n    float shadow module. This dict can be used to compare and compute the module\n    level quantization error.\n\n    This function first call prepare_model_with_stubs() to swap the quantized\n    module that we want to compare with the Shadow module, which takes quantized\n    module, corresponding float module and logger as input, and creates a forward\n    path inside to make the float module to shadow quantized module sharing the\n    same input. The logger can be customizable, default logger is ShadowLogger\n    and it will save the outputs of the quantized module and float module that\n    can be used to compute the module level quantization error.\n\n    Example usage::\n\n        module_swap_list = [torchvision.models.quantization.resnet.QuantizableBasicBlock]\n        ob_dict = compare_model_stub(float_model,qmodel,module_swap_list, data)\n        for key in ob_dict:\n            print(key, compute_error(ob_dict[key]['float'], ob_dict[key]['quantized'].dequantize()))\n\n    Args:\n        float_model: float model used to generate the q_model\n        q_model: model quantized from float_model\n        module_swap_list: list of float module types at which shadow modules will\n            be attached.\n        data: input data used to run the prepared q_model\n        logger_cls: type of logger to be used in shadow module to process the outputs of\n            quantized module and its float shadow module\n    "
    torch._C._log_api_usage_once('quantization_api._numeric_suite.compare_model_stub')
    prepare_model_with_stubs(float_model, q_model, module_swap_list, logger_cls)
    q_model(*data)
    ob_dict = get_logger_dict(q_model)
    return ob_dict

def get_matching_activations(float_module: nn.Module, q_module: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    if False:
        print('Hello World!')
    "Find the matching activation between float and quantized modules.\n\n    Args:\n        float_module: float module used to generate the q_module\n        q_module: module quantized from float_module\n\n    Return:\n        act_dict: dict with key corresponding to quantized module names and each\n        entry being a dictionary with two keys 'float' and 'quantized', containing\n        the matching float and quantized activations\n    "
    torch._C._log_api_usage_once('quantization_api._numeric_suite.get_matching_activations')
    float_dict = get_logger_dict(float_module)
    quantized_dict = get_logger_dict(q_module)
    act_dict: Dict[str, Dict] = {}
    for key in quantized_dict:
        if len(quantized_dict[key]['tensor_val']) == 0:
            continue
        match_key = _find_match(sorted(float_dict, reverse=True), key, 'stats')
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]['float'] = float_dict[match_key]['tensor_val']
            act_dict[key]['quantized'] = quantized_dict[key]['tensor_val']
    return act_dict

def prepare_model_outputs(float_module: nn.Module, q_module: nn.Module, logger_cls=OutputLogger, allow_list=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Prepare the model by attaching the logger to both float module\n    and quantized module if they are in the allow_list.\n\n    Args:\n        float_module: float module used to generate the q_module\n        q_module: module quantized from float_module\n        logger_cls: type of logger to be attached to float_module and q_module\n        allow_list: list of module types to attach logger\n    '
    torch._C._log_api_usage_once('quantization_api._numeric_suite.prepare_model_outputs')
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()
    qconfig_debug = torch.ao.quantization.QConfig(activation=logger_cls, weight=None)
    float_module.qconfig = qconfig_debug
    prepare(float_module, inplace=True, allow_list=allow_list, prepare_custom_config_dict={})
    q_module.qconfig = qconfig_debug
    prepare(q_module, inplace=True, allow_list=allow_list, observer_non_leaf_module_list=NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST, prepare_custom_config_dict={})

def compare_model_outputs(float_model: nn.Module, q_model: nn.Module, *data, logger_cls=OutputLogger, allow_list=None) -> Dict[str, Dict[str, torch.Tensor]]:
    if False:
        for i in range(10):
            print('nop')
    "Compare output activations between float and quantized models at\n    corresponding locations for the same input. Return a dict with key corresponding\n    to quantized module names and each entry being a dictionary with two keys\n    'float' and 'quantized', containing the activations of quantized model and\n    float model at matching locations. This dict can be used to compare and\n    compute the propagation quantization error.\n\n    Example usage::\n\n        act_compare_dict = compare_model_outputs(float_model, qmodel, data)\n        for key in act_compare_dict:\n            print(\n                key,\n                compute_error(\n                    act_compare_dict[key]['float'],\n                    act_compare_dict[key]['quantized'].dequantize()\n                )\n            )\n\n    Args:\n        float_model: float model used to generate the q_model\n        q_model: model quantized from float_model\n        data: input data used to run the prepared float_model and q_model\n        logger_cls: type of logger to be attached to float_module and q_module\n        allow_list: list of module types to attach logger\n\n    Return:\n        act_compare_dict: dict with key corresponding to quantized module names\n        and each entry being a dictionary with two keys 'float' and 'quantized',\n        containing the matching float and quantized activations\n    "
    torch._C._log_api_usage_once('quantization_api._numeric_suite.compare_model_outputs')
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()
    prepare_model_outputs(float_model, q_model, logger_cls, allow_list)
    float_model(*data)
    q_model(*data)
    act_compare_dict = get_matching_activations(float_model, q_model)
    return act_compare_dict