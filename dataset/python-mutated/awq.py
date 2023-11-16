"""AWQ (Activation aware Weight Quantization) integration file"""
from ..utils import is_auto_awq_available, is_torch_available
from ..utils.quantization_config import AwqBackendPackingMethod, AWQLinearVersion
if is_torch_available():
    import torch.nn as nn

def replace_with_awq_linear(model, modules_to_not_convert=None, quantization_config=None, current_key_name=None, has_been_replaced=False) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Public method that recursively replaces the Linear layers of the given model with AWQ quantized layers.\n    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the\n    conversion has been successfull or not.\n\n    During the module replacement, we also infer the backend to use through the `quantization_config` object.\n\n    Args:\n        model (`torch.nn.Module`):\n            The model to convert, can be any `torch.nn.Module` instance.\n        quantization_config (`AwqConfig`):\n            The quantization config object that contains the quantization parameters.\n        modules_to_not_convert (`list`, *optional*):\n            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be\n            converted.\n        current_key_name (`list`, *optional*):\n            A list that contains the current key name. This is used for recursion and should not be passed by the user.\n        has_been_replaced (`bool`, *optional*):\n            A boolean that indicates if the conversion has been successful or not. This is used for recursion and\n            should not be passed by the user.\n    '
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    backend = quantization_config.backend
    if not is_auto_awq_available():
        raise ValueError('AWQ (either `autoawq` or `llmawq`) is not available. Please install it with `pip install autoawq` or check out the installation guide in https://github.com/mit-han-lab/llm-awq')
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
    elif backend == AwqBackendPackingMethod.LLMAWQ:
        from awq.quantize.qmodule import WQLinear
    if backend == AwqBackendPackingMethod.AUTOAWQ:
        target_cls = WQLinear_GEMM if quantization_config.version == AWQLinearVersion.GEMM else WQLinear_GEMV
    else:
        target_cls = WQLinear
    for (name, module) in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            if not any((key in '.'.join(current_key_name) for key in modules_to_not_convert)):
                in_features = module.in_features
                out_features = module.out_features
                model._modules[name] = target_cls(w_bit=quantization_config.bits, group_size=quantization_config.group_size, in_features=in_features, out_features=out_features, bias=module.bias is not None, dev=module.weight.device)
                has_been_replaced = True
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            (_, has_been_replaced) = replace_with_awq_linear(module, modules_to_not_convert=modules_to_not_convert, current_key_name=current_key_name, quantization_config=quantization_config, has_been_replaced=has_been_replaced)
        current_key_name.pop(-1)
    return (model, has_been_replaced)