import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module
__all__ = ['script_qconfig', 'script_qconfig_dict', 'fuse_conv_bn_jit', 'prepare_jit', 'prepare_dynamic_jit', 'convert_jit', 'convert_dynamic_jit', 'quantize_jit', 'quantize_dynamic_jit']

def _check_is_script_module(model):
    if False:
        return 10
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError('input must be a script module, got: ' + str(type(model)))

def _check_forward_method(model):
    if False:
        while True:
            i = 10
    if not model._c._has_method('forward'):
        raise ValueError('input script module does not have forward method')

def script_qconfig(qconfig):
    if False:
        while True:
            i = 10
    'Instantiate the activation and weight observer modules and script\n    them, these observer module instances will be deepcopied during\n    prepare_jit step.\n    '
    return QConfig(activation=torch.jit.script(qconfig.activation())._c, weight=torch.jit.script(qconfig.weight())._c)

def script_qconfig_dict(qconfig_dict):
    if False:
        for i in range(10):
            print('nop')
    'Helper function used by `prepare_jit`.\n    Apply `script_qconfig` for all entries in `qconfig_dict` that is\n    not None.\n    '
    return {k: script_qconfig(v) if v else None for (k, v) in qconfig_dict.items()}

def fuse_conv_bn_jit(model, inplace=False):
    if False:
        while True:
            i = 10
    ' Fuse conv - bn module\n    Works for eval model only.\n\n    Args:\n        model: TorchScript model from scripting or tracing\n    '
    torch._C._log_api_usage_once('quantization_api.quantize_jit.fuse_conv_bn_jit')
    model_c = model._c
    model_c = torch._C._jit_pass_fold_convbn(model_c)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model

def _prepare_jit(model, qconfig_dict, inplace=False, quant_type=QuantType.STATIC):
    if False:
        return 10
    _check_is_script_module(model)
    _check_forward_method(model)
    if not all((isinstance(x, str) for x in qconfig_dict.keys())):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observers(model._c, 'forward', scripted_qconfig_dict, inplace, quant_type)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model

def _prepare_ondevice_jit(model, qconfig_dict, method_name='forward', inplace=False, quant_type=QuantType.STATIC):
    if False:
        i = 10
        return i + 15
    _check_is_script_module(model)
    if not all((isinstance(x, str) for x in qconfig_dict.keys())):
        raise ValueError('qconfig_dict should only contain names(str) as keys.')
    scripted_qconfig_dict = script_qconfig_dict(qconfig_dict)
    method_graph = model._c._get_method(method_name).graph
    torch._C._jit_pass_inline(method_graph)
    model = fuse_conv_bn_jit(model, inplace)
    model_c = torch._C._jit_pass_insert_observer_method_for_ondevice_ptq(model._c, method_name, scripted_qconfig_dict, inplace, quant_type)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model

def prepare_jit(model, qconfig_dict, inplace=False):
    if False:
        while True:
            i = 10
    torch._C._log_api_usage_once('quantization_api.quantize_jit.prepare_jit')
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.STATIC)

def prepare_dynamic_jit(model, qconfig_dict, inplace=False):
    if False:
        while True:
            i = 10
    torch._C._log_api_usage_once('quantization_api.quantize_jit.prepare_dynamic_jit')
    return _prepare_jit(model, qconfig_dict, inplace, quant_type=QuantType.DYNAMIC)

def _prepare_ondevice_dynamic_jit(model, qconfig_dict, method_name='forward', inplace=False):
    if False:
        for i in range(10):
            print('nop')
    return _prepare_ondevice_jit(model, qconfig_dict, method_name, inplace, quant_type=QuantType.DYNAMIC)

def _convert_jit(model, inplace=False, debug=False, quant_type=QuantType.STATIC, preserved_attrs=None):
    if False:
        i = 10
        return i + 15
    _check_is_script_module(model)
    model.eval()
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant(model_c, 'forward', inplace, debug, quant_type)
    if not debug:
        is_xpu = all((p.device.type == 'xpu' for p in model.parameters()))
        if not is_xpu:
            model.cpu()
        if preserved_attrs is None:
            preserved_attrs = []
        model_c = torch._C._jit_pass_quant_finalize(model_c, quant_type, preserved_attrs)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model

def _convert_ondevice_jit(model, method_name, inplace=False, debug=False, quant_type=QuantType.STATIC):
    if False:
        print('Hello World!')
    _check_is_script_module(model)
    assert quant_type == QuantType.DYNAMIC, 'This API, while should work for static quant, is only tested for dynamic quant.'
    assert not method_name.startswith('observe_'), 'Pass in valid method to be quantized, e.g. forward'
    observe_method_name = 'observe_' + method_name
    quantize_method_name = 'quantize_' + method_name
    model_c = model._c
    model_c = torch._C._jit_pass_insert_quant_dequant_for_ondevice_ptq(model._c, observe_method_name, inplace, debug, QuantType.DYNAMIC)
    model_c = torch._C._jit_pass_quant_finalize_for_ondevice_ptq(model_c, QuantType.DYNAMIC, quantize_method_name)
    if inplace:
        model._reconstruct(model_c)
    else:
        model = wrap_cpp_module(model_c)
    return model

def convert_jit(model, inplace=False, debug=False, preserved_attrs=None):
    if False:
        print('Hello World!')
    torch._C._log_api_usage_once('quantization_api.quantize_jit.convert_jit')
    return _convert_jit(model, inplace, debug, quant_type=QuantType.STATIC, preserved_attrs=preserved_attrs)

def convert_dynamic_jit(model, inplace=False, debug=False, preserved_attrs=None):
    if False:
        i = 10
        return i + 15
    torch._C._log_api_usage_once('quantization_api.quantize_jit.convert_dynamic_jit')
    return _convert_jit(model, inplace, debug, quant_type=QuantType.DYNAMIC, preserved_attrs=preserved_attrs)

def _convert_ondevice_dynamic_jit(model, method_name, inplace=False, debug=False):
    if False:
        for i in range(10):
            print('nop')
    return _convert_ondevice_jit(model, method_name, inplace, debug, quant_type=QuantType.DYNAMIC)

def _quantize_ondevice_dynamic_jit_impl(model, qconfig_dict, method_name, inplace=False):
    if False:
        for i in range(10):
            print('nop')
    model = _prepare_ondevice_dynamic_jit(model, qconfig_dict, method_name, inplace)
    model = _convert_ondevice_dynamic_jit(model, method_name, inplace)
    return model

def _quantize_jit(model, qconfig_dict, run_fn=None, run_args=None, inplace=False, debug=False, quant_type=QuantType.STATIC):
    if False:
        return 10
    if quant_type == QuantType.DYNAMIC:
        model = prepare_dynamic_jit(model, qconfig_dict, inplace)
        model = convert_dynamic_jit(model, True, debug)
    else:
        assert run_fn, 'Must provide calibration function for post training static quantization'
        assert run_args, 'Must provide calibration dataset for post training static quantization'
        model = prepare_jit(model, qconfig_dict, inplace)
        run_fn(model, *run_args)
        model = convert_jit(model, True, debug)
    torch._C._jit_pass_constant_propagation(model.graph)
    torch._C._jit_pass_dce(model.graph)
    return model

def quantize_jit(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    if False:
        while True:
            i = 10
    "Quantize the input float TorchScript model with\n    post training static quantization.\n\n    First it will prepare the model for calibration, then it calls\n    `run_fn` which will run the calibration step, after that we will\n    convert the model to a quantized model.\n\n    Args:\n        `model`: input float TorchScript model\n        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and\n        qconfig for that module as value, empty key means the qconfig will be applied\n        to whole model unless it's overwritten by more specific configurations, the\n        qconfig for each module is either found in the dictionary or fallback to\n         the qconfig of parent module.\n\n        Right now qconfig_dict is the only way to configure how the model is quantized,\n        and it is done in the granularity of module, that is, we only support one type\n        of qconfig for each torch.nn.Module, and the qconfig for sub module will\n        override the qconfig for parent module, empty string means global configuration.\n        `run_fn`: a calibration function for calibrating the prepared model\n        `run_args`: positional arguments for `run_fn`\n        `inplace`: carry out model transformations in-place, the original module is\n        mutated\n        `debug`: flag for producing a debug friendly model (preserve weight attribute)\n\n    Return:\n        Quantized TorchSciprt model.\n\n    Example:\n    ```python\n    import torch\n    from torch.ao.quantization import get_default_qconfig\n    from torch.ao.quantization import quantize_jit\n\n    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)\n    qconfig = get_default_qconfig('fbgemm')\n    def calibrate(model, data_loader):\n        model.eval()\n        with torch.no_grad():\n            for image, target in data_loader:\n                model(image)\n\n    quantized_model = quantize_jit(\n        ts_model,\n        {'': qconfig},\n        calibrate,\n        [data_loader_test])\n    ```\n    "
    torch._C._log_api_usage_once('quantization_api.quantize_jit.quantize_jit')
    return _quantize_jit(model, qconfig_dict, run_fn, run_args, inplace, debug, quant_type=QuantType.STATIC)

def quantize_dynamic_jit(model, qconfig_dict, inplace=False, debug=False):
    if False:
        print('Hello World!')
    "Quantize the input float TorchScript model with\n    post training dynamic quantization.\n    Currently only qint8 quantization of torch.nn.Linear is supported.\n\n    Args:\n        `model`: input float TorchScript model\n        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and\n        qconfig for that module as value, please see detailed\n        descriptions in :func:`~torch.ao.quantization.quantize_jit`\n        `inplace`: carry out model transformations in-place, the original module is\n        mutated\n        `debug`: flag for producing a debug friendly model (preserve weight attribute)\n\n    Return:\n        Quantized TorchSciprt model.\n\n    Example:\n    ```python\n    import torch\n    from torch.ao.quantization import per_channel_dynamic_qconfig\n    from torch.ao.quantization import quantize_dynamic_jit\n\n    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)\n    qconfig = get_default_qconfig('fbgemm')\n    def calibrate(model, data_loader):\n        model.eval()\n        with torch.no_grad():\n            for image, target in data_loader:\n                model(image)\n\n    quantized_model = quantize_dynamic_jit(\n        ts_model,\n        {'': qconfig},\n        calibrate,\n        [data_loader_test])\n    ```\n    "
    torch._C._log_api_usage_once('quantization_api.quantize_jit.quantize_dynamic_jit')
    return _quantize_jit(model, qconfig_dict, inplace=inplace, debug=debug, quant_type=QuantType.DYNAMIC)

def _quantize_ondevice_dynamic_jit(model, qconfig_dict, method_name='forward', inplace=False):
    if False:
        return 10
    "Prepares the input float TorchScript model with\n    *on-device* post training dynamic quantization.\n    Currently only qint8 quantization of torch.nn.Linear is supported.\n\n    Args:\n        `model`: input float TorchScript model\n        `qconfig_dict`: qconfig_dict is a dictionary with names of sub modules as key and\n        qconfig for that module as value, please see detailed\n        `method_name`: Name of the method within the model, to be prepared for quantization\n        descriptions in :func:`~torch.ao.quantization.quantize_jit`\n        `inplace`: carry out model transformations in-place, the original module is\n        mutated\n\n    Return:\n        TorchScript model that is ready for on device quantization.\n        This means that the returned\n        model has:\n        - Method is inlined.\n        - Model has observer modules inserted in the model.\n        - Model has packed params inserted in the model. However they are empty as in they dont\n          contain valid quantized weights.\n        - observe_<method_name> is added that observe the values to be quantized.\n        - reset_observers_<method_name> to reset observers.\n        - quantize_<method_name> is added to the model.\n          - This method extract scale, zero points.\n          - Quantizes observed weights.\n          - Creates packed params from it and update the attribute of the model with the new values\n            for the packed params.\n          - Reset the original fp32 weights with empty tensor using SetAttr.\n        - quantized_<method_name> is added to the model.\n          - This method uses quantized weights and quantized linear ops instead of fp32 op.\n          - This method should be used for inference post PTQ.\n        - Note that all method's signatures should be the same as method_name.\n\n        Later on device:\n        - Run reset_observers_<method_name>\n        - Run observe_<method_name>\n        - Run quantize_<method_name>\n        - Now model can be saved and loaded later.\n        - Run model with quantized_<method_name>\n\n    Example:\n    ```python\n    import torch\n    from torch.ao.quantization import per_channel_dynamic_qconfig\n    from torch.ao.quantization.quantize_jit import _quantize_ondevice_dynamic_jit\n\n    ts_model = torch.jit.script(float_model.eval())  # or torch.jit.trace(float_model, input)\n    qconfig = get_default_qconfig('fbgemm')\n    quant_ready_model = _quantize_ondevice_dynamic_jit(\n        ts_model,\n        {'': qconfig},\n        'forward',\n        True)\n    ```\n    "
    return _quantize_ondevice_dynamic_jit_impl(model, qconfig_dict, method_name, inplace=inplace)