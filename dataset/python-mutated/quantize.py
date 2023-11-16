import copy
import itertools
import warnings
import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization.quantization_mappings import get_default_dynamic_quant_module_mappings, get_default_static_quant_module_mappings, get_default_static_quant_reference_module_mappings, get_default_qat_module_mappings, get_default_qconfig_propagation_list, no_observer_set, _has_special_act_post_process, _get_special_act_post_process
from .utils import get_qparam_dict, has_no_children_ignoring_parametrizations
from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper
from torch.ao.quantization.qconfig import _add_module_to_qconfig_obs_ctr, default_dynamic_qconfig, float16_dynamic_qconfig, float_qparams_weight_only_qconfig, float_qparams_weight_only_qconfig_4bit, _activation_is_memoryless
from torch.nn.utils.parametrize import type_before_parametrizations
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.observer import _is_activation_post_process as is_activation_post_process
__all__ = ['get_default_custom_config_dict', 'propagate_qconfig_', 'add_quant_dequant', 'prepare', 'quantize', 'quantize_dynamic', 'prepare_qat', 'quantize_qat', 'convert', 'swap_module']
_DEFAULT_CUSTOM_CONFIG_DICT = {'float_to_observed_custom_module_class': {nn.LSTM: nn.quantizable.LSTM, nn.MultiheadAttention: nn.quantizable.MultiheadAttention}, 'observed_to_quantized_custom_module_class': {nn.quantizable.LSTM: nn.quantized.LSTM, nn.quantizable.MultiheadAttention: nn.quantized.MultiheadAttention}}

def get_default_custom_config_dict():
    if False:
        for i in range(10):
            print('nop')
    'Defines the default custom config dict.\n    '
    return _DEFAULT_CUSTOM_CONFIG_DICT

def _propagate_qconfig_helper(module, qconfig_dict, qconfig_parent=None, prefix='', prepare_custom_config_dict=None):
    if False:
        return 10
    'This is a helper function for `propagate_qconfig_`\n\n    Args:\n        module: input module\n        qconfig_dict: dictionary that maps from name of submodule to quantization\n                     configuration\n        qconfig_parent: quantization config of parent module, we will fallback to\n                       this config when there is no specified config for current\n                       module\n        prefix: corresponding prefix of the current module, used as key in\n                qconfig_dict\n        prepare_custom_config_dict: dictionary for custom handling of modules\n                                    see docs for :func:`~torch.ao.quantization.prepare_fx`\n\n    Return:\n        None, module is modified inplace with qconfig attached\n    '
    module_qconfig = qconfig_dict.get(type_before_parametrizations(module), qconfig_parent)
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)
    module_qconfig = getattr(module, 'qconfig', module_qconfig)
    torch.ao.quantization.qconfig._assert_valid_qconfig(module_qconfig, module)
    qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(module_qconfig, module)
    module.qconfig = qconfig_with_device_check
    for (name, child) in module.named_children():
        module_prefix = prefix + '.' + name if prefix else name
        if prepare_custom_config_dict is None or not (name in prepare_custom_config_dict.get('non_traceable_module_name', []) or type(child) in prepare_custom_config_dict.get('non_traceable_module_class', [])):
            _propagate_qconfig_helper(child, qconfig_dict, qconfig_with_device_check, module_prefix)

def propagate_qconfig_(module, qconfig_dict=None, prepare_custom_config_dict=None):
    if False:
        for i in range(10):
            print('nop')
    'Propagate qconfig through the module hierarchy and assign `qconfig`\n    attribute on each leaf module\n\n    Args:\n        module: input module\n        qconfig_dict: dictionary that maps from name or type of submodule to\n            quantization configuration, qconfig applies to all submodules of a\n            given module unless qconfig for the submodules are specified (when\n            the submodule already has qconfig attribute)\n        prepare_custom_config_dict: dictionary for custom handling of modules\n            see docs for :func:`~torch.ao.quantization.prepare_fx`\n\n    Return:\n        None, module is modified inplace with qconfig attached\n    '
    if qconfig_dict is None:
        qconfig_dict = {}
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    _propagate_qconfig_helper(module, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)

def _observer_forward_hook(self, input, output):
    if False:
        while True:
            i = 10
    'Forward hook that calls observer on the output\n    '
    return self.activation_post_process(output)

def _observer_forward_pre_hook(self, input):
    if False:
        for i in range(10):
            print('nop')
    'Forward pre hook that calls observer on the output\n    '
    return self.activation_post_process(input[0])

def _register_activation_post_process_hook(module, pre_hook=False):
    if False:
        while True:
            i = 10
    assert hasattr(module, 'activation_post_process'), 'Expect activation_post_process attribute already attached to the module'
    if pre_hook:
        handle = module.register_forward_pre_hook(_observer_forward_pre_hook, prepend=True)
    else:
        handle = module.register_forward_hook(_observer_forward_hook, prepend=True)

def _add_observer_(module, qconfig_propagation_list=None, non_leaf_module_list=None, device=None, custom_module_class_mapping=None):
    if False:
        print('Hello World!')
    'Add observer for the leaf child of the module.\n\n    This function insert observer module to all leaf child module that\n    has a valid qconfig attribute.\n\n    Args:\n        module: input module with qconfig attributes for all the leaf modules that we want to quantize\n        qconfig_propagation_list: a list of quantizable modules that will have observers added to them\n            if they are leaf nodes\n        device: parent device, if any\n        non_leaf_module_list: list of non-leaf modules we want to add observer\n\n    Return:\n        None, module is modified inplace with added observer modules and forward_hooks\n    '
    if qconfig_propagation_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()
    if custom_module_class_mapping is None:
        custom_module_class_mapping = {}
    if device is None:
        devices = _get_unique_devices_(module)
        assert len(devices) <= 1, f'_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}'
        device = next(iter(devices)) if len(devices) > 0 else None

    def get_activation_post_process(qconfig, device, special_act_post_process=None):
        if False:
            while True:
                i = 10
        activation = qconfig.activation() if special_act_post_process is None else special_act_post_process()
        if device is not None:
            activation.to(device)
        return activation

    def needs_observation(m):
        if False:
            print('Hello World!')
        return hasattr(m, 'qconfig') and m.qconfig is not None

    def insert_activation_post_process(m, special_act_post_process=None):
        if False:
            return 10
        ' Adds an activation post process module and register\n        a pre or post hook that calls the module\n        '
        if needs_observation(m) and (not isinstance(m, DeQuantStub)):
            m.add_module('activation_post_process', get_activation_post_process(m.qconfig, device, special_act_post_process))
            _register_activation_post_process_hook(m, pre_hook=_activation_is_memoryless(m.qconfig))
    for (name, child) in module.named_children():
        if type_before_parametrizations(child) in [nn.Dropout]:
            continue
        elif issubclass(type_before_parametrizations(child), (nnq.FloatFunctional, nnq.QFunctional)):
            if needs_observation(child):
                assert hasattr(child, 'activation_post_process'), f'functional class {type_before_parametrizations(child)} has no pre-defined `activation_post_process`'
                child.activation_post_process = get_activation_post_process(child.qconfig, device)
        elif isinstance(child, _FusedModule):
            if needs_observation(child):
                insert_activation_post_process(child)
        elif non_leaf_module_list is not None and type_before_parametrizations(child) in non_leaf_module_list:
            if needs_observation(child):
                insert_activation_post_process(child)
        elif _has_special_act_post_process(child):
            special_act_post_process = _get_special_act_post_process(child)
            insert_activation_post_process(child, special_act_post_process)
        elif needs_observation(child) and type_before_parametrizations(child) in custom_module_class_mapping:
            observed_child = custom_module_class_mapping[type_before_parametrizations(child)].from_float(child)
            setattr(module, name, observed_child)
            if custom_module_class_mapping[type_before_parametrizations(child)] not in no_observer_set():
                insert_activation_post_process(observed_child)
        else:
            _add_observer_(child, qconfig_propagation_list, non_leaf_module_list, device, custom_module_class_mapping)
    if has_no_children_ignoring_parametrizations(module) and (not isinstance(module, torch.nn.Sequential)) and (type_before_parametrizations(module) in qconfig_propagation_list):
        insert_activation_post_process(module)

def _get_unique_devices_(module):
    if False:
        print('Hello World!')
    return {p.device for p in module.parameters()} | {p.device for p in module.buffers()}

def add_quant_dequant(module):
    if False:
        for i in range(10):
            print('nop')
    'Wrap the leaf child module in QuantWrapper if it has a valid qconfig\n    Note that this function will modify the children of module inplace and it\n    can return a new module which wraps the input module as well.\n\n    Args:\n        module: input module with qconfig attributes for all the leaf modules\n        that we want to quantize\n\n    Return:\n        Either the inplace modified module with submodules wrapped in\n        `QuantWrapper` based on qconfig or a new `QuantWrapper` module which\n        wraps the input module, the latter case only happens when the input\n        module is a leaf module and we want to quantize it.\n    '
    if has_no_children_ignoring_parametrizations(module) and hasattr(module, 'qconfig') and module.qconfig:
        return QuantWrapper(module)
    for (name, child) in module.named_children():
        module._modules[name] = add_quant_dequant(child)
    return module

def prepare(model, inplace=False, allow_list=None, observer_non_leaf_module_list=None, prepare_custom_config_dict=None):
    if False:
        i = 10
        return i + 15
    'Prepares a copy of the model for quantization calibration or quantization-aware training.\n\n    Quantization configuration should be assigned preemptively\n    to individual submodules in `.qconfig` attribute.\n\n    The model will be attached with observer or fake quant modules, and qconfig\n    will be propagated.\n\n    Args:\n        `model`: input model to be modified in-place\n        `inplace`: carry out model transformations in-place, the original module is mutated\n        `allow_list`: list of quantizable modules\n        `observer_non_leaf_module_list`: list of non-leaf modules we want to add observer\n        `prepare_custom_config_dict`: customization configuration dictionary for prepare function\n\n    .. code-block:: python\n\n       # Example of prepare_custom_config_dict:\n       prepare_custom_config_dict = {\n           # user will manually define the corresponding observed\n           # module class which has a from_float class method that converts\n           # float custom module to observed custom module\n           "float_to_observed_custom_module_class": {\n               CustomModule: ObservedCustomModule\n           }\n        }\n\n    '
    torch._C._log_api_usage_once('quantization_api.quantize.prepare')
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = prepare_custom_config_dict.get('float_to_observed_custom_module_class', {})
    if not inplace:
        model = copy.deepcopy(model)
    qconfig_propagation_list = allow_list
    if allow_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()
    propagate_qconfig_(model, qconfig_dict=None)
    if not any((hasattr(m, 'qconfig') and m.qconfig for m in model.modules())):
        warnings.warn('None of the submodule got qconfig applied. Make sure you passed correct configuration through `qconfig_dict` or by assigning the `.qconfig` attribute directly on submodules')
    _add_observer_(model, qconfig_propagation_list, observer_non_leaf_module_list, custom_module_class_mapping=custom_module_class_mapping)
    return model

def _remove_activation_post_process(module):
    if False:
        return 10
    if hasattr(module, 'activation_post_process') and _is_activation_post_process(module.activation_post_process):
        delattr(module, 'activation_post_process')

    def remove_hooks(pre_hook=False):
        if False:
            return 10
        hook_map = module._forward_pre_hooks if pre_hook else module._forward_hooks
        observer_hook = _observer_forward_pre_hook if pre_hook else _observer_forward_hook
        handle_ids_to_remove = set()
        for (handle_id, hook_fn) in hook_map.items():
            if hook_fn is observer_hook:
                handle_ids_to_remove.add(handle_id)
        for handle_id in handle_ids_to_remove:
            hook_map.pop(handle_id)
    remove_hooks(pre_hook=True)
    remove_hooks(pre_hook=False)

def _remove_qconfig(module):
    if False:
        for i in range(10):
            print('nop')
    'Clean up the qconfig left in the module so that new qconfig can be\n    propagated.\n\n    Args:\n        module: module to be cleaned up\n    '
    for child in module.children():
        _remove_qconfig(child)
    if hasattr(module, 'qconfig'):
        del module.qconfig
    _remove_activation_post_process(module)

def quantize(model, run_fn, run_args, mapping=None, inplace=False):
    if False:
        while True:
            i = 10
    'Quantize the input float model with post training static quantization.\n\n    First it will prepare the model for calibration, then it calls\n    `run_fn` which will run the calibration step, after that we will\n    convert the model to a quantized model.\n\n    Args:\n        model: input float model\n        run_fn: a calibration function for calibrating the prepared model\n        run_args: positional arguments for `run_fn`\n        inplace: carry out model transformations in-place, the original module is mutated\n        mapping: correspondence between original module types and quantized counterparts\n\n    Return:\n        Quantized model.\n    '
    torch._C._log_api_usage_once('quantization_api.quantize.quantize')
    if mapping is None:
        mapping = get_default_static_quant_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    prepare(model, inplace=True)
    run_fn(model, *run_args)
    convert(model, mapping, inplace=True)
    return model

def quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False):
    if False:
        i = 10
        return i + 15
    'Converts a float model to dynamic (i.e. weights-only) quantized model.\n\n    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.\n\n    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization\n    by default is performed for layers with large weights size - i.e. Linear and RNN variants.\n\n    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.\n    If `qconfig` is provided, the `dtype` argument is ignored.\n\n    Args:\n        model: input model\n        qconfig_spec: Either:\n\n            - A dictionary that maps from name or type of submodule to quantization\n              configuration, qconfig applies to all submodules of a given\n              module unless qconfig for the submodules are specified (when the\n              submodule already has qconfig attribute). Entries in the dictionary\n              need to be QConfig instances.\n\n            - A set of types and/or submodule names to apply dynamic quantization to,\n              in which case the `dtype` argument is used to specify the bit-width\n\n        inplace: carry out model transformations in-place, the original module is mutated\n        mapping: maps type of a submodule to a type of corresponding dynamically quantized version\n            with which the submodule needs to be replaced\n\n    '
    torch._C._log_api_usage_once('quantization_api.quantize.quantize_dynamic')
    if qconfig_spec is None:
        if dtype == torch.qint8:
            qconfig_spec = {nn.Linear: default_dynamic_qconfig, nn.LSTM: default_dynamic_qconfig, nn.GRU: default_dynamic_qconfig, nn.LSTMCell: default_dynamic_qconfig, nn.RNNCell: default_dynamic_qconfig, nn.GRUCell: default_dynamic_qconfig}
        elif dtype == torch.float16:
            qconfig_spec = {nn.Linear: float16_dynamic_qconfig, nn.LSTM: float16_dynamic_qconfig, nn.GRU: float16_dynamic_qconfig, nn.LSTMCell: float16_dynamic_qconfig, nn.RNNCell: float16_dynamic_qconfig, nn.GRUCell: float16_dynamic_qconfig}
        elif dtype == torch.quint8:
            qconfig_spec = {nn.EmbeddingBag: float_qparams_weight_only_qconfig, nn.Embedding: float_qparams_weight_only_qconfig}
        elif dtype == torch.quint4x2:
            qconfig_spec = {nn.EmbeddingBag: float_qparams_weight_only_qconfig_4bit}
        else:
            raise ValueError(f"Don't know how to quantize with default settings for {dtype}. Provide full qconfig please")
    elif isinstance(qconfig_spec, set):
        if dtype is torch.qint8:
            default_qconfig = default_dynamic_qconfig
        elif dtype is torch.float16:
            default_qconfig = float16_dynamic_qconfig
        elif dtype is torch.quint8:
            default_qconfig = float_qparams_weight_only_qconfig
        elif dtype is torch.quint4x2:
            default_qconfig = float_qparams_weight_only_qconfig_4bit
        else:
            raise RuntimeError('Unknown dtype specified for quantize_dynamic: ', str(dtype))
        qconfig_spec = dict(zip(qconfig_spec, itertools.repeat(default_qconfig)))
    if mapping is None:
        mapping = get_default_dynamic_quant_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    propagate_qconfig_(model, qconfig_spec)
    convert(model, mapping, inplace=True)
    return model

def prepare_qat(model, mapping=None, inplace=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepares a copy of the model for quantization calibration or\n    quantization-aware training and converts it to quantized version.\n\n    Quantization configuration should be assigned preemptively\n    to individual submodules in `.qconfig` attribute.\n\n    Args:\n        model: input model to be modified in-place\n        mapping: dictionary that maps float modules to quantized modules to be\n                 replaced.\n        inplace: carry out model transformations in-place, the original module\n                 is mutated\n    '
    torch._C._log_api_usage_once('quantization_api.quantize.prepare_qat')
    assert model.training, 'prepare_qat only works on models in training mode'
    if mapping is None:
        mapping = get_default_qat_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)
    propagate_qconfig_(model, qconfig_dict=None)
    convert(model, mapping=mapping, inplace=True, remove_qconfig=False)
    prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)
    return model

def quantize_qat(model, run_fn, run_args, inplace=False):
    if False:
        return 10
    'Do quantization aware training and output a quantized model\n\n    Args:\n        model: input model\n        run_fn: a function for evaluating the prepared model, can be a\n                function that simply runs the prepared model or a training\n                loop\n        run_args: positional arguments for `run_fn`\n\n    Return:\n        Quantized model.\n    '
    torch._C._log_api_usage_once('quantization_api.quantize.quantize_qat')
    if not inplace:
        model = copy.deepcopy(model)
    model.train()
    prepare_qat(model, inplace=True)
    run_fn(model, *run_args)
    convert(model, inplace=True)
    return model

def convert(module, mapping=None, inplace=False, remove_qconfig=True, is_reference=False, convert_custom_config_dict=None):
    if False:
        for i in range(10):
            print('nop')
    'Converts submodules in input module to a different module according to `mapping`\n    by calling `from_float` method on the target module class. And remove qconfig at the\n    end if remove_qconfig is set to True.\n\n    Args:\n        `module`: prepared and calibrated module\n        `mapping`: a dictionary that maps from source module type to target\n                   module type, can be overwritten to allow swapping user defined\n                   Modules\n        `inplace`: carry out model transformations in-place, the original module\n                   is mutated\n        `convert_custom_config_dict`: custom configuration dictionary for convert function\n\n    .. code-block:: python\n\n       # Example of convert_custom_config_dict:\n       convert_custom_config_dict = {\n           # user will manually define the corresponding quantized\n           # module class which has a from_observed class method that converts\n           # observed custom module to quantized custom module\n           "observed_to_quantized_custom_module_class": {\n               ObservedCustomModule: QuantizedCustomModule\n           }\n       }\n\n    '
    torch._C._log_api_usage_once('quantization_api.quantize.convert')
    if not inplace:
        module = copy.deepcopy(module)
    _convert(module, mapping, inplace=True, is_reference=is_reference, convert_custom_config_dict=convert_custom_config_dict)
    if remove_qconfig:
        _remove_qconfig(module)
    return module

def _convert(module, mapping=None, inplace=False, is_reference=False, convert_custom_config_dict=None):
    if False:
        for i in range(10):
            print('nop')
    'Converts submodules in input module to a different module according to `mapping`\n    by calling `from_float` method on the target module class\n\n    Args:\n        module: input module\n        mapping: a dictionary that maps from source module type to target\n                 module type, can be overwritten to allow swapping user defined\n                 Modules\n        inplace: carry out model transformations in-place, the original module\n                 is mutated\n        is_reference: a flag to enable quantized reference module\n\n    '
    if mapping is None:
        mapping = get_default_static_quant_reference_module_mappings() if is_reference else get_default_static_quant_module_mappings()
    if convert_custom_config_dict is None:
        convert_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = convert_custom_config_dict.get('observed_to_quantized_custom_module_class', {})
    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for (name, mod) in module.named_children():
        if not isinstance(mod, _FusedModule) and type_before_parametrizations(mod) not in custom_module_class_mapping:
            _convert(mod, mapping, True, is_reference, convert_custom_config_dict)
        reassign[name] = swap_module(mod, mapping, custom_module_class_mapping)
    for (key, value) in reassign.items():
        module._modules[key] = value
    return module

def swap_module(mod, mapping, custom_module_class_mapping):
    if False:
        for i in range(10):
            print('nop')
    'Swaps the module if it has a quantized counterpart and it has an\n    `observer` attached.\n\n    Args:\n        mod: input module\n        mapping: a dictionary that maps from nn module to nnq module\n\n    Return:\n        The corresponding quantized module of `mod`\n    '
    new_mod = mod
    if hasattr(mod, 'qconfig') and mod.qconfig is not None:
        swapped = False
        if type_before_parametrizations(mod) in custom_module_class_mapping:
            new_mod = custom_module_class_mapping[type_before_parametrizations(mod)].from_observed(mod)
            swapped = True
        elif type_before_parametrizations(mod) in mapping:
            qmod = mapping[type_before_parametrizations(mod)]
            if hasattr(qmod, '_IS_REFERENCE') and qmod._IS_REFERENCE:
                assert mod.qconfig is not None
                weight_post_process = mod.qconfig.weight()
                weight_post_process(mod.weight)
                weight_qparams = get_qparam_dict(weight_post_process)
                new_mod = qmod.from_float(mod, weight_qparams)
            else:
                new_mod = qmod.from_float(mod)
            swapped = True
        if swapped:
            for pre_hook_fn in mod._forward_pre_hooks.values():
                new_mod.register_forward_pre_hook(pre_hook_fn)
            for hook_fn in mod._forward_hooks.values():
                if hook_fn is not _observer_forward_hook:
                    new_mod.register_forward_hook(hook_fn)
            devices = _get_unique_devices_(mod)
            assert len(devices) <= 1, f'swap_module only works with cpu or single-device CUDA modules, but got devices {devices}'
            device = next(iter(devices)) if len(devices) > 0 else None
            if device:
                new_mod.to(device)
    return new_mod

def _get_observer_dict(mod, target_dict, prefix=''):
    if False:
        return 10
    'Traverse the modules and save all observers into dict.\n    This is mainly used for quantization accuracy debug\n    Args:\n        mod: the top module we want to save all observers\n        prefix: the prefix for the current module\n        target_dict: the dictionary used to save all the observers\n    '

    def get_prefix(prefix):
        if False:
            print('Hello World!')
        return prefix if prefix == '' else prefix + '.'
    if hasattr(mod, 'activation_post_process'):
        target_dict[get_prefix(prefix) + 'activation_post_process'] = mod.activation_post_process
    for (name, child) in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_observer_dict(child, target_dict, module_prefix)