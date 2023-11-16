from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import FakeQuantize, FakeQuantizeBase, default_fake_quant, default_dynamic_fake_quant, default_per_channel_weight_fake_quant, default_weight_fake_quant, default_fused_act_fake_quant, default_fused_wt_fake_quant, FusedMovingAvgObsFakeQuantize, default_fused_per_channel_wt_fake_quant, default_embedding_fake_quant, default_embedding_fake_quant_4bit, fused_wt_fake_quant_range_neg_127_to_127, fused_per_channel_wt_fake_quant_range_neg_127_to_127
from .observer import _PartialWrapper, MinMaxObserver, HistogramObserver, MovingAverageMinMaxObserver, NoopObserver, PlaceholderObserver, ReuseInputObserver, default_debug_observer, default_dynamic_quant_observer, default_float_qparams_observer, default_float_qparams_observer_4bit, default_observer, default_per_channel_weight_observer, default_placeholder_observer, default_weight_observer, weight_observer_range_neg_127_to_127, per_channel_weight_observer_range_neg_127_to_127, default_reuse_input_observer, ObserverBase
import warnings
import copy
__all__ = ['QConfig', 'QConfigDynamic', 'default_qconfig', 'default_debug_qconfig', 'default_per_channel_qconfig', 'default_dynamic_qconfig', 'float16_dynamic_qconfig', 'float16_static_qconfig', 'per_channel_dynamic_qconfig', 'float_qparams_weight_only_qconfig', 'float_qparams_weight_only_qconfig_4bit', 'default_quint8_weight_qconfig', 'default_qat_qconfig', 'default_dynamic_qat_qconfig', 'default_weight_only_qconfig', 'default_activation_only_qconfig', 'default_qat_qconfig_v2', 'default_reuse_input_qconfig', 'default_symmetric_qnnpack_qconfig', 'default_per_channel_symmetric_qnnpack_qconfig', 'default_symmetric_qnnpack_qat_qconfig', 'default_per_channel_symmetric_qnnpack_qat_qconfig', 'default_embedding_qat_qconfig', 'default_embedding_qat_qconfig_4bit', 'get_default_qconfig', 'get_default_qat_qconfig', 'get_default_qconfig_dict', 'get_default_qat_qconfig_dict', 'QConfigAny', 'qconfig_equals']

class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8))

    """

    def __new__(cls, activation, weight):
        if False:
            return 10
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError('QConfig received observer instance, please pass observer class instead. ' + 'Use MyObserver.with_args(x=1) to override arguments to constructor if needed')
        return super().__new__(cls, activation, weight)

class QConfigDynamic(namedtuple('QConfigDynamic', ['activation', 'weight'])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """

    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity):
        if False:
            print('Hello World!')
        if isinstance(weight, nn.Module):
            raise ValueError('QConfigDynamic received observer instance, please pass observer class instead. ' + 'Use MyObserver.with_args(x=1) to override arguments to constructor if needed')
        warnings.warn('QConfigDynamic is going to be deprecated in PyTorch 1.12, please use QConfig instead')
        return super().__new__(cls, activation, weight)
default_qconfig = QConfig(activation=default_observer, weight=default_weight_observer)
'\nDefault qconfig configuration.\n'
default_debug_qconfig = QConfig(weight=default_weight_observer, activation=default_debug_observer)
'\nDefault qconfig configuration for debugging.\n'
default_per_channel_qconfig = QConfig(activation=default_observer, weight=default_per_channel_weight_observer)
'\nDefault qconfig configuration for per channel weight quantization.\n'
default_dynamic_qconfig = QConfig(activation=default_dynamic_quant_observer, weight=default_weight_observer)
'\nDefault dynamic qconfig.\n'
float16_dynamic_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float16, is_dynamic=True), weight=PlaceholderObserver.with_args(dtype=torch.float16))
'\nDynamic qconfig with weights quantized to `torch.float16`.\n'
float16_static_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float16), weight=PlaceholderObserver.with_args(dtype=torch.float16))
'\nDynamic qconfig with both activations and weights quantized to `torch.float16`.\n'
per_channel_dynamic_qconfig = QConfig(activation=default_dynamic_quant_observer, weight=default_per_channel_weight_observer)
'\nDynamic qconfig with weights quantized per channel.\n'
float_qparams_weight_only_qconfig = QConfig(activation=default_placeholder_observer, weight=default_float_qparams_observer)
'\nDynamic qconfig with weights quantized with a floating point zero_point.\n'
float_qparams_weight_only_qconfig_4bit = QConfig(activation=default_placeholder_observer, weight=default_float_qparams_observer_4bit)
default_qat_qconfig = QConfig(activation=default_fake_quant, weight=default_weight_fake_quant)
'\nDefault qconfig for QAT.\n'
default_dynamic_qat_qconfig = QConfig(activation=default_dynamic_fake_quant, weight=default_weight_fake_quant)
'\nDefault qconfig for dynamic QAT.\n'
default_weight_only_qconfig = QConfig(activation=torch.nn.Identity, weight=default_weight_fake_quant)
'\nDefault qconfig for quantizing weights only.\n'
default_activation_only_qconfig = QConfig(activation=default_fake_quant, weight=torch.nn.Identity)
'\nDefault qconfig for quantizing activations only.\n'
default_qat_qconfig_v2 = QConfig(activation=default_fused_act_fake_quant, weight=default_fused_wt_fake_quant)
'\nFused version of `default_qat_config`, has performance benefits.\n'
default_reuse_input_qconfig = QConfig(activation=default_reuse_input_observer, weight=NoopObserver)
'\nDefault qconfig for operators that reuse the observers from input Tensor, e.g. reshape\n'

def get_default_qconfig(backend='x86', version=0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the default PTQ qconfig for the specified backend.\n\n    Args:\n      * `backend` (str): a string representing the target backend. Currently supports\n        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.\n\n    Return:\n        qconfig\n    '
    supported_backends = ['fbgemm', 'x86', 'qnnpack', 'onednn']
    if backend not in supported_backends:
        raise AssertionError('backend: ' + str(backend) + f' not supported. backend must be one of {supported_backends}')
    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True), weight=default_per_channel_weight_observer)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False), weight=default_weight_observer)
        elif backend == 'onednn':
            if not torch.cpu._is_cpu_support_vnni():
                warnings.warn('Default qconfig of oneDNN backend with reduce_range of false may have accuracy issues on CPU without Vector Neural Network Instruction support.')
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False), weight=default_per_channel_weight_observer)
        elif backend == 'x86':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True), weight=default_per_channel_weight_observer)
        else:
            qconfig = default_qconfig
    else:
        raise AssertionError('Version number: ' + str(version) + ' in get_default_qconfig is not supported. Version number must be 0')
    return qconfig
"\nDefault, symmetric PTQ qconfig for the specified backend. And a per_channel\nvariant of the same.\n\nSymmetric here applies to signed weights with zero point = 0, and additional\nvalue restrictions. The activations are also signed 8-bit integers with this\nqconfig.\n\n    * Once this change is merged [as of 3/17/22], with backend or qengine =\n    'qnnpack', some quantized operators with this symmetric qconfig may use\n    operators from xnnpack library.\n\n        ** Support to use xnnpack ops with `qnnpack` backed for asymmetric\n        qconfig (returned by get_default_qconfig()) is not available yet.\n\n    * This qconfig uses signed activations and weights. Weights have added\n    restrictions such as zero point is forced to be 0, making the weights\n    symmetric, hence the name. And the 8-bit quantized values are\n    restricting to to [-127, +127], excluding -128.\n\n    * xnnpack has a requantization scale value restriction, 0x1p-32 <=\n    requantization_scale < 256.0 where, `requantization_scale = (input_scale\n    * kernel_scale) / (output_scale)`. Using this eps (w/ assumed max value\n    of 256) is to prevent requantization_scale to go below xnnpack lower\n    threshold.\n"
default_symmetric_qnnpack_qconfig = QConfig(activation=HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False, eps=2 ** (-12)), weight=weight_observer_range_neg_127_to_127)
default_per_channel_symmetric_qnnpack_qconfig = QConfig(activation=HistogramObserver.with_args(dtype=torch.qint8, reduce_range=False, eps=2 ** (-12)), weight=per_channel_weight_observer_range_neg_127_to_127)
default_embedding_qat_qconfig = QConfig(activation=NoopObserver.with_args(dtype=torch.float32), weight=default_embedding_fake_quant)
default_embedding_qat_qconfig_4bit = QConfig(activation=NoopObserver.with_args(dtype=torch.float32), weight=default_embedding_fake_quant_4bit)
default_quint8_weight_qconfig = QConfig(activation=HistogramObserver, weight=MinMaxObserver)

def get_default_qat_qconfig(backend='x86', version=1):
    if False:
        return 10
    '\n    Returns the default QAT qconfig for the specified backend.\n\n    Args:\n      * `backend` (str): a string representing the target backend. Currently supports\n        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.\n      * `version`: version, for backwards compatibility. Can be `None` or `1`.\n\n    Return:\n        qconfig\n    '
    supported_backends = ['fbgemm', 'x86', 'qnnpack', 'onednn']
    if backend not in supported_backends:
        raise AssertionError('backend: ' + str(backend) + f' not supported. backend must be one of {supported_backends}')
    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True), weight=default_per_channel_weight_fake_quant)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=False), weight=default_weight_fake_quant)
        elif backend == 'onednn':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255), weight=default_per_channel_weight_fake_quant)
        elif backend == 'x86':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True), weight=default_per_channel_weight_fake_quant)
        else:
            qconfig = default_qat_qconfig
    elif version == 1:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True), weight=default_fused_per_channel_wt_fake_quant)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=False), weight=default_fused_wt_fake_quant)
        elif backend == 'onednn':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255), weight=default_fused_per_channel_wt_fake_quant)
        elif backend == 'x86':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True), weight=default_fused_per_channel_wt_fake_quant)
        else:
            qconfig = default_qat_qconfig_v2
    else:
        raise AssertionError('Version number: ' + str(version) + 'in get_default_qat_qconfig is not supported. Version number must be 0 or 1')
    return qconfig
'\nDefault symmetric QAT qconfig for qnnpack. And its per channel weight variant.\n'
default_symmetric_qnnpack_qat_qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127, dtype=torch.qint8, reduce_range=False, eps=2 ** (-12)), weight=fused_wt_fake_quant_range_neg_127_to_127)
default_per_channel_symmetric_qnnpack_qat_qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127, dtype=torch.qint8, reduce_range=False, eps=2 ** (-12)), weight=fused_per_channel_wt_fake_quant_range_neg_127_to_127)
_default_fp32_placeholder_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float32), weight=PlaceholderObserver.with_args(dtype=torch.float32))
_default_quint8_placeholder_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.quint8), weight=None)

def get_default_qconfig_dict(backend='x86', version=0):
    if False:
        while True:
            i = 10
    warnings.warn('torch.ao.quantization.get_default_qconfig_dict is deprecated and will be removed in a future version. Please use torch.ao.quantization.get_default_qconfig_mapping instead.')
    return torch.ao.quantization.get_default_qconfig_mapping(backend, version).to_dict()

def get_default_qat_qconfig_dict(backend='x86', version=1):
    if False:
        return 10
    warnings.warn('torch.ao.quantization.get_default_qat_qconfig_dict is deprecated and will be removed in a future version. Please use torch.ao.quantization.get_default_qat_qconfig_mapping instead.')
    return torch.ao.quantization.get_default_qat_qconfig_mapping(backend, version).to_dict()

def _assert_valid_qconfig(qconfig: Optional[QConfig], mod: torch.nn.Module) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Verifies that this `qconfig` is valid.\n    '
    if qconfig is None:
        return
    is_conv_transpose_mod = isinstance(mod, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d))
    if is_conv_transpose_mod:
        if qconfig.weight is None:
            return
        example_observer = qconfig.weight()
        is_per_channel = isinstance(example_observer, (torch.ao.quantization.PerChannelMinMaxObserver, torch.ao.quantization.MovingAveragePerChannelMinMaxObserver))
        assert not is_per_channel, 'Per channel weight observer is not supported yet for ConvTranspose{n}d.'
QConfigAny = Optional[QConfig]
QConfigAny.__module__ = 'torch.ao.quantization.qconfig'

def _add_module_to_qconfig_obs_ctr(qconfig: QConfigAny, module: Optional[nn.Module]) -> Any:
    if False:
        for i in range(10):
            print('nop')
    "This is a helper function for use in quantization prepare that updates a qconfig so that\n    the constructors stored in the qconfig will create observers on the same device that\n    'module' is on. This is intended to be used when the qconfigs are propagated to each\n    module in order to avoid potential device alignment issues.\n\n    Args:\n        qconfig: QConfig with obs constructors stored in activation and weight\n        module: module which the qconfig is related to\n\n    Return:\n        qconfig: configured so that obs constructors set to construct on the same device as module\n    "
    if module is None or qconfig is None or qconfig._fields != ('activation', 'weight'):
        return qconfig

    def get_factory_kwargs_based_on_module_device():
        if False:
            while True:
                i = 10
        assert isinstance(module, torch.nn.Module)
        devices = {p.device for p in module.parameters()} | {p.device for p in module.buffers()}
        device = next(iter(devices)) if len(devices) > 0 else None
        return None if device is None else {'device': device}

    def configure_constructor_to_put_obs_on_module_device(original_constructor):
        if False:
            while True:
                i = 10
        try:
            check = original_constructor.with_args(factory_kwargs=None)
            check()
            return original_constructor.with_callable_args(factory_kwargs=get_factory_kwargs_based_on_module_device)
        except AttributeError:
            return original_constructor
        except TypeError:
            return original_constructor
    activation = configure_constructor_to_put_obs_on_module_device(qconfig.activation)
    weight = configure_constructor_to_put_obs_on_module_device(qconfig.weight)
    return QConfig(activation, weight)
_ObserverOrFakeQuantizeConstructor = Union[_PartialWrapper, Type[ObserverBase], Type[FakeQuantizeBase]]

def _obs_or_fq_ctr_equals(obs_or_fq1: _ObserverOrFakeQuantizeConstructor, obs_or_fq2: _ObserverOrFakeQuantizeConstructor):
    if False:
        print('Hello World!')
    if isinstance(obs_or_fq1, _PartialWrapper) and isinstance(obs_or_fq2, _PartialWrapper):
        return _partial_wrapper_equals(obs_or_fq1, obs_or_fq2)
    return obs_or_fq1 == obs_or_fq2

def _partial_wrapper_equals(obs_or_fq1: _PartialWrapper, obs_or_fq2: _PartialWrapper):
    if False:
        i = 10
        return i + 15
    '\n    Return whether the two partial wrappers are equal,\n    '
    obs_or_fq1_keywords = copy.copy(obs_or_fq1.p.keywords)
    obs_or_fq2_keywords = copy.copy(obs_or_fq2.p.keywords)
    keywords_equal = True
    if 'observer' in obs_or_fq1_keywords and 'observer' in obs_or_fq2_keywords:
        keywords_equal = keywords_equal and _obs_or_fq_ctr_equals(obs_or_fq1_keywords['observer'], obs_or_fq2_keywords['observer'])
        obs_or_fq1_keywords.pop('observer')
        obs_or_fq2_keywords.pop('observer')
    keywords_equal = keywords_equal and obs_or_fq1_keywords == obs_or_fq2_keywords
    return obs_or_fq1.p.func == obs_or_fq2.p.func and obs_or_fq1.p.args == obs_or_fq2.p.args and keywords_equal

def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns `True` if `q1` equals `q2`, and `False` otherwise.\n    '
    if q1 is None or q2 is None:
        return q1 == q2
    else:
        assert q1 is not None and q2 is not None
        try:
            activation_same = _obs_or_fq_ctr_equals(q1.activation, q2.activation)
            weight_same = _obs_or_fq_ctr_equals(q1.weight, q2.weight)
            return activation_same and weight_same
        except AttributeError:
            return q1 == q2

def _activation_is_memoryless(qconfig: QConfig):
    if False:
        print('Hello World!')
    '\n    Return whether the observer for activations defined in the given QConfig is memoryless.\n    This means a MovingAverage observer with averaging constant equal to 1.\n    '

    def _is_memoryless(observer):
        if False:
            while True:
                i = 10
        return hasattr(observer, 'averaging_constant') and observer.averaging_constant == 1
    act = qconfig.activation()
    if isinstance(act, FakeQuantizeBase) and hasattr(act, 'activation_post_process'):
        return _is_memoryless(act.activation_post_process)
    else:
        return _is_memoryless(act)

def _is_reuse_input_qconfig(qconfig: Optional[QConfig]):
    if False:
        print('Hello World!')
    return qconfig is not None and isinstance(qconfig.activation(), ReuseInputObserver) and isinstance(qconfig.weight(), NoopObserver)