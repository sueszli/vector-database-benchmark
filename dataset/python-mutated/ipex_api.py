from typing import Any
from bigdl.nano.utils.common import invalidInputError

def ipex_optimize(model: Any, optimizers: Any=None, dtype: Any=None, inplace: bool=False, weights_prepack: Any=None):
    if False:
        for i in range(10):
            print('nop')
    import intel_extension_for_pytorch as ipex
    training = model.training
    if optimizers is not None and (not isinstance(optimizers, (list, tuple))):
        model.train()
        optimizer = optimizers
    elif optimizers is None or len(optimizers) == 0:
        model.eval()
        optimizer = None
    elif len(optimizers) == 1:
        model.train()
        optimizer = optimizers[0]
    else:
        invalidInputError(False, 'Ipex does not support more than one optimizers.')
    ret = ipex.optimize(model=model, dtype=dtype, optimizer=optimizer, inplace=inplace, weights_prepack=weights_prepack)
    model.train(training)
    return ret

def PytorchIPEXJITModel(model, input_sample=None, use_ipex=False, use_jit=False, channels_last=None, thread_num=None, inplace=False, jit_strict=True, jit_method=None, weights_prepack=None, enable_onednn=False, example_kwarg_inputs=None):
    if False:
        print('Hello World!')
    '\n    :param model: the model(nn.module) to be transform.\n    :param input_sample: torch tensor indicate the data sample to be used\n            for tracing.\n    :param use_ipex: if use ipex to optimize the model\n    :param use_jit: if use jit to accelerate the model\n    :param channels_last: if set model and data to be channels-last mode.\n    :param thread_num: the thread num allocated for this model.\n    :param inplace: whether to perform inplace optimization. Default: ``False``.\n    :param jit_strict: Whether recording your mutable container types.\n    :param jit_method: use ``jit.trace`` or ``jit.script`` to\n           convert a model to TorchScript.\n    :param weights_prepack: Whether to perform weight prepack for convolution and linear\n           to avoid oneDNN weights reorder. The default value is None. Explicitly setting\n           this knob overwrites the configuration set by level knob. Only valid when\n           ``use_ipex=True``, otherwise will be ignored.\n    :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph\n           API, which provides a flexible API for aggressive fusion. Default to\n           ``False``, only valid when use_jit is ``True``, otherwise will be ignored.\n    :param example_kwarg_inputs: keyword arguments of example inputs that will be passed\n           to ``torch.jit.trace``. Default to None. Either this argument or input_sample\n           should be specified when use_jit is ``True`` and torch > 2.0,\n           otherwise will be ignored.\n    '
    from .ipex_inference_model import PytorchIPEXJITModel
    return PytorchIPEXJITModel(model, input_sample=input_sample, use_ipex=use_ipex, use_jit=use_jit, channels_last=channels_last, thread_num=thread_num, inplace=inplace, jit_strict=jit_strict, jit_method=jit_method, weights_prepack=weights_prepack, enable_onednn=enable_onednn, example_kwarg_inputs=example_kwarg_inputs)

def PytorchIPEXJITBF16Model(model, input_sample=None, use_ipex=False, use_jit=False, channels_last=None, thread_num=None, inplace=False, jit_strict=True, jit_method=None, weights_prepack=None, enable_onednn=False, example_kwarg_inputs=None):
    if False:
        i = 10
        return i + 15
    '\n    :param model: the model(nn.module) to be transform.\n    :param input_sample: torch tensor indicate the data sample to be used\n            for tracing.\n    :param use_ipex: if use ipex to optimize the model\n    :param use_jit: if use jit to accelerate the model\n    :param channels_last: if set model and data to be channels-last mode.\n    :param thread_num: the thread num allocated for this model.\n    :param inplace: whether to perform inplace optimization. Default: ``False``.\n    :param jit_strict: Whether recording your mutable container types.\n    :param jit_method: use ``jit.trace`` or ``jit.script`` to\n           convert a model to TorchScript.\n    :param weights_prepack: Whether to perform weight prepack for convolution and linear\n           to avoid oneDNN weights reorder. The default value is None. Explicitly setting\n           this knob overwrites the configuration set by level knob. Only valid when\n           ``use_ipex=True``, otherwise will be ignored.\n    :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph\n           API, which provides a flexible API for aggressive fusion. Default to\n           ``False``, only valid when use_jit is ``True``, otherwise will be ignored.\n    :param example_kwarg_inputs: keyword arguments of example inputs that will be passed\n           to ``torch.jit.trace``. Default to None. Either this argument or input_sample\n           should be specified when use_jit is ``True`` and torch > 2.0,\n           otherwise will be ignored.\n    '
    from .ipex_inference_bf16_model import PytorchIPEXJITBF16Model
    return PytorchIPEXJITBF16Model(model, input_sample=input_sample, use_ipex=use_ipex, use_jit=use_jit, channels_last=channels_last, thread_num=thread_num, inplace=inplace, jit_strict=jit_strict, jit_method=jit_method, weights_prepack=weights_prepack, enable_onednn=enable_onednn, example_kwarg_inputs=example_kwarg_inputs)

def PytorchIPEXQuantizationModel(model, calib_data, q_config=None, input_sample=None, channels_last=None, thread_num=None, inplace=False, jit_strict=True, example_kwarg_inputs=None, enable_onednn=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param model: the model(nn.module) to be transform.\n    :param calib_data: calibration data is required for static quantization.\n    :param q_config: describes how to quantize a layer or a part of the network\n            by providing settings (observer classes) for activations and weights\n            respectively.\n    :param input_sample: torch tensor indicate the data sample to be used\n            for tracing.\n    :param channels_last: if set model and data to be channels-last mode.\n    :param thread_num: the thread num allocated for this model.\n    :param inplace: whether to perform inplace optimization. Default: ``False``.\n    :param jit_strict: Whether recording your mutable container types.\n    :param example_kwarg_inputs: keyword arguments of example inputs that will be passed\n           to ``torch.jit.trace``. Default to None. Either this argument or input_sample\n           should be specified when use_jit is ``True`` and torch > 2.0,\n           otherwise will be ignored.\n    :param enable_onednn: Whether to use PyTorch JIT graph fuser based on oneDNN Graph\n           API, which provides a flexible API for aggressive fusion. Default to\n           ``False``.\n    '
    from .ipex_quantization_model import PytorchIPEXQuantizationModel
    return PytorchIPEXQuantizationModel(model, calib_data, q_config=q_config, input_sample=input_sample, channels_last=channels_last, thread_num=thread_num, inplace=inplace, jit_strict=jit_strict, example_kwarg_inputs=example_kwarg_inputs, enable_onednn=enable_onednn)

def PytorchIPEXPUModel(model, thread_num=None, precision='fp32', use_ipex=False):
    if False:
        print('Hello World!')
    '\n    :param model: the model(nn.module) to be transform.\n    :param thread_num: the thread num allocated for this model.\n    '
    from .ipex_inference_xpu_model import PytorchIPEXPUModel
    return PytorchIPEXPUModel(model, thread_num=thread_num, precision=precision, use_ipex=use_ipex)

def load_ipexjit_model(path, model, inplace=False, input_sample=None):
    if False:
        i = 10
        return i + 15
    from .ipex_inference_model import PytorchIPEXJITModel
    return PytorchIPEXJITModel._load(path, model, inplace=inplace, input_sample=input_sample)

def load_ipexjitbf16_model(path, model, inplace=False, input_sample=None):
    if False:
        while True:
            i = 10
    from .ipex_inference_bf16_model import PytorchIPEXJITBF16Model
    return PytorchIPEXJITBF16Model._load(path, model, inplace=inplace, input_sample=input_sample)

def load_ipex_quantization_model(path, model, inplace=False):
    if False:
        for i in range(10):
            print('nop')
    from .ipex_quantization_model import PytorchIPEXQuantizationModel
    return PytorchIPEXQuantizationModel._load(path, model, inplace=inplace)

def load_ipex_xpu_model(path, model, inplace=False):
    if False:
        for i in range(10):
            print('nop')
    from .ipex_inference_xpu_model import PytorchIPEXPUModel
    return PytorchIPEXPUModel._load(path, model, inplace=inplace)