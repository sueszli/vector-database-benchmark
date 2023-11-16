import logging
import warnings
from functorch.compile import make_boxed_func
from ..backends.common import aot_autograd
from .registry import register_backend, register_experimental_backend
log = logging.getLogger(__name__)

@register_experimental_backend
def torchxla_trivial(gm, fake_tensor_inputs):
    if False:
        return 10
    return gm

@register_experimental_backend
def torchxla_trace_once(model, fake_tensor_inputs):
    if False:
        i = 10
        return i + 15
    warnings.warn('This backend will be deprecated in 2.2, please use `openxla` backend instead')
    return xla_backend_helper(model, fake_tensor_inputs)

@register_backend
def openxla_eval(model, fake_tensor_inputs):
    if False:
        for i in range(10):
            print('nop')
    return xla_backend_helper(model, fake_tensor_inputs, boxed=False)

def openxla_eval_boxed(model, fake_tensor_inputs):
    if False:
        i = 10
        return i + 15
    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)

def xla_backend_helper(model, fake_tensor_inputs, boxed=False):
    if False:
        return 10
    try:
        import torch_xla.core.dynamo_bridge as bridge
    except ImportError as e:
        raise ImportError('Please follow the instruction in https://github.com/pytorch/xla#pytorchxla to install torch_xla') from e
    compiled_graph = None

    def fwd(*args):
        if False:
            return 10
        nonlocal model
        nonlocal compiled_graph
        if compiled_graph is None:
            compiled_graph = bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)
    return make_boxed_func(fwd) if boxed else fwd
aot_torchxla_trivial = aot_autograd(fw_compiler=torchxla_trivial)
register_experimental_backend(name='aot_torchxla_trivial', compiler_fn=aot_torchxla_trivial)
aot_torchxla_trace_once = aot_autograd(fw_compiler=torchxla_trace_once)
register_experimental_backend(name='aot_torchxla_trace_once', compiler_fn=aot_torchxla_trace_once)
openxla = aot_autograd(fw_compiler=openxla_eval_boxed)
register_backend(name='openxla', compiler_fn=openxla)