from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode
__all__ = ['trace_wrapped']

def trace_wrapped(*args, fn):
    if False:
        print('Hello World!')
    return _trace_wrapped_op(*args, fn=fn)
_trace_wrapped_op = HigherOrderOperator('trace_wrapped')

def _assert_meta(grad, size, stride, dtype):
    if False:
        return 10
    assert grad.size() == size, 'size mismatch'
    assert grad.stride() == stride, 'stride mismatch'
    assert grad.dtype == dtype, 'dtype mismatch'
    return grad

@_trace_wrapped_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(mode, *args, fn):
    if False:
        print('Hello World!')
    import torch
    assert len(args) == 1
    grad = args[0]
    assert isinstance(grad, torch.Tensor)

    def self_invoke(*args):
        if False:
            i = 10
            return i + 15
        return _trace_wrapped_op(*args, fn=fn)
    proxy_args = (mode.tracer.unwrap_proxy(grad),)
    out_proxy = mode.tracer.create_proxy('call_function', self_invoke, proxy_args, {}, name='trace_wrapped')
    grad = torch.zeros_like(grad)
    grad = track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    proxy_args = (mode.tracer.unwrap_proxy(grad), grad.size(), grad.stride(), grad.dtype)
    out_proxy = mode.tracer.create_proxy('call_function', _assert_meta, proxy_args, {}, name='assert')
    grad = torch.empty_like(grad)
    grad = track_tensor_tree(grad, out_proxy, constant=None, tracer=mode.tracer)
    return grad

@_trace_wrapped_op.py_impl(FakeTensorMode)
def inner_fake(*args, fn):
    if False:
        i = 10
        return i + 15
    raise RuntimeError('This op should never be invoked here')

@_trace_wrapped_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _trace_wrapped_op_dense(*args, fn):
    if False:
        print('Hello World!')
    mode = _get_current_dispatch_mode()
    assert mode is None, 'Mode should never be enabled for CPU/CUDA key'
    return fn(*args)
_trace_wrapped_op.py_impl(DispatchKey.Autograd)(autograd_not_implemented(_trace_wrapped_op, deferred_error=True))

@_trace_wrapped_op.py_functionalize_impl
def _trace_wrapped_functionalized(ctx, *args, fn):
    if False:
        i = 10
        return i + 15
    unwrapped_args = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        return ctx.wrap_tensors(_trace_wrapped_op(*unwrapped_args, fn=fn))