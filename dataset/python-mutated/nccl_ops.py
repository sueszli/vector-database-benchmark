"""Ops for GPU collective operations implemented using NVIDIA nccl."""
import threading
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nccl_ops
_module_lock = threading.Lock()
_shared_name_counter = 0

def all_sum(tensors):
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of tensors with the all-reduce sum across `tensors`.\n\n  The computation is done with an all-reduce operation, so if only some of the\n  returned tensors are evaluated then the computation will hang.\n\n  Args:\n    tensors: The input tensors across which to sum; must be assigned\n      to GPU devices.\n\n  Returns:\n    List of tensors, each with the sum of the input tensors, where tensor i has\n    the same device as `tensors[i]`.\n  '
    return _apply_all_reduce('sum', tensors)

@ops.RegisterGradient('NcclAllReduce')
def _all_sum_grad(op, grad):
    if False:
        for i in range(10):
            print('nop')
    'The gradients for `all_sum`.\n\n  Args:\n    op: The `all_sum` `Operation` that we are differentiating.\n    grad: Gradient with respect to the output of the `all_sum` op.\n\n  Returns:\n    The gradient with respect to the output of `all_sum`.\n\n  Raises:\n    LookupError: If `reduction` is not `sum`.\n  '
    if op.get_attr('reduction') != b'sum':
        raise LookupError('No gradient defined for NcclAllReduce except for reduction="sum".')
    _check_device(grad, expected=op.device)
    num_devices = op.get_attr('num_devices')
    shared_name = op.get_attr('shared_name') + b'_grad'
    with ops.device(op.device):
        return gen_nccl_ops.nccl_all_reduce(input=grad, reduction='sum', num_devices=num_devices, shared_name=shared_name)

def all_prod(tensors):
    if False:
        print('Hello World!')
    'Returns a list of tensors with the all-reduce product across `tensors`.\n\n  The computation is done with an all-reduce operation, so if only some of the\n  returned tensors are evaluated then the computation will hang.\n\n  Args:\n    tensors: The input tensors across which to multiply; must be assigned\n      to GPU devices.\n\n  Returns:\n    List of tensors, each with the product of the input tensors, where tensor i\n    has the same device as `tensors[i]`.\n  '
    return _apply_all_reduce('prod', tensors)

def all_min(tensors):
    if False:
        i = 10
        return i + 15
    'Returns a list of tensors with the all-reduce min across `tensors`.\n\n  The computation is done with an all-reduce operation, so if only some of the\n  returned tensors are evaluated then the computation will hang.\n\n  Args:\n    tensors: The input tensors across which to reduce; must be assigned\n      to GPU devices.\n\n  Returns:\n    List of tensors, each with the minimum of the input tensors, where tensor i\n    has the same device as `tensors[i]`.\n  '
    return _apply_all_reduce('min', tensors)

def all_max(tensors):
    if False:
        while True:
            i = 10
    'Returns a list of tensors with the all-reduce max across `tensors`.\n\n  The computation is done with an all-reduce operation, so if only some of the\n  returned tensors are evaluated then the computation will hang.\n\n  Args:\n    tensors: The input tensors across which to reduce; must be assigned\n      to GPU devices.\n\n  Returns:\n    List of tensors, each with the maximum of the input tensors, where tensor i\n    has the same device as `tensors[i]`.\n  '
    return _apply_all_reduce('max', tensors)

def reduce_sum(tensors):
    if False:
        while True:
            i = 10
    'Returns a tensor with the reduce sum across `tensors`.\n\n  The computation is done with a reduce operation, so only one tensor is\n  returned.\n\n  Args:\n    tensors: The input tensors across which to sum; must be assigned\n      to GPU devices.\n\n  Returns:\n    A tensor containing the sum of the input tensors.\n\n  Raises:\n    LookupError: If context is not currently using a GPU device.\n  '
    return _apply_reduce('sum', tensors)

@ops.RegisterGradient('NcclReduce')
def _reduce_sum_grad(op, grad):
    if False:
        while True:
            i = 10
    'The gradients for input `Operation` of `reduce_sum`.\n\n  Args:\n    op: The `sum send` `Operation` that we are differentiating.\n    grad: Gradient with respect to the output of the `reduce_sum` op.\n\n  Returns:\n    The gradient with respect to the input of `reduce_sum` op.\n\n  Raises:\n    LookupError: If the reduction attribute of op is not `sum`.\n  '
    if op.get_attr('reduction') != b'sum':
        raise LookupError('No gradient defined for NcclAllReduce except for reduction="sum".')
    _check_device(grad, expected=op.device)
    with ops.device(op.device):
        result = gen_nccl_ops.nccl_broadcast(input=grad, shape=grad.shape)
    return [result] * len(op.inputs)

def broadcast(tensor):
    if False:
        i = 10
        return i + 15
    'Returns a tensor that can be efficiently transferred to other devices.\n\n  Args:\n    tensor: The tensor to send; must be assigned to a GPU device.\n\n  Returns:\n    A tensor with the value of `src_tensor`, which can be used as input to\n    ops on other GPU devices.\n  '
    _check_device(tensor)
    with ops.device(tensor.device):
        return gen_nccl_ops.nccl_broadcast(input=tensor, shape=tensor.shape)

@ops.RegisterGradient('NcclBroadcast')
def _broadcast_grad(op, accumulated_grad):
    if False:
        i = 10
        return i + 15
    'The gradients for input `Operation` of `broadcast`.\n\n  Args:\n    op: The `broadcast send` `Operation` that we are differentiating.\n    accumulated_grad: Accumulated gradients with respect to the output of the\n      `broadcast` op.\n\n  Returns:\n    Gradients with respect to the input of `broadcast`.\n  '
    grads = [t for t in accumulated_grad.op.inputs]
    for t in grads:
        _check_device(t)
    with ops.device(op.device):
        return gen_nccl_ops.nccl_reduce(input=grads, reduction='sum')

def _apply_all_reduce(reduction, tensors):
    if False:
        for i in range(10):
            print('nop')
    'Helper function for all_* functions.'
    if not tensors:
        raise ValueError('Must pass >0 tensors to all reduce operations')
    shared_name = _get_shared_name()

    def _all_reduce():
        if False:
            i = 10
            return i + 15
        'Call nccl allreduce.'
        res = []
        for t in tensors:
            _check_device(t)
            with ops.device(t.device):
                res.append(gen_nccl_ops.nccl_all_reduce(input=t, reduction=reduction, num_devices=len(tensors), shared_name=shared_name))
        return res
    if context.executing_eagerly():
        return def_function.function(_all_reduce)()
    else:
        return _all_reduce()

def _apply_reduce(reduction, tensors):
    if False:
        for i in range(10):
            print('nop')
    'Helper function for reduce_* functions.'
    if not tensors:
        raise ValueError('Must pass >0 tensors to reduce operations')
    for t in tensors:
        _check_device(t)
    result = gen_nccl_ops.nccl_reduce(input=tensors, reduction=reduction)
    try:
        next((t for t in tensors if t.device == result.device))
    except StopIteration:
        raise ValueError('One input tensor must be assigned to current device')
    return result

def _get_shared_name():
    if False:
        while True:
            i = 10
    global _shared_name_counter
    with _module_lock:
        val = _shared_name_counter
        _shared_name_counter += 1
    return 'c%s' % val

def _check_device(tensor, expected=None):
    if False:
        return 10
    if not device.canonical_name(tensor.device):
        raise ValueError(f'Device assignment for tensor={tensor} required for nccl collective ops')
    if expected and expected != tensor.device:
        raise ValueError(f'Expected device {expected}, got {tensor.device} for tensor={tensor}.')