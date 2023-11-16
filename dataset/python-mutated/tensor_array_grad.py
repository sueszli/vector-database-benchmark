"""Gradients for operators defined in tensor_array_ops.py."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
ops.NotDifferentiable('TensorArray')
ops.NotDifferentiable('TensorArrayGrad')
ops.NotDifferentiable('TensorArraySize')
ops.NotDifferentiable('TensorArrayClose')
ops.NotDifferentiable('TensorArrayV2')
ops.NotDifferentiable('TensorArrayGradV2')
ops.NotDifferentiable('TensorArraySizeV2')
ops.NotDifferentiable('TensorArrayCloseV2')
ops.NotDifferentiable('TensorArrayV3')
ops.NotDifferentiable('TensorArrayGradV3')
ops.NotDifferentiable('TensorArrayGradWithShape')
ops.NotDifferentiable('TensorArraySizeV3')
ops.NotDifferentiable('TensorArrayCloseV3')

def _GetGradSource(op_or_tensor):
    if False:
        while True:
            i = 10
    "Identify which call to tf.gradients created this gradient op or tensor.\n\n  TensorArray gradient calls use an accumulator TensorArray object.  If\n  multiple gradients are calculated and run in the same session, the multiple\n  gradient nodes may accidentally flow through the same accumulator TensorArray.\n  This double counting breaks the TensorArray gradient flow.\n\n  The solution is to identify which gradient call this particular\n  TensorArray*Grad is being called in, by looking at the input gradient\n  tensor's name, and create or lookup an accumulator gradient TensorArray\n  associated with this specific call.  This solves any confusion and ensures\n  different gradients from the same forward graph get their own accumulators.\n\n  This function creates the unique label associated with the tf.gradients call\n  that is used to create the gradient TensorArray.\n\n  Args:\n    op_or_tensor: `Tensor` or `Operation` which is an input to a\n      TensorArray*Grad call.\n\n  Returns:\n    A python string, the unique label associated with this particular\n    gradients calculation.\n\n  Raises:\n    ValueError: If not called within a gradients calculation.\n  "
    name_tokens = op_or_tensor.name.split('/')
    grad_pos = [i for (i, x) in enumerate(name_tokens) if x.startswith('gradients')]
    if not grad_pos:
        raise ValueError(f"Expected op/tensor name to start with gradients (excluding scope), got: {op_or_tensor.name}. This means that a tf.gradients op with this op in its dependency path has a custom name that does not start with 'gradients'. Please make sure all calls to tf.gradients that have non-empty `name` arguments use names that start with 'gradients'.")
    return '/'.join(name_tokens[:grad_pos[-1] + 1])

@ops.RegisterGradient('TensorArrayRead')
@ops.RegisterGradient('TensorArrayReadV2')
@ops.RegisterGradient('TensorArrayReadV3')
def _TensorArrayReadGrad(op: ops.Operation, grad):
    if False:
        return 10
    'Gradient for TensorArrayRead.\n\n  Args:\n    op: Forward TensorArrayRead op.\n    grad: Gradient `Tensor` to TensorArrayRead.\n\n  Returns:\n    A flow `Tensor`, which can be used in control dependencies to\n    force the write of `grad` to the gradient `TensorArray`.\n  '
    handle = op.inputs[0]
    index = op.inputs[1]
    flow = op.inputs[2]
    dtype = op.get_attr('dtype')
    grad_source = _GetGradSource(grad)
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    w_g = g.write(index, grad)
    return [None, None, w_g.flow]

@ops.RegisterGradient('TensorArrayWrite')
@ops.RegisterGradient('TensorArrayWriteV2')
@ops.RegisterGradient('TensorArrayWriteV3')
def _TensorArrayWriteGrad(op: ops.Operation, flow):
    if False:
        for i in range(10):
            print('nop')
    'Gradient for TensorArrayWrite.\n\n  Args:\n    op: Forward TensorArrayWrite op.\n    flow: Gradient `Tensor` flow to TensorArrayWrite.\n\n  Returns:\n    A grad `Tensor`, the gradient created in an upstream ReadGrad or PackGrad.\n  '
    handle = op.inputs[0]
    index = op.inputs[1]
    dtype = op.get_attr('T')
    grad_source = _GetGradSource(flow)
    flow_out = array_ops.identity(op.outputs[0], 'flow_out')
    with ops.control_dependencies([flow_out]):
        flow = array_ops.identity(flow, 'write_barrier')
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    grad = g.read(index)
    return [None, None, grad, flow]

@ops.RegisterGradient('TensorArrayGather')
@ops.RegisterGradient('TensorArrayGatherV2')
@ops.RegisterGradient('TensorArrayGatherV3')
def _TensorArrayGatherGrad(op: ops.Operation, grad):
    if False:
        i = 10
        return i + 15
    'Gradient for TensorArrayGather.\n\n  Args:\n    op: Forward TensorArrayGather op.\n    grad: Gradient `Tensor` to TensorArrayGather.\n\n  Returns:\n    A flow `Tensor`, which can be used in control dependencies to\n    force the write of `grad` to the gradient `TensorArray`.\n  '
    handle = op.inputs[0]
    indices = op.inputs[1]
    flow = op.inputs[2]
    dtype = op.get_attr('dtype')
    grad_source = _GetGradSource(grad)
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    u_g = g.scatter(indices, grad)
    return [None, None, u_g.flow]

@ops.RegisterGradient('TensorArrayScatter')
@ops.RegisterGradient('TensorArrayScatterV2')
@ops.RegisterGradient('TensorArrayScatterV3')
def _TensorArrayScatterGrad(op: ops.Operation, flow):
    if False:
        print('Hello World!')
    'Gradient for TensorArrayScatter.\n\n  Args:\n    op: Forward TensorArrayScatter op.\n    flow: Gradient `Tensor` flow to TensorArrayScatter.\n\n  Returns:\n    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.\n  '
    handle = op.inputs[0]
    indices = op.inputs[1]
    dtype = op.get_attr('T')
    grad_source = _GetGradSource(flow)
    flow_out = array_ops.identity(op.outputs[0], 'flow_out')
    with ops.control_dependencies([flow_out]):
        flow = array_ops.identity(flow, 'write_barrier')
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    grad = g.gather(indices)
    return [None, None, grad, flow]

@ops.RegisterGradient('TensorArrayConcat')
@ops.RegisterGradient('TensorArrayConcatV2')
@ops.RegisterGradient('TensorArrayConcatV3')
def _TensorArrayConcatGrad(op: ops.Operation, grad, unused_lengths_grad):
    if False:
        return 10
    'Gradient for TensorArrayConcat.\n\n  Args:\n    op: Forward TensorArrayConcat op.\n    grad: Gradient `Tensor` to TensorArrayConcat.\n\n  Returns:\n    A flow `Tensor`, which can be used in control dependencies to\n    force the write of `grad` to the gradient `TensorArray`.\n  '
    handle = op.inputs[0]
    flow = op.inputs[1]
    lengths = op.outputs[1]
    dtype = op.get_attr('dtype')
    grad_source = _GetGradSource(grad)
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    u_g = g.split(grad, lengths=lengths)
    return [None, u_g.flow]

@ops.RegisterGradient('TensorArraySplit')
@ops.RegisterGradient('TensorArraySplitV2')
@ops.RegisterGradient('TensorArraySplitV3')
def _TensorArraySplitGrad(op: ops.Operation, flow):
    if False:
        return 10
    'Gradient for TensorArraySplit.\n\n  Args:\n    op: Forward TensorArraySplit op.\n    flow: Gradient `Tensor` flow to TensorArraySplit.\n\n  Returns:\n    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.\n  '
    handle = op.inputs[0]
    dtype = op.get_attr('T')
    grad_source = _GetGradSource(flow)
    flow_out = array_ops.identity(op.outputs[0], 'flow_out')
    with ops.control_dependencies([flow_out]):
        flow = array_ops.identity(flow, 'write_barrier')
    g = tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow, colocate_with_first_write_call=False).grad(source=grad_source, flow=flow)
    grad = g.concat()
    return [None, grad, None, flow]