"""Gradients for operators defined in control_flow_ops.py."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.control_flow_ops import *

def _SwitchGrad(op, *grad):
    if False:
        while True:
            i = 10
    'Gradients for a Switch op is calculated using a Merge op.\n\n  If the switch is a loop switch, it will be visited twice. We create\n  the merge on the first visit, and update the other input of the merge\n  on the second visit. A next_iteration is also added on second visit.\n  '
    graph = ops.get_default_graph()
    op_ctxt = op._get_control_flow_context()
    grad_ctxt = graph._get_control_flow_context()
    if isinstance(op_ctxt, WhileContext):
        merge_grad = grad_ctxt.grad_state.switch_map.get(op)
        if merge_grad is not None:
            if grad[1] is not None:
                control_flow_ops._AddNextAndBackEdge(merge_grad, grad[1], enforce_shape_invariant=False)
            return (None, None)
        elif grad[0] is not None:
            merge_grad = merge([grad[0], grad[0]], name='b_switch')[0]
            grad_ctxt.grad_state.switch_map[op] = merge_grad
            return (merge_grad, None)
        else:
            return (None, None)
    elif isinstance(op_ctxt, CondContext):
        zero_grad = grad[1 - op_ctxt.branch]
        if zero_grad is None:
            if op.inputs[0].dtype == dtypes.resource:
                return (merge([grad[op_ctxt.branch]] * 2, name='cond_resource_grad')[0], None)
            return (None, None)
        return (merge(grad, name='cond_grad')[0], None)
    else:
        false_grad = switch(grad[0], op.inputs[1])[0]
        true_grad = switch(grad[1], op.inputs[1])[1]
        return (merge([false_grad, true_grad])[0], None)
ops.RegisterGradient('Switch')(_SwitchGrad)
ops.RegisterGradient('RefSwitch')(_SwitchGrad)

@ops.RegisterGradient('Merge')
def _MergeGrad(op, grad, _):
    if False:
        return 10
    'Gradients for a Merge op are calculated using a Switch op.'
    input_op = op.inputs[0].op
    graph = ops.get_default_graph()
    op_ctxt = control_flow_util.GetOutputContext(input_op)
    grad_ctxt = graph._get_control_flow_context()
    if isinstance(op_ctxt, WhileContext):
        return control_flow_ops._SwitchRefOrTensor(grad, grad_ctxt.pivot)
    elif isinstance(op_ctxt, CondContext):
        pred = op_ctxt.pred
        if grad_ctxt and grad_ctxt.grad_state:
            grad_state = grad_ctxt.grad_state
            real_pred = grad_state.history_map.get(pred.name)
            if real_pred is None:
                grad_ctxt = grad_state.grad_context
                grad_ctxt.Exit()
                history_pred = grad_state.AddForwardAccumulator(pred)
                grad_ctxt.Enter()
                real_pred = grad_state.AddBackpropAccumulatedValue(history_pred, pred)
                grad_state.history_map[pred.name] = real_pred
            pred = real_pred
        return control_flow_ops._SwitchRefOrTensor(grad, pred, name='cond_grad')
    else:
        num_inputs = len(op.inputs)
        cond = [math_ops.equal(op.outputs[1], i) for i in range(num_inputs)]
        return [control_flow_ops._SwitchRefOrTensor(grad, cond[i])[1] for i in range(num_inputs)]

@ops.RegisterGradient('RefMerge')
def _RefMergeGrad(op, grad, _):
    if False:
        for i in range(10):
            print('nop')
    return _MergeGrad(op, grad, _)

@ops.RegisterGradient('Exit')
def _ExitGrad(op, grad):
    if False:
        return 10
    'Gradients for an exit op are calculated using an Enter op.'
    graph = ops.get_default_graph()
    op_ctxt = op._get_control_flow_context()
    grad_ctxt = graph._get_control_flow_context()
    if not grad_ctxt.back_prop:
        return None
    if op_ctxt.grad_state:
        raise TypeError('Second-order gradient for while loops not supported.')
    if isinstance(grad, tensor.Tensor):
        grad_ctxt.AddName(grad.name)
    else:
        if not isinstance(grad, (indexed_slices.IndexedSlices, sparse_tensor.SparseTensor)):
            raise TypeError(f'Type {type(grad)} not supported, must be either`indexed_slices.IndexedSlices` or `SparseTensor`.')
        grad_ctxt.AddName(grad.values.name)
        grad_ctxt.AddName(grad.indices.name)
        dense_shape = grad.dense_shape
        if dense_shape is not None:
            grad_ctxt.AddName(dense_shape.name)
    grad_ctxt.Enter()
    result = control_flow_ops._Enter(grad, grad_ctxt.name, is_constant=False, parallel_iterations=grad_ctxt.parallel_iterations, name='b_exit')
    grad_ctxt.loop_enters.append(result)
    grad_ctxt.Exit()
    return result
ops.RegisterGradient('RefExit')(_ExitGrad)

@ops.RegisterGradient('NextIteration')
def _NextIterationGrad(_, grad):
    if False:
        for i in range(10):
            print('nop')
    'A forward next_iteration is translated into a backprop identity.\n\n  Note that the backprop next_iteration is added in switch grad.\n  '
    return grad

@ops.RegisterGradient('RefNextIteration')
def _RefNextIterationGrad(_, grad):
    if False:
        while True:
            i = 10
    return _NextIterationGrad(_, grad)

@ops.RegisterGradient('Enter')
def _EnterGrad(op, grad):
    if False:
        while True:
            i = 10
    'Gradients for an Enter are calculated using an Exit op.\n\n  For loop variables, grad is the gradient so just add an exit.\n  For loop invariants, we need to add an accumulator loop.\n  '
    graph = ops.get_default_graph()
    grad_ctxt = graph._get_control_flow_context()
    if grad_ctxt is None:
        return grad
    if not grad_ctxt.back_prop:
        return grad
    if grad_ctxt.grad_state is None:
        return grad
    if op.get_attr('is_constant'):
        if isinstance(grad, tensor.Tensor):
            result = grad_ctxt.AddBackpropAccumulator(op, grad)
        elif isinstance(grad, indexed_slices.IndexedSlices):
            result = grad_ctxt.AddBackpropIndexedSlicesAccumulator(op, grad)
        else:
            raise TypeError(f'Type {type(grad)} not supported,must be Tensor or Indexed Slices')
    else:
        result = exit(grad)
        grad_ctxt.loop_exits.append(result)
        grad_ctxt.ExitResult([result])
    return result

@ops.RegisterGradient('RefEnter')
def _RefEnterGrad(op, grad):
    if False:
        print('Hello World!')
    return _EnterGrad(op, grad)

@ops.RegisterGradient('LoopCond')
def _LoopCondGrad(_):
    if False:
        print('Hello World!')
    'Stop backprop for the predicate of a while loop.'
    return None