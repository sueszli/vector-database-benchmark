"""Utilities for managing state of v1 control flow for computing gradients."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops

def _GetMaxSizeFromNestedMaximumIterations(value, while_ctxt):
    if False:
        print('Hello World!')
    "Calculate a max_size for use by stack ops inside an XLA while_loop.\n\n  Args:\n    value: The value inside the while_loop forward context.  Used for printing\n      error messages.\n    while_ctxt: The forward context inside which value resides.  This does not\n      always match the value's immediate context, as `value` may be inside e.g.\n      a cond context inside the while_loop.\n\n  Returns:\n    A tensor containing the `max_size` to feed to a Stack initializer.\n\n  Raises:\n    ValueError: If `value` is nested inside a `while_loop` that either\n      lacks a `maximum_iterations` parameter, or the `maximum_iterations`\n      parameter:\n\n        - is inside a `while_loop` that is a parent of the calling context, and\n        - cannot be evaluated at graph build time to a constant.\n  "
    value_name = value.name
    curr_ctxt = ops.get_default_graph()._get_control_flow_context()
    curr_ctxt_name = curr_ctxt.name if curr_ctxt is not None else ''
    max_size = constant_op.constant(1)
    while while_ctxt not in (None, curr_ctxt):
        max_iter = while_ctxt.maximum_iterations
        if max_iter is None:
            raise ValueError("Cannot create a gradient accumulator for tensor '%s' inside XLA while_loop because maximum_iterations was not passed to the tf.while_loop call ('%s')." % (value_name, while_ctxt.name))
        max_iter_ctxt = max_iter.op._get_control_flow_context()
        if util.IsContainingContext(curr_ctxt, max_iter_ctxt):
            max_size *= max_iter
        else:
            const_max_iter = tensor_util.constant_value(max_iter)
            if const_max_iter is None:
                raise ValueError("Cannot create a gradient accumulator for tensor '%s' inside XLA while_loop. maximum_iterations tensor '%s' for while_loop context '%s' must be statically known (e.g. a constant value or known shape dimension), or be defined at or outside the while loop context '%s' (currently defined in '%s')." % (value_name, max_iter.name, while_ctxt.name, curr_ctxt_name, max_iter_ctxt.name))
            max_size *= const_max_iter
        while_ctxt = util.GetContainingWhileContext(while_ctxt.outer_context, stop_ctxt=curr_ctxt)
    return max_size

class _GradLoopState:
    """The state used for constructing the gradient graph for a while loop.

  We create a _GradLoopState for each while loop in forward and its
  corresponding while loop in backprop. This gives us access to both
  the forward and the backprop WhileContexts.

  During the construction of gradient graph, any time when we detect
  a forward value that is needed for backprop, we create a history
  accumulator and add it to `history_map`. Any time when we backprop
  a loop switch op (in _SwitchGrad), we add the grad merge op in
  `switch_map`.
  """

    def __init__(self, forward_ctxt, outer_grad_state):
        if False:
            for i in range(10):
                print('nop')
        self._outer_grad_state = None
        self._forward_context = None
        self._forward_index = None
        self._forward_sync = None
        self._grad_context = None
        self._grad_index = None
        self._grad_sync = None
        self._history_map = {}
        self._switch_map = {}
        self._unused_exits = []
        self._deferred_exits = []
        self._forward_loop_exits = list(forward_ctxt.loop_exits)
        self._pending_exits_count = len(forward_ctxt.loop_exits)
        self._outer_grad_state = outer_grad_state
        if outer_grad_state:
            outer_forward_ctxt = outer_grad_state.forward_context
        else:
            if not hasattr(forward_ctxt, 'outer_context'):
                raise ValueError('Failed to call gradients on a while loop withoutproperly serializing graph via MetaGraphDef')
            outer_forward_ctxt = forward_ctxt.outer_context
        with forward_ctxt._graph.as_default():
            if outer_forward_ctxt:
                outer_forward_ctxt.Enter()
            (cnt, forward_index) = forward_ctxt.AddForwardLoopCounter(outer_grad_state)
            if outer_forward_ctxt:
                outer_forward_ctxt.Exit()
        self._forward_context = forward_ctxt
        self._forward_index = forward_index
        if outer_grad_state:
            outer_forward_ctxt.AddName(cnt.name)
            history_cnt = outer_grad_state.AddForwardAccumulator(cnt)
            outer_grad_ctxt = outer_grad_state.grad_context
            outer_grad_ctxt.Enter()
            self._grad_context = control_flow_ops.WhileContext(maximum_iterations=forward_ctxt.maximum_iterations, parallel_iterations=forward_ctxt.parallel_iterations, back_prop=forward_ctxt.back_prop, swap_memory=forward_ctxt.swap_memory, name=forward_ctxt.name, grad_state=self)
            real_cnt = outer_grad_state.AddBackpropAccumulatedValue(history_cnt, cnt)
            self._grad_index = self._grad_context.AddBackpropLoopCounter(real_cnt, outer_grad_state)
            outer_grad_ctxt.Exit()
        else:
            if outer_forward_ctxt:
                outer_forward_ctxt.Enter()
            self._grad_context = control_flow_ops.WhileContext(maximum_iterations=forward_ctxt.maximum_iterations, parallel_iterations=forward_ctxt.parallel_iterations, back_prop=forward_ctxt.back_prop, swap_memory=forward_ctxt.swap_memory, name=forward_ctxt.name, grad_state=self)
            self._grad_index = self._grad_context.AddBackpropLoopCounter(cnt, outer_grad_state)
            if outer_forward_ctxt:
                outer_forward_ctxt.Exit()

    @property
    def outer_grad_state(self):
        if False:
            i = 10
            return i + 15
        'The grad loop state for outer loop.'
        return self._outer_grad_state

    @property
    def forward_context(self):
        if False:
            i = 10
            return i + 15
        'The while loop context for forward.'
        return self._forward_context

    @property
    def forward_index(self):
        if False:
            return 10
        'The loop index of forward loop.'
        return self._forward_index

    @property
    def forward_sync(self):
        if False:
            i = 10
            return i + 15
        'A control trigger node for synchronization in the forward loop.\n\n    One main use is to keep the push ops of a stack executed in the\n    iteration order.\n    '
        if self._forward_sync is None:
            with ops.control_dependencies(None):
                self._forward_sync = control_flow_ops.control_trigger(name='f_sync')
            self._forward_sync._set_control_flow_context(self._forward_context)
            self._forward_index.op._add_control_input(self._forward_sync)
        return self._forward_sync

    @property
    def grad_context(self):
        if False:
            return 10
        'The corresponding WhileContext for gradient.'
        return self._grad_context

    @property
    def grad_index(self):
        if False:
            print('Hello World!')
        'The loop index of backprop loop.'
        return self._grad_index

    @property
    def grad_sync(self):
        if False:
            for i in range(10):
                print('nop')
        'A control trigger node for synchronization in the grad loop.\n\n    One main use is to keep the pop ops of a stack executed in the\n    iteration order.\n    '
        if self._grad_sync is None:
            with ops.control_dependencies(None):
                self._grad_sync = control_flow_ops.control_trigger(name='b_sync')
            self._grad_sync._set_control_flow_context(self._grad_context)
            self._grad_index.op._add_control_input(self._grad_sync)
            if self._grad_context.outer_context:
                self._grad_context.outer_context.AddInnerOp(self._grad_sync)
        return self._grad_sync

    @property
    def history_map(self):
        if False:
            return 10
        'The map that records all the tensors needed for backprop.'
        return self._history_map

    @property
    def switch_map(self):
        if False:
            for i in range(10):
                print('nop')
        'The map that records all the Switch ops for the while loop.'
        return self._switch_map

    @property
    def unused_exits(self):
        if False:
            while True:
                i = 10
        'The list of "unused" exits.'
        return self._unused_exits

    @property
    def deferred_exits(self):
        if False:
            print('Hello World!')
        'The list of "deferred" exits.'
        return self._deferred_exits

    @property
    def forward_loop_exits(self):
        if False:
            print('Hello World!')
        'The list of exits of the forward loop.'
        return self._forward_loop_exits

    @property
    def pending_exits_count(self):
        if False:
            return 10
        "The number of exits we expect to see but haven't."
        return self._pending_exits_count

    @pending_exits_count.setter
    def pending_exits_count(self, cnt):
        if False:
            i = 10
            return i + 15
        'Set the pending count to cnt.'
        self._pending_exits_count = cnt

    def AddForwardAccumulator(self, value, dead_branch=False):
        if False:
            while True:
                i = 10
        "Add an accumulator for each forward tensor that is needed in backprop.\n\n    This is added to the forward loop at the first time when a tensor\n    in the forward loop is used by backprop gradient computation loop.\n    We create an accumulator that accumulates the value of tensor at each\n    iteration. Called in the control flow context where gradients() is called.\n\n    The pseudocode is:\n    ```\n      acc = stack();\n      while (_pivot) {\n        acc = stack_push(acc, value);\n      }\n    ```\n\n    We make sure that the stack push op in one iteration is executed before\n    next iteration. This is achieved by adding a control edge from\n    `forward_index.op.inputs[0].op` to the push op, and another control\n    edge from the push op to either `forward_index.op` or `forward_sync`.\n\n    Args:\n      value: The source tensor in forward that is to be accumulated.\n      dead_branch: True iff the tensor is on a dead branch of a cond.\n\n    Returns:\n      The stack that contains the accumulated history of the tensor.\n\n    Raises:\n      TypeError: For internal errors involving the value condition context.\n      ValueError: If `value` is inside a XLA scope and a valid max size\n        for the stack can't be found.\n    "
        with self._forward_index.graph.as_default():
            curr_ctxt = ops.get_default_graph()._get_control_flow_context()
            with ops.control_dependencies(None):
                if curr_ctxt:
                    curr_ctxt.Enter()
                with ops.colocate_with(value):
                    if not util.IsInXLAContext(value.op):
                        max_size = constant_op.constant(-1, dtypes.int32)
                    else:
                        max_size = _GetMaxSizeFromNestedMaximumIterations(value, self.forward_context)
                    acc = gen_data_flow_ops.stack_v2(max_size=max_size, elem_type=value.dtype.base_dtype, name='f_acc')
                if curr_ctxt:
                    curr_ctxt.Exit()
                enter_acc = self.forward_context.AddValue(acc)
                swap_enabled = self.forward_context.swap_memory
                value_ctxt = util.GetOutputContext(value.op)
                if value_ctxt == self.forward_context:
                    self.forward_context.Enter()
                    push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                    self.forward_context.Exit()
                    self.forward_index.op._add_control_input(push.op)
                else:
                    if not isinstance(value_ctxt, control_flow_ops.CondContext):
                        raise TypeError('value_ctxt is not a CondContext: %s' % value_ctxt)
                    if dead_branch:
                        value_ctxt.outer_context.Enter()
                        push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                        value_ctxt.outer_context.Exit()
                        push.op._set_control_flow_context(value_ctxt)
                    else:
                        value_ctxt.Enter()
                        push = gen_data_flow_ops.stack_push_v2(enter_acc, value, swap_memory=swap_enabled)
                        value_ctxt.Exit()
                    self.forward_sync._add_control_input(push.op)
                add_op = self.forward_index.op.inputs[0].op
                push.op._add_control_input(add_op)
                return acc

    def AddBackpropAccumulatedValue(self, history_value, value, dead_branch=False):
        if False:
            i = 10
            return i + 15
        'Add the getter for an accumulated value in the grad context.\n\n    This is added to the backprop loop. Called in the grad context to\n    get the value of an accumulated value. The stack pop op must be guarded\n    by the pred of the controlling cond.\n\n    Args:\n      history_value: The history (a stack) of a value.\n      value: The value that is pushed onto the stack.\n      dead_branch: True iff the tensor is on a dead branch of a cond.\n\n    Returns:\n      The current value (the top of the stack).\n    '
        history_ctxt = history_value.op._get_control_flow_context()
        cond_ctxt = None
        value_ctxt = value.op._get_control_flow_context()
        while value_ctxt and value_ctxt != history_ctxt:
            if isinstance(value_ctxt, control_flow_ops.CondContext):
                cond_ctxt = value_ctxt
                break
            value_ctxt = value_ctxt.outer_context
        with ops.control_dependencies(None):
            self.grad_context.Enter()
            if cond_ctxt:
                grad_state = self
                pred = None
                while pred is None and grad_state:
                    pred = grad_state.history_map.get(cond_ctxt.pred.name)
                    grad_state = grad_state.outer_grad_state
                if pred is None:
                    pred = cond_ctxt.pred
                branch = 1 - cond_ctxt.branch if dead_branch else cond_ctxt.branch
                history_value = control_flow_ops._SwitchRefOrTensor(history_value, pred)[branch]
            pop = gen_data_flow_ops.stack_pop_v2(history_value, value.dtype.base_dtype)
            pop.set_shape(value.get_shape())
            self.grad_context.Exit()
        parallel_iterations = self.grad_context.parallel_iterations
        if parallel_iterations > 1:
            self.grad_sync._add_control_input(pop.op)
        return pop

    def GetRealValue(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Get the real value of `value`.\n\n    If backprop "uses" a value produced by forward inference, an accumulator\n    is added in the forward loop to accumulate its values.  We use the\n    accumulated value. This method must be called in the grad loop context.\n    `value` must be in forward and needed for backprop.\n\n    Args:\n      value: A tensor to be captured.\n\n    Returns:\n      The same tensor obtained from the saved history.\n    '
        assert value.op.type not in ['Variable', 'VariableV2']
        real_value = self._history_map.get(value.name)
        if real_value is None:
            cur_value = value
            cur_grad_state = self
            while True:
                enter_op = util.GetLoopConstantEnter(cur_value)
                if enter_op:
                    cur_value = enter_op.inputs[0]
                    cur_grad_state = cur_grad_state.outer_grad_state
                    if cur_grad_state is None:
                        real_value = self._grad_context.AddValue(cur_value)
                        break
                elif constant_op.is_constant(cur_value):
                    real_value = constant_op.constant(tensor_util.constant_value(cur_value), dtype=cur_value.dtype)
                    break
                else:
                    self._grad_context.Exit()
                    history_value = cur_grad_state.AddForwardAccumulator(cur_value)
                    self._grad_context.Enter()
                    break
            if real_value is None:
                real_value = cur_grad_state.AddBackpropAccumulatedValue(history_value, cur_value)
                if cur_grad_state != self:
                    real_value = self._grad_context.AddValue(real_value)
            self._history_map[value.name] = real_value
        return real_value

class _ControlFlowState:
    """Maintain the mapping from the loops to their grad states."""

    def __init__(self):
        if False:
            print('Hello World!')
        self._map = {}

    def GetGradState(self, op: ops.Operation, before):
        if False:
            for i in range(10):
                print('nop')
        "Return the grad state for this op if it's in a forward loop context."
        if before and util.IsLoopExit(op):
            forward_ctxt = op._get_control_flow_context()
            forward_ctxt = forward_ctxt.outer_context
            if forward_ctxt:
                forward_ctxt = forward_ctxt.GetWhileContext()
        else:
            forward_ctxt = util.GetWhileContext(op)
        if forward_ctxt:
            return self._map.get(forward_ctxt)
        return None

    def ProcessUnusedLoopExits(self, pending_count, to_ops_set):
        if False:
            while True:
                i = 10
        'Process all the "unused" loop exits.\n\n    The "unused" exits of the loops are added to `unused_exits`. An exit is\n    unused if its pending_count is 0. If there is an exit with real gradient,\n    all these deferred exits will enter the backprop loop with zero gradient.\n    Otherwise, they will enter the backprop loop with None. As an example,\n    people often write:\n\n    ```python\n    v1, _ = tf.while_loop(p, b, [x1, x2])\n    result = gradients(v1, x1)\n    ```\n\n    The exit node for x2 is not included by the betweenness analysis. But we\n    need to backprop x2 if x2 is involved in computing v1.\n\n    Args:\n      pending_count: The number of backprop inputs for every op.\n      to_ops_set: The set of ops for ys in gradients(ys, xs)\n\n    Returns:\n      The set of unused loop exits that we know at this point we need\n      to backprop.\n    '
        loop_exits = []
        for grad_state in self._map.values():
            for y in grad_state.forward_loop_exits:
                if pending_count[y.op] == 0:
                    grad_state.pending_exits_count -= 1
                    if y.op not in to_ops_set:
                        grad_state.unused_exits.append(y)
                    if grad_state.pending_exits_count == 0:
                        loop_exits.extend(grad_state.unused_exits)
            for y in grad_state.forward_context.loop_enters:
                if pending_count[y.op] == 0:
                    pending_count[y.op] = 1
        return loop_exits

    def EnterGradWhileContext(self, op, before):
        if False:
            for i in range(10):
                print('nop')
        'Enter the WhileContext for gradient computation.'
        grad_state = self.GetGradState(op, before)
        if grad_state:
            grad_state.grad_context.Enter()

    def ExitGradWhileContext(self, op, before):
        if False:
            for i in range(10):
                print('nop')
        'Exit the WhileContext for gradient computation.'
        grad_state = self.GetGradState(op, before)
        if grad_state:
            grad_state.grad_context.Exit()

    def AddWhileContext(self, op, between_op_list, between_ops):
        if False:
            while True:
                i = 10
        'Add the grad state for the while loop that op belongs to.\n\n    Note that op is an Exit, and this method must be called in\n    the control flow context where gradients() is called.\n\n    Note that this method modifies `between_op_list` and `between_ops`.\n    '
        forward_ctxt = util.GetWhileContext(op)
        grad_state = self._map.get(forward_ctxt)
        if grad_state is None:
            outer_forward_ctxt = forward_ctxt.outer_context
            if outer_forward_ctxt:
                outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
            outer_grad_state = None
            if outer_forward_ctxt:
                outer_grad_state = self._map.get(outer_forward_ctxt)
            grad_state = _GradLoopState(forward_ctxt, outer_grad_state)
            self._map[forward_ctxt] = grad_state
            for loop_exit in grad_state.forward_loop_exits:
                if loop_exit.op not in between_ops:
                    between_ops.add(loop_exit.op)
                    between_op_list.append(loop_exit.op)

    def ZerosLikeForExit(self, val):
        if False:
            print('Hello World!')
        'Create zeros_like gradient for a loop exit.\n\n    If the result of a loop variable is not used but is involved in\n    computing the result of some needed loop variable, we create a\n    zero-valued tensor that is fed as gradient for the Exit node of that\n    loop variable. Note that val.op is an Exit, and this method must be\n    called in the control flow context where gradients() is called.\n\n    Args:\n      val: The output tensor of an Exit op.\n\n    Returns:\n      A zero tensor of the same shape of val.\n    '
        val_shape = val.get_shape()
        forward_ctxt = val.op._get_control_flow_context()
        outer_forward_ctxt = forward_ctxt.outer_context
        if outer_forward_ctxt:
            outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
        outer_grad_state = None
        if outer_forward_ctxt:
            outer_grad_state = self._map.get(outer_forward_ctxt)
        if outer_grad_state:
            if val_shape.is_fully_defined():
                outer_grad_state.grad_context.Enter()
                result = array_ops.zeros(val_shape.dims, val.dtype)
                outer_grad_state.grad_context.Exit()
            else:
                forward_ctxt.outer_context.Enter()
                shape = array_ops.shape_internal(val, optimize=False)
                forward_ctxt.outer_context.Exit()
                history_shape = outer_grad_state.AddForwardAccumulator(shape)
                outer_grad_ctxt = outer_grad_state.grad_context
                outer_grad_ctxt.Enter()
                real_shape = outer_grad_state.AddBackpropAccumulatedValue(history_shape, shape)
                result = array_ops.zeros(real_shape, val.dtype)
                outer_grad_ctxt.Exit()
        elif val_shape.is_fully_defined():
            result = array_ops.zeros(val_shape.dims, val.dtype)
        else:
            result = array_ops.zeros_like(val, optimize=False)
        return result

    def ZerosLikeV1WhileLoop(self, op, index):
        if False:
            while True:
                i = 10
        'Create zeros_like for the specified output of an op.\n\n    If op is in a while loop that is part of gradients(), this method\n    must be called in its grad loop context.\n\n    Args:\n      op: A tensorflow operation.\n      index: the index for a specific output of the op.\n\n    Returns:\n      A zero tensor of the same shape of op.outputs[index].\n    '
        if util.IsLoopSwitch(op):
            return None
        if op.graph.building_function:
            return array_ops.zeros_like(op.outputs[index])
        dead_branch = util.IsSwitch(op)
        forward_ctxt = util.GetWhileContext(op)
        grad_state = self._map.get(forward_ctxt)
        if grad_state is None:
            return ZerosLike(op, index)
        op_ctxt = op._get_control_flow_context()
        val = ops.convert_to_tensor(op.outputs[index], name='tensor')
        shape = val.get_shape()
        if shape.is_fully_defined():
            if val.dtype == dtypes.resource:
                result = array_ops.zeros(resource_variable_ops.variable_shape(val), dtype=default_gradient.get_zeros_dtype(val))
            else:
                result = constant_op.constant(0, shape=shape.dims, dtype=val.dtype)
            if dead_branch:
                pred = grad_state.history_map.get(op_ctxt.pred.name)
                branch = op_ctxt.branch
                result = control_flow_ops._SwitchRefOrTensor(result, pred)[1 - branch]
        else:
            if dead_branch:
                pred = op_ctxt.pred
                branch = op_ctxt.branch
                op_ctxt.outer_context.Enter()
                val = control_flow_ops._SwitchRefOrTensor(op.inputs[0], pred)[1 - branch]
                zeros_shape = array_ops.shape_internal(val, optimize=False)
                op_ctxt.outer_context.Exit()
                val.op._set_control_flow_context(op_ctxt)
                zeros_shape.op._set_control_flow_context(op_ctxt)
            else:
                op_ctxt.Enter()
                zeros_shape = array_ops.shape_internal(val, optimize=False)
                op_ctxt.Exit()
            grad_state.grad_context.Exit()
            history_zeros_shape = grad_state.AddForwardAccumulator(zeros_shape, dead_branch=dead_branch)
            grad_state.grad_context.Enter()
            shape = grad_state.AddBackpropAccumulatedValue(history_zeros_shape, zeros_shape, dead_branch)
            result = array_ops.zeros(shape, val.dtype)
        return result

    def PostProcessing(self):
        if False:
            while True:
                i = 10
        "Perform postprocessing at the end of gradients().\n\n    We have created the gradient graph at this point. So this function\n    can be used to perform any postprocessing on the gradient graph.\n    We currently perform the following postprocessing:\n      1. Patch the gradient graph if the output of a loop variable\n         doesn't depend on its input.\n    "
        for (_, grad_state) in self._map.items():
            for (_, b_merge) in grad_state.switch_map.items():
                if b_merge.op.inputs[0] == b_merge.op.inputs[1]:
                    dtype = b_merge.op.inputs[0].dtype
                    shape = b_merge.op.inputs[0].get_shape()
                    if shape.is_fully_defined():
                        grad_state.grad_context.Enter()
                        grad_val = constant_op.constant(0, dtype=dtype, shape=shape)
                        next_grad_val = control_flow_ops._NextIteration(grad_val)
                        grad_state.grad_context.Exit()
                    else:
                        outer_grad_ctxt = grad_state.grad_context.outer_context
                        if outer_grad_ctxt:
                            outer_grad_ctxt.Enter()
                        enter_grad_op = b_merge.op.inputs[0].op
                        enter_grad = enter_grad_op.inputs[0]
                        grad_shape = array_ops.shape_internal(enter_grad, optimize=False)
                        grad_val = array_ops.zeros(grad_shape)
                        if outer_grad_ctxt:
                            outer_grad_ctxt.Exit()
                        grad_state.grad_context.Enter()
                        next_grad_val = control_flow_ops._NextIteration(grad_val)
                        grad_state.grad_context.Exit()
                    b_merge.op._update_input(1, next_grad_val)

def MaybeCreateControlFlowState(between_op_list, between_ops, colocate_gradients_with_ops):
    if False:
        i = 10
        return i + 15
    'Create the state for all the while loops involved in one gradients().\n\n  We create a _ControlFlowState when there are while loops involved in\n  gradients(). In gradients(), control flow logic is only invoked when\n  the _ControlFlowState is not None.\n\n  Note that this method modifies `between_op_list` and `between_ops`.\n  '
    loop_state = None
    for op in between_op_list:
        if util.IsLoopExit(op):
            if loop_state is None:
                loop_state = _ControlFlowState()
            if colocate_gradients_with_ops:
                with ops.colocate_with(op):
                    loop_state.AddWhileContext(op, between_op_list, between_ops)
            else:
                loop_state.AddWhileContext(op, between_op_list, between_ops)
    return loop_state

def _ZerosLikeV1(op, index):
    if False:
        return 10
    'Branch of ZerosLike for TF1.'
    val = op.outputs[index]
    op_ctxt = op._get_control_flow_context()
    if op_ctxt:
        pred = op_ctxt.pred
        branch = op_ctxt.branch
        switch_val = control_flow_ops.switch(op.inputs[0], pred)[1 - branch]
        pivot = array_ops.identity(switch_val)
        if val.dtype == dtypes.resource:
            with ops.control_dependencies([pivot]):
                return array_ops.zeros(gen_resource_variable_ops.variable_shape(switch_val), dtype=default_gradient.get_zeros_dtype(val))
        zeros_shape = array_ops.shape_internal(switch_val, optimize=False)
        with ops.control_dependencies([pivot]):
            return array_ops.zeros(zeros_shape, dtype=val.dtype)
    else:
        return array_ops.zeros_like(val, optimize=False)

def _ZerosLikeV2(op, index):
    if False:
        return 10
    'Branch of ZerosLike for TF2.'
    val = op.outputs[index]
    if val.dtype == dtypes.resource:
        return array_ops.zeros(gen_resource_variable_ops.variable_shape(val), dtype=default_gradient.get_zeros_dtype(val))
    if isinstance(val.op.graph, control_flow_v2_func_graphs.WhileBodyFuncGraph) and val.dtype != dtypes.variant:
        if val.shape.is_fully_defined():
            return constant_op.constant(0, shape=val.shape.dims, dtype=val.dtype)
        else:
            zeros_shape = array_ops.shape_internal(val, optimize=False)
            return array_ops.zeros(zeros_shape, val.dtype)
    else:
        return array_ops.zeros_like(val, optimize=False)

def ZerosLike(op, index):
    if False:
        while True:
            i = 10
    'Create zeros_like for the specified output of an op.'
    if not util.IsSwitch(op):
        return _ZerosLikeV2(op, index)
    else:
        return _ZerosLikeV1(op, index)