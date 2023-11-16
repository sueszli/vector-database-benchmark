from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.var import Var
import logging
import numpy as np
import copy
from collections import defaultdict
DEBUG = False
'\nDescription of the Pass:\n\nThe pass is divided into 3 phases\n\n1st phase: information gathering\n---------------------------------\n\n- We first plug in Identity ops for all output nodes. This allows us to treat all ops uniformly during traversal.\n- Block is traversed in the topological order, starting from the ops connected to the inputs.\n- During the traversal, a value is associated with every var in the block\n- This value can be either of type "HypotheticalValue" or "LazyTransposeHypotheticalValue"\n- Main purpose of type "HypotheticalValue" is to essentially indicate that it is NOT of type  "LazyTransposeHypotheticalValue"\n- "LazyTransposeHypotheticalValue" represents either one or multiple transpose ops with the same perm value. This information\nis stored in this class. It also wraps a "HypotheticalValue" that was the last hypothetical value which was generated\nprior to the origin of "LazyTransposeHypotheticalValue"\n\n- Each op decides which type of hypothetical value to associate with its output vars, based on its op type,\nits attributes, and the types of the hypothetical values of its input vars\n- Ops are classified into 4 categories: unary like, axis update, transpose and materialize (all the rest)\n- Transpose ops: these are the ops from which a "LazyTransposeHypotheticalValue" originate.\n    - If the input to it is a "HypotheticalValue", its output will be a "LazyTransposeHypotheticalValue",\n        indicating that this transpose op is available to get cancelled downstream\n    - If the input to it is a "LazyTransposeHypotheticalValue", then it is checked whether this op cancels it or not\n    - If it cancels it, a "HypotheticalValue" value is generated at the output and the information about this transpose cancellation\n      is recorded in the dictionary "transpose_op_to_cancel_ops"\n    - If it does not cancel, the current transpose op is categrorized as a materialize op and hence the information in\n       dictionary "transpose_op_to_materialize_ops" is updated accordingly. The output of the op is now mapped to a\n       "HypotheticalValue"\n- Unary like ops: they simply transfer their input hypothetical value type to the output.\n- Axis update ops: if a transpose can pass through them, they are treated like a unary op and the dictionary\n   "transpose_op_to_axis_update_ops" is updated. If the op cannot be updated in any manner to\n   allow a transpose to pass through, this op is then categorized as a materialize op and handled accordingly\n- Materialzie ops: All "LazyTransposeHypotheticalValue" input vars, if present, materialize here. Output of this op\n  is always of type "HypotheticalValue". If the input is a "LazyTransposeHypotheticalValue", update the dict\n  "transpose_op_to_materialize_ops"\n- To treat an op like a unary op, add its type to "UNARY_LIKE_OP_TYPES". In future changes we want to make this process\nautomatic, by automatically detecting an op as a unary like by its "traits"\n\n- To treat an op like axis update op, add a class specific to the op implementing the class "transform_axis_update_ops"\nFor examples, see classes "transform_concat", "transform_pad" etc. The dictionary AXIS_UPDATE_OPS is automatically filled\nin by the decorator "register_axis_update_op"\n\n\n2nd phase: making a determination on which transpose ops to remove from the graph\n---------------------------------------------------------------------------------\n- All Transpose ops that have a corresponding compliment op in dict "transpose_op_to_cancel_ops" is a candidate\n- However, we need to make sure of two things\n    - if a transpose op is removed then all its cancel ops in "transpose_op_to_cancel_ops" must be also removed,\n      to ensure correctness of the graph. Same is true in the reverse direction as well,\n      that is, for every cancel op that is removed all its parent transpose ops upstream, must also be removed.\n    - transpose ops should only be removed if the number of cancel ops is greater than the number of transpose ops\n      that would get freshly introduced to the block as a result of materialization ops. Right now in the algorithm\n      each materialization op/output var (dicts "transpose_op_to_materialize_ops"/"old_output_vars"),\n      results in one more transpose op, although this can be further optimized in the future\n\n- To resolve this, we recognize that nodes consisting of sets (a) and (b) form a bipartitle graph, where,\n(a) == starting transpose ops (originators of "LazyTransposeHypotheticalValue")\nand (b) == set of transpose cancel ops and materialize ops\n- in this bipartite graph, we find all the connected components\n- for each connected component, either the whole set of transpose ops in it are removed/materialized or none\nof them are touched\n- thus for each set, a determination is made based on counting the number of cancel ops and materialize ops\n- Based on this the final set of transpose ops to be removed is updated\n\n\n3rd phase: transforming the graph\n----------------------------------\n- Transpose starting ops and the cancel ops are removed\n- Axis update ops, affected by these transpose ops, are updated\n- Transposes are materialized, i.e. added just before the materialize ops, which are linked to the starting transpose ops.\n  The starting transpose op can get materialized (inserted) multiple times, before each of the "materialize ops" downstream.\n- Block outputs are handled similar to the materialize ops\n- Type inference on all ops is invoked after all the transformations\n- All Identity ops that are plugged into the graph to treat outputs as materialized are removed.\n\nDebugging:\n------------\nIf the debug flag is set to True, the block before and after the transformation is plotted,\nwith transpose nodes highlighted\n\n'
UNARY_LIKE_OP_TYPES = set(['relu', 'log', 'relu6', 'abs', 'acos', 'asin', 'atan', 'atanh', 'ceil', 'clip', 'cos', 'cosh', 'erf', 'exp', 'exp2', 'floor', 'identity', 'logical_not', 'round', 'rsqrt', 'sign', 'sin', 'sinh', 'sqrt', 'square', 'tan', 'tanh', 'threshold', 'clamped_relu', 'elu', 'gelu', 'leaky_relu', 'linear_activation', 'scaled_tanh', 'sigmoid', 'sigmoid_hard', 'softplus', 'softplus_parametric', 'softsign', 'thresholded_relu'])
AXIS_UPDATE_OPS = {}

def _do_transposes_cancel(perm1, perm2):
    if False:
        while True:
            i = 10
    if len(perm1) != len(perm2):
        return False
    x = list(range(len(perm1)))
    x1 = [x[i] for i in perm1]
    x2 = [x1[i] for i in perm2]
    if x == x2:
        return True
    return False

def _get_input_vars(op, only_nonconst_vars=False):
    if False:
        print('Hello World!')
    '\n    :return: List[Var]\n    '
    input_vars = []
    for (name, val) in op.inputs.items():
        if isinstance(val, Var):
            if only_nonconst_vars:
                if val.op and val.op.op_type == 'const':
                    continue
            input_vars.append(val)
        elif isinstance(val, (list, tuple)):
            for var in val:
                if not isinstance(var, Var):
                    msg = "transpose optimization pass: unrecognized input type of op='{}', input='{}'"
                    raise ValueError(msg.format(op.name, name))
                if only_nonconst_vars:
                    if var.op and var.op.op_type == 'const':
                        continue
                input_vars.append(var)
        else:
            msg = "transpose optimization pass: unrecognized input type of op='{}', input='{}'"
            raise ValueError(msg.format(op.name, name))
    return input_vars

def register_axis_update_op(cls=None, similar_ops=[]):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param similar_ops: these ops share the same "update" and\n    "can_transpose_pass" methods as the base class.\n    For example: the class "transform_reduce_mean" corresponding to\n    op "reduce_mean" can be shared with other ops such as\n    "reduce_prod", "reduce_sum" etc\n    '

    def class_wrapper(op_update_cls):
        if False:
            for i in range(10):
                print('nop')
        cls_name = op_update_cls.__name__
        op_type = cls_name[len('transform_'):]
        if op_type in AXIS_UPDATE_OPS:
            raise ValueError("Update class for op '{}' already defined".format(op_update_cls))
        AXIS_UPDATE_OPS[op_type] = op_update_cls
        for similar_op_type in similar_ops:
            if similar_op_type in AXIS_UPDATE_OPS:
                raise ValueError("Update class for op of type '{}' already defined".format(op_type))
            AXIS_UPDATE_OPS[similar_op_type] = op_update_cls
        return op_update_cls
    if cls is None:
        return class_wrapper
    return class_wrapper

class transform_axis_update_ops(object):
    """
    Parent class for every axis update op's class

    An axis update op is an op that can be updated, such that it can allow a transpose layer to "pass" through it.
    That is,

    op(transpose(x)) == transpose(op_updated(x))

    where
    "op" : original op,
    "op_updated": op after being updated.

    Example:

    if x is a tensor of rank 2, and transpose has perm=[1,0],
    then

    reduce_mean[axis=1](transpose(x)) == transpose(reduce_mean[axis=0](x))

    here reduce_mean op with axis=1 can be updated to a reduce_mean op with axis=0,
    to allow the transpose to "pass" through it, i.e. get applied after it.

    """

    def __init__(self, op, transpose_axes):
        if False:
            print('Hello World!')
        self.op = op
        self.transpose_axes = transpose_axes

    def can_transpose_pass(self):
        if False:
            while True:
                i = 10
        '\n        Each "axis" op must determine whether it can act like a unary op\n        and allow the transpose to pass through.\n        Return True if it can allow the transpose to pass through, otherwise return False.\n\n        :return: bool\n        '
        raise NotImplementedError('This function must be implemented by each op')

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A method that updates some attribute of the axis op,\n        based on the transpose axes value.\n        This method only gets called if "can_transpose_pass" returns True.\n\n        Update the op such that the output %i2 should be equal to %o2\n\n        Before:\n        %i_1 = transpose_op(%i_0, perm=transpose_axes)\n        %i2 = op(%i1)\n\n        After:\n        %o1 = op_updated(%i0)\n        %o2 = transpose_op(%o1, perm=transpose_axes)\n\n        :return: None\n        '
        raise NotImplementedError('This function must be implemented by each op')

@register_axis_update_op()
class transform_concat(transform_axis_update_ops):

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super(transform_concat, self).__init__(**kwargs)
        self.axis_var = self.op.inputs['axis']

    def can_transpose_pass(self):
        if False:
            return 10
        if self.axis_var.val is not None:
            return True
        return False

    def update(self):
        if False:
            i = 10
            return i + 15
        new_axis_val = self.transpose_axes[self.axis_var.val]
        inputs = list(self.op.inputs['values'])
        transpose_perm_for_const = [0] * len(self.transpose_axes)
        for (i, axis) in enumerate(self.transpose_axes):
            transpose_perm_for_const[axis] = i
        for input_var in inputs:
            if input_var.op.op_type == 'const':
                const_val = input_var.val
                new_const_val = np.transpose(const_val, transpose_perm_for_const)
                with self.op.enclosing_block:
                    new_const_input_var = mb.const(val=new_const_val, mode='immediate_value', before_op=self.op)
                self.op.enclosing_block.replace_uses_of_var_after_op(anchor_op=new_const_input_var.op, end_op=self.op, old_var=input_var, new_var=new_const_input_var, no_check_var_types=True)
        with self.op.enclosing_block:
            new_axis_var = mb.const(val=new_axis_val, mode='immediate_value', before_op=self.op)
        self.op.enclosing_block.replace_uses_of_var_after_op(anchor_op=new_axis_var.op, end_op=self.op, old_var=self.axis_var, new_var=new_axis_var, no_check_var_types=True)

@register_axis_update_op()
class transform_pad(transform_axis_update_ops):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(transform_pad, self).__init__(**kwargs)
        self.pad_var = self.op.inputs['pad']
        self.pad_op = self.pad_var.op
        self.mode = self.op.mode.val
        self.pad_amounts_new = None

    def _compute_new_pad_values(self):
        if False:
            for i in range(10):
                print('nop')
        pad_amounts = np.reshape(self.pad_var.val, [-1, 2])
        rank_diff = len(self.transpose_axes) - pad_amounts.shape[0]
        self.pad_amounts_new = copy.deepcopy(pad_amounts)
        self.pad_amounts_new = np.concatenate((np.zeros(2 * rank_diff).reshape(-1, 2), self.pad_amounts_new))
        self.pad_amounts_new = self.pad_amounts_new.astype(pad_amounts.dtype)
        pad_amounts = np.concatenate((np.zeros(2 * rank_diff).reshape(-1, 2), pad_amounts))
        for (i, axis) in enumerate(self.transpose_axes):
            self.pad_amounts_new[axis][0] = pad_amounts[i][0]
            self.pad_amounts_new[axis][1] = pad_amounts[i][1]
        top_rows = self.pad_amounts_new[:rank_diff, :]
        if not np.all(top_rows == 0):
            return False
        self.pad_amounts_new = self.pad_amounts_new[rank_diff:, :]
        self.pad_amounts_new = self.pad_amounts_new.flatten()
        return True

    def can_transpose_pass(self):
        if False:
            print('Hello World!')
        if len(_get_input_vars(self.op, only_nonconst_vars=True)) != 1 or self.pad_op.op_type != 'const':
            return False
        if len(self.transpose_axes) < 2:
            return False
        if not self._compute_new_pad_values():
            return False
        if self.mode != 'constant' and (not np.all(self.pad_amounts_new[:-4] == 0)):
            return False
        return True

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        self._compute_new_pad_values()
        with self.op.enclosing_block:
            new_pad_var = mb.const(val=self.pad_amounts_new, mode='immediate_value', before_op=self.op)
        self.op.enclosing_block.replace_uses_of_var_after_op(anchor_op=new_pad_var.op, end_op=self.op, old_var=self.pad_var, new_var=new_pad_var, no_check_var_types=True)

@register_axis_update_op(similar_ops=['reduce_l1_norm', 'reduce_l2_norm', 'reduce_max', 'reduce_log_sum', 'reduce_log_sum_exp', 'reduce_min', 'reduce_prod', 'reduce_sum', 'reduce_sum_square'])
class transform_reduce_mean(transform_axis_update_ops):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(transform_reduce_mean, self).__init__(**kwargs)
        self.axes_var = self.op.inputs['axes']
        self.axes_op = self.axes_var.op

    def can_transpose_pass(self):
        if False:
            print('Hello World!')
        if self.op.inputs['keep_dims'].val:
            if self.axes_op.op_type == 'const':
                return True
        return False

    def update(self):
        if False:
            return 10
        old_axes_val = self.axes_var.val
        new_axes_val = [0] * len(old_axes_val)
        for (i, axis) in enumerate(old_axes_val):
            new_axes_val[i] = self.transpose_axes[axis]
        with self.op.enclosing_block:
            new_axis_var = mb.const(val=new_axes_val, mode='immediate_value', before_op=self.op)
        self.op.enclosing_block.replace_uses_of_var_after_op(anchor_op=new_axis_var.op, end_op=self.op, old_var=self.axes_var, new_var=new_axis_var, no_check_var_types=True)

@register_axis_update_op(similar_ops=['mul', 'sub', 'real_div', 'maximum', 'minimum'])
class transform_add(transform_axis_update_ops):

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(transform_add, self).__init__(**kwargs)

    def can_transpose_pass(self):
        if False:
            i = 10
            return i + 15
        const_input = None
        if self.op.inputs['x'].op and self.op.inputs['x'].op.op_type == 'const':
            const_input = self.op.inputs['x']
            other_input = self.op.inputs['y']
        if self.op.inputs['y'].op and self.op.inputs['y'].op.op_type == 'const':
            if const_input is not None:
                return False
            const_input = self.op.inputs['y']
            other_input = self.op.inputs['x']
        if const_input is None:
            return True
        if not isinstance(const_input.val, (np.ndarray, np.generic)):
            return False
        rank_const_input = len(const_input.val.shape)
        rank_other_input = len(other_input.shape) if other_input.shape else 0
        if rank_const_input <= 1 and rank_other_input > 0:
            return True
        return False

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        if len(_get_input_vars(self.op, only_nonconst_vars=True)) == 2:
            return
        for input_var in _get_input_vars(self.op):
            if input_var.op and input_var.op.op_type == 'const':
                const_input_var = input_var
                break
        const_value = const_input_var.val
        if len(const_value.shape) == 0:
            return
        rank = len(self.transpose_axes)
        new_shape = [1] * rank
        new_shape[self.transpose_axes[-1]] = const_value.shape[0]
        new_const_val = np.reshape(const_value, new_shape)
        with self.op.enclosing_block:
            new_const_var = mb.const(val=new_const_val, mode=const_input_var.op.mode, before_op=self.op)
        self.op.enclosing_block.replace_uses_of_var_after_op(anchor_op=new_const_var.op, end_op=self.op, old_var=const_input_var, new_var=new_const_var, no_check_var_types=True)

class HypotheticalValue(object):

    def __init__(self, var=None):
        if False:
            print('Hello World!')
        self.value = var

class LazyTransposeHypotheticalValue(object):

    def __init__(self, hypothetical_value, transpose_ops, perm):
        if False:
            return 10
        self.wrapped_hypothetical_value = hypothetical_value
        if not isinstance(hypothetical_value, HypotheticalValue):
            raise ValueError('transpose optimization pass: incorrect type passed for hypothetical_value')
        for op in transpose_ops:
            if op.op_type != 'transpose':
                raise ValueError('transpose optimization pass: LazyTransposeHypotheticalValue can only be made with transpose ops')
            perm_op = list(op.inputs['perm'].val)
            if perm_op != perm:
                raise ValueError("transpose optimization pass: LazyTransposeHypotheticalValue can only be made with transpose ops with the same 'perm' values")
        self.perm = perm
        self.transpose_ops = transpose_ops

class TransposeOptimization(object):

    def __init__(self, block):
        if False:
            i = 10
            return i + 15
        self.block = block
        self.var_to_hypothetical_value = {}
        for (_, input_var) in block.inputs.items():
            self.var_to_hypothetical_value[input_var] = HypotheticalValue(input_var)
        self.transpose_op_to_cancel_ops = defaultdict(lambda : [])
        self.transpose_op_to_materialize_ops = defaultdict(lambda : [])
        self.transpose_op_to_axis_update_ops = defaultdict(lambda : [])
        self.ops_updated = set()
        self.materialized_ops_handled = set()
        self.transpose_ops_removed = set()
        self.old_output_vars = []
        self.output_sink_ops = []
        self._add_output_sinks()

    def _add_output_sinks(self):
        if False:
            print('Hello World!')
        self.old_output_vars = {var: var.name for var in self.block.outputs}
        new_outputs = []
        output_sinks_var = {}
        for out_var in self.block.outputs:
            with self.block:
                if out_var not in output_sinks_var:
                    out_sink = mb.identity(x=out_var)
                    output_sinks_var[out_var] = out_sink
                else:
                    out_sink = output_sinks_var[out_var]
                new_outputs.append(out_sink)
                self.output_sink_ops.append(out_sink.op)
        self.block.set_outputs(new_outputs)

    def _visit_unary_like_op(self, op, input_var=None):
        if False:
            print('Hello World!')
        if input_var is None:
            input_var = op.inputs['x']
        if len(op.outputs) > 1:
            msg = "transpose optimization pass: op '{}', of type = '{}', has multiple outputs, hence itcannot be handled like a unary op"
            raise ValueError(msg.format(op.name, op.op_type))
        self.var_to_hypothetical_value[op.outputs[0]] = self.var_to_hypothetical_value[input_var]

    def _visit_materialize_op(self, op):
        if False:
            return 10
        for out_var in op.outputs:
            self.var_to_hypothetical_value[out_var] = HypotheticalValue(out_var)
        for input_var in _get_input_vars(op):
            input_hypothetical_value = self.var_to_hypothetical_value[input_var]
            if isinstance(input_hypothetical_value, LazyTransposeHypotheticalValue):
                all_lazy_transpose_ops = input_hypothetical_value.transpose_ops
                for transpose_op in all_lazy_transpose_ops:
                    self.transpose_op_to_materialize_ops[transpose_op].append((op, input_var))

    def _visit_axis_update_op(self, op):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that all non constant inputs are of type LazyTransposeHypotheticalValue with the same perm value\n        This check is common for all "axis update" ops.\n        '
        input_vars = _get_input_vars(op, only_nonconst_vars=True)
        perm = None
        for (i, var) in enumerate(input_vars):
            hypothetical_value = self.var_to_hypothetical_value[var]
            if not isinstance(hypothetical_value, LazyTransposeHypotheticalValue):
                self._visit_materialize_op(op)
                return
            if i == 0:
                perm = hypothetical_value.perm
            elif perm != hypothetical_value.perm:
                self._visit_materialize_op(op)
                return
        op_cls = AXIS_UPDATE_OPS.get(op.op_type, None)
        if op_cls is None:
            raise ValueError("Transform class for op of type '{}' not found".format(op.op_type))
        if not op_cls(**{'op': op, 'transpose_axes': perm}).can_transpose_pass():
            self._visit_materialize_op(op)
            return
        all_lazy_transpose_ops = set()
        for var in input_vars:
            input_hypothetical_value = self.var_to_hypothetical_value[var]
            all_lazy_transpose_ops.update(input_hypothetical_value.transpose_ops)
        for transpose_op in all_lazy_transpose_ops:
            self.transpose_op_to_axis_update_ops[transpose_op].append(op)
        self.var_to_hypothetical_value[op.outputs[0]] = LazyTransposeHypotheticalValue(input_hypothetical_value.wrapped_hypothetical_value, all_lazy_transpose_ops, perm)

    def _visit_transpose_op(self, op):
        if False:
            return 10
        input_var = op.inputs['x']
        if op.inputs['perm'].val is None:
            self._visit_materialize_op(op)
            return
        perm = list(op.inputs['perm'].val)
        input_hypothetical_value = self.var_to_hypothetical_value[input_var]
        '\n        There are 3 cases to handle:\n\n        1. input type == HypotheticalValue\n        2. input type == LazyTransposeHypotheticalValue and this op is the transpose compliment of it\n        3. input type == LazyTransposeHypotheticalValue and this op is NOT the transpose compliment of it\n        '
        if isinstance(input_hypothetical_value, HypotheticalValue):
            if op.outputs[0] in self.old_output_vars:
                self._visit_materialize_op(op)
            else:
                self.var_to_hypothetical_value[op.outputs[0]] = LazyTransposeHypotheticalValue(input_hypothetical_value, set([op]), perm)
            return
        do_cancel = _do_transposes_cancel(input_hypothetical_value.perm, perm)
        if do_cancel:
            self.var_to_hypothetical_value[op.outputs[0]] = input_hypothetical_value.wrapped_hypothetical_value
            all_lazy_transpose_ops = input_hypothetical_value.transpose_ops
            for transpose_op in all_lazy_transpose_ops:
                self.transpose_op_to_cancel_ops[transpose_op].append(op)
        else:
            self._visit_materialize_op(op)

    def _visit_op(self, op):
        if False:
            return 10
        input_vars = _get_input_vars(op)
        for var in input_vars:
            assert var in self.var_to_hypothetical_value, "transpose optimization pass: hypothetical value for var '{}', not found".format(var.name)
        if op in self.output_sink_ops:
            self._visit_materialize_op(op)
        elif op.op_type in UNARY_LIKE_OP_TYPES:
            self._visit_unary_like_op(op)
        elif op.op_type in AXIS_UPDATE_OPS:
            self._visit_axis_update_op(op)
        elif op.op_type == 'transpose':
            self._visit_transpose_op(op)
        elif op.op_type == 'const':
            self.var_to_hypothetical_value[op.outputs[0]] = HypotheticalValue(op.outputs[0])
        else:
            self._visit_materialize_op(op)

    def block_traversal(self):
        if False:
            return 10
        for op in self.block.operations:
            self._visit_op(op)

    def _verify_cancellable_transposes(self):
        if False:
            for i in range(10):
                print('nop')
        transpose_cancel_ops_to_starting_transpose_set = defaultdict(lambda : set())
        for (op, cancel_ops_list) in self.transpose_op_to_cancel_ops.items():
            for cancel_op in cancel_ops_list:
                transpose_cancel_ops_to_starting_transpose_set[cancel_op].update(set([op]))
        for op in transpose_cancel_ops_to_starting_transpose_set:
            assert op not in self.transpose_op_to_cancel_ops, "transpose reduction optimization: transpose op '{}' cannot be both a starting and cancel op".format(op.name)
        materizalize_ops_to_starting_transpose_set = defaultdict(lambda : set())
        for (op, materialize_ops) in self.transpose_op_to_materialize_ops.items():
            for (materialize_op, edge) in materialize_ops:
                materizalize_ops_to_starting_transpose_set[materialize_op].update(set([op]))
                if op not in self.transpose_op_to_cancel_ops:
                    self.transpose_op_to_cancel_ops[op] = []
        connected_components = []
        visited = {}
        for op in list(self.transpose_op_to_cancel_ops.keys()):
            if op in visited:
                continue
            visited[op] = 1
            set_a = set([op])
            set_b1 = set()
            set_b2 = set()
            queue = []
            queue.extend(self.transpose_op_to_cancel_ops[op])
            if op in self.transpose_op_to_materialize_ops:
                materialize_ops_list = list(list(zip(*self.transpose_op_to_materialize_ops[op]))[0])
                queue.extend(materialize_ops_list)
            while len(queue) > 0:
                o = queue.pop(0)
                visited[o] = 1
                if o in self.transpose_op_to_cancel_ops:
                    set_a.update(set([o]))
                    for neighbor_op in self.transpose_op_to_cancel_ops[o]:
                        if neighbor_op not in visited:
                            queue.append(neighbor_op)
                    if o in self.transpose_op_to_materialize_ops:
                        materialize_ops_list = list(list(zip(*self.transpose_op_to_materialize_ops[o]))[0])
                        for neighbor_op in materialize_ops_list:
                            if neighbor_op not in visited:
                                queue.append(neighbor_op)
                elif o in transpose_cancel_ops_to_starting_transpose_set:
                    set_b1.update(set([o]))
                    for neighbor_op in transpose_cancel_ops_to_starting_transpose_set[o]:
                        if neighbor_op not in visited:
                            queue.append(neighbor_op)
                else:
                    set_b2.update(set([o]))
                    for neighbor_op in materizalize_ops_to_starting_transpose_set[o]:
                        if neighbor_op not in visited:
                            queue.append(neighbor_op)
            connected_components.append((set_a, set_b1, set_b2))
        starting_ops_to_remove = set()
        for (op_set, op_cancel_set, materialize_op_set) in connected_components:
            block_output = False
            for op in op_set:
                if op.outputs[0] in self.block.outputs:
                    starting_ops_to_remove.update(op_set)
                    block_output = True
                    break
            if block_output:
                continue
            materizalize_set = set(list(materialize_op_set))
            if len(materizalize_set) >= len(op_set) + len(op_cancel_set):
                starting_ops_to_remove.update(op_set)
        for op in starting_ops_to_remove:
            self.transpose_op_to_cancel_ops.pop(op, None)

    def _remove_transpose_ops(self, starting_transpose_op):
        if False:
            while True:
                i = 10
        perm = list(starting_transpose_op.inputs['perm'].val)
        starting_transpose_op_out_var = starting_transpose_op.outputs[0]
        starting_transpose_op_input_var = starting_transpose_op.inputs['x']
        for op in self.transpose_op_to_axis_update_ops.get(starting_transpose_op, []):
            if op not in self.ops_updated:
                op_cls = AXIS_UPDATE_OPS.get(op.op_type, None)
                op_cls(**{'op': op, 'transpose_axes': perm}).update()
                self.ops_updated.add(op)
        to_be_removed_ops = []
        name_changed_vars = set()
        for op in [starting_transpose_op] + self.transpose_op_to_cancel_ops[starting_transpose_op]:
            if op in self.transpose_ops_removed:
                continue
            to_be_removed_ops.append(op)
            self.transpose_ops_removed.add(op)
            input_var = op.inputs['x']
            output_var = op.outputs[0]
            parent_op = input_var.op
            if output_var in self.old_output_vars:
                if input_var not in name_changed_vars:
                    input_var.name = output_var.name
                    input_var.op.name = output_var.op.name
                    name_changed_vars.update([input_var])
                else:
                    with self.block:
                        input_var = mb.identity(x=input_var, before_op=op, name=output_var.name)
                        parent_op = input_var.op
            self.block.replace_uses_of_var_after_op(anchor_op=parent_op, old_var=output_var, new_var=input_var, no_check_var_types=True)
        '\n        Insert a transpose op JUST before each one of the materialize ops\n        i.e.\n        Given:  %i1 = op(...)\n                ...\n                ... = materialize_op(..., %i1 ,...)\n                ...\n\n        Result: %i1 = op(...)\n                ...\n                %i2 = transpose_op(%i1, %perm)\n                ... = materialize_op(..., %i2 ,...)\n                ...\n        '
        for (op, input_var) in self.transpose_op_to_materialize_ops.get(starting_transpose_op, []):
            if (op, input_var) in self.materialized_ops_handled:
                continue
            self.materialized_ops_handled.add((op, input_var))
            with self.block:
                if input_var == starting_transpose_op_out_var:
                    if op in self.output_sink_ops:
                        continue
                    i1 = starting_transpose_op_input_var
                else:
                    i1 = input_var
                if op in self.output_sink_ops:
                    if i1 not in name_changed_vars:
                        x = mb.transpose(x=i1, perm=perm, before_op=op, name=i1.name)
                        i1.name = '_before_transpose_op_' + x.op.name
                        i1.op.name = '_before_transpose_op_' + x.op.name
                    else:
                        x = mb.transpose(x=i1, perm=perm, before_op=op, name=self.old_output_vars[i1])
                else:
                    x = mb.transpose(x=i1, perm=perm, before_op=op)
            self.block.replace_uses_of_var_after_op(anchor_op=x.op, end_op=op, old_var=i1, new_var=x, no_check_var_types=True)
        self.block.remove_ops(to_be_removed_ops)

    def apply_transform(self):
        if False:
            while True:
                i = 10
        '\n        Take in the data collected during graph traversal\n        and transform the graph by cancelling out transpose ops that can be removed.\n        '
        logging.debug('Block before optimize transpose transform:\n{}'.format(self.block))
        if DEBUG:
            import graphviz
            graphviz.Source(self.block.get_dot_string(highlight_debug_op_names=[], highlight_debug_op_types=['transpose'])).view(filename='/tmp/block_before_reduce_transpose')
        '\n        First check which transposes can be cancelled.\n        After this function call we get an updated dictionary "transpose_op_to_cancel_ops"\n        with only the transpose ops that can really be cancelled in the graph\n        Reasons to not cancel:\n        - materialize_ops are greater than cancel_ops, so removing transpose will instead end up increasing the count of transposes\n        - removing a transpose op can only be successful, if all of its cancel ops are removed, removing all the cancel ops\n          is only successful if all of their starting transpose ops are removed and so on. This check is also done in\n           "_verify_cancellable_transposes()"\n        '
        self._verify_cancellable_transposes()
        for transpose_op in self.transpose_op_to_cancel_ops:
            self._remove_transpose_ops(transpose_op)
        self.block.set_outputs([sink_op.x for sink_op in self.output_sink_ops])
        self.block.remove_ops(list(self.output_sink_ops))
        if DEBUG:
            graphviz.Source(self.block.get_dot_string(highlight_debug_op_names=[], highlight_debug_op_types=['transpose'])).view(filename='/tmp/block_after_reduce_transpose')
        logging.debug('Block after optimize transpose transform:\n{}'.format(self.block))
        for op in self.block.operations:
            op.type_value_inference(overwrite_output=True)

def reduce_transposes_block(block):
    if False:
        print('Hello World!')
    '\n    Only apply the optimization if the block is flat,\n    i.e, it does not contain any op which contains a sub-block.\n    TODO:\n    Removing transposes and transpose compliments requires re-running\n    type inference for the set of ops in between the fused transpose ops,\n    which is simpler to do when all the ops in the block are free of sub blocks.\n    The case of transpose fusion with sub-block containing ops needs to be handled with more care and test cases.\n    '
    for op in list(block.operations):
        if len(op.blocks) > 0:
            return
    opt_transposes = TransposeOptimization(block)
    opt_transposes.block_traversal()
    opt_transposes.apply_transform()

@register_pass(namespace='common')
def reduce_transposes(prog):
    if False:
        print('Hello World!')
    for (f_name, f) in prog.functions.items():
        reduce_transposes_block(f)