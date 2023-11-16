import os
import shutil
import sys
import traceback
import six
import chainer

def _reduce(grad_list):
    if False:
        return 10
    if not grad_list:
        return None
    if len(grad_list) >= 2:
        grad_list[:] = [chainer.functions.add(*grad_list)]
    return grad_list[0]

def _pure(grad):
    if False:
        while True:
            i = 10
    return [] if grad is None else [grad]

def _pop_or_none(grad_list):
    if False:
        for i in range(10):
            print('nop')
    return grad_list.pop() if grad_list else None

def _grad_var_from_alive_node(node):
    if False:
        for i in range(10):
            print('nop')
    var = node.get_variable_or_none()
    if var is None:
        return None
    else:
        gv = var.grad_var
        var.grad_var = None
        return gv

class GradTable(object):
    """Dict of nodes to references of gradients

    The gradients are stored as references to them in the backprop process. The
    current implementation uses lists. Keep the lengths of lists <= 1 for the
    strict accumulation of gradients. Leave them to accumulate gradients
    lazily.

    Args:
        accumulate_grad_inputs (bool): Fallback to grad_var of input variables.
            However, the current implementation reproduces the legacy behavior,
            i.e. to read ``grad_var`` of node when the node has not been added.

    """

    def __init__(self, accumulate_grad_inputs=False):
        if False:
            for i in range(10):
                print('nop')
        self.grads = {}
        self._load_if_new = accumulate_grad_inputs

    def __setitem__(self, node, grad):
        if False:
            return 10
        assert node is not None
        self.grads[node] = _pure(grad)

    def accumulate(self, node, grad):
        if False:
            i = 10
            return i + 15
        self.get_as_list(node).append(grad)

    def get_as_list(self, node):
        if False:
            print('Hello World!')
        assert node is not None
        grads = self.grads
        if node not in grads:
            if self._load_if_new and node.creator_node is None:
                node._check_old_style_gradient()
                grads[node] = _pure(_grad_var_from_alive_node(node))
            else:
                grads[node] = []
        return grads[node]

    def pop(self, node):
        if False:
            while True:
                i = 10
        if node is None:
            return None
        grads = self.grads
        if node in grads:
            return _reduce(grads.pop(node))
        if self._load_if_new:
            return _grad_var_from_alive_node(node)
        else:
            return None

    def assert_no_grads(self):
        if False:
            for i in range(10):
                print('nop')
        for gx in self.grads.values():
            assert gx == []

def backprop_step(func, target_input_indexes, grad_outputs, grad_inputs, is_debug):
    if False:
        while True:
            i = 10
    'Accumulates gradients of a FunctionNode\n\n    This routine is used by :meth:`chainer.Variable.backward` and\n    :func:`chainer.grad`.\n\n    Args:\n        func (~chainer.FunctionNode): The function for which gradients are\n            accumulated.\n        target_input_indexes (tuple of int): Sorted indices of the inputs\n            that require gradients. It is guaranteed that this tuple contains\n            at least one element.\n        grad_outputs (tuple of Variable): Gradients w.r.t. the output\n            variables. If the gradient w.r.t. an output variable is not\n            given, the corresponding element is ``None``.\n        grad_inputs (dict): References of the gradients w.r.t. the input\n            variables.\n        is_debug (bool): ``True`` if the debug mode is enabled.\n\n    '
    if is_debug:
        assert isinstance(target_input_indexes, tuple)
        assert target_input_indexes == tuple(sorted(target_input_indexes))
        assert isinstance(grad_outputs, tuple)
    if func.backward_accumulate.__code__ is not chainer.FunctionNode.backward_accumulate.__code__:
        grad_inputs_tuple = tuple([_pop_or_none(grad_inputs[func.inputs[i]]) for i in target_input_indexes])
        try:
            gxs = func.backward_accumulate(target_input_indexes, grad_outputs, grad_inputs_tuple)
        except Exception as e:
            _reraise_with_stack(func, e)
    else:
        try:
            gxs = func.backward(target_input_indexes, grad_outputs)
        except Exception as e:
            _reraise_with_stack(func, e)
        if is_debug:
            for gx in gxs:
                if not (gx is None or isinstance(gx, chainer.Variable)):
                    raise ValueError(func._get_error_message('type of gradients returned from backward is incorrect: {} != expected {}'.format(type(gx), chainer.Variable)))
        len_gxs = len(gxs)
        if len_gxs == len(func.inputs):
            gxs = tuple([gxs[i] for i in target_input_indexes])
        elif len_gxs != len(target_input_indexes):
            msg = 'number of gradients returned from backward is incorrect: '
            if len(func.inputs) == len(target_input_indexes):
                msg += '%s != expected %s' % (len_gxs, len(func.inputs))
            else:
                msg += '%s != expected %s or %s' % (len_gxs, len(func.inputs), len(target_input_indexes))
            raise ValueError(func._get_error_message(msg))
    for (i, gx) in six.moves.zip(target_input_indexes, gxs):
        if gx is None or gx.raw_array is None:
            continue
        grad_inputs[func.inputs[i]].append(gx)
        if is_debug:
            node_x = func.inputs[i]
            g_input_list = grad_inputs[node_x]
            if gx.shape != node_x.shape:
                raise ValueError(func._get_error_message('shape of gradients returned from backward is incorrect: input-index={}, actual {} != expected {}'.format(i, gx.shape, node_x.shape)))
            if gx is not None and g_input_list:
                g_input = g_input_list[0]
                if gx.shape != g_input.shape:
                    raise ValueError(func._get_error_message('shape of gradients returned from backward is incorrect: input-index={}, actual {} != expected {}'.format(i, gx.shape, g_input.shape)))
                if gx.dtype != g_input.dtype:
                    raise ValueError(func._get_error_message('dtype of gradients returned from backward is incorrect: input-index={}, actual {} != expected {}'.format(i, gx.dtype, g_input.dtype)))
    del gxs
    if is_debug:

        def iter_gxs(gxs):
            if False:
                for i in range(10):
                    print('nop')
            for gx in gxs:
                for gx_elem in gx:
                    yield gx_elem
        for gx in iter_gxs(grad_inputs.values()):
            if chainer.backend._contains_nan(gx.data):
                raise RuntimeError('NaN is detected on backward computation of {}'.format(func.label))
    if not func.lazy_grad_sum:
        for gx in grad_inputs.values():
            _reduce(gx)

def _get_columns():
    if False:
        return 10
    if sys.version_info >= (3, 3):
        (cols, rows) = shutil.get_terminal_size()
        return cols
    return int(os.getenv('COLUMNS', 80))

def _reraise_with_stack(func, e):
    if False:
        i = 10
        return i + 15
    if func.stack is not None:
        additional_message = '\n{}\nStacktrace of the function is below:\n{}'.format('-' * _get_columns(), ''.join(traceback.format_list(func.stack[:-1])))
        if e.args:
            e.args = (e.args[0] + additional_message,) + e.args[1:]
        else:
            e.args = (additional_message,)
    raise