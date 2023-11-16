"""Utility functions for control flow.

This file is copied from tensorflow/python/ops/control_flow_util.py.
"""
from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import variables

def InXlaContext(graph):
    if False:
        while True:
            i = 10
    ctxt = graph._get_control_flow_context()
    return GetContainingXLAContext(ctxt) is not None

def GraphOrParentsInXlaContext(graph):
    if False:
        print('Hello World!')
    while True:
        if InXlaContext(graph):
            return True
        try:
            graph = graph.outer_graph
        except AttributeError:
            return False

def IsInWhileLoop(op):
    if False:
        for i in range(10):
            print('nop')
    ctxt = op._get_control_flow_context()
    return GetContainingWhileContext(ctxt) is not None

def GetContainingWhileContext(ctxt, stop_ctxt=None):
    if False:
        i = 10
        return i + 15
    'Returns the first ancestor WhileContext of `ctxt`.\n\n  Returns `ctxt` if `ctxt` is a WhileContext, or None if `ctxt` is not in a\n  while loop.\n\n  Args:\n    ctxt: ControlFlowContext\n    stop_ctxt: ControlFlowContext, optional. If provided, the search will end\n      if it sees stop_ctxt.\n\n  Returns:\n    `ctxt` if `ctxt` is a WhileContext, the most nested WhileContext containing\n    `ctxt`, or None if `ctxt` is not in a while loop.  If `stop_ctxt` is not\n    `None`, this returns `ctxt` if it matches `stop_ctxt` in its traversal.\n  '
    while ctxt:
        if ctxt.IsWhileContext() or ctxt == stop_ctxt:
            return ctxt
        ctxt = ctxt.outer_context
    return None

def GetContainingXLAContext(ctxt):
    if False:
        i = 10
        return i + 15
    'Returns the first ancestor XLAContext of `ctxt`.\n\n  Returns `ctxt` if `ctxt` is a XLAContext, or None if `ctxt` is not in a\n  while loop.\n\n  Args:\n    ctxt: ControlFlowContext\n\n  Returns:\n    `ctxt` if `ctxt` is a XLAContext, the most nested XLAContext containing\n    `ctxt`, or None if `ctxt` is not in a while loop.\n  '
    while ctxt:
        if ctxt.IsXLAContext():
            return ctxt
        ctxt = ctxt.outer_context
    return None

def smart_cond(pred, true_fn=None, false_fn=None, name=None):
    if False:
        print('Hello World!')
    'Return either `true_fn()` if predicate `pred` is true else `false_fn()`.\n\n  If `pred` is a bool or has a constant value, we return either `true_fn()`\n  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.\n\n  Args:\n    pred: A scalar determining whether to return the result of `true_fn` or\n      `false_fn`.\n    true_fn: The callable to be performed if pred is true.\n    false_fn: The callable to be performed if pred is false.\n    name: Optional name prefix when using `tf.cond`.\n\n  Returns:\n    Tensors returned by the call to either `true_fn` or `false_fn`.\n\n  Raises:\n    TypeError: If `true_fn` or `false_fn` is not callable.\n  '
    if isinstance(pred, variables.Variable):
        return cond.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
    return smart_module.smart_cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)

def constant_value(pred):
    if False:
        for i in range(10):
            print('nop')
    'Return the bool value for `pred`, or None if `pred` had a dynamic value.\n\n  Args:\n    pred: A scalar, either a Python bool or a TensorFlow boolean variable\n      or tensor, or the Python integer 1 or 0.\n\n  Returns:\n    True or False if `pred` has a constant boolean value, None otherwise.\n\n  Raises:\n    TypeError: If `pred` is not a Variable, Tensor or bool, or Python\n      integer 1 or 0.\n  '
    if isinstance(pred, tensor.Tensor):
        return tensor_util.constant_value(pred)
    if pred in {0, 1}:
        return bool(pred)
    if isinstance(pred, bool):
        return pred
    if isinstance(pred, variables.Variable):
        return None
    raise TypeError('`pred` must be a Tensor, or a Python bool, or 1 or 0. Found instead: %s' % type(pred))