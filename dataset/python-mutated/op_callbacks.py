"""Unified callbacks op execution and creation under eager and graph modes."""
from tensorflow.python.eager import context
from tensorflow.python.eager import execute

def add_op_callback(callback_fn):
    if False:
        return 10
    'Add a thread-local callback that intercepts op execution and op creation.\n\n  The `callback_fn` will be invoked immediately after any of the three types\n  of events:\n    - The execution of an TensorFlow operation ("op" for short hereafter)\n      under eager mode,\n    - The execution of a FuncGraph under eager mode,\n    - The creation of an op during graph construction (e.g., in\n      @tf.function-decorated Python functions).\n\n  Known limitations:\n    1. Under graph mode, overriding the output tensors of control-flow ops,\n       including "If", "StatelessIf" and "While", may cause errors\n       (b/139668453). Overriding other tensors in a graph consisting of such\n       control-flow ops is okay.\n    2. Under eager mode, calling eager ops from the callback function itself\n       may lead to recursion stack overflow. This can be prevented by\n       returning from the callback function immediately on encountering the\n       op type involved (b/140334369).\n\n  Args:\n    callback_fn: A callback_fn that has the following signature:\n      def callback_fn(op_type,\n                      inputs,\n                      attrs,\n                      outputs,\n                      op_name=None,\n                      graph=None):\n        # op_type: The type of the op, as a string. E.g., "MatMul".\n        #          For the special case of FuncGraph execution, op_type\n        #          takes the name of the graph name, e.g.,\n        #          "__inference_my_func_24".\n        # inputs: (`tuple` of `Tensor`s) Input tensors to the op or the\n        #         FuncGraph.\n        #         - In eager execution, these are `EagerTensor`s.\n        #         - In graph construction, these are non-eager `Tensor`s\n        #           that form the inputs to the just-created op.\n        # attrs: The attributes of the op or FuncGraph of which the execution\n        #        or creation caused the current invocation of the callback.\n        #        This is applicable to both eager- and graph-based execution,\n        #        as well as graph construction.\n        #        This is a tuple of alternating attribute keys and attribute\n        #        values. E.g., `(\'adjoint_a\', False, \'adjoint_b\', False)`.\n        # outputs: (`tuple of `Tensor`s) Output tensors from the op or\n        #          FuncGraph.\n        #          In eager execution, these are `EagerTensor`s.\n        #          In graph construction, these are non-eager `Tensor`s that\n        #          are the outputs of the just-created op.\n        # op_name: Name of the op.\n        #          - If the current invocation of the callback is due to the\n        #            eager execution of an op or FuncGraph, this will be\n        #            `None`, as op names are meaningless in eager execution.\n        #          - In graph construction, this is the name of the op, e.g.,\n        #            "MatMul_2".\n        # graph: The graph that the op belongs to (if any).\n        #        - In eager execution of an op or FuncGraph, this is `None`.\n        #        - In graph construction, this is the op\'s enclosing graph\n        #          as a `tf.Graph` object.\n        #\n        # Return values:\n        #   This callback function is expected to return `None` or\n        #   a `list` or `tuple` of `Tensor`s with its length matching\n        #   `len(outputs)`, in the order that corresponds to that of the\n        #   `outputs` argument.\n        #   If the return value is `None`, downstream execution or graph\n        #   construction will be unaffected.\n        #   However, if the return value is a `list` or `tuple` of `Tensor`s,\n        #   - In eager execution, these returned `Tensor`s should be\n        #     `EagerTensor`s. Their values will replace the original values of\n        #     `outputs` for downstream eager execution. (*Not implemented yet*).\n        #   - In graph construction, these returned `Tensor`s should be\n        #     non-eager `Tensor`s. Their values will replace the original\n        #     `outputs` for downstream graph construction.\n\n  Raises:\n    ValueEror: If `callback_fn` is `None` or not callable.\n  '
    if callback_fn is None:
        raise ValueError('Passed callback function cannot be None.')
    if not callable(callback_fn):
        raise ValueError(f'Callback function passed to op_callback() is expected to be callable, but got {callback_fn} of type {type(callback_fn)}.')
    ctx = context.context()
    ctx.add_op_callback(callback_fn)
    if ctx.executing_eagerly():
        execute.execute = execute.execute_with_callbacks

def should_invoke_op_callbacks():
    if False:
        i = 10
        return i + 15
    'Determine if op callbacks are present and should be invoked.\n\n  Returns:\n    A thread-local result (boolean) indicating whether any op callback(s) exist\n    and should be invoked.\n  '
    ctx = context.context()
    return ctx.op_callbacks and (not ctx.invoking_op_callbacks)

def remove_op_callback(op_callback):
    if False:
        for i in range(10):
            print('nop')
    'Remove an already-added op callback.\n\n  Args:\n    op_callback: The op callback to be removed.\n\n  Raises:\n    KeyError: If `op_callback` has not been registered using `add_op_callback()`\n      before.\n  '
    ctx = context.context()
    ctx.remove_op_callback(op_callback)
    if ctx.executing_eagerly() and (not ctx.op_callbacks):
        execute.execute = execute.quick_execute

def clear_op_callbacks():
    if False:
        while True:
            i = 10
    'Clear all op callbacks registered in the current thread.'
    for callback in context.context().op_callbacks:
        remove_op_callback(callback)

def invoke_op_callbacks(op_type, inputs, attrs, outputs, op_name=None, graph=None):
    if False:
        for i in range(10):
            print('nop')
    'Invoke the callbacks that exist in the current scope (if any).\n\n  If no callbacks are present in the current scope, this method returns\n  immediately.\n\n  Args:\n    op_type: Type of the operation (e.g., "MatMul").\n    inputs: Input tensors to the op. These are `EagerTensor`s in the case of\n      eager execution of ops or `FuncGraph`s, and are non-eager `Tensor`s in the\n      case of graph construction.\n    attrs: Attributes of the op, as `tuple` of alternating keys and values.\n    outputs: Output tensors from the op. These are `EagerTensor`s in the case of\n      eager execution and are non-eager `Tensor`s in the case of graph\n      construction.\n    op_name: Name of the op. Applicable if and only if this method is invoked\n      due to the graph construction of an op or the eager execution of a\n      `FuncGraph`.\n    graph: The graph involved (if any).\n      - In the case if the eager execution of an op or FuncGraph, this is\n        `None`.\n      - In the case of the graph construction of an op, this is the `tf.Graph`\n        object being built.\n\n  Returns:\n    `None`, or a `list` or `tuple` of output tenors that will override the\n    original (input) `outputs`.\n  '
    ctx = context.context()
    if ctx.op_callbacks:
        ctx.invoking_op_callbacks = True
        try:
            if isinstance(attrs, dict):
                attrs_list = []
                for key in attrs:
                    attrs_list.append(key)
                    attrs_list.append(attrs[key])
                attrs_tuple = tuple(attrs_list)
            else:
                attrs_tuple = attrs
            new_outputs = outputs
            for callback in ctx.op_callbacks:
                new_outputs = callback(op_type, inputs, attrs_tuple, new_outputs, op_name=op_name, graph=graph)
                if new_outputs is not None and len(new_outputs) != len(outputs):
                    raise ValueError(f'The op callback returned {len(new_outputs)} tensors, which does not match the original number of outputs of op {op_name} ({len(outputs)}).')
            return new_outputs
        finally:
            ctx.invoking_op_callbacks = False
    else:
        return outputs