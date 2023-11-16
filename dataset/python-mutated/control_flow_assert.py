"""Assert functions for Control Flow Operations."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export

def _summarize_eager(tensor, summarize=None):
    if False:
        while True:
            i = 10
    'Returns a summarized string representation of eager `tensor`.\n\n  Args:\n    tensor: EagerTensor to summarize\n    summarize: Include these many first elements of `array`\n  '
    if summarize is None:
        summarize = 3
    elif summarize < 0:
        summarize = array_ops.size(tensor)
    if tensor._rank():
        flat = tensor.numpy().reshape((-1,))
        lst = [str(x) for x in flat[:summarize]]
        if len(lst) < flat.size:
            lst.append('...')
    elif gen_math_ops.not_equal(summarize, 0):
        lst = [str(tensor.numpy())]
    else:
        lst = []
    return ', '.join(lst)

@tf_export('debugging.Assert', 'Assert')
@dispatch.add_dispatch_support
@tf_should_use.should_use_result
def Assert(condition, data, summarize=None, name=None):
    if False:
        print('Hello World!')
    'Asserts that the given condition is true.\n\n  If `condition` evaluates to false, print the list of tensors in `data`.\n  `summarize` determines how many entries of the tensors to print.\n\n  Args:\n    condition: The condition to evaluate.\n    data: The tensors to print out when condition is false.\n    summarize: Print this many entries of each tensor.\n    name: A name for this operation (optional).\n\n  Returns:\n    assert_op: An `Operation` that, when executed, raises a\n    `tf.errors.InvalidArgumentError` if `condition` is not true.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    @compatibility(TF1)\n    When in TF V1 mode (that is, outside `tf.function`) Assert needs a control\n    dependency on the output to ensure the assertion executes:\n\n  ```python\n  # Ensure maximum element of x is smaller or equal to 1\n  assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])\n  with tf.control_dependencies([assert_op]):\n    ... code using x ...\n  ```\n\n    @end_compatibility\n  '
    if context.executing_eagerly():
        if not condition:
            xs = ops.convert_n_to_tensor(data)
            data_str = [_summarize_eager(x, summarize) for x in xs]
            raise errors.InvalidArgumentError(node_def=None, op=None, message="Expected '%s' to be true. Summarized data: %s" % (condition, '\n'.join(data_str)))
        return
    with ops.name_scope(name, 'Assert', [condition, data]) as name:
        xs = ops.convert_n_to_tensor(data)
        if all((x.dtype in {dtypes.string, dtypes.int32} for x in xs)):
            return gen_logging_ops._assert(condition, data, summarize, name='Assert')
        else:
            condition = ops.convert_to_tensor(condition, name='Condition')

            def true_assert():
                if False:
                    for i in range(10):
                        print('nop')
                return gen_logging_ops._assert(condition, data, summarize, name='Assert')
            guarded_assert = cond.cond(condition, gen_control_flow_ops.no_op, true_assert, name='AssertGuard')
            if context.executing_eagerly():
                return
            return guarded_assert.op