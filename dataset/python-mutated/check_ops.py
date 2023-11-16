"""Asserts and Boolean Checks."""
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
NUMERIC_TYPES = frozenset([dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64, dtypes.qint8, dtypes.qint16, dtypes.qint32, dtypes.quint8, dtypes.quint16, dtypes.complex64, dtypes.complex128, dtypes.bfloat16])
__all__ = ['assert_negative', 'assert_positive', 'assert_proper_iterable', 'assert_non_negative', 'assert_non_positive', 'assert_equal', 'assert_none_equal', 'assert_near', 'assert_integer', 'assert_less', 'assert_less_equal', 'assert_greater', 'assert_greater_equal', 'assert_rank', 'assert_rank_at_least', 'assert_rank_in', 'assert_same_float_dtype', 'assert_scalar', 'assert_type', 'assert_shapes', 'is_non_decreasing', 'is_numeric_tensor', 'is_strictly_increasing']

def _maybe_constant_value_string(t):
    if False:
        i = 10
        return i + 15
    if not isinstance(t, tensor_lib.Tensor):
        return str(t)
    const_t = tensor_util.constant_value(t)
    if const_t is not None:
        return str(const_t)
    return t

def _assert_static(condition, data):
    if False:
        print('Hello World!')
    'Raises a InvalidArgumentError with as much information as possible.'
    if not condition:
        data_static = [_maybe_constant_value_string(x) for x in data]
        raise errors.InvalidArgumentError(node_def=None, op=None, message='\n'.join(data_static))

def _shape_and_dtype_str(tensor):
    if False:
        return 10
    "Returns a string containing tensor's shape and dtype."
    return 'shape=%s dtype=%s' % (tensor.shape, tensor.dtype.name)

def _unary_assert_doc(sym, sym_name):
    if False:
        i = 10
        return i + 15
    'Common docstring for assert_* ops that evaluate a unary predicate over every element of a tensor.\n\n  Args:\n    sym: Mathematical symbol for the check performed on each element, i.e. "> 0"\n    sym_name: English-language name for the op described by sym\n\n  Returns:\n    Decorator that adds the appropriate docstring to the function for symbol\n    `sym`.\n  '

    def _decorator(func):
        if False:
            return 10
        'Generated decorator that adds the appropriate docstring to the function for symbol `sym`.\n\n    Args:\n      func: Function for a TensorFlow op\n\n    Returns:\n      Version of `func` with documentation attached.\n    '
        opname = func.__name__
        cap_sym_name = sym_name.capitalize()
        func.__doc__ = '\n    Assert the condition `x {sym}` holds element-wise.\n\n    When running in graph mode, you should add a dependency on this operation\n    to ensure that it runs. Example of adding a dependency to an operation:\n\n    ```python\n    with tf.control_dependencies([tf.debugging.{opname}(x, y)]):\n      output = tf.reduce_sum(x)\n    ```\n\n    {sym_name} means, for every element `x[i]` of `x`, we have `x[i] {sym}`.\n    If `x` is empty this is trivially satisfied.\n\n    Args:\n      x:  Numeric `Tensor`.\n      data:  The tensors to print out if the condition is False.  Defaults to\n        error message and first few entries of `x`.\n      summarize: Print this many entries of each tensor.\n      message: A string to prefix to the default message.\n      name: A name for this operation (optional).  Defaults to "{opname}".\n\n    Returns:\n      Op that raises `InvalidArgumentError` if `x {sym}` is False.\n      @compatibility(eager)\n        returns None\n      @end_compatibility\n\n    Raises:\n      InvalidArgumentError: if the check can be performed immediately and\n        `x {sym}` is False. The check can be performed immediately during\n        eager execution or if `x` is statically known.\n    '.format(sym=sym, sym_name=cap_sym_name, opname=opname)
        return func
    return _decorator

def _binary_assert_doc(sym, test_var):
    if False:
        print('Hello World!')
    'Common docstring for most of the v1 assert_* ops that compare two tensors element-wise.\n\n  Args:\n    sym: Binary operation symbol, i.e. "=="\n    test_var: a string that represents the variable in the right-hand side of\n      binary operator of the test case\n\n  Returns:\n    Decorator that adds the appropriate docstring to the function for\n  symbol `sym`.\n  '

    def _decorator(func):
        if False:
            print('Hello World!')
        'Generated decorator that adds the appropriate docstring to the function for symbol `sym`.\n\n    Args:\n      func: Function for a TensorFlow op\n\n    Returns:\n      A version of `func` with documentation attached.\n    '
        opname = func.__name__
        func.__doc__ = '\n    Assert the condition `x {sym} y` holds element-wise.\n\n    This condition holds if for every pair of (possibly broadcast) elements\n    `x[i]`, `y[i]`, we have `x[i] {sym} y[i]`.\n    If both `x` and `y` are empty, this is trivially satisfied.\n\n    When running in graph mode, you should add a dependency on this operation\n    to ensure that it runs. Example of adding a dependency to an operation:\n\n    ```python\n    with tf.control_dependencies([tf.compat.v1.{opname}(x, y)]):\n      output = tf.reduce_sum(x)\n    ```\n\n    Args:\n      x:  Numeric `Tensor`.\n      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.\n      data:  The tensors to print out if the condition is False.  Defaults to\n        error message and first few entries of `x`, `y`.\n      summarize: Print this many entries of each tensor.\n      message: A string to prefix to the default message.\n      name: A name for this operation (optional).  Defaults to "{opname}".\n\n    Returns:\n      Op that raises `InvalidArgumentError` if `x {sym} y` is False.\n\n    Raises:\n      InvalidArgumentError: if the check can be performed immediately and\n        `x {sym} y` is False. The check can be performed immediately during\n        eager execution or if `x` and `y` are statically known.\n\n    @compatibility(TF2)\n    `tf.compat.v1.{opname}` is compatible with eager execution and\n    `tf.function`.\n    Please use `tf.debugging.{opname}` instead when migrating to TF2. Apart\n    from `data`, all arguments are supported with the same argument name.\n\n    If you want to ensure the assert statements run before the\n    potentially-invalid computation, please use `tf.control_dependencies`,\n    as tf.function auto-control dependencies are insufficient for assert\n    statements.\n\n    #### Structural Mapping to Native TF2\n\n    Before:\n\n    ```python\n    tf.compat.v1.{opname}(\n      x=x, y=y, data=data, summarize=summarize,\n      message=message, name=name)\n    ```\n\n    After:\n\n    ```python\n    tf.debugging.{opname}(\n      x=x, y=y, message=message,\n      summarize=summarize, name=name)\n    ```\n\n    #### TF1 & TF2 Usage Example\n\n    TF1:\n\n    >>> g = tf.Graph()\n    >>> with g.as_default():\n    ...   a = tf.compat.v1.placeholder(tf.float32, [2])\n    ...   b = tf.compat.v1.placeholder(tf.float32, [2])\n    ...   result = tf.compat.v1.{opname}(a, b,\n    ...     message=\'"a {sym} b" does not hold for the given inputs\')\n    ...   with tf.compat.v1.control_dependencies([result]):\n    ...     sum_node = a + b\n    >>> sess = tf.compat.v1.Session(graph=g)\n    >>> val = sess.run(sum_node, feed_dict={{a: [1, 2], b:{test_var}}})\n\n\n    TF2:\n\n    >>> a = tf.Variable([1, 2], dtype=tf.float32)\n    >>> b = tf.Variable({test_var}, dtype=tf.float32)\n    >>> assert_op = tf.debugging.{opname}(a, b, message=\n    ...   \'"a {sym} b" does not hold for the given inputs\')\n    >>> # When working with tf.control_dependencies\n    >>> with tf.control_dependencies([assert_op]):\n    ...   val = a + b\n\n    @end_compatibility\n    '.format(sym=sym, opname=opname, test_var=test_var)
        return func
    return _decorator

def _binary_assert_doc_v2(sym, opname, test_var):
    if False:
        return 10
    'Common docstring for v2 assert_* ops that compare two tensors element-wise.\n\n  Args:\n    sym: Binary operation symbol, i.e. "=="\n    opname: Name for the symbol, i.e. "assert_equal"\n    test_var: A number used in the docstring example\n\n  Returns:\n    Decorator that adds the appropriate docstring to the function for\n  symbol `sym`.\n  '

    def _decorator(func):
        if False:
            return 10
        'Decorator that adds docstring to the function for symbol `sym`.\n\n    Args:\n      func: Function for a TensorFlow op\n\n    Returns:\n      A version of `func` with documentation attached.\n    '
        func.__doc__ = '\n    Assert the condition `x {sym} y` holds element-wise.\n\n    This Op checks that `x[i] {sym} y[i]` holds for every pair of (possibly\n    broadcast) elements of `x` and `y`. If both `x` and `y` are empty, this is\n    trivially satisfied.\n\n    If `x` {sym} `y` does not hold, `message`, as well as the first `summarize`\n    entries of `x` and `y` are printed, and `InvalidArgumentError` is raised.\n\n    When using inside `tf.function`, this API takes effects during execution.\n    It\'s recommended to use this API with `tf.control_dependencies` to\n    ensure the correct execution order.\n\n    In the following example, without `tf.control_dependencies`, errors may\n    not be raised at all.\n    Check `tf.control_dependencies` for more details.\n\n    >>> def check_size(x):\n    ...   with tf.control_dependencies([\n    ...       tf.debugging.{opname}(tf.size(x), {test_var},\n    ...                       message=\'Bad tensor size\')]):\n    ...     return x\n\n    >>> check_size(tf.ones([2, 3], tf.float32))\n    Traceback (most recent call last):\n       ...\n    InvalidArgumentError: ...\n\n    Args:\n      x:  Numeric `Tensor`.\n      y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.\n      message: A string to prefix to the default message. (optional)\n      summarize: Print this many entries of each tensor. (optional)\n      name: A name for this operation (optional).  Defaults to "{opname}".\n\n    Returns:\n      Op that raises `InvalidArgumentError` if `x {sym} y` is False. This can\n        be used with `tf.control_dependencies` inside of `tf.function`s to\n        block followup computation until the check has executed.\n      @compatibility(eager)\n      returns None\n      @end_compatibility\n\n    Raises:\n      InvalidArgumentError: if the check can be performed immediately and\n        `x == y` is False. The check can be performed immediately during eager\n        execution or if `x` and `y` are statically known.\n    '.format(sym=sym, opname=opname, test_var=test_var)
        return func
    return _decorator

def _make_assert_msg_data(sym, x, y, summarize, test_op):
    if False:
        for i in range(10):
            print('nop')
    'Subroutine of _binary_assert that generates the components of the default error message when running in eager mode.\n\n  Args:\n    sym: Mathematical symbol for the test to apply to pairs of tensor elements,\n      i.e. "=="\n    x: First input to the assertion after applying `convert_to_tensor()`\n    y: Second input to the assertion\n    summarize: Value of the "summarize" parameter to the original assert_* call;\n      tells how many elements of each tensor to print.\n    test_op: TensorFlow op that returns a Boolean tensor with True in each\n      position where the assertion is satisfied.\n\n  Returns:\n    List of tensors and scalars that, when stringified and concatenated,\n    will produce the error message string.\n  '
    data = []
    data.append('Condition x %s y did not hold.' % sym)
    if summarize > 0:
        if x.shape == y.shape and x.shape.as_list():
            mask = math_ops.logical_not(test_op)
            indices = array_ops.where(mask)
            indices_np = indices.numpy()
            x_vals = array_ops.boolean_mask(x, mask)
            y_vals = array_ops.boolean_mask(y, mask)
            num_vals = min(summarize, indices_np.shape[0])
            data.append('Indices of first %d different values:' % num_vals)
            data.append(indices_np[:num_vals])
            data.append('Corresponding x values:')
            data.append(x_vals.numpy().reshape((-1,))[:num_vals])
            data.append('Corresponding y values:')
            data.append(y_vals.numpy().reshape((-1,))[:num_vals])
        x_np = x.numpy().reshape((-1,))
        y_np = y.numpy().reshape((-1,))
        x_sum = min(x_np.size, summarize)
        y_sum = min(y_np.size, summarize)
        data.append('First %d elements of x:' % x_sum)
        data.append(x_np[:x_sum])
        data.append('First %d elements of y:' % y_sum)
        data.append(y_np[:y_sum])
    return data

def _pretty_print(data_item, summarize):
    if False:
        while True:
            i = 10
    'Format a data item for use in an error message in eager mode.\n\n  Args:\n    data_item: One of the items in the "data" argument to an assert_* function.\n      Can be a Tensor or a scalar value.\n    summarize: How many elements to retain of each tensor-valued entry in data.\n\n  Returns:\n    An appropriate string representation of data_item\n  '
    if isinstance(data_item, tensor_lib.Tensor):
        arr = data_item.numpy()
        if np.isscalar(arr):
            return str(arr)
        else:
            flat = arr.reshape((-1,))
            lst = [str(x) for x in flat[:summarize]]
            if len(lst) < flat.size:
                lst.append('...')
            return str(lst)
    else:
        return str(data_item)

def _binary_assert(sym, opname, op_func, static_func, x, y, data, summarize, message, name):
    if False:
        while True:
            i = 10
    'Generic binary elementwise assertion.\n\n  Implements the behavior described in _binary_assert_doc() above.\n  Args:\n    sym: Mathematical symbol for the test to apply to pairs of tensor elements,\n      i.e. "=="\n    opname: Name of the assert op in the public API, i.e. "assert_equal"\n    op_func: Function that, if passed the two Tensor inputs to the assertion (x\n      and y), will return the test to be passed to reduce_all() i.e.\n    static_func: Function that, if passed numpy ndarray versions of the two\n      inputs to the assertion, will return a Boolean ndarray with containing\n      True in all positions where the assertion PASSES.\n      i.e. np.equal for assert_equal()\n    x:  Numeric `Tensor`.\n    y:  Numeric `Tensor`, same dtype as and broadcastable to `x`.\n    data:  The tensors to print out if the condition is False.  Defaults to\n      error message and first few entries of `x`, `y`.\n    summarize: Print this many entries of each tensor.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).  Defaults to the value of\n      `opname`.\n\n  Returns:\n    See docstring template in _binary_assert_doc().\n  '
    with ops.name_scope(name, opname, [x, y, data]):
        x = ops.convert_to_tensor(x, name='x')
        y = ops.convert_to_tensor(y, name='y')
        if context.executing_eagerly():
            test_op = op_func(x, y)
            condition = math_ops.reduce_all(test_op)
            if condition:
                return
            if summarize is None:
                summarize = 3
            elif summarize < 0:
                summarize = 1000000000.0
            if data is None:
                data = _make_assert_msg_data(sym, x, y, summarize, test_op)
            if message is not None:
                data = [message] + list(data)
            raise errors.InvalidArgumentError(node_def=None, op=None, message='\n'.join((_pretty_print(d, summarize) for d in data)))
        else:
            if data is None:
                data = ['Condition x %s y did not hold element-wise:' % sym, 'x (%s) = ' % x.name, x, 'y (%s) = ' % y.name, y]
            if message is not None:
                data = [message] + list(data)
            condition = math_ops.reduce_all(op_func(x, y))
            x_static = tensor_util.constant_value(x)
            y_static = tensor_util.constant_value(y)
            if x_static is not None and y_static is not None:
                condition_static = np.all(static_func(x_static, y_static))
                _assert_static(condition_static, data)
            return control_flow_assert.Assert(condition, data, summarize=summarize)

@tf_export('debugging.assert_proper_iterable', v1=['debugging.assert_proper_iterable', 'assert_proper_iterable'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_proper_iterable')
def assert_proper_iterable(values):
    if False:
        while True:
            i = 10
    'Static assert that values is a "proper" iterable.\n\n  `Ops` that expect iterables of `Tensor` can call this to validate input.\n  Useful since `Tensor`, `ndarray`, byte/text type are all iterables themselves.\n\n  Args:\n    values:  Object to be checked.\n\n  Raises:\n    TypeError:  If `values` is not iterable or is one of\n      `Tensor`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.\n  '
    unintentional_iterables = (tensor_lib.Tensor, sparse_tensor.SparseTensor, np.ndarray) + compat.bytes_or_text_types
    if isinstance(values, unintentional_iterables):
        raise TypeError('Expected argument "values" to be a "proper" iterable.  Found: %s' % type(values))
    if not hasattr(values, '__iter__'):
        raise TypeError('Expected argument "values" to be iterable.  Found: %s' % type(values))

@tf_export('debugging.assert_negative', v1=[])
@dispatch.add_dispatch_support
def assert_negative_v2(x, message=None, summarize=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Assert the condition `x < 0` holds element-wise.\n\n  This Op checks that `x[i] < 0` holds for every element of `x`. If `x` is\n  empty, this is trivially satisfied.\n\n  If `x` is not negative everywhere, `message`, as well as the first `summarize`\n  entries of `x` are printed, and `InvalidArgumentError` is raised.\n\n  Args:\n    x:  Numeric `Tensor`.\n    message: A string to prefix to the default message.\n    summarize: Print this many entries of each tensor.\n    name: A name for this operation (optional).  Defaults to "assert_negative".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless `x` is all negative. This can be\n      used with `tf.control_dependencies` inside of `tf.function`s to block\n      followup computation until the check has executed.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    InvalidArgumentError: if the check can be performed immediately and\n      `x[i] < 0` is False. The check can be performed immediately during eager\n      execution or if `x` is statically known.\n  '
    return assert_negative(x=x, message=message, summarize=summarize, name=name)

@tf_export(v1=['debugging.assert_negative', 'assert_negative'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_negative')
@_unary_assert_doc('< 0', 'negative')
def assert_negative(x, data=None, summarize=None, message=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    message = _message_prefix(message)
    with ops.name_scope(name, 'assert_negative', [x, data]):
        x = ops.convert_to_tensor(x, name='x')
        if data is None:
            if context.executing_eagerly():
                name = _shape_and_dtype_str(x)
            else:
                name = x.name
            data = [message, 'Condition x < 0 did not hold element-wise:', 'x (%s) = ' % name, x]
        zero = ops.convert_to_tensor(0, dtype=x.dtype)
        return assert_less(x, zero, data=data, summarize=summarize)

@tf_export('debugging.assert_positive', v1=[])
@dispatch.add_dispatch_support
def assert_positive_v2(x, message=None, summarize=None, name=None):
    if False:
        return 10
    'Assert the condition `x > 0` holds element-wise.\n\n  This Op checks that `x[i] > 0` holds for every element of `x`. If `x` is\n  empty, this is trivially satisfied.\n\n  If `x` is not positive everywhere, `message`, as well as the first `summarize`\n  entries of `x` are printed, and `InvalidArgumentError` is raised.\n\n  Args:\n    x:  Numeric `Tensor`.\n    message: A string to prefix to the default message.\n    summarize: Print this many entries of each tensor.\n    name: A name for this operation (optional). Defaults to "assert_positive".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless `x` is all positive. This can be\n      used with `tf.control_dependencies` inside of `tf.function`s to block\n      followup computation until the check has executed.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    InvalidArgumentError: if the check can be performed immediately and\n      `x[i] > 0` is False. The check can be performed immediately during eager\n      execution or if `x` is statically known.\n  '
    return assert_positive(x=x, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_positive', 'assert_positive'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_positive')
@_unary_assert_doc('> 0', 'positive')
def assert_positive(x, data=None, summarize=None, message=None, name=None):
    if False:
        return 10
    message = _message_prefix(message)
    with ops.name_scope(name, 'assert_positive', [x, data]):
        x = ops.convert_to_tensor(x, name='x')
        if data is None:
            if context.executing_eagerly():
                name = _shape_and_dtype_str(x)
            else:
                name = x.name
            data = [message, 'Condition x > 0 did not hold element-wise:', 'x (%s) = ' % name, x]
        zero = ops.convert_to_tensor(0, dtype=x.dtype)
        return assert_less(zero, x, data=data, summarize=summarize)

@tf_export('debugging.assert_non_negative', v1=[])
@dispatch.add_dispatch_support
def assert_non_negative_v2(x, message=None, summarize=None, name=None):
    if False:
        i = 10
        return i + 15
    'Assert the condition `x >= 0` holds element-wise.\n\n  This Op checks that `x[i] >= 0` holds for every element of `x`. If `x` is\n  empty, this is trivially satisfied.\n\n  If `x` is not >= 0 everywhere, `message`, as well as the first `summarize`\n  entries of `x` are printed, and `InvalidArgumentError` is raised.\n\n  Args:\n    x:  Numeric `Tensor`.\n    message: A string to prefix to the default message.\n    summarize: Print this many entries of each tensor.\n    name: A name for this operation (optional).  Defaults to\n      "assert_non_negative".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless `x` is all non-negative. This can\n      be used with `tf.control_dependencies` inside of `tf.function`s to block\n      followup computation until the check has executed.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    InvalidArgumentError: if the check can be performed immediately and\n      `x[i] >= 0` is False. The check can be performed immediately during eager\n      execution or if `x` is statically known.\n  '
    return assert_non_negative(x=x, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_non_negative', 'assert_non_negative'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_non_negative')
@_unary_assert_doc('>= 0', 'non-negative')
def assert_non_negative(x, data=None, summarize=None, message=None, name=None):
    if False:
        print('Hello World!')
    message = _message_prefix(message)
    with ops.name_scope(name, 'assert_non_negative', [x, data]):
        x = ops.convert_to_tensor(x, name='x')
        if data is None:
            if context.executing_eagerly():
                name = _shape_and_dtype_str(x)
            else:
                name = x.name
            data = [message, 'Condition x >= 0 did not hold element-wise:', 'x (%s) = ' % name, x]
        zero = ops.convert_to_tensor(0, dtype=x.dtype)
        return assert_less_equal(zero, x, data=data, summarize=summarize)

@tf_export('debugging.assert_non_positive', v1=[])
@dispatch.add_dispatch_support
def assert_non_positive_v2(x, message=None, summarize=None, name=None):
    if False:
        return 10
    'Assert the condition `x <= 0` holds element-wise.\n\n  This Op checks that `x[i] <= 0` holds for every element of `x`. If `x` is\n  empty, this is trivially satisfied.\n\n  If `x` is not <= 0 everywhere, `message`, as well as the first `summarize`\n  entries of `x` are printed, and `InvalidArgumentError` is raised.\n\n  Args:\n    x:  Numeric `Tensor`.\n    message: A string to prefix to the default message.\n    summarize: Print this many entries of each tensor.\n    name: A name for this operation (optional).  Defaults to\n      "assert_non_positive".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless `x` is all non-positive. This can\n      be used with `tf.control_dependencies` inside of `tf.function`s to block\n      followup computation until the check has executed.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    InvalidArgumentError: if the check can be performed immediately and\n      `x[i] <= 0` is False. The check can be performed immediately during eager\n      execution or if `x` is statically known.\n  '
    return assert_non_positive(x=x, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_non_positive', 'assert_non_positive'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_non_positive')
@_unary_assert_doc('<= 0', 'non-positive')
def assert_non_positive(x, data=None, summarize=None, message=None, name=None):
    if False:
        return 10
    message = _message_prefix(message)
    with ops.name_scope(name, 'assert_non_positive', [x, data]):
        x = ops.convert_to_tensor(x, name='x')
        if data is None:
            if context.executing_eagerly():
                name = _shape_and_dtype_str(x)
            else:
                name = x.name
            data = [message, 'Condition x <= 0 did not hold element-wise:x (%s) = ' % name, x]
        zero = ops.convert_to_tensor(0, dtype=x.dtype)
        return assert_less_equal(x, zero, data=data, summarize=summarize)

@tf_export('debugging.assert_equal', 'assert_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('==', 'assert_equal', 3)
def assert_equal_v2(x, y, message=None, summarize=None, name=None):
    if False:
        return 10
    return assert_equal(x=x, y=y, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_equal', 'assert_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('==', '[1, 2]')
def assert_equal(x, y, data=None, summarize=None, message=None, name=None):
    if False:
        i = 10
        return i + 15
    with ops.name_scope(name, 'assert_equal', [x, y, data]):
        if x is y:
            return None if context.executing_eagerly() else control_flow_ops.no_op()
    return _binary_assert('==', 'assert_equal', math_ops.equal, np.equal, x, y, data, summarize, message, name)

@tf_export('debugging.assert_none_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('!=', 'assert_none_equal', 6)
def assert_none_equal_v2(x, y, summarize=None, message=None, name=None):
    if False:
        return 10
    return assert_none_equal(x=x, y=y, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_none_equal', 'assert_none_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_none_equal')
@_binary_assert_doc('!=', '[2, 1]')
def assert_none_equal(x, y, data=None, summarize=None, message=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    return _binary_assert('!=', 'assert_none_equal', math_ops.not_equal, np.not_equal, x, y, data, summarize, message, name)

@tf_export('debugging.assert_near', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
def assert_near_v2(x, y, rtol=None, atol=None, message=None, summarize=None, name=None):
    if False:
        i = 10
        return i + 15
    'Assert the condition `x` and `y` are close element-wise.\n\n  This Op checks that `x[i] - y[i] < atol + rtol * tf.abs(y[i])` holds for every\n  pair of (possibly broadcast) elements of `x` and `y`. If both `x` and `y` are\n  empty, this is trivially satisfied.\n\n  If any elements of `x` and `y` are not close, `message`, as well as the first\n  `summarize` entries of `x` and `y` are printed, and `InvalidArgumentError`\n  is raised.\n\n  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest\n  representable positive number such that `1 + eps != 1`.  This is about\n  `1.2e-6` in `32bit`, `2.22e-15` in `64bit`, and `0.00977` in `16bit`.\n  See `numpy.finfo`.\n\n  Args:\n    x: Float or complex `Tensor`.\n    y: Float or complex `Tensor`, same dtype as and broadcastable to `x`.\n    rtol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.\n      The relative tolerance.  Default is `10 * eps`.\n    atol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.\n      The absolute tolerance.  Default is `10 * eps`.\n    message: A string to prefix to the default message.\n    summarize: Print this many entries of each tensor.\n    name: A name for this operation (optional).  Defaults to "assert_near".\n\n  Returns:\n    Op that raises `InvalidArgumentError` if `x` and `y` are not close enough.\n      This can be used with `tf.control_dependencies` inside of `tf.function`s\n      to block followup computation until the check has executed.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    InvalidArgumentError: if the check can be performed immediately and\n      `x != y` is False for any pair of elements in `x` and `y`. The check can\n      be performed immediately during eager execution or if `x` and `y` are\n      statically known.\n\n  @compatibility(numpy)\n  Similar to `numpy.testing.assert_allclose`, except tolerance depends on data\n  type. This is due to the fact that `TensorFlow` is often used with `32bit`,\n  `64bit`, and even `16bit` data.\n  @end_compatibility\n  '
    return assert_near(x=x, y=y, rtol=rtol, atol=atol, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_near', 'assert_near'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_near')
def assert_near(x, y, rtol=None, atol=None, data=None, summarize=None, message=None, name=None):
    if False:
        return 10
    'Assert the condition `x` and `y` are close element-wise.\n\n  Example of adding a dependency to an operation:\n\n  ```python\n  with tf.control_dependencies([tf.compat.v1.assert_near(x, y)]):\n    output = tf.reduce_sum(x)\n  ```\n\n  This condition holds if for every pair of (possibly broadcast) elements\n  `x[i]`, `y[i]`, we have\n\n  ```tf.abs(x[i] - y[i]) <= atol + rtol * tf.abs(y[i])```.\n\n  If both `x` and `y` are empty, this is trivially satisfied.\n\n  The default `atol` and `rtol` is `10 * eps`, where `eps` is the smallest\n  representable positive number such that `1 + eps != 1`.  This is about\n  `1.2e-6` in `32bit`, `2.22e-15` in `64bit`, and `0.00977` in `16bit`.\n  See `numpy.finfo`.\n\n  Args:\n    x:  Float or complex `Tensor`.\n    y:  Float or complex `Tensor`, same `dtype` as, and broadcastable to, `x`.\n    rtol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.\n      The relative tolerance.  Default is `10 * eps`.\n    atol:  `Tensor`.  Same `dtype` as, and broadcastable to, `x`.\n      The absolute tolerance.  Default is `10 * eps`.\n    data:  The tensors to print out if the condition is False.  Defaults to\n      error message and first few entries of `x`, `y`.\n    summarize: Print this many entries of each tensor.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).  Defaults to "assert_near".\n\n  Returns:\n    Op that raises `InvalidArgumentError` if `x` and `y` are not close enough.\n\n  @compatibility(numpy)\n  Similar to `numpy.testing.assert_allclose`, except tolerance depends on data\n  type. This is due to the fact that `TensorFlow` is often used with `32bit`,\n  `64bit`, and even `16bit` data.\n  @end_compatibility\n  '
    message = _message_prefix(message)
    with ops.name_scope(name, 'assert_near', [x, y, rtol, atol, data]):
        x = ops.convert_to_tensor(x, name='x')
        y = ops.convert_to_tensor(y, name='y', dtype=x.dtype)
        dtype = x.dtype
        if dtype.is_complex:
            dtype = dtype.real_dtype
        eps = np.finfo(dtype.as_numpy_dtype).eps
        rtol = 10 * eps if rtol is None else rtol
        atol = 10 * eps if atol is None else atol
        rtol = ops.convert_to_tensor(rtol, name='rtol', dtype=dtype)
        atol = ops.convert_to_tensor(atol, name='atol', dtype=dtype)
        if context.executing_eagerly():
            x_name = _shape_and_dtype_str(x)
            y_name = _shape_and_dtype_str(y)
        else:
            x_name = x.name
            y_name = y.name
        if data is None:
            data = [message, 'x and y not equal to tolerance rtol = %s, atol = %s' % (rtol, atol), 'x (%s) = ' % x_name, x, 'y (%s) = ' % y_name, y]
        tol = atol + rtol * math_ops.abs(y)
        diff = math_ops.abs(x - y)
        condition = math_ops.reduce_all(math_ops.less(diff, tol))
        return control_flow_assert.Assert(condition, data, summarize=summarize)

@tf_export('debugging.assert_less', 'assert_less', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('<', 'assert_less', 3)
def assert_less_v2(x, y, message=None, summarize=None, name=None):
    if False:
        while True:
            i = 10
    return assert_less(x=x, y=y, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_less', 'assert_less'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('<', '[2, 3]')
def assert_less(x, y, data=None, summarize=None, message=None, name=None):
    if False:
        while True:
            i = 10
    return _binary_assert('<', 'assert_less', math_ops.less, np.less, x, y, data, summarize, message, name)

@tf_export('debugging.assert_less_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('<=', 'assert_less_equal', 3)
def assert_less_equal_v2(x, y, message=None, summarize=None, name=None):
    if False:
        return 10
    return assert_less_equal(x=x, y=y, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_less_equal', 'assert_less_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_less_equal')
@_binary_assert_doc('<=', '[1, 3]')
def assert_less_equal(x, y, data=None, summarize=None, message=None, name=None):
    if False:
        while True:
            i = 10
    return _binary_assert('<=', 'assert_less_equal', math_ops.less_equal, np.less_equal, x, y, data, summarize, message, name)

@tf_export('debugging.assert_greater', 'assert_greater', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('>', 'assert_greater', 9)
def assert_greater_v2(x, y, message=None, summarize=None, name=None):
    if False:
        i = 10
        return i + 15
    return assert_greater(x=x, y=y, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_greater', 'assert_greater'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc('>', '[0, 1]')
def assert_greater(x, y, data=None, summarize=None, message=None, name=None):
    if False:
        return 10
    return _binary_assert('>', 'assert_greater', math_ops.greater, np.greater, x, y, data, summarize, message, name)

@tf_export('debugging.assert_greater_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('>=', 'assert_greater_equal', 9)
def assert_greater_equal_v2(x, y, message=None, summarize=None, name=None):
    if False:
        while True:
            i = 10
    return assert_greater_equal(x=x, y=y, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_greater_equal', 'assert_greater_equal'])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_greater_equal')
@_binary_assert_doc('>=', '[1, 0]')
def assert_greater_equal(x, y, data=None, summarize=None, message=None, name=None):
    if False:
        return 10
    return _binary_assert('>=', 'assert_greater_equal', math_ops.greater_equal, np.greater_equal, x, y, data, summarize, message, name)

def _assert_rank_condition(x, rank, static_condition, dynamic_condition, data, summarize):
    if False:
        print('Hello World!')
    'Assert `x` has a rank that satisfies a given condition.\n\n  Args:\n    x:  Numeric `Tensor`.\n    rank:  Scalar `Tensor`.\n    static_condition:   A python function that takes `[actual_rank, given_rank]`\n      and returns `True` if the condition is satisfied, `False` otherwise.\n    dynamic_condition:  An `op` that takes [actual_rank, given_rank] and return\n      `True` if the condition is satisfied, `False` otherwise.\n    data:  The tensors to print out if the condition is false.  Defaults to\n      error message and first few entries of `x`.\n    summarize: Print this many entries of each tensor.\n\n  Returns:\n    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.\n\n  Raises:\n    ValueError:  If static checks determine `x` fails static_condition.\n  '
    assert_type(rank, dtypes.int32)
    rank_static = tensor_util.constant_value(rank)
    if rank_static is not None:
        if rank_static.ndim != 0:
            raise ValueError('Rank must be a scalar.')
        x_rank_static = x.get_shape().ndims
        if x_rank_static is not None:
            if not static_condition(x_rank_static, rank_static):
                raise ValueError('Static rank condition failed', x_rank_static, rank_static)
            return control_flow_ops.no_op(name='static_checks_determined_all_ok')
    condition = dynamic_condition(array_ops.rank(x), rank)
    if rank_static is None:
        this_data = ['Rank must be a scalar. Received rank: ', rank]
        rank_check = assert_rank(rank, 0, data=this_data)
        condition = control_flow_ops.with_dependencies([rank_check], condition)
    return control_flow_assert.Assert(condition, data, summarize=summarize)

@tf_export('debugging.assert_rank', 'assert_rank', v1=[])
@dispatch.add_dispatch_support
def assert_rank_v2(x, rank, message=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Assert that `x` has rank equal to `rank`.\n\n  This Op checks that the rank of `x` is equal to `rank`.\n\n  If `x` has a different rank, `message`, as well as the shape of `x` are\n  printed, and `InvalidArgumentError` is raised.\n\n  Args:\n    x: `Tensor`.\n    rank: Scalar integer `Tensor`.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional). Defaults to\n      "assert_rank".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless `x` has specified rank.\n    If static checks determine `x` has correct rank, a `no_op` is returned.\n    This can be used with `tf.control_dependencies` inside of `tf.function`s\n    to block followup computation until the check has executed.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    InvalidArgumentError: if the check can be performed immediately and\n      `x` does not have rank `rank`. The check can be performed immediately\n      during eager execution or if the shape of `x` is statically known.\n  '
    return assert_rank(x=x, rank=rank, message=message, name=name)

@tf_export(v1=['debugging.assert_rank', 'assert_rank'])
@dispatch.add_dispatch_support
def assert_rank(x, rank, data=None, summarize=None, message=None, name=None):
    if False:
        print('Hello World!')
    'Assert `x` has rank equal to `rank`.\n\n  Example of adding a dependency to an operation:\n\n  ```python\n  with tf.control_dependencies([tf.compat.v1.assert_rank(x, 2)]):\n    output = tf.reduce_sum(x)\n  ```\n\n  Args:\n    x:  Numeric `Tensor`.\n    rank:  Scalar integer `Tensor`.\n    data:  The tensors to print out if the condition is False.  Defaults to\n      error message and the shape of `x`.\n    summarize: Print this many entries of each tensor.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).  Defaults to "assert_rank".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless `x` has specified rank.\n    If static checks determine `x` has correct rank, a `no_op` is returned.\n\n  Raises:\n    ValueError:  If static checks determine `x` has wrong rank.\n  '
    with ops.name_scope(name, 'assert_rank', (x, rank) + tuple(data or [])):
        if not isinstance(x, sparse_tensor.SparseTensor):
            x = ops.convert_to_tensor(x, name='x')
        rank = ops.convert_to_tensor(rank, name='rank')
        message = _message_prefix(message)
        static_condition = lambda actual_rank, given_rank: actual_rank == given_rank
        dynamic_condition = math_ops.equal
        if context.executing_eagerly() or isinstance(x, sparse_tensor.SparseTensor):
            name = ''
        else:
            name = x.name
        if data is None:
            data = [message, 'Tensor %s must have rank' % name, rank, 'Received shape: ', array_ops.shape(x)]
        try:
            assert_op = _assert_rank_condition(x, rank, static_condition, dynamic_condition, data, summarize)
        except ValueError as e:
            if e.args[0] == 'Static rank condition failed':
                raise ValueError('%sTensor %s must have rank %d.  Received rank %d, shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
            else:
                raise ValueError(e.args[0])
    return assert_op

@tf_export('debugging.assert_rank_at_least', v1=[])
@dispatch.add_dispatch_support
def assert_rank_at_least_v2(x, rank, message=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Assert that `x` has rank of at least `rank`.\n\n  This Op checks that the rank of `x` is greater or equal to `rank`.\n\n  If `x` has a rank lower than `rank`, `message`, as well as the shape of `x`\n  are printed, and `InvalidArgumentError` is raised.\n\n  Args:\n    x: `Tensor`.\n    rank: Scalar integer `Tensor`.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).  Defaults to\n      "assert_rank_at_least".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.\n    If static checks determine `x` has correct rank, a `no_op` is returned.\n    This can be used with `tf.control_dependencies` inside of `tf.function`s\n    to block followup computation until the check has executed.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    InvalidArgumentError: `x` does not have rank at least `rank`, but the rank\n      cannot be statically determined.\n    ValueError: If static checks determine `x` has mismatched rank.\n  '
    return assert_rank_at_least(x=x, rank=rank, message=message, name=name)

@tf_export(v1=['debugging.assert_rank_at_least', 'assert_rank_at_least'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_rank_at_least')
def assert_rank_at_least(x, rank, data=None, summarize=None, message=None, name=None):
    if False:
        i = 10
        return i + 15
    'Assert `x` has rank equal to `rank` or higher.\n\n  Example of adding a dependency to an operation:\n\n  ```python\n  with tf.control_dependencies([tf.compat.v1.assert_rank_at_least(x, 2)]):\n    output = tf.reduce_sum(x)\n  ```\n\n  Args:\n    x:  Numeric `Tensor`.\n    rank:  Scalar `Tensor`.\n    data:  The tensors to print out if the condition is False.  Defaults to\n      error message and first few entries of `x`.\n    summarize: Print this many entries of each tensor.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).\n      Defaults to "assert_rank_at_least".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless `x` has specified rank or higher.\n    If static checks determine `x` has correct rank, a `no_op` is returned.\n\n  Raises:\n    ValueError:  If static checks determine `x` has wrong rank.\n  '
    with ops.name_scope(name, 'assert_rank_at_least', (x, rank) + tuple(data or [])):
        x = ops.convert_to_tensor(x, name='x')
        rank = ops.convert_to_tensor(rank, name='rank')
        message = _message_prefix(message)
        static_condition = lambda actual_rank, given_rank: actual_rank >= given_rank
        dynamic_condition = math_ops.greater_equal
        if context.executing_eagerly():
            name = ''
        else:
            name = x.name
        if data is None:
            data = [message, 'Tensor %s must have rank at least' % name, rank, 'Received shape: ', array_ops.shape(x)]
        try:
            assert_op = _assert_rank_condition(x, rank, static_condition, dynamic_condition, data, summarize)
        except ValueError as e:
            if e.args[0] == 'Static rank condition failed':
                raise ValueError('%sTensor %s must have rank at least %d.  Received rank %d, shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
            else:
                raise
    return assert_op

def _static_rank_in(actual_rank, given_ranks):
    if False:
        while True:
            i = 10
    return actual_rank in given_ranks

def _dynamic_rank_in(actual_rank, given_ranks):
    if False:
        return 10
    if len(given_ranks) < 1:
        return ops.convert_to_tensor(False)
    result = math_ops.equal(given_ranks[0], actual_rank)
    for given_rank in given_ranks[1:]:
        result = math_ops.logical_or(result, math_ops.equal(given_rank, actual_rank))
    return result

def _assert_ranks_condition(x, ranks, static_condition, dynamic_condition, data, summarize):
    if False:
        print('Hello World!')
    'Assert `x` has a rank that satisfies a given condition.\n\n  Args:\n    x:  Numeric `Tensor`.\n    ranks:  Scalar `Tensor`.\n    static_condition:   A python function that takes\n      `[actual_rank, given_ranks]` and returns `True` if the condition is\n      satisfied, `False` otherwise.\n    dynamic_condition:  An `op` that takes [actual_rank, given_ranks]\n      and return `True` if the condition is satisfied, `False` otherwise.\n    data:  The tensors to print out if the condition is false.  Defaults to\n      error message and first few entries of `x`.\n    summarize: Print this many entries of each tensor.\n\n  Returns:\n    Op raising `InvalidArgumentError` if `x` fails dynamic_condition.\n\n  Raises:\n    ValueError:  If static checks determine `x` fails static_condition.\n  '
    for rank in ranks:
        assert_type(rank, dtypes.int32)
    ranks_static = tuple([tensor_util.constant_value(rank) for rank in ranks])
    if not any((r is None for r in ranks_static)):
        for rank_static in ranks_static:
            if rank_static.ndim != 0:
                raise ValueError('Rank must be a scalar.')
        x_rank_static = x.get_shape().ndims
        if x_rank_static is not None:
            if not static_condition(x_rank_static, ranks_static):
                raise ValueError('Static rank condition failed', x_rank_static, ranks_static)
            return control_flow_ops.no_op(name='static_checks_determined_all_ok')
    condition = dynamic_condition(array_ops.rank(x), ranks)
    for (rank, rank_static) in zip(ranks, ranks_static):
        if rank_static is None:
            this_data = ['Rank must be a scalar. Received rank: ', rank]
            rank_check = assert_rank(rank, 0, data=this_data)
            condition = control_flow_ops.with_dependencies([rank_check], condition)
    return control_flow_assert.Assert(condition, data, summarize=summarize)

@tf_export('debugging.assert_rank_in', v1=[])
@dispatch.add_dispatch_support
def assert_rank_in_v2(x, ranks, message=None, name=None):
    if False:
        print('Hello World!')
    'Assert that `x` has a rank in `ranks`.\n\n  This Op checks that the rank of `x` is in `ranks`.\n\n  If `x` has a different rank, `message`, as well as the shape of `x` are\n  printed, and `InvalidArgumentError` is raised.\n\n  Args:\n    x: `Tensor`.\n    ranks: `Iterable` of scalar `Tensor` objects.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional). Defaults to "assert_rank_in".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.\n    If static checks determine `x` has matching rank, a `no_op` is returned.\n    This can be used with `tf.control_dependencies` inside of `tf.function`s\n    to block followup computation until the check has executed.\n    @compatibility(eager)\n    returns None\n    @end_compatibility\n\n  Raises:\n    InvalidArgumentError: `x` does not have rank in `ranks`, but the rank cannot\n      be statically determined.\n    ValueError: If static checks determine `x` has mismatched rank.\n  '
    return assert_rank_in(x=x, ranks=ranks, message=message, name=name)

@tf_export(v1=['debugging.assert_rank_in', 'assert_rank_in'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_rank_in')
def assert_rank_in(x, ranks, data=None, summarize=None, message=None, name=None):
    if False:
        i = 10
        return i + 15
    'Assert `x` has rank in `ranks`.\n\n  Example of adding a dependency to an operation:\n\n  ```python\n  with tf.control_dependencies([tf.compat.v1.assert_rank_in(x, (2, 4))]):\n    output = tf.reduce_sum(x)\n  ```\n\n  Args:\n    x:  Numeric `Tensor`.\n    ranks:  Iterable of scalar `Tensor` objects.\n    data:  The tensors to print out if the condition is False.  Defaults to\n      error message and first few entries of `x`.\n    summarize: Print this many entries of each tensor.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).\n      Defaults to "assert_rank_in".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.\n    If static checks determine `x` has matching rank, a `no_op` is returned.\n\n  Raises:\n    ValueError:  If static checks determine `x` has mismatched rank.\n  '
    with ops.name_scope(name, 'assert_rank_in', (x,) + tuple(ranks) + tuple(data or [])):
        if not isinstance(x, sparse_tensor.SparseTensor):
            x = ops.convert_to_tensor(x, name='x')
        ranks = tuple([ops.convert_to_tensor(rank, name='rank') for rank in ranks])
        message = _message_prefix(message)
        if context.executing_eagerly() or isinstance(x, sparse_tensor.SparseTensor):
            name = ''
        else:
            name = x.name
        if data is None:
            data = [message, 'Tensor %s must have rank in' % name] + list(ranks) + ['Received shape: ', array_ops.shape(x)]
        try:
            assert_op = _assert_ranks_condition(x, ranks, _static_rank_in, _dynamic_rank_in, data, summarize)
        except ValueError as e:
            if e.args[0] == 'Static rank condition failed':
                raise ValueError('%sTensor %s must have rank in %s.  Received rank %d, shape %s' % (message, name, e.args[2], e.args[1], x.get_shape()))
            else:
                raise
    return assert_op

@tf_export('debugging.assert_integer', v1=[])
@dispatch.add_dispatch_support
def assert_integer_v2(x, message=None, name=None):
    if False:
        while True:
            i = 10
    'Assert that `x` is of integer dtype.\n\n  If `x` has a non-integer type, `message`, as well as the dtype of `x` are\n  printed, and `InvalidArgumentError` is raised.\n\n  This can always be checked statically, so this method returns nothing.\n\n  Args:\n    x: A `Tensor`.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional). Defaults to "assert_integer".\n\n  Raises:\n    TypeError:  If `x.dtype` is not a non-quantized integer type.\n  '
    assert_integer(x=x, message=message, name=name)

@tf_export(v1=['debugging.assert_integer', 'assert_integer'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_integer')
def assert_integer(x, message=None, name=None):
    if False:
        while True:
            i = 10
    'Assert that `x` is of integer dtype.\n\n  Example of adding a dependency to an operation:\n\n  ```python\n  with tf.control_dependencies([tf.compat.v1.assert_integer(x)]):\n    output = tf.reduce_sum(x)\n  ```\n\n  Args:\n    x: `Tensor` whose basetype is integer and is not quantized.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).  Defaults to "assert_integer".\n\n  Raises:\n    TypeError:  If `x.dtype` is anything other than non-quantized integer.\n\n  Returns:\n    A `no_op` that does nothing.  Type can be determined statically.\n  '
    with ops.name_scope(name, 'assert_integer', [x]):
        x = ops.convert_to_tensor(x, name='x')
        if not x.dtype.is_integer:
            if context.executing_eagerly():
                name = 'tensor'
            else:
                name = x.name
            err_msg = '%sExpected "x" to be integer type.  Found: %s of dtype %s' % (_message_prefix(message), name, x.dtype)
            raise TypeError(err_msg)
        return control_flow_ops.no_op('statically_determined_was_integer')

@tf_export('debugging.assert_type', v1=[])
@dispatch.add_dispatch_support
def assert_type_v2(tensor, tf_type, message=None, name=None):
    if False:
        while True:
            i = 10
    'Asserts that the given `Tensor` is of the specified type.\n\n  This can always be checked statically, so this method returns nothing.\n\n  Example:\n\n  >>> a = tf.Variable(1.0)\n  >>> tf.debugging.assert_type(a, tf_type= tf.float32)\n\n  >>> b = tf.constant(21)\n  >>> tf.debugging.assert_type(b, tf_type=tf.bool)\n  Traceback (most recent call last):\n  ...\n  TypeError: ...\n\n  >>> c = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2],\n  ...  dense_shape=[3, 4])\n  >>> tf.debugging.assert_type(c, tf_type= tf.int32)\n\n  Args:\n    tensor: A `Tensor`, `SparseTensor` or `tf.Variable` .\n    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,\n      etc).\n    message: A string to prefix to the default message.\n    name:  A name for this operation. Defaults to "assert_type"\n\n  Raises:\n    TypeError: If the tensor\'s data type doesn\'t match `tf_type`.\n  '
    assert_type(tensor=tensor, tf_type=tf_type, message=message, name=name)

@tf_export(v1=['debugging.assert_type', 'assert_type'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_type')
def assert_type(tensor, tf_type, message=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Statically asserts that the given `Tensor` is of the specified type.\n\n  Args:\n    tensor: A `Tensor` or `SparseTensor`.\n    tf_type: A tensorflow type (`dtypes.float32`, `tf.int64`, `dtypes.bool`,\n      etc).\n    message: A string to prefix to the default message.\n    name:  A name to give this `Op`.  Defaults to "assert_type"\n\n  Raises:\n    TypeError: If the tensors data type doesn\'t match `tf_type`.\n\n  Returns:\n    A `no_op` that does nothing.  Type can be determined statically.\n  '
    tf_type = dtypes.as_dtype(tf_type)
    with ops.name_scope(name, 'assert_type', [tensor]):
        if not isinstance(tensor, sparse_tensor.SparseTensor):
            tensor = ops.convert_to_tensor(tensor, name='tensor')
        if tensor.dtype != tf_type:
            raise TypeError(f"{_message_prefix(message)}{getattr(tensor, 'name', 'tensor')} must be of type {tf_type!r}; got {tensor.dtype!r}")
        return control_flow_ops.no_op('statically_determined_correct_type')

def _dimension_sizes(x):
    if False:
        while True:
            i = 10
    'Gets the dimension sizes of a tensor `x`.\n\n  If a size can be determined statically it is returned as an integer,\n  otherwise as a tensor.\n\n  If `x` is a scalar it is treated as rank 1 size 1.\n\n  Args:\n    x: A `Tensor`.\n\n  Returns:\n    Dimension sizes.\n  '
    dynamic_shape = array_ops.shape(x)
    rank = x.get_shape().rank
    rank_is_known = rank is not None
    if rank_is_known and rank == 0:
        return (1,)
    if rank_is_known and rank > 0:
        static_shape = x.get_shape().as_list()
        sizes = [int(size) if size is not None else dynamic_shape[i] for (i, size) in enumerate(static_shape)]
        return sizes
    has_rank_zero = math_ops.equal(array_ops.rank(x), 0)
    return cond.cond(has_rank_zero, lambda : array_ops.constant([1]), lambda : dynamic_shape)

def _symbolic_dimension_sizes(symbolic_shape):
    if False:
        return 10
    if not symbolic_shape:
        return tuple([1])
    return symbolic_shape

def _has_known_value(dimension_size):
    if False:
        while True:
            i = 10
    not_none = dimension_size is not None
    try:
        int(dimension_size)
        can_be_parsed_as_int = True
    except (ValueError, TypeError):
        can_be_parsed_as_int = False
    return not_none and can_be_parsed_as_int

def _is_symbol_for_any_size(symbol):
    if False:
        return 10
    return symbol in [None, '.']
_TensorDimSizes = collections.namedtuple('_TensorDimSizes', ['x', 'unspecified_dim', 'actual_sizes', 'symbolic_sizes'])

@tf_export('debugging.assert_shapes', v1=[])
@dispatch.add_dispatch_support
def assert_shapes_v2(shapes, data=None, summarize=None, message=None, name=None):
    if False:
        while True:
            i = 10
    'Assert tensor shapes and dimension size relationships between tensors.\n\n  This Op checks that a collection of tensors shape relationships\n  satisfies given constraints.\n\n  Example:\n\n  >>> n = 10\n  >>> q = 3\n  >>> d = 7\n  >>> x = tf.zeros([n,q])\n  >>> y = tf.ones([n,d])\n  >>> param = tf.Variable([1.0, 2.0, 3.0])\n  >>> scalar = 1.0\n  >>> tf.debugging.assert_shapes([\n  ...  (x, (\'N\', \'Q\')),\n  ...  (y, (\'N\', \'D\')),\n  ...  (param, (\'Q\',)),\n  ...  (scalar, ()),\n  ... ])\n\n  >>> tf.debugging.assert_shapes([\n  ...   (x, (\'N\', \'D\')),\n  ...   (y, (\'N\', \'D\'))\n  ... ])\n  Traceback (most recent call last):\n  ...\n  ValueError: ...\n\n  If `x`, `y`, `param` or `scalar` does not have a shape that satisfies\n  all specified constraints, `message`, as well as the first `summarize` entries\n  of the first encountered violating tensor are printed, and\n  `InvalidArgumentError` is raised.\n\n  Size entries in the specified shapes are checked against other entries by\n  their __hash__, except:\n    - a size entry is interpreted as an explicit size if it can be parsed as an\n      integer primitive.\n    - a size entry is interpreted as *any* size if it is None or \'.\'.\n\n  If the first entry of a shape is `...` (type `Ellipsis`) or \'*\' that indicates\n  a variable number of outer dimensions of unspecified size, i.e. the constraint\n  applies to the inner-most dimensions only.\n\n  Scalar tensors and specified shapes of length zero (excluding the \'inner-most\'\n  prefix) are both treated as having a single dimension of size one.\n\n  Args:\n    shapes: dictionary with (`Tensor` to shape) items, or a list of\n      (`Tensor`, shape) tuples. A shape must be an iterable.\n    data: The tensors to print out if the condition is False.  Defaults to error\n      message and first few entries of the violating tensor.\n    summarize: Print this many entries of the tensor.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).  Defaults to "assert_shapes".\n\n  Raises:\n    ValueError:  If static checks determine any shape constraint is violated.\n  '
    assert_shapes(shapes, data=data, summarize=summarize, message=message, name=name)

@tf_export(v1=['debugging.assert_shapes'])
@dispatch.add_dispatch_support
def assert_shapes(shapes, data=None, summarize=None, message=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Assert tensor shapes and dimension size relationships between tensors.\n\n  This Op checks that a collection of tensors shape relationships\n  satisfies given constraints.\n\n  Example:\n\n  >>> n = 10\n  >>> q = 3\n  >>> d = 7\n  >>> x = tf.zeros([n,q])\n  >>> y = tf.ones([n,d])\n  >>> param = tf.Variable([1.0, 2.0, 3.0])\n  >>> scalar = 1.0\n  >>> tf.debugging.assert_shapes([\n  ...  (x, (\'N\', \'Q\')),\n  ...  (y, (\'N\', \'D\')),\n  ...  (param, (\'Q\',)),\n  ...  (scalar, ()),\n  ... ])\n\n  >>> tf.debugging.assert_shapes([\n  ...   (x, (\'N\', \'D\')),\n  ...   (y, (\'N\', \'D\'))\n  ... ])\n  Traceback (most recent call last):\n  ...\n  ValueError: ...\n\n  Example of adding a dependency to an operation:\n\n  ```python\n  with tf.control_dependencies([tf.assert_shapes(shapes)]):\n    output = tf.matmul(x, y, transpose_a=True)\n  ```\n\n  If `x`, `y`, `param` or `scalar` does not have a shape that satisfies\n  all specified constraints, `message`, as well as the first `summarize` entries\n  of the first encountered violating tensor are printed, and\n  `InvalidArgumentError` is raised.\n\n  Size entries in the specified shapes are checked against other entries by\n  their __hash__, except:\n    - a size entry is interpreted as an explicit size if it can be parsed as an\n      integer primitive.\n    - a size entry is interpreted as *any* size if it is None or \'.\'.\n\n  If the first entry of a shape is `...` (type `Ellipsis`) or \'*\' that indicates\n  a variable number of outer dimensions of unspecified size, i.e. the constraint\n  applies to the inner-most dimensions only.\n\n  Scalar tensors and specified shapes of length zero (excluding the \'inner-most\'\n  prefix) are both treated as having a single dimension of size one.\n\n  Args:\n    shapes: A list of (`Tensor`, `shape`) tuples, wherein `shape` is the\n      expected shape of `Tensor`. See the example code above. The `shape` must\n      be an iterable. Each element of the iterable can be either a concrete\n      integer value or a string that abstractly represents the dimension.\n      For example,\n        - `(\'N\', \'Q\')` specifies a 2D shape wherein the first and second\n          dimensions of shape may or may not be equal.\n        - `(\'N\', \'N\', \'Q\')` specifies a 3D shape wherein the first and second\n          dimensions are equal.\n        - `(1, \'N\')` specifies a 2D shape wherein the first dimension is\n          exactly 1 and the second dimension can be any value.\n      Note that the abstract dimension letters take effect across different\n      tuple elements of the list. For example,\n      `tf.debugging.assert_shapes([(x, (\'N\', \'A\')), (y, (\'N\', \'B\'))]` asserts\n      that both `x` and `y` are rank-2 tensors and their first dimensions are\n      equal (`N`).\n      `shape` can also be a `tf.TensorShape`.\n    data: The tensors to print out if the condition is False.  Defaults to error\n      message and first few entries of the violating tensor.\n    summarize: Print this many entries of the tensor.\n    message: A string to prefix to the default message.\n    name: A name for this operation (optional).  Defaults to "assert_shapes".\n\n  Returns:\n    Op raising `InvalidArgumentError` unless all shape constraints are\n    satisfied.\n    If static checks determine all constraints are satisfied, a `no_op` is\n    returned.\n\n  Raises:\n    ValueError:  If static checks determine any shape constraint is violated.\n  '
    if isinstance(shapes, dict):
        shapes = shapes.items()
    message_prefix = _message_prefix(message)
    with ops.name_scope(name, 'assert_shapes', [shapes, data]):
        shape_constraints = [(x if isinstance(x, sparse_tensor.SparseTensor) else ops.convert_to_tensor(x), s) for (x, s) in shapes if s is not None]
        executing_eagerly = context.executing_eagerly()

        def tensor_name(x):
            if False:
                return 10
            if executing_eagerly or isinstance(x, sparse_tensor.SparseTensor):
                return _shape_and_dtype_str(x)
            return x.name
        tensor_dim_sizes = []
        for (tensor, symbolic_shape) in shape_constraints:
            is_iterable = hasattr(symbolic_shape, '__iter__') or hasattr(symbolic_shape, '__getitem__')
            if not is_iterable:
                raise ValueError('%sTensor %s.  Specified shape must be an iterable.  An iterable has the attribute `__iter__` or `__getitem__`.  Received specified shape: %s' % (message_prefix, tensor_name(tensor), symbolic_shape))
            symbolic_shape_tuple = tuple(symbolic_shape)
            tensors_specified_innermost = False
            for (i, symbol) in enumerate(symbolic_shape_tuple):
                if symbol not in [Ellipsis, '*']:
                    continue
                if i != 0:
                    raise ValueError('%sTensor %s specified shape index %d.  Symbol `...` or `*` for a variable number of unspecified dimensions is only allowed as the first entry' % (message_prefix, tensor_name(tensor), i))
                tensors_specified_innermost = True
            tensor_dim_sizes.append(_TensorDimSizes(tensor, tensors_specified_innermost, _dimension_sizes(tensor), _symbolic_dimension_sizes(symbolic_shape_tuple[1:] if tensors_specified_innermost else symbolic_shape_tuple)))
        rank_assertions = []
        for sizes in tensor_dim_sizes:
            rank = len(sizes.symbolic_sizes)
            rank_zero_or_one = rank in [0, 1]
            if sizes.unspecified_dim:
                if rank_zero_or_one:
                    continue
                assertion = assert_rank_at_least(x=sizes.x, rank=rank, data=data, summarize=summarize, message=message, name=name)
            elif rank_zero_or_one:
                assertion = assert_rank_in(x=sizes.x, ranks=[0, 1], data=data, summarize=summarize, message=message, name=name)
            else:
                assertion = assert_rank(x=sizes.x, rank=rank, data=data, summarize=summarize, message=message, name=name)
            rank_assertions.append(assertion)
        size_assertions = []
        size_specifications = {}
        for sizes in tensor_dim_sizes:
            for (i, size_symbol) in enumerate(sizes.symbolic_sizes):
                if _is_symbol_for_any_size(size_symbol):
                    continue
                if sizes.unspecified_dim:
                    tensor_dim = i - len(sizes.symbolic_sizes)
                else:
                    tensor_dim = i
                if size_symbol in size_specifications or _has_known_value(size_symbol):
                    if _has_known_value(size_symbol):
                        specified_size = int(size_symbol)
                        size_check_message = 'Specified explicitly'
                    else:
                        (specified_size, specified_by_y, specified_at_dim) = size_specifications[size_symbol]
                        size_check_message = 'Specified by tensor %s dimension %d' % (tensor_name(specified_by_y), specified_at_dim)
                    with ops.control_dependencies(rank_assertions):
                        actual_size = sizes.actual_sizes[tensor_dim]
                    if _has_known_value(actual_size) and _has_known_value(specified_size):
                        if int(actual_size) != int(specified_size):
                            raise ValueError('%s%s.  Tensor %s dimension %s must have size %d.  Received size %d, shape %s' % (message_prefix, size_check_message, tensor_name(sizes.x), tensor_dim, specified_size, actual_size, sizes.x.get_shape()))
                        continue
                    condition = math_ops.equal(ops.convert_to_tensor(actual_size), ops.convert_to_tensor(specified_size))
                    data_ = data
                    if data is None:
                        data_ = [message_prefix, size_check_message, 'Tensor %s dimension' % tensor_name(sizes.x), tensor_dim, 'must have size', specified_size, 'Received shape: ', array_ops.shape(sizes.x)]
                    size_assertions.append(control_flow_assert.Assert(condition, data_, summarize=summarize))
                else:
                    with ops.control_dependencies(rank_assertions):
                        size = sizes.actual_sizes[tensor_dim]
                    size_specifications[size_symbol] = (size, sizes.x, tensor_dim)
    with ops.control_dependencies(rank_assertions):
        shapes_assertion = control_flow_ops.group(size_assertions)
    return shapes_assertion

def _get_results_for_monotonic_comparison(x, compare_op):
    if False:
        return 10
    'Gets the difference x[1:] - x[:-1].'
    x = array_ops.reshape(x, [-1])
    if not is_numeric_tensor(x):
        raise TypeError('Expected x to be numeric, instead found: %s' % x)
    is_shorter_than_two = math_ops.less(array_ops.size(x), 2)
    short_result = lambda : ops.convert_to_tensor([], dtype=bool)
    s_len = array_ops.shape(x) - 1
    diff = lambda : compare_op(array_ops.strided_slice(x, [1], [1] + s_len), array_ops.strided_slice(x, [0], s_len))
    return cond.cond(is_shorter_than_two, short_result, diff)

@tf_export('debugging.is_numeric_tensor', v1=['debugging.is_numeric_tensor', 'is_numeric_tensor'])
@deprecation.deprecated_endpoints('is_numeric_tensor')
def is_numeric_tensor(tensor):
    if False:
        print('Hello World!')
    'Returns `True` if the elements of `tensor` are numbers.\n\n  Specifically, returns `True` if the dtype of `tensor` is one of the following:\n\n  * `tf.float16`\n  * `tf.float32`\n  * `tf.float64`\n  * `tf.int8`\n  * `tf.int16`\n  * `tf.int32`\n  * `tf.int64`\n  * `tf.uint8`\n  * `tf.uint16`\n  * `tf.uint32`\n  * `tf.uint64`\n  * `tf.qint8`\n  * `tf.qint16`\n  * `tf.qint32`\n  * `tf.quint8`\n  * `tf.quint16`\n  * `tf.complex64`\n  * `tf.complex128`\n  * `tf.bfloat16`\n\n  Returns `False` if `tensor` is of a non-numeric type or if `tensor` is not\n  a `tf.Tensor` object.\n  '
    return isinstance(tensor, tensor_lib.Tensor) and tensor.dtype in NUMERIC_TYPES

@tf_export('math.is_non_decreasing', v1=['math.is_non_decreasing', 'debugging.is_non_decreasing', 'is_non_decreasing'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('debugging.is_non_decreasing', 'is_non_decreasing')
def is_non_decreasing(x, name=None):
    if False:
        print('Hello World!')
    'Returns `True` if `x` is non-decreasing.\n\n  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`\n  is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.\n  If `x` has less than two elements, it is trivially non-decreasing.\n\n  See also:  `is_strictly_increasing`\n\n  >>> x1 = tf.constant([1.0, 1.0, 3.0])\n  >>> tf.math.is_non_decreasing(x1)\n  <tf.Tensor: shape=(), dtype=bool, numpy=True>\n  >>> x2 = tf.constant([3.0, 1.0, 2.0])\n  >>> tf.math.is_non_decreasing(x2)\n  <tf.Tensor: shape=(), dtype=bool, numpy=False>\n\n  Args:\n    x: Numeric `Tensor`.\n    name: A name for this operation (optional).  Defaults to "is_non_decreasing"\n\n  Returns:\n    Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.\n\n  Raises:\n    TypeError: if `x` is not a numeric tensor.\n  '
    with ops.name_scope(name, 'is_non_decreasing', [x]):
        diff = _get_results_for_monotonic_comparison(x, math_ops.greater_equal)
        return math_ops.reduce_all(diff)

@tf_export('math.is_strictly_increasing', v1=['math.is_strictly_increasing', 'debugging.is_strictly_increasing', 'is_strictly_increasing'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('debugging.is_strictly_increasing', 'is_strictly_increasing')
def is_strictly_increasing(x, name=None):
    if False:
        i = 10
        return i + 15
    'Returns `True` if `x` is strictly increasing.\n\n  Elements of `x` are compared in row-major order.  The tensor `[x[0],...]`\n  is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.\n  If `x` has less than two elements, it is trivially strictly increasing.\n\n  See also:  `is_non_decreasing`\n\n  >>> x1 = tf.constant([1.0, 2.0, 3.0])\n  >>> tf.math.is_strictly_increasing(x1)\n  <tf.Tensor: shape=(), dtype=bool, numpy=True>\n  >>> x2 = tf.constant([3.0, 1.0, 2.0])\n  >>> tf.math.is_strictly_increasing(x2)\n  <tf.Tensor: shape=(), dtype=bool, numpy=False>\n\n  Args:\n    x: Numeric `Tensor`.\n    name: A name for this operation (optional).\n      Defaults to "is_strictly_increasing"\n\n  Returns:\n    Boolean `Tensor`, equal to `True` iff `x` is strictly increasing.\n\n  Raises:\n    TypeError: if `x` is not a numeric tensor.\n  '
    with ops.name_scope(name, 'is_strictly_increasing', [x]):
        diff = _get_results_for_monotonic_comparison(x, math_ops.greater)
        return math_ops.reduce_all(diff)

def _assert_same_base_type(items, expected_type=None):
    if False:
        while True:
            i = 10
    'Asserts all items are of the same base type.\n\n  Args:\n    items: List of graph items (e.g., `Variable`, `Tensor`, `SparseTensor`,\n        `Operation`, or `IndexedSlices`). Can include `None` elements, which\n        will be ignored.\n    expected_type: Expected type. If not specified, assert all items are\n        of the same base type.\n\n  Returns:\n    Validated type, or none if neither expected_type nor items provided.\n\n  Raises:\n    ValueError: If any types do not match.\n  '
    original_expected_type = expected_type
    mismatch = False
    for item in items:
        if item is not None:
            item_type = item.dtype.base_dtype
            if not expected_type:
                expected_type = item_type
            elif expected_type != item_type:
                mismatch = True
                break
    if mismatch:
        expected_type = original_expected_type
        original_item_str = None
        for item in items:
            if item is not None:
                item_type = item.dtype.base_dtype
                if not expected_type:
                    expected_type = item_type
                    original_item_str = item.name if hasattr(item, 'name') else str(item)
                elif expected_type != item_type:
                    raise ValueError('%s, type=%s, must be of the same type (%s)%s.' % (item.name if hasattr(item, 'name') else str(item), item_type, expected_type, ' as %s' % original_item_str if original_item_str else ''))
        return expected_type
    else:
        return expected_type

@tf_export('debugging.assert_same_float_dtype', v1=['debugging.assert_same_float_dtype', 'assert_same_float_dtype'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_same_float_dtype')
def assert_same_float_dtype(tensors=None, dtype=None):
    if False:
        for i in range(10):
            print('nop')
    'Validate and return float type based on `tensors` and `dtype`.\n\n  For ops such as matrix multiplication, inputs and weights must be of the\n  same float type. This function validates that all `tensors` are the same type,\n  validates that type is `dtype` (if supplied), and returns the type. Type must\n  be a floating point type. If neither `tensors` nor `dtype` is supplied,\n  the function will return `dtypes.float32`.\n\n  Args:\n    tensors: Tensors of input values. Can include `None` elements, which will be\n        ignored.\n    dtype: Expected type.\n\n  Returns:\n    Validated type.\n\n  Raises:\n    ValueError: if neither `tensors` nor `dtype` is supplied, or result is not\n        float, or the common type of the inputs is not a floating point type.\n  '
    if tensors:
        dtype = _assert_same_base_type(tensors, dtype)
    if not dtype:
        dtype = dtypes.float32
    elif not dtype.is_floating:
        raise ValueError('Expected floating point type, got %s.' % dtype)
    return dtype

@tf_export('debugging.assert_scalar', v1=[])
@dispatch.add_dispatch_support
def assert_scalar_v2(tensor, message=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Asserts that the given `tensor` is a scalar.\n\n  This function raises `ValueError` unless it can be certain that the given\n  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is\n  unknown.\n\n  This is always checked statically, so this method returns nothing.\n\n  Args:\n    tensor: A `Tensor`.\n    message: A string to prefix to the default message.\n    name:  A name for this operation. Defaults to "assert_scalar"\n\n  Raises:\n    ValueError: If the tensor is not scalar (rank 0), or if its shape is\n      unknown.\n  '
    assert_scalar(tensor=tensor, message=message, name=name)

@tf_export(v1=['debugging.assert_scalar', 'assert_scalar'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('assert_scalar')
def assert_scalar(tensor, name=None, message=None):
    if False:
        i = 10
        return i + 15
    'Asserts that the given `tensor` is a scalar (i.e. zero-dimensional).\n\n  This function raises `ValueError` unless it can be certain that the given\n  `tensor` is a scalar. `ValueError` is also raised if the shape of `tensor` is\n  unknown.\n\n  Args:\n    tensor: A `Tensor`.\n    name:  A name for this operation. Defaults to "assert_scalar"\n    message: A string to prefix to the default message.\n\n  Returns:\n    The input tensor (potentially converted to a `Tensor`).\n\n  Raises:\n    ValueError: If the tensor is not scalar (rank 0), or if its shape is\n      unknown.\n  '
    with ops.name_scope(name, 'assert_scalar', [tensor]) as name_scope:
        tensor = ops.convert_to_tensor(tensor, name=name_scope)
        shape = tensor.get_shape()
        message = _message_prefix(message)
        if shape.ndims != 0:
            if context.executing_eagerly():
                raise ValueError('%sExpected scalar shape, saw shape: %s.' % (message, shape))
            else:
                raise ValueError('%sExpected scalar shape for %s, saw shape: %s.' % (message, tensor.name, shape))
        return tensor

def _message_prefix(message):
    if False:
        while True:
            i = 10
    if message:
        return '%s.  ' % message
    return ''

@tf_export('ensure_shape')
@dispatch.add_dispatch_support
def ensure_shape(x, shape, name=None):
    if False:
        i = 10
        return i + 15
    'Updates the shape of a tensor and checks at runtime that the shape holds.\n\n  When executed, this operation asserts that the input tensor `x`\'s shape\n  is compatible with the `shape` argument.\n  See `tf.TensorShape.is_compatible_with` for details.\n\n  >>> x = tf.constant([[1, 2, 3],\n  ...                  [4, 5, 6]])\n  >>> x = tf.ensure_shape(x, [2, 3])\n\n  Use `None` for unknown dimensions:\n\n  >>> x = tf.ensure_shape(x, [None, 3])\n  >>> x = tf.ensure_shape(x, [2, None])\n\n  If the tensor\'s shape is not compatible with the `shape` argument, an error\n  is raised:\n\n  >>> x = tf.ensure_shape(x, [5])\n  Traceback (most recent call last):\n  ...\n  tf.errors.InvalidArgumentError: Shape of tensor dummy_input [3] is not\n    compatible with expected shape [5]. [Op:EnsureShape]\n\n  During graph construction (typically tracing a `tf.function`),\n  `tf.ensure_shape` updates the static-shape of the **result** tensor by\n  merging the two shapes. See `tf.TensorShape.merge_with` for details.\n\n  This is most useful when **you** know a shape that can\'t be determined\n  statically by TensorFlow.\n\n  The following trivial `tf.function` prints the input tensor\'s\n  static-shape before and after `ensure_shape` is applied.\n\n  >>> @tf.function\n  ... def f(tensor):\n  ...   print("Static-shape before:", tensor.shape)\n  ...   tensor = tf.ensure_shape(tensor, [None, 3])\n  ...   print("Static-shape after:", tensor.shape)\n  ...   return tensor\n\n  This lets you see the effect of `tf.ensure_shape` when the function is traced:\n  >>> cf = f.get_concrete_function(tf.TensorSpec([None, None]))\n  Static-shape before: (None, None)\n  Static-shape after: (None, 3)\n\n  >>> cf(tf.zeros([3, 3])) # Passes\n  >>> cf(tf.constant([1, 2, 3])) # fails\n  Traceback (most recent call last):\n  ...\n  InvalidArgumentError:  Shape of tensor x [3] is not compatible with expected shape [3,3].\n\n  The above example raises `tf.errors.InvalidArgumentError`, because `x`\'s\n  shape, `(3,)`, is not compatible with the `shape` argument, `(None, 3)`\n\n  Inside a `tf.function` or `v1.Graph` context it checks both the buildtime and\n  runtime shapes. This is stricter than `tf.Tensor.set_shape` which only\n  checks the buildtime shape.\n\n  Note: This differs from `tf.Tensor.set_shape` in that it sets the static shape\n  of the resulting tensor and enforces it at runtime, raising an error if the\n  tensor\'s runtime shape is incompatible with the specified shape.\n  `tf.Tensor.set_shape` sets the static shape of the tensor without enforcing it\n  at runtime, which may result in inconsistencies between the statically-known\n  shape of tensors and the runtime value of tensors.\n\n  For example, of loading images of a known size:\n\n  >>> @tf.function\n  ... def decode_image(png):\n  ...   image = tf.image.decode_png(png, channels=3)\n  ...   # the `print` executes during tracing.\n  ...   print("Initial shape: ", image.shape)\n  ...   image = tf.ensure_shape(image,[28, 28, 3])\n  ...   print("Final shape: ", image.shape)\n  ...   return image\n\n  When tracing a function, no ops are being executed, shapes may be unknown.\n  See the [Concrete Functions Guide](https://www.tensorflow.org/guide/concrete_function)\n  for details.\n\n  >>> concrete_decode = decode_image.get_concrete_function(\n  ...     tf.TensorSpec([], dtype=tf.string))\n  Initial shape:  (None, None, 3)\n  Final shape:  (28, 28, 3)\n\n  >>> image = tf.random.uniform(maxval=255, shape=[28, 28, 3], dtype=tf.int32)\n  >>> image = tf.cast(image,tf.uint8)\n  >>> png = tf.image.encode_png(image)\n  >>> image2 = concrete_decode(png)\n  >>> print(image2.shape)\n  (28, 28, 3)\n\n  >>> image = tf.concat([image,image], axis=0)\n  >>> print(image.shape)\n  (56, 28, 3)\n  >>> png = tf.image.encode_png(image)\n  >>> image2 = concrete_decode(png)\n  Traceback (most recent call last):\n  ...\n  tf.errors.InvalidArgumentError:  Shape of tensor DecodePng [56,28,3] is not\n    compatible with expected shape [28,28,3].\n\n  Caution: if you don\'t use the result of `tf.ensure_shape` the check may not\n  run.\n\n  >>> @tf.function\n  ... def bad_decode_image(png):\n  ...   image = tf.image.decode_png(png, channels=3)\n  ...   # the `print` executes during tracing.\n  ...   print("Initial shape: ", image.shape)\n  ...   # BAD: forgot to use the returned tensor.\n  ...   tf.ensure_shape(image,[28, 28, 3])\n  ...   print("Final shape: ", image.shape)\n  ...   return image\n\n  >>> image = bad_decode_image(png)\n  Initial shape:  (None, None, 3)\n  Final shape:  (None, None, 3)\n  >>> print(image.shape)\n  (56, 28, 3)\n\n  Args:\n    x: A `Tensor`.\n    shape: A `TensorShape` representing the shape of this tensor, a\n      `TensorShapeProto`, a list, a tuple, or None.\n    name: A name for this operation (optional). Defaults to "EnsureShape".\n\n  Returns:\n    A `Tensor`. Has the same type and contents as `x`.\n\n  Raises:\n    tf.errors.InvalidArgumentError: If `shape` is incompatible with the shape\n    of `x`.\n  '
    if not isinstance(shape, tensor_shape.TensorShape):
        shape = tensor_shape.TensorShape(shape)
    return array_ops.ensure_shape(x, shape, name=name)

@ops.RegisterGradient('EnsureShape')
def _ensure_shape_grad(op, grad):
    if False:
        while True:
            i = 10
    del op
    return grad