"""Functional operations."""
import re
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['map_fn'])
@deprecation.deprecated_args(None, 'Use fn_output_signature instead', 'dtype')
def map_fn(fn, elems, dtype=None, parallel_iterations=None, back_prop=True, swap_memory=False, infer_shape=True, name=None, fn_output_signature=None):
    if False:
        print('Hello World!')
    'Transforms `elems` by applying `fn` to each element unstacked on axis 0.\n\n  See also `tf.scan`.\n\n  `map_fn` unstacks `elems` on axis 0 to obtain a sequence of elements;\n  calls `fn` to transform each element; and then stacks the transformed\n  values back together.\n\n  #### Mapping functions with single-Tensor inputs and outputs\n\n  If `elems` is a single tensor and `fn`\'s signature is `tf.Tensor->tf.Tensor`,\n  then `map_fn(fn, elems)` is equivalent to\n  `tf.stack([fn(elem) for elem in tf.unstack(elems)])`.  E.g.:\n\n  >>> tf.map_fn(fn=lambda t: tf.range(t, t + 3), elems=tf.constant([3, 5, 2]))\n  <tf.Tensor: shape=(3, 3), dtype=int32, numpy=\n    array([[3, 4, 5],\n           [5, 6, 7],\n           [2, 3, 4]], dtype=int32)>\n\n  `map_fn(fn, elems).shape = [elems.shape[0]] + fn(elems[0]).shape`.\n\n  #### Mapping functions with multi-arity inputs and outputs\n\n  `map_fn` also supports functions with multi-arity inputs and outputs:\n\n  * If `elems` is a tuple (or nested structure) of tensors, then those tensors\n    must all have the same outer-dimension size (`num_elems`); and `fn` is\n    used to transform each tuple (or structure) of corresponding slices from\n    `elems`.  E.g., if `elems` is a tuple `(t1, t2, t3)`, then `fn` is used to\n    transform each tuple of slices `(t1[i], t2[i], t3[i])`\n    (where `0 <= i < num_elems`).\n\n  * If `fn` returns a tuple (or nested structure) of tensors, then the\n    result is formed by stacking corresponding elements from those structures.\n\n  #### Specifying `fn`\'s output signature\n\n  If `fn`\'s input and output signatures are different, then the output\n  signature must be specified using `fn_output_signature`.  (The input and\n  output signatures are differ if their structures, dtypes, or tensor types do\n  not match).  E.g.:\n\n  >>> tf.map_fn(fn=tf.strings.length,  # input & output have different dtypes\n  ...           elems=tf.constant(["hello", "moon"]),\n  ...           fn_output_signature=tf.int32)\n  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([5, 4], dtype=int32)>\n  >>> tf.map_fn(fn=tf.strings.join,  # input & output have different structures\n  ...           elems=[tf.constant([\'The\', \'A\']), tf.constant([\'Dog\', \'Cat\'])],\n  ...           fn_output_signature=tf.string)\n  <tf.Tensor: shape=(2,), dtype=string,\n   numpy=array([b\'TheDog\', b\'ACat\'], dtype=object)>\n\n  `fn_output_signature` can be specified using any of the following:\n\n  * A `tf.DType` or `tf.TensorSpec` (to describe a `tf.Tensor`)\n  * A `tf.RaggedTensorSpec` (to describe a `tf.RaggedTensor`)\n  * A `tf.SparseTensorSpec` (to describe a `tf.sparse.SparseTensor`)\n  * A (possibly nested) tuple, list, or dict containing the above types.\n\n  #### RaggedTensors\n\n  `map_fn` supports `tf.RaggedTensor` inputs and outputs.  In particular:\n\n  * If `elems` is a `RaggedTensor`, then `fn` will be called with each\n    row of that ragged tensor.\n    * If `elems` has only one ragged dimension, then the values passed to\n      `fn` will be `tf.Tensor`s.\n    * If `elems` has multiple ragged dimensions, then the values passed to\n      `fn` will be `tf.RaggedTensor`s with one fewer ragged dimension.\n\n  * If the result of `map_fn` should be a `RaggedTensor`, then use a\n    `tf.RaggedTensorSpec` to specify `fn_output_signature`.\n    * If `fn` returns `tf.Tensor`s with varying sizes, then use a\n      `tf.RaggedTensorSpec` with `ragged_rank=0` to combine them into a\n      single ragged tensor (which will have ragged_rank=1).\n    * If `fn` returns `tf.RaggedTensor`s, then use a `tf.RaggedTensorSpec`\n      with the same `ragged_rank`.\n\n  >>> # Example: RaggedTensor input\n  >>> rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])\n  >>> tf.map_fn(tf.reduce_sum, rt, fn_output_signature=tf.int32)\n  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([6, 0, 9, 6], dtype=int32)>\n\n  >>> # Example: RaggedTensor output\n  >>> elems = tf.constant([3, 5, 0, 2])\n  >>> tf.map_fn(tf.range, elems,\n  ...           fn_output_signature=tf.RaggedTensorSpec(shape=[None],\n  ...                                                   dtype=tf.int32))\n  <tf.RaggedTensor [[0, 1, 2], [0, 1, 2, 3, 4], [], [0, 1]]>\n\n  Note: `map_fn` should only be used if you need to map a function over the\n  *rows* of a `RaggedTensor`.  If you wish to map a function over the\n  individual values, then you should use:\n\n  * `tf.ragged.map_flat_values(fn, rt)`\n    (if fn is expressible as TensorFlow ops)\n  * `rt.with_flat_values(map_fn(fn, rt.flat_values))`\n    (otherwise)\n\n  E.g.:\n\n  >>> rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])\n  >>> tf.ragged.map_flat_values(lambda x: x + 2, rt)\n  <tf.RaggedTensor [[3, 4, 5], [], [6, 7], [8]]>\n\n  #### SparseTensors\n\n  `map_fn` supports `tf.sparse.SparseTensor` inputs and outputs.  In particular:\n\n  * If `elems` is a `SparseTensor`, then `fn` will be called with each row\n    of that sparse tensor. In particular, the value passed to `fn` will be a\n    `tf.sparse.SparseTensor` with one fewer dimension than `elems`.\n\n  * If the result of `map_fn` should be a `SparseTensor`, then use a\n    `tf.SparseTensorSpec` to specify `fn_output_signature`.  The individual\n    `SparseTensor`s returned by `fn` will be stacked into a single\n    `SparseTensor` with one more dimension.\n\n  >>> # Example: SparseTensor input\n  >>> st = tf.sparse.SparseTensor([[0, 0], [2, 0], [2, 1]], [2, 3, 4], [4, 4])\n  >>> tf.map_fn(tf.sparse.reduce_sum, st, fn_output_signature=tf.int32)\n  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([2, 0, 7, 0], dtype=int32)>\n\n  >>> # Example: SparseTensor output\n  >>> tf.sparse.to_dense(\n  ...     tf.map_fn(tf.sparse.eye, tf.constant([2, 3]),\n  ...               fn_output_signature=tf.SparseTensorSpec(None, tf.float32)))\n  <tf.Tensor: shape=(2, 3, 3), dtype=float32, numpy=\n    array([[[1., 0., 0.],\n            [0., 1., 0.],\n            [0., 0., 0.]],\n           [[1., 0., 0.],\n            [0., 1., 0.],\n            [0., 0., 1.]]], dtype=float32)>\n\n  Note: `map_fn` should only be used if you need to map a function over the\n  *rows* of a `SparseTensor`.  If you wish to map a function over the nonzero\n  values, then you should use:\n\n  * If the function is expressible as TensorFlow ops, use:\n    ```python\n    tf.sparse.SparseTensor(st.indices, fn(st.values), st.dense_shape)\n    ```\n  * Otherwise, use:\n    ```python\n    tf.sparse.SparseTensor(st.indices, tf.map_fn(fn, st.values),\n                           st.dense_shape)\n    ```\n\n  #### `map_fn` vs. vectorized operations\n\n  `map_fn` will apply the operations used by `fn` to each element of `elems`,\n  resulting in `O(elems.shape[0])` total operations.  This is somewhat\n  mitigated by the fact that `map_fn` can process elements in parallel.\n  However, a transform expressed using `map_fn` is still typically less\n  efficient than an equivalent transform expressed using vectorized operations.\n\n  `map_fn` should typically only be used if one of the following is true:\n\n  * It is difficult or expensive to express the desired transform with\n    vectorized operations.\n  * `fn` creates large intermediate values, so an equivalent vectorized\n    transform would take too much memory.\n  * Processing elements in parallel is more efficient than an equivalent\n    vectorized transform.\n  * Efficiency of the transform is not critical, and using `map_fn` is\n    more readable.\n\n  E.g., the example given above that maps `fn=lambda t: tf.range(t, t + 3)`\n  across `elems` could be rewritten more efficiently using vectorized ops:\n\n  >>> elems = tf.constant([3, 5, 2])\n  >>> tf.range(3) + tf.expand_dims(elems, 1)\n  <tf.Tensor: shape=(3, 3), dtype=int32, numpy=\n    array([[3, 4, 5],\n           [5, 6, 7],\n           [2, 3, 4]], dtype=int32)>\n\n  In some cases, `tf.vectorized_map` can be used to automatically convert a\n  function to a vectorized equivalent.\n\n  #### Eager execution\n\n  When executing eagerly, `map_fn` does not execute in parallel even if\n  `parallel_iterations` is set to a value > 1. You can still get the\n  performance benefits of running a function in parallel by using the\n  `tf.function` decorator:\n\n  >>> fn=lambda t: tf.range(t, t + 3)\n  >>> @tf.function\n  ... def func(elems):\n  ...   return tf.map_fn(fn, elems, parallel_iterations=3)\n  >>> func(tf.constant([3, 5, 2]))\n  <tf.Tensor: shape=(3, 3), dtype=int32, numpy=\n    array([[3, 4, 5],\n           [5, 6, 7],\n           [2, 3, 4]], dtype=int32)>\n\n\n  Note: if you use the `tf.function` decorator, any non-TensorFlow Python\n  code that you may have written in your function won\'t get executed. See\n  `tf.function` for more  details. The recommendation would be to debug without\n  `tf.function` but switch to it to get performance benefits of running `map_fn`\n  in parallel.\n\n  Args:\n    fn: The callable to be performed.  It accepts one argument, which will have\n      the same (possibly nested) structure as `elems`.  Its output must have the\n      same structure as `fn_output_signature` if one is provided; otherwise it\n      must have the same structure as `elems`.\n    elems: A tensor or (possibly nested) sequence of tensors, each of which will\n      be unstacked along their first dimension.  `fn` will be applied to the\n      nested sequence of the resulting slices.  `elems` may include ragged and\n      sparse tensors. `elems` must consist of at least one tensor.\n    dtype: Deprecated: Equivalent to `fn_output_signature`.\n    parallel_iterations: (optional) The number of iterations allowed to run in\n      parallel. When graph building, the default value is 10. While executing\n      eagerly, the default value is set to 1.\n    back_prop: (optional) False disables support for back propagation.\n    swap_memory: (optional) True enables GPU-CPU memory swapping.\n    infer_shape: (optional) False disables tests for consistent output shapes.\n    name: (optional) Name prefix for the returned tensors.\n    fn_output_signature: The output signature of `fn`. Must be specified if\n      `fn`\'s input and output signatures are different (i.e., if their\n      structures, dtypes, or tensor types do not match).\n      `fn_output_signature` can be specified using any of the following:\n\n      * A `tf.DType` or `tf.TensorSpec` (to describe a `tf.Tensor`)\n      * A `tf.RaggedTensorSpec` (to describe a `tf.RaggedTensor`)\n      * A `tf.SparseTensorSpec` (to describe a `tf.sparse.SparseTensor`)\n      * A (possibly nested) tuple, list, or dict containing the above types.\n\n  Returns:\n    A tensor or (possibly nested) sequence of tensors.  Each tensor stacks the\n    results of applying `fn` to tensors unstacked from `elems` along the first\n    dimension, from first to last.  The result may include ragged and sparse\n    tensors.\n\n  Raises:\n    TypeError: if `fn` is not callable or the structure of the output of\n      `fn` and `fn_output_signature` do not match.\n    ValueError: if the lengths of the output of `fn` and `fn_output_signature`\n      do not match, or if the `elems` does not contain any tensor.\n\n  Examples:\n\n    >>> elems = np.array([1, 2, 3, 4, 5, 6])\n    >>> tf.map_fn(lambda x: x * x, elems)\n    <tf.Tensor: shape=(6,), dtype=int64, numpy=array([ 1,  4,  9, 16, 25, 36])>\n\n    >>> elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))\n    >>> tf.map_fn(lambda x: x[0] * x[1], elems, fn_output_signature=tf.int64)\n    <tf.Tensor: shape=(3,), dtype=int64, numpy=array([-1,  2, -3])>\n\n    >>> elems = np.array([1, 2, 3])\n    >>> tf.map_fn(lambda x: (x, -x), elems,\n    ...          fn_output_signature=(tf.int64, tf.int64))\n    (<tf.Tensor: shape=(3,), dtype=int64, numpy=array([1, 2, 3])>,\n     <tf.Tensor: shape=(3,), dtype=int64, numpy=array([-1, -2, -3])>)\n  '
    if fn_output_signature is None:
        fn_output_signature = dtype
    if not callable(fn):
        raise TypeError(f'The provided function {fn.__name__} is not callable.fn must be callable.')
    in_graph_mode = not context.executing_eagerly()
    if in_graph_mode and (not parallel_iterations):
        parallel_iterations = 10
    elif not in_graph_mode and (not parallel_iterations):
        parallel_iterations = 1
    elif not in_graph_mode and parallel_iterations > 1:
        logging.log_first_n(logging.WARN, 'Setting parallel_iterations > 1 has no effect when executing eagerly. Consider calling map_fn with tf.function to execute fn in parallel.', 1)
        parallel_iterations = 1
    elems = variable_utils.convert_variables_to_tensors(elems)
    elems_flat = nest.flatten(elems)
    if len(elems_flat) == 0:
        raise ValueError('elems must be a Tensor or (possibly nested) sequence of Tensors. Got {}, which does not contain any Tensors.'.format(elems))
    elems_flat_signature = [type_spec.type_spec_from_value(e) for e in elems_flat]
    elems_unflatten = lambda x: nest.pack_sequence_as(elems, x)
    if fn_output_signature is None:
        result_flat_signature = [_most_general_type_spec(e)._unbatch() for e in elems_flat]
        result_unflatten = elems_unflatten
    else:
        result_flat_signature = [_dtype_to_spec(d) for d in nest.flatten(fn_output_signature)]
        result_unflatten = lambda x: nest.pack_sequence_as(fn_output_signature, x)
    with ops.name_scope(name, 'map', elems_flat):
        if in_graph_mode:
            varscope = vs.get_variable_scope()
            varscope_caching_device_was_none = False
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)
                varscope_caching_device_was_none = True
        elems_flat = [ops.convert_to_tensor_or_composite(t, name='elem') for t in elems_flat]
        first_elem = elems_flat[0]
        if hasattr(first_elem, 'shape'):
            elems_static_shape = first_elem.shape
            if elems_static_shape.ndims is not None and elems_static_shape.ndims < 1:
                raise ValueError('Elements in elems must be 1+ dimensional Tensors, not scalars')
        elems_batchable = _elems_flat_to_batchable(elems_flat)
        n_static = tensor_shape.Dimension(tensor_shape.dimension_value(elems_batchable[0].get_shape().with_rank_at_least(1)[0]))
        for tensor in elems_batchable[1:]:
            n_static.assert_is_compatible_with(tensor_shape.Dimension(tensor_shape.dimension_value(tensor.get_shape().with_rank_at_least(1)[0])))
        n = n_static.value or array_ops.shape(elems_batchable[0])[0]
        elems_batchable_ta = [tensor_array_ops.TensorArray(dtype=t.dtype, size=n, dynamic_size=False, infer_shape=True) for t in elems_batchable]
        elems_batchable_ta = [ta.unstack(t) for (ta, t) in zip(elems_batchable_ta, elems_batchable)]
        i = constant_op.constant(0)
        result_batchable_tensor_spec = _result_flat_signature_to_batchable_tensor_spec(result_flat_signature)
        result_batchable_ta = []
        for spec in result_batchable_tensor_spec:
            result_batchable_ta.append(tensor_array_ops.TensorArray(dtype=spec.dtype, size=n, dynamic_size=False, infer_shape=infer_shape, element_shape=spec.shape))

        def compute(i, tas):
            if False:
                while True:
                    i = 10
            "The loop body of map_fn.\n\n      Args:\n        i: the loop counter\n        tas: the flat TensorArray accumulator list\n\n      Returns:\n        (i + 1, tas): the updated counter + updated TensorArrays\n\n      Raises:\n        TypeError: if fn_output_signature and result_value structure don't match\n        ValueType: if fn_output_signature and result_value lengths don't match\n      "
            elems_value_batchable = [ta.read(i) for ta in elems_batchable_ta]
            elems_value_flat = _elems_value_batchable_to_flat(elems_value_batchable, elems_flat_signature)
            elems_value = elems_unflatten(elems_value_flat)
            ag_ctx = autograph_ctx.control_status_ctx()
            autographed_fn = autograph.tf_convert(fn, ag_ctx)
            result_value = autographed_fn(elems_value)
            nest.assert_same_structure(fn_output_signature or elems, result_value)
            result_value_flat = nest.flatten(result_value)
            result_value_batchable = _result_value_flat_to_batchable(result_value_flat, result_flat_signature)
            tas = [ta.write(i, value) for (ta, value) in zip(tas, result_value_batchable)]
            return (i + 1, tas)
        (_, r_a) = while_loop.while_loop(lambda i, _: i < n, compute, (i, result_batchable_ta), parallel_iterations=parallel_iterations, back_prop=back_prop, swap_memory=swap_memory, maximum_iterations=n)
        result_batchable = [r.stack() for r in r_a]
        for r in result_batchable:
            r.set_shape(tensor_shape.TensorShape(n_static).concatenate(r.get_shape()[1:]))
        if in_graph_mode and varscope_caching_device_was_none:
            varscope.set_caching_device(None)
        result_flat = _result_batchable_to_flat(result_batchable, result_flat_signature, n_static)
        result = result_unflatten(result_flat)
        return result

def _dtype_to_spec(d):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(d, type_spec.TypeSpec):
        d = tensor_spec.TensorSpec(None, d)
    return d

def _most_general_type_spec(elem):
    if False:
        for i in range(10):
            print('nop')
    'Returns the most general TypeSpec for elem.'
    if isinstance(elem, composite_tensor.CompositeTensor):
        try:
            spec = elem._shape_invariant_to_type_spec(tensor_shape.TensorShape(None))
        except NotImplementedError:
            spec = type_spec.type_spec_from_value(elem)
    else:
        spec = type_spec.type_spec_from_value(elem)
        if isinstance(spec, tensor_spec.TensorSpec):
            spec = tensor_spec.TensorSpec(None, spec.dtype)
    return spec

def _result_flat_signature_to_batchable_tensor_spec(result_flat_signature):
    if False:
        while True:
            i = 10
    'Converts result_flat_signature -> result_batchable_tensor_specs.'
    tensor_specs = []
    for spec in result_flat_signature:
        if not isinstance(spec, type_spec.BatchableTypeSpec):
            raise TypeError('map_fn can not generate %s outputs' % (spec,))
        tensor_specs.extend(spec._flat_tensor_specs)
    return tensor_specs

def _elems_flat_to_batchable(elems_flat):
    if False:
        return 10
    'Converts elems_flat -> elems_batchable.'
    elems_batchable = []
    for elems_tensor in elems_flat:
        spec = type_spec.type_spec_from_value(elems_tensor)
        if not isinstance(spec, type_spec.BatchableTypeSpec):
            raise TypeError('map_fn can not consume %s inputs: got %r' % (spec, elems_tensor))
        elems_batchable.extend(spec._to_batched_tensor_list(elems_tensor))
    return elems_batchable

def _elems_value_batchable_to_flat(elems_value_batchable, elems_flat_signature):
    if False:
        for i in range(10):
            print('nop')
    'Converts elems_value_batchable -> elems_value_flat.'
    elems_value_flat = []
    i = 0
    for spec in elems_flat_signature:
        spec = spec._unbatch()
        tensor_list = elems_value_batchable[i:i + len(spec._flat_tensor_specs)]
        elems_value_flat.append(spec._from_compatible_tensor_list(tensor_list))
        i += len(tensor_list)
    assert i == len(elems_value_batchable)
    return elems_value_flat

def _result_value_flat_to_batchable(result_value_flat, result_flat_signature):
    if False:
        for i in range(10):
            print('nop')
    'Converts result_value_flat -> result_value_batchable.'
    result_value_batchable = []
    for (r_value, r_spec) in zip(result_value_flat, result_flat_signature):
        if isinstance(r_spec, tensor_spec.TensorSpec):
            result_value_batchable.append(r_value)
        else:
            if not r_spec.is_compatible_with(r_value):
                raise ValueError('Error in map_fn:\n  Expected `fn` to return a:\n    %s\n  But it returned a:\n    %s\n    (value=%s)\n  To fix, update the `fn_output_signature` (or `dtype`) argument to `map_fn`.' % (r_spec, type_spec.type_spec_from_value(r_value), r_value))
            result_value_batchable.extend(r_spec._to_tensor_list(r_value))
    return result_value_batchable

def _result_batchable_to_flat(result_batchable, result_flat_signature, batch_size):
    if False:
        for i in range(10):
            print('nop')
    'Converts result_batchable -> result_flat.'
    result_flat = []
    i = 0
    for spec in result_flat_signature:
        num_tensors = len(spec._flat_tensor_specs)
        result_flat.append(spec._batch(batch_size)._from_compatible_tensor_list(result_batchable[i:i + num_tensors]))
        i += num_tensors
    assert i == len(result_batchable)
    return result_flat

@tf_export('map_fn', v1=[])
@deprecation.deprecated_arg_values(None, 'back_prop=False is deprecated. Consider using tf.stop_gradient instead.\nInstead of:\nresults = tf.map_fn(fn, elems, back_prop=False)\nUse:\nresults = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))', warn_once=True, back_prop=False)
@deprecation.deprecated_args(None, 'Use fn_output_signature instead', 'dtype')
def map_fn_v2(fn, elems, dtype=None, parallel_iterations=None, back_prop=True, swap_memory=False, infer_shape=True, name=None, fn_output_signature=None):
    if False:
        i = 10
        return i + 15
    'Transform `elems` by applying `fn` to each element unstacked on axis 0.'
    if fn_output_signature is None:
        fn_output_signature = dtype
    return map_fn(fn=fn, elems=elems, fn_output_signature=fn_output_signature, parallel_iterations=parallel_iterations, back_prop=back_prop, swap_memory=swap_memory, infer_shape=infer_shape, name=name)
map_fn_v2.__doc__ = re.sub('(  back_prop: \\(optional\\) )(.*)', '\\1Deprecated: prefer using `tf.stop_gradient` instead.  \\2', map_fn.__doc__)
assert 'prefer using `tf.stop_gradient` instead' in map_fn_v2.__doc__