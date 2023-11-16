"""Utils for supporting the DRAGNN runtime from the TF side."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import re
import tensorflow as tf
from dragnn.python import network_units
from syntaxnet.util import check

def add_hooks(component, cell_subgraph_spec):
    if False:
        print('Hello World!')
    'Adds "hook" nodes to the graph, for use by the runtime.\n\n  The runtime hook nodes are not on the path to any required output, and will\n  not be called when running TF-based DRAGNN.  As long as the TF graph is not\n  pruned, however, the DRAGNN runtime can call them.\n\n  Runtime hook nodes can perform any TF computation.  Possible uses include:\n    * Applying stable names to existing tensors (e.g., via tf.identity()).\n    * Converting variable data from a TF-friendly or training-friendly format\n      into a runtime-friendly format.\n\n  NB: There are several restrictions on the context in which this function is\n  called.  In brief, call ComponentBuilderBase._add_runtime_hooks() at the top\n  of each ComponentBuilderSubclass.build_*() method.  In detail, this:\n    * Must be called in the variable scope of the |component|, so variable\n      references in component.get_variable() work.\n    * Must be called, possibly transitively, from one of the |component|\'s\n      build_*() methods, so MasterBuilder.read_from_avg is set properly for\n      component.get_variable().\n    * Must not be called from within a tf.while_loop(), or the hook nodes will\n      not work.  In particular, NetworkUnitInterface.create() is called from a\n      tf.while_loop() in DynamicComponentBuilder.\n\n  Args:\n    component: Component for which to add hooks.\n    cell_subgraph_spec: CellSubgraphSpec for which to add hooks.\n  '
    for (channel_id, feature_spec) in enumerate(component.spec.linked_feature):
        if feature_spec.embedding_dim != -1:
            _add_hooks_for_linked_embedding_matrix(component, channel_id)
    for (channel_id, feature_spec) in enumerate(component.spec.fixed_feature):
        if feature_spec.embedding_dim != -1:
            _add_hooks_for_fixed_embedding_matrix(component, channel_id)
    for params in component.network.params:
        _add_hooks_for_trainable_params(component, params)
    for parameter_getter in component.network.derived_params:
        _add_hooks_for_derived_parameter(parameter_getter)
    _add_hook_node(tf.constant(cell_subgraph_spec.SerializeToString(), tf.string), '{}/EXPORT/CellSubgraphSpec'.format(component.name))

def _blocked_and_dtype_transformations(tensor):
    if False:
        print('Hello World!')
    'Yields variants of a tensor, for standard blocking/dtype variants.\n\n  Args:\n    tensor (tf.Tensor): Input tensor.\n\n  Yields:\n    (modified_tensor, suffix) pairs, where `modified_tensor` is a transformed\n    version of the input, and `suffix` is a string like "/blocked32".\n  '
    for blocking_level in (32, 48):
        blocked = make_padded_blocked_matrix(tensor, blocking_level)
        bfloat16_blocked = tf.to_bfloat16(bfloat16_permutation(blocked))
        yield (blocked, '/blocked{}'.format(blocking_level))
        yield (bfloat16_blocked, '/blocked{}/bfloat16'.format(blocking_level))

def _add_hooks_for_linked_embedding_matrix(component, channel_id):
    if False:
        while True:
            i = 10
    "Adds runtime hooks for a linked embedding matrix.\n\n  The computation performed by network_units.pass_through_embedding_matrix() is\n  equivalent to the following:\n\n    for i in range(stride):\n      if step_idx[i] == -1:\n        outputs[i,:] = out_of_bounds_vector\n      else:\n        outputs[i,:] = tf.matmul(act_block[i,:], weight_matrix)\n\n  The implementation uses clever arithmetic to do this in one matmul per batch.\n  Specifically, the weight_matrix is extended with the out_of_bounds_vector and\n  each activation vector is extended with a 0/1 out-of-bounds indicator.  Then,\n  multiplying the two suffices, assuming that act_block[i,:] is set to zero for\n  out-of-bounds links.\n\n  While this works well for training and high-throughput batched computation, it\n  isn't the best for the runtime:\n    * Appending a 0/1 indicator to the input activation vector requires a copy.\n      Ideally, we could use the input activation vector by reference alone.\n    * In order to access to the |out_of_bounds_vector| as a contiguous array,\n      the runtime must load the linked embedding matrix in row-major format,\n      which may not be the fastest format for arithmetic.\n    * The dimensions of the extended-by-1 matrix and vector are likely to be\n      pessimal.  Most dimensions are specified as 2^n, and adding one element\n      produces maximal padding on the trailing elements, which in turn wastes\n      memory, reduces cache utilization, etc.\n\n  Therefore, in the runtime we split the linked embedding matrix into a separate\n  weight matrix and out-of-bounds vector.\n\n  Args:\n    component: Component for which to add hooks.\n    channel_id: Linked embedding channel for which to add hooks.\n  "
    var_name = network_units.linked_embeddings_name(channel_id)
    extended_matrix = component.get_variable(var_name)
    extended_num_rows = tf.shape(extended_matrix)[0]
    (matrix, vector) = tf.split(extended_matrix, [extended_num_rows - 1, 1], 0)
    transposed = tf.transpose(matrix)
    hook_name = functools.partial(_get_hook_name, component, var_name)
    _add_hook_node(matrix, hook_name('/weights'))
    _add_hook_node(transposed, hook_name('/weights/transposed'))
    for (blocked, blocked_suffix) in _blocked_and_dtype_transformations(matrix):
        blocked_name = hook_name('/weights/matrix' + blocked_suffix)
        _add_hook_node(blocked, blocked_name)
    for (blocked, blocked_suffix) in _blocked_and_dtype_transformations(transposed):
        blocked_name = hook_name('/weights/transposed' + blocked_suffix)
        _add_hook_node(blocked, blocked_name)
    _add_hook_node(tf.shape(transposed), hook_name('/weights/transposed/shape'))
    _add_hook_node(vector, _get_hook_name(component, var_name, '/out_of_bounds'))

def _add_hooks_for_fixed_embedding_matrix(component, channel_id):
    if False:
        for i in range(10):
            print('nop')
    'Adds runtime hooks for a fixed embedding matrix.\n\n  The hooks remove the last row from the embedding matrix.  The extra row was\n  probably intended for out-of-vocabulary items, but those are handled in the\n  feature system and the extra row is never used.\n\n  Args:\n    component: Component for which to add hooks.\n    channel_id: Fixed embedding channel for which to add hooks.\n  '
    var_name = network_units.fixed_embeddings_name(channel_id)
    extended_matrix = component.get_variable(var_name)
    extended_num_rows = tf.shape(extended_matrix)[0]
    matrix = tf.slice(extended_matrix, [0, 0], [extended_num_rows - 1, -1])
    _add_hook_node(matrix, _get_hook_name(component, var_name, '/trimmed'))

def _add_hooks_for_derived_parameter(getter):
    if False:
        i = 10
        return i + 15
    'Adds hooks for derived parameters.\n\n  Derived parameters are typically slight format modifications of regular\n  parameters, exposed because doing the computation in Python is more convenient\n  than as VariableStore wrappers.\n\n  Args:\n    getter: Function which, when called, will return the derived tensor.\n  '
    parameter = getter()
    full_name = parameter.op.name

    def _hook_name(base_name):
        if False:
            print('Hello World!')
        'Returns a hook node name constructed from a base name.'
        return full_name + base_name
    if parameter.shape.ndims != 2:
        tf.logging.info('Not adding matrix hooks for derived parameter %s', full_name)
        return
    _add_hook_node(tf.transpose(parameter), _hook_name('/transposed'))
    for (blocked, blocked_suffix) in _blocked_and_dtype_transformations(parameter):
        _add_hook_node(blocked, _hook_name('/matrix' + blocked_suffix))

def _add_hooks_for_trainable_params(component, params):
    if False:
        while True:
            i = 10
    'Adds runtime hooks for a variable of trainable parameters.\n\n  Ignores parameters that are not statically-deducible as matrices.\n\n  Args:\n    component: Component for which to add hooks.\n    params: Variable for which to add hooks.\n  '
    full_name = params.op.name
    matrix = component.get_variable(var_params=params)
    if params.shape.ndims != 2:
        tf.logging.info('Not adding hooks for trainable params %s', full_name)
        return
    suffix = re.sub('^' + re.escape(full_name), '', matrix.op.name)
    check.Ne(suffix, matrix.op.name, 'Failed to find suffix for params %s' % full_name)

    def _hook_name(base_name):
        if False:
            return 10
        'Returns a hook node name constructed from a base name.'
        return full_name + base_name + suffix
    transposed = tf.transpose(matrix)
    _add_hook_node(matrix, _hook_name('/matrix'))
    _add_hook_node(transposed, _hook_name('/transposed'))
    for (blocked, blocked_suffix) in _blocked_and_dtype_transformations(matrix):
        _add_hook_node(blocked, _hook_name('/matrix' + blocked_suffix))
    for (blocked, blocked_suffix) in _blocked_and_dtype_transformations(transposed):
        _add_hook_node(blocked, _hook_name('/transposed' + blocked_suffix))
    _add_hook_node(tf.shape(matrix), _hook_name('/matrix/shape'))
    _add_hook_node(tf.shape(transposed), _hook_name('/transposed/shape'))

def make_padded_blocked_matrix(matrix, block_size):
    if False:
        return 10
    'Converts a matrix to padded column-blocked format.\n\n  For example, given a [64,127] matrix and block_size=16, this function returns\n  an [8,64,16] tensor where the 8 inner sub-matrices, when concatenated left to\n  right, re-constitute the original matrix.  Note that the 8th sub-matrix has a\n  final column of padding.\n\n  Args:\n    matrix: The matrix to convert.\n    block_size: The number of columns per block.\n\n  Returns:\n    Padded column-blocked matrix.\n  '
    shape = tf.shape(matrix)
    num_rows = shape[0]
    num_columns = shape[1]
    last_block_size = num_columns % block_size
    padding_size = (block_size - last_block_size) % block_size
    num_blocks = (num_columns + padding_size) // block_size
    padded = tf.pad(matrix, [[0, 0], [0, padding_size]])
    transposed = tf.transpose(padded)
    blocked = tf.reshape(transposed, [num_blocks, block_size, num_rows])
    return tf.transpose(blocked, [0, 2, 1])

def bfloat16_permutation(tensor):
    if False:
        i = 10
        return i + 15
    "Permutes values in the last dimension of a tensor.\n\n  This permutation is used so that we can directly use unpacklo/unpackhi AVX2\n  instructions on the matrix coefficients. These unpacking instructions\n  effectively permute the data. See FastUnpackPermutation() and\n  AvxFloatVecArray::Load(const TruncatedFloat16 *) in avx_vector_array.h for\n  more details.\n\n  Args:\n    tensor: Blocked matrix, the result of make_padded_blocked_matrix(). Must\n      have its last dimension a multiple of 16.\n\n  Returns:\n    Permuted matrix, suitable for calling tf.to_bfloat16() on. For testing\n    convenience we don't do so in this method.\n\n  Raises:\n    ValueError: If the matrix's block dimension is not a multiple of 16.\n  "
    orig_shape = tensor.shape
    if tensor.shape[-1] % 16 != 0:
        raise ValueError('Bad block dimension, must be divisible by 16')
    permutation = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
    indices = tf.constant([16 * (i // 16) + permutation[i % 16] for i in xrange(orig_shape[-1])])
    return tf.gather(tensor, indices, axis=len(orig_shape) - 1)

def _get_hook_name(component, variable_name, suffix):
    if False:
        for i in range(10):
            print('nop')
    'Builds the name of a hook node.\n\n  Specifically, the name of the hook node is:\n\n    <component.name>/<variable_name><suffix><remainder>\n\n  where <remainder> is whatever follows <variable_name> in the name of the op\n  that produces the named variable.  Recall that component.get_variable() may\n  return either the original variable or its moving average.  These might have\n  names like:\n\n    foo_component/bar_variable\n    foo_component/bar_variable/ExponentialMovingAverage\n\n  In the examples above, the <remainder> is "" for the original variable and\n  "/ExponentialMovingAverage" for its moving average.  Calling this function\n  with suffix="/baz_suffix" in either case would add hook nodes named:\n\n    foo_component/bar_variable/baz_suffix\n    foo_component/bar_variable/baz_suffix/ExponentialMovingAverage\n\n  Note that the suffix is inserted after the variable name, not necessarily at\n  the end of the entire op name.\n\n  Args:\n    component: Component that the hook node belongs to.\n    variable_name: Variable that the hook node name is based on.\n    suffix: Suffix to append to the variable name.\n\n  Returns:\n    Name of the hook node.\n  '
    variable = component.get_variable(variable_name)
    full_name = variable.op.name
    prefix = component.name + '/' + variable_name
    hook_name = re.sub('^' + re.escape(prefix), prefix + suffix, full_name)
    check.Ne(full_name, hook_name, 'Failed to match expected variable prefix "{}" in variable "{}"'.format(prefix, full_name))
    return hook_name

def _add_hook_node(tensor, fully_qualified_name):
    if False:
        i = 10
        return i + 15
    'Adds a hook node that outputs a tensor with a fully-qualified name.'
    with tf.name_scope(None):
        tf.identity(tensor, name=fully_qualified_name)