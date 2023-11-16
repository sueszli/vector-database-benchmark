"""A wrapper for gen_multiplex_3_op.py.

This defines a public API and provides a docstring for the C++ Op defined by
multiplex_3_kernel.cc
"""
import tensorflow as tf
from tensorflow.examples.custom_ops_doc.multiplex_2.multiplex_2_op import examples_multiplex_dense
from tensorflow.python.platform import resource_loader
_multiplex_3_module = tf.load_op_library(resource_loader.get_path_to_datafile('multiplex_3_kernel.so'))
examples_multiplex_sparse = _multiplex_3_module.examples_multiplex_sparse

@tf.experimental.dispatch_for_api(examples_multiplex_dense)
def multiplex_sparse(cond: tf.SparseTensor, a: tf.SparseTensor, b: tf.SparseTensor, name=None):
    if False:
        i = 10
        return i + 15
    "Return elements chosen from `a` or `b` depending on `cond`.\n\n\n  This is similar to `np.where` and `tf.where`, but simplified to only handle\n  the case of rank 1 sparse tensors, no optional parameters, no broadcasting,\n  etc..\n\n  >>> cond = tf.SparseTensor(\n  ...     indices=[[1], [3], [6]], values=[True, False, True], dense_shape=[7])\n  >>> a = tf.sparse.from_dense(['', 'a0', '', 'a1', '', 'a2', ''])\n  >>> b = tf.sparse.from_dense(['b0', '', 'b1', 'b2', '', '', 'b3'])\n  >>> multiplex_3_op.multiplex_sparse(cond, a, b)\n  SparseTensorValue(indices=array([[0],\n    [1],\n    [2],\n    [3]]), values=array([b'b0', b'a0', b'b1', b'b2'], dtype=object),\n    dense_shape=array([7]))\n  Args:\n    cond: tf.SparseTensor of type bool. Where True, yield `a`, otherwise yield\n      `b`.\n    a: tf.SparseTensor with the same type and shape as `b`.\n    b: tf.SparseTensor with the same type and shape as `a`.\n    name: An optional name for the op.\n\n  Returns:\n    A tf.SparseTensor with elements from `a` where `cond` is True, and elements\n    from `b` elsewhere.\n  "
    (indices, values, shape) = examples_multiplex_sparse(cond_indices=cond.indices, cond_values=cond.values, cond_shape=cond.dense_shape, a_indices=a.indices, a_values=a.values, a_shape=a.dense_shape, b_indices=b.indices, b_values=b.values, b_shape=b.dense_shape, name=name)
    return tf.SparseTensor(indices, values, shape)