"""A wrapper for gen_multiplex_4_op.py.

This defines a public API (and provides a docstring for it) for the C++ Op
defined by multiplex_4_kernel.cc
"""
import tensorflow as tf
from tensorflow.python.platform import resource_loader
_multiplex_4_module = tf.load_op_library(resource_loader.get_path_to_datafile('multiplex_4_kernel.so'))
examples_multiplex_dense = _multiplex_4_module.examples_multiplex_dense

def multiplex(cond, a, b, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Return elements chosen from `a` or `b` depending on `cond`.\n\n  This is similar to `np.where` and `tf.where` if `cond` and `a` are tensors.\n  This is similar to `np.select` if `cond` and `a` are lists of tensors.\n  In either case, this is simplified to only handle the case of dense tensors,\n  no optional parameters, no broadcasting, etc..\n\n  >>> multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])\n  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)>\n\n  >>> a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)\n  >>> a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)\n  >>> a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)\n  >>> b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)\n  >>> cond1 = tf.constant([False, False, True, False, False], dtype=bool)\n  >>> cond2 = tf.constant([False, False, False, False, True], dtype=bool)\n  >>> cond3 = tf.constant([True, False, True, False, True], dtype=bool)\n  >>> multiplex_4_op.multiplex([cond1, cond2, cond3], [a1, a2, a3], b)\n  <tf.Tensor: shape=(5,), ... numpy=array([ 11, 102,   3, 104,  10], ...)>\n\n  Args:\n    cond: tf.Tensor or list of tf.Tensor of type bool. Where True, yield `a`.\n      When muliple corresponding `cond` elements are true, the first one yield\n      based on the first one encountered.\n    a: tf.Tensor or list of tf.Tensor, each with the same type and shape as `b`.\n    b: tf.Tensor or list of tf.Tensor with the same type and shape as `a`. Yield\n      `b` if all corresponding `cond` values is False.\n    name: An optional name for the op.\n\n  Returns:\n    A tf.Tensor with elements from `a` where `cond` is True, and elements\n    from `b` elsewhere.\n  '
    if not isinstance(cond, (list, tuple)):
        return examples_multiplex_dense(cond=[cond], a_values=[a], b_values=b, name=name)
    return examples_multiplex_dense(cond=cond, a_values=a, b_values=b, name=name)