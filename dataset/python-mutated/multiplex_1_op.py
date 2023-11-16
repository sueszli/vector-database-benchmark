"""A wrapper for gen_multiplex_1_op.py.

This defines a public API and provides a docstring for the C++ Op defined by
multiplex_1_kernel.cc
"""
import tensorflow as tf
from tensorflow.python.platform import resource_loader
_multiplex_1_module = tf.load_op_library(resource_loader.get_path_to_datafile('multiplex_1_kernel.so'))
examples_multiplex_dense = _multiplex_1_module.examples1_multiplex_dense

def multiplex(cond, a, b, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Return elements chosen from `a` or `b` depending on `cond`.\n\n  This is similar to `np.where` and `tf.where`, but simplified to only handle\n  the case of dense tensors, no optional parameters, no broadcasting, etc..\n\n  >>> multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])\n  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)>\n\n  Args:\n    cond: tf.Tensor of type bool. Where True, yield `a`, otherwise yield `b`.\n    a: tf.Tensor with the same type and shape as `b`.\n    b: tf.Tensor with the same type and shape as `a`.\n    name: An optional name for the op.\n\n  Returns:\n    A tf.Tensor with elements from `a` where `cond` is True, and elements\n    from `b` elsewhere.\n  '
    return examples_multiplex_dense(cond=cond, a_values=a, b_values=b, name=name)