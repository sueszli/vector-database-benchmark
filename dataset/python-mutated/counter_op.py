"""The implementation of `tf.data.Dataset.counter`."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops

def _counter(start, step, dtype, name=None):
    if False:
        return 10
    with ops.name_scope('counter'):
        start = ops.convert_to_tensor(start, dtype=dtype, name='start')
        step = ops.convert_to_tensor(step, dtype=dtype, name='step')
        return dataset_ops.Dataset.from_tensors(0, name=name).repeat(None).scan(start, lambda state, _: (state + step, state))