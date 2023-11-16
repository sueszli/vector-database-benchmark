"""Base class for Tensor wrappers."""
import abc
import tensorflow.compat.v2 as tf

class TensorWrapper(metaclass=abc.ABCMeta):
    """Base class for Tensor wrappers.

  Implements ops that manipulate the backing tensors of Tensor wrappers
  (e.g. DateTensor, PeriodTensor). These ops are mostly about reshaping the
  backing tensors, such as tf.reshape, tf.expand_dims, tf.stack, etc. Also
  includes indexing and slicing.

  Inheritors must implement _apply_op(self, op_fn) and provide a static method
  _apply_sequence_to_tensor_op(op_fn, tensors). For example:

  ```python
  class MyWrapper(TensorWrapper):
    def __init__(self, backing_tensor):
       self._backing_tensor = backing_tensor

    def _apply_op(self, op_fn):
      new_backing_tensor = op_fn(self._backing_tensor)
      return MyWrapper(new_backing_tensor)

    @staticmethod
    def _apply_sequence_to_tensor_op(op_fn, tensors):
      new_backing_tensor = op_fn([t._backing_tensor for t in tensors])
      return MyWrapper(new_backing_tensor)
  ```

  Then 'MyWrapper` can be used as follows:

  ```python
  m1 = MyWrapper(tf.constant([[1, 2, 3], [4, 5, 6]]))
  m2 = MyWrapper(...)
  m3 = m1[0, 1:-1]
  m4 = m1.expand_dims(axis=-1)
  m5 = MyWrapper.concat((m1, m2), axis=-1)
  # etc.
  ```
  """

    @classmethod
    def concat(cls, tensor_wrappers, axis):
        if False:
            for i in range(10):
                print('nop')
        'See tf.concat.'
        cls._validate_tensor_types(tensor_wrappers, 'concat')
        return cls._apply_sequence_to_tensor_op(lambda ts: tf.concat(ts, axis), tensor_wrappers)

    @classmethod
    def stack(cls, tensor_wrappers, axis=0):
        if False:
            while True:
                i = 10
        'See tf.stack.'
        cls._validate_tensor_types(tensor_wrappers, 'stack')
        return cls._apply_sequence_to_tensor_op(lambda ts: tf.stack(ts, axis), tensor_wrappers)

    @classmethod
    def where(cls, condition, tensor_wrapper_1, tensor_wrapper_2):
        if False:
            i = 10
            return i + 15
        'See tf.where. Only three-argument version is supported here.'
        tensor_wrappers = [tensor_wrapper_1, tensor_wrapper_2]
        cls._validate_tensor_types(tensor_wrappers, 'where')
        return cls._apply_sequence_to_tensor_op(lambda ts: tf.compat.v2.where(condition, ts[0], ts[1]), tensor_wrappers)

    @classmethod
    def _validate_tensor_types(cls, tensor_wrappers, function_name):
        if False:
            while True:
                i = 10
        for tensor in tensor_wrappers:
            if not isinstance(tensor, cls):
                raise ValueError('{}.{} cannot be applied to {}'.format(cls.__name__, function_name, type(tensor).__name__))

    def expand_dims(self, axis):
        if False:
            for i in range(10):
                print('nop')
        'See tf.expand_dims.'
        return self._apply_op(lambda t: tf.expand_dims(t, axis))

    def reshape(self, shape):
        if False:
            return 10
        'See tf.reshape.'
        return self._apply_op(lambda t: tf.reshape(t, shape))

    def identity(self):
        if False:
            i = 10
            return i + 15
        'See tf.identity.'
        return self._apply_op(tf.identity)

    def broadcast_to(self, shape):
        if False:
            for i in range(10):
                print('nop')
        'See tf.broadcast_to.'
        return self._apply_op(lambda t: tf.broadcast_to(t, shape))

    def transpose(self, perm=None):
        if False:
            for i in range(10):
                print('nop')
        'See tf.transpose.'
        return self._apply_op(lambda t: tf.transpose(t, perm))

    def squeeze(self, axis=None):
        if False:
            print('Hello World!')
        'See tf.squeeze.'
        return self._apply_op(lambda t: tf.squeeze(t, axis))

    def boolean_mask(self, mask, axis=None):
        if False:
            while True:
                i = 10
        'See tf.boolean_mask.'
        return self._apply_op(lambda t: tf.boolean_mask(t, mask, axis=axis))

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Implements indexing.'
        return self._apply_op(lambda t: t.__getitem__(key))

    def __getslice__(self, *args):
        if False:
            print('Hello World!')
        'Implements slicing.'
        return self._apply_op(lambda t: t.__getslice__(*args))

    @classmethod
    @abc.abstractmethod
    def _apply_sequence_to_tensor_op(cls, op_fn, tensor_wrappers):
        if False:
            while True:
                i = 10
        'Applies given sequence-to-tensor op.\n\n    This method is used for implementing ops that take a sequence of tensors and\n    return a new tensor, such as tf.concat and tf.stack. Implementing wrappers\n    should apply `op_fn` to the backing tensor(s) and return an new wrapper\n    instance with the combined backing tensor.\n\n    Args:\n     op_fn: Callable that applies sequence-to-tensor op to the given sequence\n       of Tensors. E.g. applies tf.concat.\n     tensor_wrappers: a sequence of tensor wrappers to be transformed. All\n       elements have the type of the implementing TensorWrapper class.\n\n    Returns:\n      A TensorWrapper instance with combined backing tensor(s).\n    '
        raise NotImplementedError()

    @abc.abstractmethod
    def _apply_op(self, op_fn):
        if False:
            return 10
        'Applies given tensor-to-tensor op.\n\n    This method is used for implementing ops that take a tensor and return a new\n    tensor, such as tf.expand_dims or tf.transpose. Implementing wrappers\n    should apply `op_fn` to the backing tensor(s) and return an new wrapper\n    instance with the updated backing tensor.\n\n    Args:\n       op_fn: Callable that applies tensor-to-tensor op to the given Tensor.\n        E.g. applies tf.expand_dims.\n\n    Returns:\n      A TensorWrapper instance with updated backing tensor(s).\n    '
        raise NotImplementedError()