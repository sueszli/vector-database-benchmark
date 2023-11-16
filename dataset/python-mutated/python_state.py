"""Utilities for including Python state in TensorFlow checkpoints."""
import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export
PYTHON_STATE = 'py_state'

@tf_export('train.experimental.PythonState')
class PythonState(base.Trackable, metaclass=abc.ABCMeta):
    """A mixin for putting Python state in an object-based checkpoint.

  This is an abstract class which allows extensions to TensorFlow's object-based
  checkpointing (see `tf.train.Checkpoint`). For example a wrapper for NumPy
  arrays:

  ```python
  import io
  import numpy

  class NumpyWrapper(tf.train.experimental.PythonState):

    def __init__(self, array):
      self.array = array

    def serialize(self):
      string_file = io.BytesIO()
      try:
        numpy.save(string_file, self.array, allow_pickle=False)
        serialized = string_file.getvalue()
      finally:
        string_file.close()
      return serialized

    def deserialize(self, string_value):
      string_file = io.BytesIO(string_value)
      try:
        self.array = numpy.load(string_file, allow_pickle=False)
      finally:
        string_file.close()
  ```

  Instances of `NumpyWrapper` are checkpointable objects, and will be saved and
  restored from checkpoints along with TensorFlow state like variables.

  ```python
  root = tf.train.Checkpoint(numpy=NumpyWrapper(numpy.array([1.])))
  save_path = root.save(prefix)
  root.numpy.array *= 2.
  assert [2.] == root.numpy.array
  root.restore(save_path)
  assert [1.] == root.numpy.array
  ```
  """

    @abc.abstractmethod
    def serialize(self):
        if False:
            while True:
                i = 10
        'Callback to serialize the object. Returns a string.'

    @abc.abstractmethod
    def deserialize(self, string_value):
        if False:
            i = 10
            return i + 15
        'Callback to deserialize the object.'

    def _serialize_to_tensors(self):
        if False:
            i = 10
            return i + 15
        'Implements Trackable._serialize_to_tensors.'
        with ops.init_scope():
            value = constant_op.constant(self.serialize(), dtype=dtypes.string)
        return {PYTHON_STATE: value}