"""Tests for tensors module."""
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test

class TensorsTest(test.TestCase):

    def _simple_tensor_array(self):
        if False:
            return 10
        return tensor_array_ops.TensorArray(dtypes.int32, size=3)

    def _simple_tensor_list(self):
        if False:
            i = 10
            return i + 15
        return list_ops.empty_tensor_list(element_shape=constant_op.constant([1]), element_dtype=dtypes.int32)

    def _simple_list_of_tensors(self):
        if False:
            while True:
                i = 10
        return [constant_op.constant(1), constant_op.constant(2)]

    def test_is_tensor_array(self):
        if False:
            return 10
        self.assertTrue(tensors.is_tensor_array(self._simple_tensor_array()))
        self.assertFalse(tensors.is_tensor_array(self._simple_tensor_list()))
        self.assertFalse(tensors.is_tensor_array(constant_op.constant(1)))
        self.assertFalse(tensors.is_tensor_array(self._simple_list_of_tensors()))
        self.assertFalse(tensors.is_tensor_array(None))

    def test_is_tensor_list(self):
        if False:
            while True:
                i = 10
        self.assertFalse(tensors.is_tensor_list(self._simple_tensor_array()))
        self.assertTrue(tensors.is_tensor_list(self._simple_tensor_list()))
        self.assertFalse(tensors.is_tensor_list(constant_op.constant(1)))
        self.assertFalse(tensors.is_tensor_list(self._simple_list_of_tensors()))
        self.assertFalse(tensors.is_tensor_list(None))

    def is_range_tensor(self):
        if False:
            while True:
                i = 10
        self.assertTrue(tensors.is_range_tensor(math_ops.range(1)))
        self.assertTrue(tensors.is_range_tensor(math_ops.range(1, 2)))
        self.assertTrue(tensors.is_range_tensor(math_ops.range(1, 2, 3)))
        self.assertFalse(tensors.is_range_tensor(None))
        self.assertFalse(tensors.is_range_tensor(constant_op.constant(range(1))))
if __name__ == '__main__':
    test.main()