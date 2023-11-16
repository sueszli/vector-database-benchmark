"""Tests for AddN."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class XlaAddNTest(xla_test.XLATestCase):

    def testAddTensorLists(self):
        if False:
            print('Hello World!')
        with self.session(), self.test_scope():
            l1 = list_ops.tensor_list_reserve(element_shape=[], element_dtype=dtypes.float32, num_elements=3)
            l2 = list_ops.tensor_list_reserve(element_shape=[], element_dtype=dtypes.float32, num_elements=3)
            l1 = list_ops.tensor_list_set_item(l1, 0, 5.0)
            l2 = list_ops.tensor_list_set_item(l2, 2, 10.0)
            l = math_ops.add_n([l1, l2])
            self.assertAllEqual(list_ops.tensor_list_stack(l, element_dtype=dtypes.float32), [5.0, 0.0, 10.0])

    def testAddTensorListsFailsIfLeadingDimsMismatch(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            l1 = list_ops.tensor_list_reserve(element_shape=[], element_dtype=dtypes.float32, num_elements=2)
            l2 = list_ops.tensor_list_reserve(element_shape=[], element_dtype=dtypes.float32, num_elements=3)
            l = math_ops.add_n([l1, l2])
            with self.assertRaisesRegex(errors.InvalidArgumentError, 'TensorList arguments to AddN must all have the same shape'):
                list_ops.tensor_list_stack(l, element_dtype=dtypes.float32).eval()

    def testAddTensorListsFailsIfElementShapesMismatch(self):
        if False:
            i = 10
            return i + 15
        with self.session() as session, self.test_scope():
            l1_element_shape = array_ops.placeholder(dtype=dtypes.int32)
            l2_element_shape = array_ops.placeholder(dtype=dtypes.int32)
            l1 = list_ops.tensor_list_reserve(element_shape=l1_element_shape, element_dtype=dtypes.float32, num_elements=3)
            l2 = list_ops.tensor_list_reserve(element_shape=l2_element_shape, element_dtype=dtypes.float32, num_elements=3)
            l = math_ops.add_n([l1, l2])
            with self.assertRaisesRegex(errors.InvalidArgumentError, 'TensorList arguments to AddN must all have the same shape'):
                session.run(list_ops.tensor_list_stack(l, element_dtype=dtypes.float32), {l1_element_shape: [], l2_element_shape: [2]})
if __name__ == '__main__':
    test.main()