"""Tests for ops which manipulate lists of tensors via bridge."""
import os
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test

class ListOpsTest(parameterized.TestCase, xla_test.XLATestCase):

    def testElementShape(self):
        if False:
            while True:
                i = 10
        with self.session() as sess, self.test_scope():
            dim = array_ops.placeholder(dtypes.int32)
            l = list_ops.empty_tensor_list(element_shape=(dim, 15), element_dtype=dtypes.float32, max_num_elements=20)
            e32 = list_ops.tensor_list_element_shape(l, shape_type=dtypes.int32)
            e64 = list_ops.tensor_list_element_shape(l, shape_type=dtypes.int64)
            self.assertAllEqual(sess.run(e32, {dim: 10}), (10, 15))
            self.assertAllEqual(sess.run(e64, {dim: 7}), (7, 15))

    def testPushPop(self):
        if False:
            i = 10
            return i + 15
        with self.session() as sess, self.test_scope():
            l = list_ops.empty_tensor_list(element_shape=(7, 15), element_dtype=dtypes.float32, max_num_elements=10)
            l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0, shape=(7, 15)))
            l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0, shape=(7, 15)))
            (l, e2) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            (_, e1) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            self.assertAllEqual(sess.run(e2), 2.0 * np.ones((7, 15)))
            self.assertAllEqual(sess.run(e1), 1.0 * np.ones((7, 15)))

    def testDoNotConstantFoldVariants(self):
        if False:
            return 10
        with self.session() as sess, self.test_scope():
            val = array_ops.placeholder(dtype=dtypes.float32)
            l = list_ops.empty_tensor_list(element_shape=(7, 15), element_dtype=dtypes.float32, max_num_elements=10)
            l = list_ops.tensor_list_push_back(l, array_ops.fill(value=val, dims=(7, 15)))
            l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0, shape=(7, 15)))
            (l, e2) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            (_, e1) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            self.assertAllEqual(sess.run(e2, {val: 1.0}), 2.0 * np.ones((7, 15)))
            self.assertAllEqual(sess.run(e1, {val: 1.0}), 1.0 * np.ones((7, 15)))

    def testPushPopSeparateLists(self):
        if False:
            i = 10
            return i + 15
        with self.session() as sess, self.test_scope():
            l = list_ops.empty_tensor_list(element_shape=[], element_dtype=dtypes.float32, max_num_elements=20)
            l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
            l2 = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
            l3 = list_ops.tensor_list_push_back(l, constant_op.constant(3.0))
            (_, e11) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            (l2, e21) = list_ops.tensor_list_pop_back(l2, element_dtype=dtypes.float32)
            (l2, e22) = list_ops.tensor_list_pop_back(l2, element_dtype=dtypes.float32)
            (l3, e31) = list_ops.tensor_list_pop_back(l3, element_dtype=dtypes.float32)
            (l3, e32) = list_ops.tensor_list_pop_back(l3, element_dtype=dtypes.float32)
            result = sess.run([e11, [e21, e22], [e31, e32]])
            self.assertEqual(result, [1.0, [2.0, 1.0], [3.0, 1.0]])

    def testEmptyTensorListNoMax(self):
        if False:
            return 10
        with self.session() as sess, self.test_scope():
            l = list_ops.empty_tensor_list(element_shape=(7, 15), element_dtype=dtypes.float32)
            l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0, shape=(7, 15)))
            (_, e) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            with self.assertRaisesRegex(errors.InvalidArgumentError, 'Set the max number of elements'):
                self.assertAllEqual(sess.run(e), 1.0 * np.ones((7, 15)))

    def testEmptyTensorListMax(self):
        if False:
            print('Hello World!')
        with self.session() as sess, self.test_scope():
            l = list_ops.empty_tensor_list(element_shape=(10, 15), element_dtype=dtypes.float32, max_num_elements=2)
            l = list_ops.tensor_list_push_back(l, array_ops.fill(value=3.0, dims=(10, 15)))
            (_, e) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            self.assertAllEqual(sess.run(e), 3.0 * np.ones((10, 15)))

    def testListFromTensor(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            t = constant_op.constant([1.0, 2.0])
            l = list_ops.tensor_list_from_tensor(t, element_shape=[])
            e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
            self.assertAllEqual(e, 1.0)
            (l, e0) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            self.assertAllEqual(e0, 2.0)
            (l, e1) = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
            self.assertAllEqual(e1, 1.0)
            self.assertAllEqual(list_ops.tensor_list_length(l), 2)

    def testGetSet(self):
        if False:
            return 10
        with self.session(), self.test_scope():
            t = constant_op.constant([1.0, 2.0])
            l = list_ops.tensor_list_from_tensor(t, element_shape=[])
            e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
            self.assertAllEqual(e0, 1.0)
            l = list_ops.tensor_list_set_item(l, 0, 3.0)
            t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
            self.assertAllEqual(t, [3.0, 2.0])

    def testSetDoesNotUpdatePushIndex(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            l = list_ops.empty_tensor_list(element_shape=[], element_dtype=dtypes.float32, max_num_elements=2)
            l = list_ops.tensor_list_set_item(l, 1, 3.0)
            l = list_ops.tensor_list_push_back(l, 5.0)
            l = list_ops.tensor_list_push_back(l, 7.0)
            t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
            self.assertAllEqual(t, [5.0, 7.0])

    def testGetSetReserved(self):
        if False:
            print('Hello World!')
        with self.session(), self.test_scope():
            l = list_ops.tensor_list_reserve(element_dtype=dtypes.float32, element_shape=[], num_elements=2)
            e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
            self.assertAllEqual(e0, 0.0)
            l = list_ops.tensor_list_set_item(l, 0, 3.0)
            t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
            self.assertAllEqual(t, [3.0, 0.0])

    def testSetStackReservedUnknownElementShape(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            l = list_ops.tensor_list_reserve(element_dtype=dtypes.float32, element_shape=None, num_elements=2)
            l = list_ops.tensor_list_set_item(l, 0, [3.0, 4.0])
            t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
            self.assertAllEqual(t, [[3.0, 4.0], [0.0, 0.0]])

    def testPushInEmptyListWithUnknownElementShape(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            l = list_ops.empty_tensor_list(element_dtype=dtypes.float32, element_shape=None, max_num_elements=2)
            l = list_ops.tensor_list_push_back(l, [3.0, 4.0])
            with self.assertRaisesRegex(errors.InternalError, 'shape'):
                l = list_ops.tensor_list_push_back(l, 5.0)
                self.evaluate(list_ops.tensor_list_stack(l, element_dtype=dtypes.float32))

    def testGetSetReservedNonScalar(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as sess, self.test_scope():
            l = list_ops.tensor_list_reserve(element_dtype=dtypes.float32, element_shape=(7, 15), num_elements=2)
            l = list_ops.tensor_list_set_item(l, 0, constant_op.constant(1.0, shape=(7, 15)))
            e1 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
            e2 = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
            self.assertAllEqual(sess.run(e1), np.ones((7, 15)))
            self.assertAllEqual(sess.run(e2), np.zeros((7, 15)))

    def testStack(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            l = list_ops.empty_tensor_list(element_dtype=dtypes.float32, element_shape=[], max_num_elements=2)
            l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
            e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
            self.assertAllEqual(e, 1.0)
            l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
            t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
            self.assertAllEqual(t.shape.as_list(), [None])
            self.assertAllEqual(t, [1.0, 2.0])

    @parameterized.named_parameters(('FlatList', [1.0, 2.0, 3.0], [], [0, 2], [1.0, 3.0]), ('NestedList', [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [2], [1], [[3.0, 4.0]]), ('EmptyIndices', [1.0, 2.0, 3.0], [], [], []))
    def testGather(self, input_list, element_shape, indices, output):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            tensor_list = list_ops.tensor_list_from_tensor(input_list, element_shape=element_shape)
            gather_t = list_ops.tensor_list_gather(tensor_list, indices, element_dtype=dtypes.float32)
            self.assertAllEqual(gather_t, output)

    def testStackWithUninitializedTensors(self):
        if False:
            return 10
        with self.session(), self.test_scope():
            l = list_ops.tensor_list_reserve(element_dtype=dtypes.float32, element_shape=[], num_elements=3)
            t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
            self.assertAllEqual(t, [0.0, 0.0, 0.0])

    def testZerosLikeForTensorList(self):
        if False:
            print('Hello World!')
        with self.session(), self.test_scope():
            l = list_ops.empty_tensor_list(element_dtype=dtypes.float32, element_shape=[], max_num_elements=2)
            l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
            z = array_ops.zeros_like(l)
            z = list_ops.tensor_list_stack(z, element_dtype=dtypes.float32)
            self.assertAllEqual(z.shape.as_list(), [None])
            self.assertAllEqual(z, [0.0, 0.0])

    def testInvalidSplitLength(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            tensor_list_split = list_ops.tensor_list_split(tensor=[1], element_shape=[-1], lengths=[0])
            with self.assertRaisesRegex(errors.UnimplementedError, 'All lengths must be positive'):
                self.evaluate(tensor_list_split)
if __name__ == '__main__':
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_min_cluster_size=2 ' + os.environ.get('TF_XLA_FLAGS', '')
    test.main()