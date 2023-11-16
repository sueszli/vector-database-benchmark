"""Tests for common shapes."""
import numpy as np
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class CommonShapesTest(test_util.TensorFlowTestCase):

    def _assert_incompatible_broadcast(self, shape1, shape2):
        if False:
            while True:
                i = 10
        if shape1.dims is not None and shape2.dims is not None:
            zeros1 = np.zeros(shape1.as_list())
            zeros2 = np.zeros(shape2.as_list())
            with self.assertRaises(ValueError):
                np.broadcast(zeros1, zeros2)
            with self.assertRaises(ValueError):
                np.broadcast(zeros2, zeros1)
        self.assertFalse(common_shapes.is_broadcast_compatible(shape1, shape2))
        self.assertFalse(common_shapes.is_broadcast_compatible(shape2, shape1))
        with self.assertRaises(ValueError):
            common_shapes.broadcast_shape(shape1, shape2)
        with self.assertRaises(ValueError):
            common_shapes.broadcast_shape(shape2, shape1)

    def _assert_broadcast(self, expected, shape1, shape2):
        if False:
            for i in range(10):
                print('nop')
        if shape1.dims is not None and shape2.dims is not None:
            expected_np = expected.as_list()
            zeros1 = np.zeros(shape1.as_list())
            zeros2 = np.zeros(shape2.as_list())
            self.assertAllEqual(expected_np, np.broadcast(zeros1, zeros2).shape)
            self.assertAllEqual(expected_np, np.broadcast(zeros2, zeros1).shape)
            self.assertEqual(expected, common_shapes.broadcast_shape(shape1, shape2))
            self.assertEqual(expected, common_shapes.broadcast_shape(shape2, shape1))
        else:
            self.assertEqual(expected, common_shapes.broadcast_shape(shape1, shape2))
            self.assertEqual(expected, common_shapes.broadcast_shape(shape2, shape1))

    def testBroadcast_one_dimension(self):
        if False:
            for i in range(10):
                print('nop')
        s1 = tensor_shape.TensorShape([5])
        s2 = tensor_shape.TensorShape([7])
        unknown = tensor_shape.unknown_shape()
        scalar = tensor_shape.TensorShape([])
        expanded_scalar = tensor_shape.TensorShape([1])
        for shape in (s1, s2, unknown, scalar, expanded_scalar):
            self._assert_broadcast(expected=shape, shape1=shape, shape2=shape)
        self._assert_broadcast(expected=s1, shape1=s1, shape2=scalar)
        self._assert_broadcast(expected=s2, shape1=s2, shape2=scalar)
        self._assert_broadcast(expected=s1, shape1=s1, shape2=expanded_scalar)
        self._assert_broadcast(expected=s2, shape1=s2, shape2=expanded_scalar)
        self._assert_broadcast(expected=unknown, shape1=s1, shape2=unknown)
        self._assert_broadcast(expected=unknown, shape1=s2, shape2=unknown)
        self._assert_broadcast(expected=expanded_scalar, shape1=scalar, shape2=expanded_scalar)
        self._assert_incompatible_broadcast(shape1=s1, shape2=s2)

    def testBroadcast_many_dimensions(self):
        if False:
            while True:
                i = 10
        unknown = tensor_shape.unknown_shape()
        shape_0 = tensor_shape.TensorShape([])
        shape_1 = tensor_shape.TensorShape([1])
        shape_4 = tensor_shape.TensorShape([4])
        shape_1x4 = tensor_shape.TensorShape([1, 4])
        shape_4x1 = tensor_shape.TensorShape([4, 1])
        shape_3x4 = tensor_shape.TensorShape([3, 4])
        shape_4x3 = tensor_shape.TensorShape([4, 3])
        for shape in (shape_0, shape_1, shape_4, shape_1x4, shape_4x1, shape_3x4, shape_4x3):
            self._assert_broadcast(expected=shape, shape1=shape, shape2=shape)
        for identity in (shape_0, shape_1):
            for shape in (shape_4, shape_1x4, shape_4x1, shape_3x4, shape_4x3):
                self._assert_broadcast(expected=shape, shape1=identity, shape2=shape)
        for shape in (shape_4, shape_1x4, shape_4x1, shape_3x4, shape_4x3):
            self._assert_broadcast(expected=unknown, shape1=shape, shape2=unknown)
        self._assert_broadcast(expected=shape_1x4, shape1=shape_4, shape2=shape_1x4)
        shape_4x4 = tensor_shape.TensorShape([4, 4])
        self._assert_broadcast(expected=shape_4x4, shape1=shape_4, shape2=shape_4x1)
        self._assert_broadcast(expected=shape_3x4, shape1=shape_4, shape2=shape_3x4)
        self._assert_incompatible_broadcast(shape1=shape_4, shape2=shape_4x3)
        self._assert_broadcast(expected=shape_4x4, shape1=shape_1x4, shape2=shape_4x1)
        self._assert_broadcast(expected=shape_3x4, shape1=shape_1x4, shape2=shape_3x4)
        self._assert_incompatible_broadcast(shape1=shape_1x4, shape2=shape_4x3)
        self._assert_incompatible_broadcast(shape1=shape_4x1, shape2=shape_3x4)
        self._assert_broadcast(expected=shape_4x3, shape1=shape_4x1, shape2=shape_4x3)
        self._assert_incompatible_broadcast(shape1=shape_3x4, shape2=shape_4x3)

    def _assert_broadcast_with_unknown_dims(self, expected, shape1, shape2):
        if False:
            print('Hello World!')
        actual_dims = common_shapes.broadcast_shape(shape1, shape2).dims
        reflexive_actual_dims = common_shapes.broadcast_shape(shape2, shape1).dims
        if actual_dims is None:
            self.assertIsNone(reflexive_actual_dims)
        elif reflexive_actual_dims is None:
            self.assertIsNone(actual_dims)
        else:
            self.assertEqual(len(actual_dims), len(reflexive_actual_dims))
            for (actual_dim, reflexive_actual_dim) in zip(actual_dims, reflexive_actual_dims):
                self.assertEqual(actual_dim.value, reflexive_actual_dim.value)
        expected_dims = expected.dims
        if expected_dims is None:
            self.assertIsNone(actual_dims)
        elif actual_dims is None:
            self.assertIsNone(expected_dims)
        else:
            self.assertEqual(len(expected_dims), len(actual_dims))
            for (expected_dim, actual_dim) in zip(expected_dims, actual_dims):
                self.assertEqual(expected_dim.value, actual_dim.value)

    def testBroadcast_unknown_dims(self):
        if False:
            print('Hello World!')
        unknown = tensor_shape.unknown_shape()
        shape_0 = tensor_shape.TensorShape([])
        shape_1 = tensor_shape.TensorShape([1])
        shape_U = tensor_shape.TensorShape([None])
        shape_1xU = tensor_shape.TensorShape([1, None])
        shape_Ux1 = tensor_shape.TensorShape([None, 1])
        shape_4xU = tensor_shape.TensorShape([4, None])
        shape_Ux4 = tensor_shape.TensorShape([None, 4])
        for shape in (shape_U, shape_1xU, shape_Ux1, shape_4xU, shape_Ux4):
            self._assert_broadcast_with_unknown_dims(expected=shape, shape1=shape, shape2=shape)
        for identity in (shape_0, shape_1):
            for shape in (shape_U, shape_1xU, shape_Ux1, shape_4xU, shape_Ux4):
                self._assert_broadcast_with_unknown_dims(expected=shape, shape1=identity, shape2=shape)
        for shape in (shape_U, shape_1xU, shape_Ux1, shape_4xU, shape_Ux4):
            self._assert_broadcast_with_unknown_dims(expected=unknown, shape1=shape, shape2=unknown)
        self._assert_broadcast_with_unknown_dims(expected=shape_1xU, shape1=shape_U, shape2=shape_1xU)
        shape_UxU = tensor_shape.TensorShape([None, None])
        self._assert_broadcast_with_unknown_dims(expected=shape_UxU, shape1=shape_U, shape2=shape_Ux1)
        self._assert_broadcast_with_unknown_dims(expected=shape_4xU, shape1=shape_U, shape2=shape_4xU)
        self._assert_broadcast_with_unknown_dims(expected=shape_Ux4, shape1=shape_U, shape2=shape_Ux4)
        self._assert_broadcast_with_unknown_dims(expected=shape_UxU, shape1=shape_1xU, shape2=shape_Ux1)
        self._assert_broadcast_with_unknown_dims(expected=shape_4xU, shape1=shape_1xU, shape2=shape_4xU)
        self._assert_broadcast_with_unknown_dims(expected=shape_Ux4, shape1=shape_1xU, shape2=shape_Ux4)
        self._assert_broadcast_with_unknown_dims(expected=shape_4xU, shape1=shape_Ux1, shape2=shape_4xU)
        self._assert_broadcast_with_unknown_dims(expected=shape_Ux4, shape1=shape_Ux1, shape2=shape_Ux4)
        shape_4x4 = tensor_shape.TensorShape([4, 4])
        self._assert_broadcast_with_unknown_dims(expected=shape_4x4, shape1=shape_4xU, shape2=shape_Ux4)
if __name__ == '__main__':
    googletest.main()