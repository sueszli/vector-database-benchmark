"""Tests for broadcast rules."""
import numpy as np
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import test

def _test_values(shape):
    if False:
        for i in range(10):
            print('nop')
    return np.reshape(np.cumsum(np.ones(shape), dtype=np.int32), newshape=shape)

class AssertBroadcastableTest(test.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        ops.reset_default_graph()

    def _test_valid(self, weights, values):
        if False:
            for i in range(10):
                print('nop')
        static_op = weights_broadcast_ops.assert_broadcastable(weights=weights, values=values)
        weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
        values_placeholder = array_ops.placeholder(dtypes_lib.float32)
        dynamic_op = weights_broadcast_ops.assert_broadcastable(weights=weights_placeholder, values=values_placeholder)
        with self.cached_session():
            static_op.run()
            dynamic_op.run(feed_dict={weights_placeholder: weights, values_placeholder: values})

    @test_util.run_deprecated_v1
    def testScalar(self):
        if False:
            print('Hello World!')
        self._test_valid(weights=5, values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def test1x1x1(self):
        if False:
            while True:
                i = 10
        self._test_valid(weights=np.asarray((5,)).reshape((1, 1, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def test1x1xN(self):
        if False:
            i = 10
            return i + 15
        self._test_valid(weights=np.asarray((5, 7, 11, 3)).reshape((1, 1, 4)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def test1xNx1(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_valid(weights=np.asarray((5, 11)).reshape((1, 2, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def test1xNxN(self):
        if False:
            print('Hello World!')
        self._test_valid(weights=np.asarray((5, 7, 11, 3, 2, 13, 7, 5)).reshape((1, 2, 4)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testNx1x1(self):
        if False:
            print('Hello World!')
        self._test_valid(weights=np.asarray((5, 7, 11)).reshape((3, 1, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testNx1xN(self):
        if False:
            while True:
                i = 10
        self._test_valid(weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3)).reshape((3, 1, 4)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testNxNxN(self):
        if False:
            return 10
        self._test_valid(weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3, 2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4)), values=_test_values((3, 2, 4)))

    def _test_invalid(self, weights, values):
        if False:
            return 10
        error_msg = 'weights can not be broadcast to values'
        with self.assertRaisesRegex(ValueError, error_msg):
            weights_broadcast_ops.assert_broadcastable(weights=weights, values=values)
        weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
        values_placeholder = array_ops.placeholder(dtypes_lib.float32)
        dynamic_op = weights_broadcast_ops.assert_broadcastable(weights=weights_placeholder, values=values_placeholder)
        with self.cached_session():
            with self.assertRaisesRegex(errors_impl.OpError, error_msg):
                dynamic_op.run(feed_dict={weights_placeholder: weights, values_placeholder: values})

    @test_util.run_deprecated_v1
    def testInvalid1(self):
        if False:
            while True:
                i = 10
        self._test_invalid(weights=np.asarray((5,)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalid1x1(self):
        if False:
            print('Hello World!')
        self._test_invalid(weights=np.asarray((5,)).reshape((1, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidPrefixMatch(self):
        if False:
            i = 10
            return i + 15
        self._test_invalid(weights=np.asarray((5, 7, 11, 3, 2, 12)).reshape((3, 2)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidSuffixMatch(self):
        if False:
            i = 10
            return i + 15
        self._test_invalid(weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5)).reshape((2, 4)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidOnesExtraDim(self):
        if False:
            i = 10
            return i + 15
        self._test_invalid(weights=np.asarray((5,)).reshape((1, 1, 1, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidPrefixMatchExtraDim(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_invalid(weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3, 2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidSuffixMatchExtraDim(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_invalid(weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3, 2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((1, 3, 2, 4)), values=_test_values((3, 2, 4)))

class BroadcastWeightsTest(test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        ops.reset_default_graph()

    def _test_valid(self, weights, values, expected):
        if False:
            i = 10
            return i + 15
        static_op = weights_broadcast_ops.broadcast_weights(weights=weights, values=values)
        weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
        values_placeholder = array_ops.placeholder(dtypes_lib.float32)
        dynamic_op = weights_broadcast_ops.broadcast_weights(weights=weights_placeholder, values=values_placeholder)
        with self.cached_session():
            self.assertAllEqual(expected, self.evaluate(static_op))
            self.assertAllEqual(expected, dynamic_op.eval(feed_dict={weights_placeholder: weights, values_placeholder: values}))

    @test_util.run_deprecated_v1
    def testScalar(self):
        if False:
            return 10
        self._test_valid(weights=5, values=_test_values((3, 2, 4)), expected=5 * np.ones((3, 2, 4)))

    @test_util.run_deprecated_v1
    def test1x1x1(self):
        if False:
            print('Hello World!')
        self._test_valid(weights=np.asarray((5,)).reshape((1, 1, 1)), values=_test_values((3, 2, 4)), expected=5 * np.ones((3, 2, 4)))

    @test_util.run_deprecated_v1
    def test1x1xN(self):
        if False:
            i = 10
            return i + 15
        weights = np.asarray((5, 7, 11, 3)).reshape((1, 1, 4))
        self._test_valid(weights=weights, values=_test_values((3, 2, 4)), expected=np.tile(weights, reps=(3, 2, 1)))

    @test_util.run_deprecated_v1
    def test1xNx1(self):
        if False:
            print('Hello World!')
        weights = np.asarray((5, 11)).reshape((1, 2, 1))
        self._test_valid(weights=weights, values=_test_values((3, 2, 4)), expected=np.tile(weights, reps=(3, 1, 4)))

    @test_util.run_deprecated_v1
    def test1xNxN(self):
        if False:
            return 10
        weights = np.asarray((5, 7, 11, 3, 2, 13, 7, 5)).reshape((1, 2, 4))
        self._test_valid(weights=weights, values=_test_values((3, 2, 4)), expected=np.tile(weights, reps=(3, 1, 1)))

    @test_util.run_deprecated_v1
    def testNx1x1(self):
        if False:
            for i in range(10):
                print('nop')
        weights = np.asarray((5, 7, 11)).reshape((3, 1, 1))
        self._test_valid(weights=weights, values=_test_values((3, 2, 4)), expected=np.tile(weights, reps=(1, 2, 4)))

    @test_util.run_deprecated_v1
    def testNx1xN(self):
        if False:
            while True:
                i = 10
        weights = np.asarray((5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3)).reshape((3, 1, 4))
        self._test_valid(weights=weights, values=_test_values((3, 2, 4)), expected=np.tile(weights, reps=(1, 2, 1)))

    @test_util.run_deprecated_v1
    def testNxNxN(self):
        if False:
            print('Hello World!')
        weights = np.asarray((5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3, 2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4))
        self._test_valid(weights=weights, values=_test_values((3, 2, 4)), expected=weights)

    def _test_invalid(self, weights, values):
        if False:
            return 10
        error_msg = 'weights can not be broadcast to values'
        with self.assertRaisesRegex(ValueError, error_msg):
            weights_broadcast_ops.broadcast_weights(weights=weights, values=values)
        weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
        values_placeholder = array_ops.placeholder(dtypes_lib.float32)
        dynamic_op = weights_broadcast_ops.broadcast_weights(weights=weights_placeholder, values=values_placeholder)
        with self.cached_session():
            with self.assertRaisesRegex(errors_impl.OpError, error_msg):
                dynamic_op.eval(feed_dict={weights_placeholder: weights, values_placeholder: values})

    @test_util.run_deprecated_v1
    def testInvalid1(self):
        if False:
            i = 10
            return i + 15
        self._test_invalid(weights=np.asarray((5,)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalid1x1(self):
        if False:
            return 10
        self._test_invalid(weights=np.asarray((5,)).reshape((1, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidPrefixMatch(self):
        if False:
            while True:
                i = 10
        self._test_invalid(weights=np.asarray((5, 7, 11, 3, 2, 12)).reshape((3, 2)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidSuffixMatch(self):
        if False:
            i = 10
            return i + 15
        self._test_invalid(weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5)).reshape((2, 4)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidOnesExtraDim(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_invalid(weights=np.asarray((5,)).reshape((1, 1, 1, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidPrefixMatchExtraDim(self):
        if False:
            print('Hello World!')
        self._test_invalid(weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3, 2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4, 1)), values=_test_values((3, 2, 4)))

    @test_util.run_deprecated_v1
    def testInvalidSuffixMatchExtraDim(self):
        if False:
            i = 10
            return i + 15
        self._test_invalid(weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3, 2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((1, 3, 2, 4)), values=_test_values((3, 2, 4)))
if __name__ == '__main__':
    test.main()