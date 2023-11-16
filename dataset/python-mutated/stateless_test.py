"""Tests for random.stateless."""
import numpy as np
import tensorflow.compat.v2 as tf
import tf_quant_finance as tff
from tensorflow.python.framework import test_util
tff_rnd = tff.math.random

class StatelessRandomOpsTest(tf.test.TestCase):

    @test_util.run_in_graph_and_eager_modes
    def testOutputIsPermutation(self):
        if False:
            while True:
                i = 10
        'Checks that stateless_random_shuffle outputs a permutation.'
        for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
            identity_permutation = tf.range(10, dtype=dtype)
            random_shuffle_seed_1 = tff_rnd.stateless_random_shuffle(identity_permutation, seed=tf.constant((1, 42), tf.int64))
            random_shuffle_seed_2 = tff_rnd.stateless_random_shuffle(identity_permutation, seed=tf.constant((2, 42), tf.int64))
            for shuffle in (random_shuffle_seed_1, random_shuffle_seed_2):
                np.testing.assert_equal(shuffle.dtype, dtype.as_numpy_dtype)
            random_shuffle_seed_1 = self.evaluate(random_shuffle_seed_1)
            random_shuffle_seed_2 = self.evaluate(random_shuffle_seed_2)
            identity_permutation = self.evaluate(identity_permutation)
            self.assertTrue(np.abs(random_shuffle_seed_1 - random_shuffle_seed_2).max())
            for shuffle in (random_shuffle_seed_1, random_shuffle_seed_2):
                self.assertAllEqual(set(shuffle), set(identity_permutation))

    @test_util.run_in_graph_and_eager_modes
    def testOutputIsStateless(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks that stateless_random_shuffle is stateless.'
        random_permutation_next_call = None
        for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
            random_permutation = tff_rnd.stateless_random_shuffle(tf.range(10, dtype=dtype), seed=(100, 42))
            random_permutation_first_call = self.evaluate(random_permutation)
            if random_permutation_next_call is not None:
                np.testing.assert_array_equal(random_permutation_first_call, random_permutation_next_call)
            random_permutation_next_call = self.evaluate(random_permutation)
            np.testing.assert_array_equal(random_permutation_first_call, random_permutation_next_call)

    @test_util.run_in_graph_and_eager_modes
    def testOutputIsIndependentOfInputValues(self):
        if False:
            for i in range(10):
                print('nop')
        'stateless_random_shuffle output is independent of input_tensor values.'
        np.random.seed(25)
        random_input = np.random.normal(size=[10])
        random_input.sort()
        for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
            random_permutation = tff_rnd.stateless_random_shuffle(tf.range(10, dtype=dtype), seed=(100, 42))
            random_permutation = self.evaluate(random_permutation)
            random_shuffle_control = tff_rnd.stateless_random_shuffle(random_input, seed=(100, 42))
            random_shuffle_control = self.evaluate(random_shuffle_control)
            np.testing.assert_array_equal(np.argsort(random_permutation), np.argsort(random_shuffle_control))

    @test_util.run_v1_only('Sessions are not available in TF2.0')
    def testOutputIsStatelessSession(self):
        if False:
            return 10
        'Checks that stateless_random_shuffle is stateless across Sessions.'
        random_permutation_next_call = None
        for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
            random_permutation = tff_rnd.stateless_random_shuffle(tf.range(10, dtype=dtype), seed=tf.constant((100, 42), tf.int64))
            with tf.compat.v1.Session() as sess:
                random_permutation_first_call = sess.run(random_permutation)
            if random_permutation_next_call is not None:
                np.testing.assert_array_equal(random_permutation_first_call, random_permutation_next_call)
            with tf.compat.v1.Session() as sess:
                random_permutation_next_call = sess.run(random_permutation)
            np.testing.assert_array_equal(random_permutation_first_call, random_permutation_next_call)

    @test_util.run_in_graph_and_eager_modes
    def testMultiDimensionalShape(self):
        if False:
            while True:
                i = 10
        'Check that stateless_random_shuffle works with multi-dim shapes.'
        for dtype in (tf.int32, tf.int64, tf.float32, tf.float64):
            input_permutation = tf.constant([[[1], [2], [3]], [[4], [5], [6]]], dtype=dtype)
            random_shuffle = tff_rnd.stateless_random_shuffle(input_permutation, seed=(1, 42))
            random_permutation_first_call = self.evaluate(random_shuffle)
            random_permutation_next_call = self.evaluate(random_shuffle)
            input_permutation = self.evaluate(input_permutation)
            np.testing.assert_equal(random_permutation_first_call.dtype, dtype.as_numpy_dtype)
            np.testing.assert_array_equal(random_permutation_first_call, random_permutation_next_call)
            np.testing.assert_equal(random_permutation_first_call.shape, input_permutation.shape)
if __name__ == '__main__':
    tf.test.main()