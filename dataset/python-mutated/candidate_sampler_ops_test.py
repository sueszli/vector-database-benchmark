"""Tests for CandidateSamplerOp."""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class RangeSamplerOpsTest(test.TestCase):
    BATCH_SIZE = 3
    NUM_TRUE = 2
    RANGE = 5
    NUM_SAMPLED = RANGE
    TRUE_LABELS = [[1, 2], [0, 4], [3, 3]]

    @test_util.run_deprecated_v1
    def testTrueCandidates(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            indices = constant_op.constant([0, 0, 1, 1, 2, 2])
            true_candidates_vec = constant_op.constant([1, 2, 0, 4, 3, 3])
            true_candidates_matrix = array_ops.reshape(true_candidates_vec, [self.BATCH_SIZE, self.NUM_TRUE])
            (indices_val, true_candidates_val) = sess.run([indices, true_candidates_matrix])
        self.assertAllEqual(indices_val, [0, 0, 1, 1, 2, 2])
        self.assertAllEqual(true_candidates_val, self.TRUE_LABELS)

    def testSampledCandidates(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            true_classes = constant_op.constant([[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
            (sampled_candidates, _, _) = candidate_sampling_ops.all_candidate_sampler(true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
            result = self.evaluate(sampled_candidates)
        expected_ids = [0, 1, 2, 3, 4]
        self.assertAllEqual(result, expected_ids)
        self.assertEqual(sampled_candidates.get_shape(), [self.NUM_SAMPLED])

    def testTrueLogExpectedCount(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            true_classes = constant_op.constant([[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
            (_, true_expected_count, _) = candidate_sampling_ops.all_candidate_sampler(true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
            true_log_expected_count = math_ops.log(true_expected_count)
            result = self.evaluate(true_log_expected_count)
        self.assertAllEqual(result, [[0.0] * self.NUM_TRUE] * self.BATCH_SIZE)
        self.assertEqual(true_expected_count.get_shape(), [self.BATCH_SIZE, self.NUM_TRUE])
        self.assertEqual(true_log_expected_count.get_shape(), [self.BATCH_SIZE, self.NUM_TRUE])

    def testSampledLogExpectedCount(self):
        if False:
            print('Hello World!')
        with self.cached_session():
            true_classes = constant_op.constant([[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
            (_, _, sampled_expected_count) = candidate_sampling_ops.all_candidate_sampler(true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
            sampled_log_expected_count = math_ops.log(sampled_expected_count)
            result = self.evaluate(sampled_log_expected_count)
        self.assertAllEqual(result, [0.0] * self.NUM_SAMPLED)
        self.assertEqual(sampled_expected_count.get_shape(), [self.NUM_SAMPLED])
        self.assertEqual(sampled_log_expected_count.get_shape(), [self.NUM_SAMPLED])

    def testAccidentalHits(self):
        if False:
            i = 10
            return i + 15
        with self.cached_session() as sess:
            true_classes = constant_op.constant([[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
            (sampled_candidates, _, _) = candidate_sampling_ops.all_candidate_sampler(true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True)
            accidental_hits = candidate_sampling_ops.compute_accidental_hits(true_classes, sampled_candidates, self.NUM_TRUE)
            (indices, ids, weights) = self.evaluate(accidental_hits)
        self.assertEqual(1, accidental_hits[0].get_shape().ndims)
        self.assertEqual(1, accidental_hits[1].get_shape().ndims)
        self.assertEqual(1, accidental_hits[2].get_shape().ndims)
        for (index, id_, weight) in zip(indices, ids, weights):
            self.assertTrue(id_ in self.TRUE_LABELS[index])
            self.assertLess(weight, -1e+37)

    @test_util.run_deprecated_v1
    def testSeed(self):
        if False:
            i = 10
            return i + 15

        def draw(seed):
            if False:
                print('Hello World!')
            with self.cached_session():
                true_classes = constant_op.constant([[1, 2], [0, 4], [3, 3]], dtype=dtypes.int64)
                (sampled, _, _) = candidate_sampling_ops.log_uniform_candidate_sampler(true_classes, self.NUM_TRUE, self.NUM_SAMPLED, True, 5, seed=seed)
                return self.evaluate(sampled)
        for seed in [1, 12, 123, 1234]:
            self.assertAllEqual(draw(seed), draw(seed))
        num_same = 0
        for _ in range(10):
            if np.allclose(draw(None), draw(None)):
                num_same += 1
        self.assertLessEqual(num_same, 2)

    def testCandidateOutOfRange(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'out of range'):
            self.evaluate(candidate_sampling_ops.log_uniform_candidate_sampler(true_classes=[[0, 10]], num_true=2, num_sampled=1000, unique=False, range_max=2))
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'out of range'):
            self.evaluate(candidate_sampling_ops.log_uniform_candidate_sampler(true_classes=[[0, -10]], num_true=2, num_sampled=1000, unique=False, range_max=2))
if __name__ == '__main__':
    test.main()