"""Tests for loss scaling utilities in tensorflow.ops.nn."""
from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl_distribute
from tensorflow.python.platform import test as test_lib

class LossUtilitiesTest(test_lib.TestCase, parameterized.TestCase):

    def testComputeAverageLossGlobalBatchSize(self):
        if False:
            return 10
        per_example_loss = [1, 2, 3, 4, 5]
        loss = nn_impl_distribute.compute_average_loss(per_example_loss, global_batch_size=10)
        self.assertEqual(self.evaluate(loss), 1.5)

    def testComputeAverageLossGlobalBatchSize_BatchSizeNonScalar(self):
        if False:
            i = 10
            return i + 15
        per_example_loss = [1, 2, 3, 4, 5]
        with self.assertRaisesWithPredicateMatch(ValueError, 'global_batch_size must be scalar'):
            nn_impl_distribute.compute_average_loss(per_example_loss, global_batch_size=[10])

    def testComputeAverageLossGlobalBatchSize_BatchSizeFloat(self):
        if False:
            print('Hello World!')
        per_example_loss = [1, 2, 3, 4, 5]
        with self.assertRaisesWithPredicateMatch(TypeError, 'global_batch_size must be an int'):
            nn_impl_distribute.compute_average_loss(per_example_loss, global_batch_size=10.0)

    def testComputeAverageLossGlobalBatchSize_BatchSizeNegative(self):
        if False:
            return 10
        per_example_loss = [1, 2, 3, 4, 5]
        with self.assertRaisesWithPredicateMatch(errors_impl.InvalidArgumentError, 'global_batch_size must be non-negative'):
            nn_impl_distribute.compute_average_loss(per_example_loss, global_batch_size=-1)

    def testComputeAverageLossGlobalBatchSize_BatchSizeZero(self):
        if False:
            for i in range(10):
                print('nop')
        per_example_loss = [1, 2, 3, 4, 5]
        loss = nn_impl_distribute.compute_average_loss(per_example_loss, global_batch_size=0)
        self.assertEqual(self.evaluate(loss), 0.0)

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_cpus], mode=['graph', 'eager']))
    def testComputeAverageLossDefaultGlobalBatchSize(self, distribution):
        if False:
            while True:
                i = 10
        per_example_loss = constant_op.constant([2.5, 6.2, 5.0])
        loss = nn_impl_distribute.compute_average_loss(per_example_loss)
        self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.0) / 3)
        with distribution.scope():
            per_replica_losses = distribution.run(nn_impl_distribute.compute_average_loss, args=(per_example_loss,))
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.0) / 3)

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_cpus], mode=['graph', 'eager']))
    def testComputeAverageLossDefaultGlobalBatchSizeEmptyBatch(self, distribution):
        if False:
            i = 10
            return i + 15
        per_example_loss = constant_op.constant([], dtypes.float32)
        loss = nn_impl_distribute.compute_average_loss(per_example_loss)
        self.assertEqual(self.evaluate(loss), 0.0)
        with distribution.scope():
            per_replica_losses = distribution.run(nn_impl_distribute.compute_average_loss, args=(per_example_loss,))
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertAllClose(self.evaluate(loss), 0.0)

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_cpus], mode=['graph', 'eager']))
    def testComputeAverageLossSampleWeights(self, distribution):
        if False:
            while True:
                i = 10
        with distribution.scope():
            per_replica_losses = distribution.run(nn_impl_distribute.compute_average_loss, args=([2.0, 4.0, 6.0],), kwargs={'sample_weight': 2})
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertAllClose(self.evaluate(loss), (2.0 + 4.0 + 6.0) * 2.0 / 3)
            per_replica_losses = distribution.run(nn_impl_distribute.compute_average_loss, args=([2.0, 4.0, 6.0],), kwargs={'sample_weight': [0.3, 0.5, 0.2]})
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertAllClose(self.evaluate(loss), (2.0 * 0.3 + 4.0 * 0.5 + 6.0 * 0.2) / 3)
            per_replica_losses = distribution.run(nn_impl_distribute.compute_average_loss, args=([[2.0, 0.5], [4.0, 1.0]],), kwargs={'sample_weight': [[0.3, 0.7], [0.2, 0.8]]})
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertAllClose(self.evaluate(loss), (2.0 * 0.3 + 0.5 * 0.7 + 4.0 * 0.2 + 1.0 * 0.8) / 2)

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_cpus], mode=['graph', 'eager']))
    def testComputeAverageLossSampleWeightsEmptyBatch(self, distribution):
        if False:
            print('Hello World!')
        empty_rank0 = constant_op.constant([], dtypes.float32)
        with distribution.scope():
            per_replica_losses = distribution.run(nn_impl_distribute.compute_average_loss, args=(empty_rank0,), kwargs={'sample_weight': 2})
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertAllClose(self.evaluate(loss), 0.0)
            per_replica_losses = distribution.run(nn_impl_distribute.compute_average_loss, args=(empty_rank0,), kwargs={'sample_weight': empty_rank0})
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertAllClose(self.evaluate(loss), 0.0)

    def testComputeAverageLossInvalidSampleWeights(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesIncompatibleShapesError((ValueError, errors_impl.InvalidArgumentError)):
            nn_impl_distribute.compute_average_loss([2.5, 6.2, 5.0], sample_weight=[0.2, 0.8], global_batch_size=10)

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_cpus], mode=['graph', 'eager']))
    def testComputeAverageLossDtype(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        with distribution.scope():
            per_example_loss = constant_op.constant([2.0, 4.0, 6.0], dtype=dtypes.float64)
            per_replica_losses = distribution.run(nn_impl_distribute.compute_average_loss, args=(per_example_loss,), kwargs={'sample_weight': 2})
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertEqual(loss.dtype, dtypes.float64)

    def testComputeAverageLossInvalidRank(self):
        if False:
            return 10
        per_example_loss = constant_op.constant(2.0)
        with self.assertRaisesRegex(ValueError, 'Invalid value passed for `per_example_loss`. Expected a tensor with at least rank 1.'):
            nn_impl_distribute.compute_average_loss(per_example_loss)
        with context.graph_mode():
            per_example_loss = array_ops.placeholder(dtype=dtypes.float32)
            loss = nn_impl_distribute.compute_average_loss(per_example_loss)
            with self.cached_session() as sess:
                with self.assertRaisesRegex(errors.InvalidArgumentError, 'Invalid value passed for `per_example_loss`. Expected a tensor with at least rank 1.'):
                    sess.run(loss, {per_example_loss: 2})

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_cpus], mode=['graph', 'eager']))
    def testComputeAverageLossInCrossReplicaContext(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        with distribution.scope():
            with self.assertRaisesRegex(RuntimeError, 'You are calling `compute_average_loss` in cross replica context'):
                nn_impl_distribute.compute_average_loss([2, 3])

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_cpus], mode=['graph', 'eager']))
    def testScaleRegularizationLoss(self, distribution):
        if False:
            return 10
        reg_losses = constant_op.constant([2.5, 6.2, 5.0])
        loss = nn_impl_distribute.scale_regularization_loss(reg_losses)
        self.assertAllClose(self.evaluate(loss), 2.5 + 6.2 + 5.0)
        with distribution.scope():
            per_replica_losses = distribution.run(nn_impl_distribute.scale_regularization_loss, args=(reg_losses,))
            loss = distribution.reduce('SUM', per_replica_losses, axis=None)
            self.assertAllClose(self.evaluate(loss), 2.5 + 6.2 + 5.0)

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_cpus], mode=['graph', 'eager']))
    def testScaleRegularizationLossInCrossReplicaContext(self, distribution):
        if False:
            return 10
        with distribution.scope():
            with self.assertRaisesRegex(RuntimeError, 'You are calling `scale_regularization_loss` in cross replica context'):
                nn_impl_distribute.scale_regularization_loss([2, 3])
if __name__ == '__main__':
    test_lib.main()