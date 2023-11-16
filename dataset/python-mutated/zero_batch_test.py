"""Test DistributionStrategy in the zero batch case."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

class NormalizationTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.one_device_strategy], mode=['graph'], fused=[True, False]))
    def testBNWithZeroBatchInputGraph(self, distribution, fused):
        if False:
            print('Hello World!')
        distribution.extended.experimental_enable_get_next_as_optional = True
        with distribution.scope(), self.cached_session() as sess:
            bn_list = []
            inputs = np.random.random((0, 4, 4, 3)) + 100
            targets = np.random.random((0, 4, 4, 3))
            inputs_placeholder = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 4, 4, 3])
            targets_placeholder = array_ops.placeholder(dtype=dtypes.float32, shape=[None, 4, 4, 3])

            def step_fn(is_training, inputs, targets=None):
                if False:
                    for i in range(10):
                        print('nop')
                bn = normalization.BatchNormalization(axis=3, epsilon=0.001, momentum=0.9, fused=fused)
                bn_list.append(bn)
                outputs = bn.apply(inputs, training=is_training)
                if not is_training:
                    return outputs
                loss = losses.mean_squared_error(targets, outputs)
                optimizer = gradient_descent.GradientDescentOptimizer(0.01)
                train_op = optimizer.minimize(loss)
                with ops.control_dependencies([train_op]):
                    return array_ops.identity(loss)
            train_op = distribution.extended.call_for_each_replica(step_fn, args=(True, inputs_placeholder, targets_placeholder))
            predict_op = distribution.extended.call_for_each_replica(step_fn, args=(False, inputs_placeholder))
            bn = bn_list[0]
            self.evaluate(variables.global_variables_initializer())
            (moving_mean, moving_var) = self.evaluate([bn.moving_mean, bn.moving_variance])
            self.assertAllEqual([0, 0, 0], moving_mean)
            self.assertAllEqual([1, 1, 1], moving_var)
            (np_gamma, np_beta) = self.evaluate([bn.gamma, bn.beta])
            self.assertAllEqual([1, 1, 1], np_gamma)
            self.assertAllEqual([0, 0, 0], np_beta)
            for _ in range(100):
                (np_output, _, _) = sess.run([train_op] + bn.updates, {inputs_placeholder: inputs, targets_placeholder: targets})
                self.assertEqual(0.0, np_output)
            (moving_mean, moving_var) = self.evaluate([bn.moving_mean, bn.moving_variance])
            self.assertAllEqual([0, 0, 0], moving_mean)
            self.assertAllEqual([1, 1, 1], moving_var)
            (np_gamma, np_beta) = self.evaluate([bn.gamma, bn.beta])
            self.assertAllEqual([1, 1, 1], np_gamma)
            self.assertAllEqual([0, 0, 0], np_beta)
            np_output = sess.run(predict_op, {inputs_placeholder: inputs})
            self.assertEqual([], np_output.tolist())

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.one_device_strategy], mode=['eager'], fused=[True, False]))
    def testBNWithZeroBatchInput(self, distribution, fused):
        if False:
            while True:
                i = 10
        distribution.extended.experimental_enable_get_next_as_optional = True
        with distribution.scope():
            inputs = np.random.random((0, 4, 4, 3)).astype(np.float32) + 100
            targets = np.random.random((0, 4, 4, 3)).astype(np.float32)
            bn = normalization.BatchNormalization(axis=3, epsilon=0.001, momentum=0.9, fused=fused)
            optimizer = gradient_descent.GradientDescentOptimizer(0.01)

            @def_function.function
            def train_step():
                if False:
                    print('Hello World!')

                def step_fn(inputs, targets):
                    if False:
                        for i in range(10):
                            print('nop')
                    with backprop.GradientTape() as tape:
                        outputs = bn.apply(inputs, training=True)
                        loss = losses.mean_squared_error(targets, outputs)
                    grads = tape.gradient(loss, bn.variables)
                    optimizer.apply_gradients(zip(grads, bn.variables))
                    return loss
                return distribution.run(step_fn, args=(inputs, targets))
            for _ in range(100):
                np_output = train_step().numpy()
                self.assertEqual(0.0, np_output)
            self.assertAllEqual([0, 0, 0], bn.moving_mean.numpy())
            self.assertAllEqual([1, 1, 1], bn.moving_variance.numpy())
            self.assertAllEqual([1, 1, 1], bn.gamma.numpy())
            self.assertAllEqual([0, 0, 0], bn.beta.numpy())

            @def_function.function
            def test_step():
                if False:
                    return 10

                def step_fn(inputs):
                    if False:
                        while True:
                            i = 10
                    outputs = bn.apply(inputs, training=False)
                    return outputs
                return distribution.run(step_fn, args=(inputs,))
            self.assertAllEqual(np.zeros(shape=(0, 4, 4, 3), dtype=np.float32), test_step().numpy())

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.one_device_strategy], mode=['eager'], fused=[True, False]))
    def testBNWithDynamicBatchInputEager(self, distribution, fused):
        if False:
            while True:
                i = 10
        distribution.extended.experimental_enable_get_next_as_optional = True
        with distribution.scope():
            inputs = np.random.random((11, 4, 4, 3)).astype(np.float32) + 100
            targets = np.random.random((11, 4, 4, 3)).astype(np.float32)
            dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets)).batch(10, drop_remainder=False).repeat()
            dataset_iterator = iter(distribution.experimental_distribute_dataset(dataset))
            bn = normalization.BatchNormalization(axis=-1, epsilon=0.001, momentum=0.9, fused=fused)
            optimizer = gradient_descent.GradientDescentOptimizer(0.01)

            @def_function.function
            def train_step(iterator):
                if False:
                    print('Hello World!')

                def step_fn(inputs):
                    if False:
                        i = 10
                        return i + 15
                    (features, targets) = inputs
                    with backprop.GradientTape() as tape:
                        outputs = bn(features, training=True)
                        loss = losses.mean_squared_error(targets, outputs)
                    grads = tape.gradient(loss, bn.variables)
                    optimizer.apply_gradients(zip(grads, bn.variables))
                    return loss
                return distribution.run(step_fn, args=(next(iterator),))
            for _ in range(100):
                train_step(dataset_iterator).numpy()
            self.assertNotAllEqual(np.ndarray([0, 0, 0]), bn.moving_mean.numpy())
            self.assertNotAllEqual(np.ndarray([1, 1, 1]), bn.moving_variance.numpy())
            self.assertNotAllEqual(np.ndarray([1, 1, 1]), bn.gamma.numpy())
            self.assertNotAllEqual(np.ndarray([0, 0, 0]), bn.beta.numpy())
if __name__ == '__main__':
    test_util.main()