"""Tests for optimizer_builder."""
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import optimizer_builder
from object_detection.protos import optimizer_pb2

class LearningRateBuilderTest(tf.test.TestCase):

    def testBuildConstantLearningRate(self):
        if False:
            while True:
                i = 10
        learning_rate_text_proto = '\n      constant_learning_rate {\n        learning_rate: 0.004\n      }\n    '
        learning_rate_proto = optimizer_pb2.LearningRate()
        text_format.Merge(learning_rate_text_proto, learning_rate_proto)
        learning_rate = optimizer_builder._create_learning_rate(learning_rate_proto)
        self.assertTrue(learning_rate.op.name.endswith('learning_rate'))
        with self.test_session():
            learning_rate_out = learning_rate.eval()
        self.assertAlmostEqual(learning_rate_out, 0.004)

    def testBuildExponentialDecayLearningRate(self):
        if False:
            for i in range(10):
                print('nop')
        learning_rate_text_proto = '\n      exponential_decay_learning_rate {\n        initial_learning_rate: 0.004\n        decay_steps: 99999\n        decay_factor: 0.85\n        staircase: false\n      }\n    '
        learning_rate_proto = optimizer_pb2.LearningRate()
        text_format.Merge(learning_rate_text_proto, learning_rate_proto)
        learning_rate = optimizer_builder._create_learning_rate(learning_rate_proto)
        self.assertTrue(learning_rate.op.name.endswith('learning_rate'))
        self.assertTrue(isinstance(learning_rate, tf.Tensor))

    def testBuildManualStepLearningRate(self):
        if False:
            print('Hello World!')
        learning_rate_text_proto = '\n      manual_step_learning_rate {\n        initial_learning_rate: 0.002\n        schedule {\n          step: 100\n          learning_rate: 0.006\n        }\n        schedule {\n          step: 90000\n          learning_rate: 0.00006\n        }\n        warmup: true\n      }\n    '
        learning_rate_proto = optimizer_pb2.LearningRate()
        text_format.Merge(learning_rate_text_proto, learning_rate_proto)
        learning_rate = optimizer_builder._create_learning_rate(learning_rate_proto)
        self.assertTrue(isinstance(learning_rate, tf.Tensor))

    def testBuildCosineDecayLearningRate(self):
        if False:
            print('Hello World!')
        learning_rate_text_proto = '\n      cosine_decay_learning_rate {\n        learning_rate_base: 0.002\n        total_steps: 20000\n        warmup_learning_rate: 0.0001\n        warmup_steps: 1000\n        hold_base_rate_steps: 20000\n      }\n    '
        learning_rate_proto = optimizer_pb2.LearningRate()
        text_format.Merge(learning_rate_text_proto, learning_rate_proto)
        learning_rate = optimizer_builder._create_learning_rate(learning_rate_proto)
        self.assertTrue(isinstance(learning_rate, tf.Tensor))

    def testRaiseErrorOnEmptyLearningRate(self):
        if False:
            while True:
                i = 10
        learning_rate_text_proto = '\n    '
        learning_rate_proto = optimizer_pb2.LearningRate()
        text_format.Merge(learning_rate_text_proto, learning_rate_proto)
        with self.assertRaises(ValueError):
            optimizer_builder._create_learning_rate(learning_rate_proto)

class OptimizerBuilderTest(tf.test.TestCase):

    def testBuildRMSPropOptimizer(self):
        if False:
            print('Hello World!')
        optimizer_text_proto = '\n      rms_prop_optimizer: {\n        learning_rate: {\n          exponential_decay_learning_rate {\n            initial_learning_rate: 0.004\n            decay_steps: 800720\n            decay_factor: 0.95\n          }\n        }\n        momentum_optimizer_value: 0.9\n        decay: 0.9\n        epsilon: 1.0\n      }\n      use_moving_average: false\n    '
        optimizer_proto = optimizer_pb2.Optimizer()
        text_format.Merge(optimizer_text_proto, optimizer_proto)
        (optimizer, _) = optimizer_builder.build(optimizer_proto)
        self.assertTrue(isinstance(optimizer, tf.train.RMSPropOptimizer))

    def testBuildMomentumOptimizer(self):
        if False:
            print('Hello World!')
        optimizer_text_proto = '\n      momentum_optimizer: {\n        learning_rate: {\n          constant_learning_rate {\n            learning_rate: 0.001\n          }\n        }\n        momentum_optimizer_value: 0.99\n      }\n      use_moving_average: false\n    '
        optimizer_proto = optimizer_pb2.Optimizer()
        text_format.Merge(optimizer_text_proto, optimizer_proto)
        (optimizer, _) = optimizer_builder.build(optimizer_proto)
        self.assertTrue(isinstance(optimizer, tf.train.MomentumOptimizer))

    def testBuildAdamOptimizer(self):
        if False:
            while True:
                i = 10
        optimizer_text_proto = '\n      adam_optimizer: {\n        learning_rate: {\n          constant_learning_rate {\n            learning_rate: 0.002\n          }\n        }\n      }\n      use_moving_average: false\n    '
        optimizer_proto = optimizer_pb2.Optimizer()
        text_format.Merge(optimizer_text_proto, optimizer_proto)
        (optimizer, _) = optimizer_builder.build(optimizer_proto)
        self.assertTrue(isinstance(optimizer, tf.train.AdamOptimizer))

    def testBuildMovingAverageOptimizer(self):
        if False:
            i = 10
            return i + 15
        optimizer_text_proto = '\n      adam_optimizer: {\n        learning_rate: {\n          constant_learning_rate {\n            learning_rate: 0.002\n          }\n        }\n      }\n      use_moving_average: True\n    '
        optimizer_proto = optimizer_pb2.Optimizer()
        text_format.Merge(optimizer_text_proto, optimizer_proto)
        (optimizer, _) = optimizer_builder.build(optimizer_proto)
        self.assertTrue(isinstance(optimizer, tf.contrib.opt.MovingAverageOptimizer))

    def testBuildMovingAverageOptimizerWithNonDefaultDecay(self):
        if False:
            print('Hello World!')
        optimizer_text_proto = '\n      adam_optimizer: {\n        learning_rate: {\n          constant_learning_rate {\n            learning_rate: 0.002\n          }\n        }\n      }\n      use_moving_average: True\n      moving_average_decay: 0.2\n    '
        optimizer_proto = optimizer_pb2.Optimizer()
        text_format.Merge(optimizer_text_proto, optimizer_proto)
        (optimizer, _) = optimizer_builder.build(optimizer_proto)
        self.assertTrue(isinstance(optimizer, tf.contrib.opt.MovingAverageOptimizer))
        self.assertAlmostEqual(optimizer._ema._decay, 0.2)

    def testBuildEmptyOptimizer(self):
        if False:
            while True:
                i = 10
        optimizer_text_proto = '\n    '
        optimizer_proto = optimizer_pb2.Optimizer()
        text_format.Merge(optimizer_text_proto, optimizer_proto)
        with self.assertRaises(ValueError):
            optimizer_builder.build(optimizer_proto)
if __name__ == '__main__':
    tf.test.main()