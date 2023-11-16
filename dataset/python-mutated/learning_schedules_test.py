"""Tests for object_detection.utils.learning_schedules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
import tensorflow as tf
from object_detection.utils import learning_schedules
from object_detection.utils import test_case

class LearningSchedulesTest(test_case.TestCase):

    def testExponentialDecayWithBurnin(self):
        if False:
            print('Hello World!')

        def graph_fn(global_step):
            if False:
                print('Hello World!')
            learning_rate_base = 1.0
            learning_rate_decay_steps = 3
            learning_rate_decay_factor = 0.1
            burnin_learning_rate = 0.5
            burnin_steps = 2
            min_learning_rate = 0.05
            learning_rate = learning_schedules.exponential_decay_with_burnin(global_step, learning_rate_base, learning_rate_decay_steps, learning_rate_decay_factor, burnin_learning_rate, burnin_steps, min_learning_rate)
            assert learning_rate.op.name.endswith('learning_rate')
            return (learning_rate,)
        output_rates = [self.execute(graph_fn, [np.array(i).astype(np.int64)]) for i in range(9)]
        exp_rates = [0.5, 0.5, 1, 1, 1, 0.1, 0.1, 0.1, 0.05]
        self.assertAllClose(output_rates, exp_rates, rtol=0.0001)

    def testCosineDecayWithWarmup(self):
        if False:
            while True:
                i = 10

        def graph_fn(global_step):
            if False:
                print('Hello World!')
            learning_rate_base = 1.0
            total_steps = 100
            warmup_learning_rate = 0.1
            warmup_steps = 9
            learning_rate = learning_schedules.cosine_decay_with_warmup(global_step, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps)
            assert learning_rate.op.name.endswith('learning_rate')
            return (learning_rate,)
        exp_rates = [0.1, 0.5, 0.9, 1.0, 0]
        input_global_steps = [0, 4, 8, 9, 100]
        output_rates = [self.execute(graph_fn, [np.array(step).astype(np.int64)]) for step in input_global_steps]
        self.assertAllClose(output_rates, exp_rates)

    def testCosineDecayAfterTotalSteps(self):
        if False:
            return 10

        def graph_fn(global_step):
            if False:
                return 10
            learning_rate_base = 1.0
            total_steps = 100
            warmup_learning_rate = 0.1
            warmup_steps = 9
            learning_rate = learning_schedules.cosine_decay_with_warmup(global_step, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps)
            assert learning_rate.op.name.endswith('learning_rate')
            return (learning_rate,)
        exp_rates = [0]
        input_global_steps = [101]
        output_rates = [self.execute(graph_fn, [np.array(step).astype(np.int64)]) for step in input_global_steps]
        self.assertAllClose(output_rates, exp_rates)

    def testCosineDecayWithHoldBaseLearningRateSteps(self):
        if False:
            print('Hello World!')

        def graph_fn(global_step):
            if False:
                for i in range(10):
                    print('nop')
            learning_rate_base = 1.0
            total_steps = 120
            warmup_learning_rate = 0.1
            warmup_steps = 9
            hold_base_rate_steps = 20
            learning_rate = learning_schedules.cosine_decay_with_warmup(global_step, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps, hold_base_rate_steps)
            assert learning_rate.op.name.endswith('learning_rate')
            return (learning_rate,)
        exp_rates = [0.1, 0.5, 0.9, 1.0, 1.0, 1.0, 0.999702, 0.874255, 0.577365, 0.0]
        input_global_steps = [0, 4, 8, 9, 10, 29, 30, 50, 70, 120]
        output_rates = [self.execute(graph_fn, [np.array(step).astype(np.int64)]) for step in input_global_steps]
        self.assertAllClose(output_rates, exp_rates)

    def testManualStepping(self):
        if False:
            for i in range(10):
                print('nop')

        def graph_fn(global_step):
            if False:
                for i in range(10):
                    print('nop')
            boundaries = [2, 3, 7]
            rates = [1.0, 2.0, 3.0, 4.0]
            learning_rate = learning_schedules.manual_stepping(global_step, boundaries, rates)
            assert learning_rate.op.name.endswith('learning_rate')
            return (learning_rate,)
        output_rates = [self.execute(graph_fn, [np.array(i).astype(np.int64)]) for i in range(10)]
        exp_rates = [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0]
        self.assertAllClose(output_rates, exp_rates)

    def testManualSteppingWithWarmup(self):
        if False:
            return 10

        def graph_fn(global_step):
            if False:
                while True:
                    i = 10
            boundaries = [4, 6, 8]
            rates = [0.02, 0.1, 0.01, 0.001]
            learning_rate = learning_schedules.manual_stepping(global_step, boundaries, rates, warmup=True)
            assert learning_rate.op.name.endswith('learning_rate')
            return (learning_rate,)
        output_rates = [self.execute(graph_fn, [np.array(i).astype(np.int64)]) for i in range(9)]
        exp_rates = [0.02, 0.04, 0.06, 0.08, 0.1, 0.1, 0.01, 0.01, 0.001]
        self.assertAllClose(output_rates, exp_rates)

    def testManualSteppingWithZeroBoundaries(self):
        if False:
            for i in range(10):
                print('nop')

        def graph_fn(global_step):
            if False:
                while True:
                    i = 10
            boundaries = []
            rates = [0.01]
            learning_rate = learning_schedules.manual_stepping(global_step, boundaries, rates)
            return (learning_rate,)
        output_rates = [self.execute(graph_fn, [np.array(i).astype(np.int64)]) for i in range(4)]
        exp_rates = [0.01] * 4
        self.assertAllClose(output_rates, exp_rates)
if __name__ == '__main__':
    tf.test.main()