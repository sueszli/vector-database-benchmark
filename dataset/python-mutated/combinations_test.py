"""Tests for tensorflow.python.distribute.combinations."""
import importlib
import os
import sys
import unittest
from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.platform import test

class ClusterCombinationTest(test.TestCase, parameterized.TestCase):

    @framework_combinations.generate(framework_combinations.combine(distribution=[combinations.NamedDistribution('HasClusterParams', lambda : None, has_chief=True, num_workers=2)]), test_combinations=(combinations.ClusterCombination(),))
    def testClusterParams(self, distribution, has_chief, num_workers):
        if False:
            print('Hello World!')
        self.assertTrue(has_chief)
        self.assertEqual(num_workers, 2)

    @framework_combinations.generate(framework_combinations.combine(distribution=[combinations.NamedDistribution('NoClusterParams', lambda : None)]), test_combinations=(combinations.ClusterCombination(),))
    def testClusterParamsHasDefault(self, distribution, has_chief, num_workers):
        if False:
            print('Hello World!')
        self.assertFalse(has_chief)
        self.assertEqual(num_workers, 1)

    @framework_combinations.generate(framework_combinations.combine(v=1), test_combinations=(combinations.ClusterCombination(),))
    def testClusterParamsNoStrategy(self, v, has_chief, num_workers):
        if False:
            while True:
                i = 10
        self.assertFalse(has_chief)
        self.assertEqual(num_workers, 1)

    @framework_combinations.generate(framework_combinations.combine(distribution=[combinations.NamedDistribution('WithClusterParams', lambda : None, has_chief=True, num_workers=2), combinations.NamedDistribution('WithoutClusterParams', lambda : None)]), test_combinations=(combinations.ClusterCombination(),))
    def testClusterParamsAreOptional(self, distribution):
        if False:
            while True:
                i = 10
        pass

    @framework_combinations.generate(framework_combinations.combine(ds1=combinations.NamedDistribution('Strategy1', lambda : None, has_chief=True, num_workers=0), ds2=combinations.NamedDistribution('Strategy2', lambda : None, has_chief=False, num_workers=1), ds3=combinations.NamedDistribution('Strategy3', lambda : None, has_chief=True, num_workers=0)), test_combinations=(combinations.ClusterCombination(),))
    def testMultipleDistributionSingleWorker(self, ds1, ds2, ds3):
        if False:
            while True:
                i = 10
        pass

    @combinations.generate(combinations.combine(num_workers=2))
    def testUseWithoutStrategy(self):
        if False:
            return 10
        self.assertNotEqual(os.getenv('TF_CONFIG'), '')

@combinations.generate(combinations.combine(num_workers=2))
class ClusterCombinationTestEnvTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        if combinations.in_main_process():
            combinations.env().tf_data_service_dispatcher = 'localhost'

    def testTfDataServiceDispatcher(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(combinations.env().tf_data_service_dispatcher, 'localhost')

    def testUpdateEnvInWorker(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            combinations.env().tf_data_service_dispatcher = 'localhost'

@unittest.expectedFailure
class ClusterParametersShouldFailTest(test.TestCase, parameterized.TestCase):

    @framework_combinations.generate(framework_combinations.combine(ds1=combinations.NamedDistribution('Strategy1', lambda : None, has_chief=True, num_workers=2), ds2=combinations.NamedDistribution('Strategy2', lambda : None, has_chief=True, num_workers=2)), test_combinations=(combinations.ClusterCombination(),))
    def testMultipleDistributionMultiWorker(self, ds1, ds2):
        if False:
            print('Hello World!')
        pass

@unittest.expectedFailure
class CombinationsExpectedFailureTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(distribution=[combinations.NamedDistribution('OneChiefOneWorker', lambda : None, has_chief=True, num_workers=1), combinations.NamedDistribution('TwoWorkers', lambda : None, has_chief=False, num_workers=2)]))
    def testMultiWorkerCanFail(self, distribution):
        if False:
            return 10
        resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertIsNone(resolver.task_id)

@unittest.expectedFailure
@combinations.generate(combinations.combine(distribution=[combinations.NamedDistribution('OneChiefOneWorker', lambda : None, has_chief=True, num_workers=1), combinations.NamedDistribution('TwoWorkers', lambda : None, has_chief=False, num_workers=2)]))
class CombinationsOnClassMultiWorkerExpectedFailureTest(test.TestCase, parameterized.TestCase):

    def test(self, distribution):
        if False:
            while True:
                i = 10
        resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
        self.assertIsNone(resolver.task_id)

class TfFunctionTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(combinations.combine(tf_function_1=combinations.tf_function, tf_function_2=combinations.no_tf_function, mode='eager'))
    def testFunc(self, tf_function_1, tf_function_2):
        if False:
            while True:
                i = 10

        @tf_function_1
        def foo():
            if False:
                while True:
                    i = 10
            self.assertFalse(context.executing_eagerly())

        @tf_function_2
        def bar():
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(context.executing_eagerly())
        foo()
        bar()

class ModuleInitializingTest(test.TestCase, parameterized.TestCase):

    def testSysArgvClearedIsFine(self):
        if False:
            i = 10
            return i + 15
        original_argv = list(sys.argv)
        sys.argv.clear()
        importlib.reload(combinations)
        sys.argv = original_argv

class ShareGPUTest(test.TestCase, parameterized.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        if combinations.in_main_process():
            num_gpus = combinations.env().total_phsyical_gpus
            if num_gpus != 2 and num_gpus != 4:
                self.skipTest('requires 2 or 4 GPUs')

    @combinations.generate(combinations.combine(num_workers=2, required_gpus=1, share_gpu=True))
    def testShareGPU(self):
        if False:
            while True:
                i = 10
        self.assertLen(context.context().list_physical_devices('GPU'), combinations.env().total_phsyical_gpus)

    @combinations.generate(combinations.combine(num_workers=2, required_gpus=1))
    def testShareGPUByDefault(self):
        if False:
            i = 10
            return i + 15
        self.assertLen(context.context().list_physical_devices('GPU'), combinations.env().total_phsyical_gpus)

    @combinations.generate(combinations.combine(num_workers=2, required_gpus=1, share_gpu=False))
    def testNotShareGPU(self):
        if False:
            return 10
        self.assertLen(context.context().list_physical_devices('GPU'), combinations.env().total_phsyical_gpus / 2)
if __name__ == '__main__':
    test_util.main()