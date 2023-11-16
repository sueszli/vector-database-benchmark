import os
import shutil
import unittest
have_torch = True
try:
    import torch
except ImportError:
    have_torch = False
from pyspark.sql import SparkSession
from pyspark.ml.torch.tests.test_distributor import TorchDistributorBaselineUnitTestsMixin, TorchDistributorLocalUnitTestsMixin, TorchDistributorDistributedUnitTestsMixin, TorchWrapperUnitTestsMixin, set_up_test_dirs, get_local_mode_conf, get_distributed_mode_conf

@unittest.skipIf(not have_torch, 'torch is required')
class TorchDistributorBaselineUnitTestsOnConnect(TorchDistributorBaselineUnitTestsMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        cls.spark = SparkSession.builder.remote('local[4]').getOrCreate()

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cls.spark.stop()

@unittest.skipIf(not have_torch, 'torch is required')
class TorchDistributorLocalUnitTestsOnConnect(TorchDistributorLocalUnitTestsMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        (cls.gpu_discovery_script_file_name, cls.mnist_dir_path) = set_up_test_dirs()
        builder = SparkSession.builder.appName(cls.__name__)
        for (k, v) in get_local_mode_conf().items():
            builder = builder.config(k, v)
        builder = builder.config('spark.driver.resource.gpu.discoveryScript', cls.gpu_discovery_script_file_name)
        cls.spark = builder.remote('local-cluster[2,2,512]').getOrCreate()

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        shutil.rmtree(cls.mnist_dir_path)
        os.unlink(cls.gpu_discovery_script_file_name)
        cls.spark.stop()

    def _get_inputs_for_test_local_training_succeeds(self):
        if False:
            while True:
                i = 10
        return [('0,1,2', 1, True, '0,1,2'), ('0,1,2', 3, True, '0,1,2'), ('0,1,2', 2, False, '0,1,2'), (None, 3, False, 'NONE')]

@unittest.skipIf(not have_torch, 'torch is required')
class TorchDistributorLocalUnitTestsIIOnConnect(TorchDistributorLocalUnitTestsMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        (cls.gpu_discovery_script_file_name, cls.mnist_dir_path) = set_up_test_dirs()
        builder = SparkSession.builder.appName(cls.__name__)
        for (k, v) in get_local_mode_conf().items():
            builder = builder.config(k, v)
        builder = builder.config('spark.driver.resource.gpu.discoveryScript', cls.gpu_discovery_script_file_name)
        cls.spark = builder.remote('local[4]').getOrCreate()

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        shutil.rmtree(cls.mnist_dir_path)
        os.unlink(cls.gpu_discovery_script_file_name)
        cls.spark.stop()

    def _get_inputs_for_test_local_training_succeeds(self):
        if False:
            while True:
                i = 10
        return [('0,1,2', 1, True, '0,1,2'), ('0,1,2', 3, True, '0,1,2'), ('0,1,2', 2, False, '0,1,2'), (None, 3, False, 'NONE')]

@unittest.skipIf(not have_torch, 'torch is required')
class TorchDistributorDistributedUnitTestsOnConnect(TorchDistributorDistributedUnitTestsMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        (cls.gpu_discovery_script_file_name, cls.mnist_dir_path) = set_up_test_dirs()
        builder = SparkSession.builder.appName(cls.__name__)
        for (k, v) in get_distributed_mode_conf().items():
            builder = builder.config(k, v)
        builder = builder.config('spark.worker.resource.gpu.discoveryScript', cls.gpu_discovery_script_file_name)
        cls.spark = builder.remote('local-cluster[2,2,512]').getOrCreate()

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        shutil.rmtree(cls.mnist_dir_path)
        os.unlink(cls.gpu_discovery_script_file_name)
        cls.spark.stop()

@unittest.skipIf(not have_torch, 'torch is required')
class TorchWrapperUnitTestsOnConnect(TorchWrapperUnitTestsMixin, unittest.TestCase):
    pass
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_parity_torch_distributor import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)