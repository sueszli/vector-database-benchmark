import unittest
from pyspark.sql import SparkSession
from pyspark.ml.torch.tests.test_data_loader import TorchDistributorDataLoaderUnitTests
have_torch = True
try:
    import torch
except ImportError:
    have_torch = False

@unittest.skipIf(not have_torch, 'torch is required')
class TorchDistributorBaselineUnitTestsOnConnect(TorchDistributorDataLoaderUnitTests):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.spark = SparkSession.builder.remote('local[1]').config('spark.default.parallelism', '1').getOrCreate()
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_parity_torch_data_loader import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)