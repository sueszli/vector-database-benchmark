import unittest
from pyspark.sql import SparkSession
from pyspark.testing.connectutils import should_test_connect, connect_requirement_message
have_torch = True
try:
    import torch
except ImportError:
    have_torch = False
if should_test_connect:
    from pyspark.ml.tests.connect.test_legacy_mode_classification import ClassificationTestsMixin

@unittest.skipIf(not should_test_connect or not have_torch, connect_requirement_message or 'torch is required')
class ClassificationTestsOnConnect(ClassificationTestsMixin, unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        self.spark = SparkSession.builder.remote('local[2]').config('spark.connect.copyFromLocalToFs.allowDestLocal', 'true').getOrCreate()

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        self.spark.stop()
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_connect_classification import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)