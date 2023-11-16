import unittest
from pyspark.sql import SparkSession
from pyspark.testing.connectutils import should_test_connect, connect_requirement_message
if should_test_connect:
    from pyspark.ml.tests.connect.test_legacy_mode_pipeline import PipelineTestsMixin

@unittest.skipIf(not should_test_connect, connect_requirement_message)
class PipelineTestsOnConnect(PipelineTestsMixin, unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.spark = SparkSession.builder.remote('local[2]').config('spark.connect.copyFromLocalToFs.allowDestLocal', 'true').getOrCreate()

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        self.spark.stop()
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_connect_pipeline import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)