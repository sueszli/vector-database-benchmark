import unittest
from pyspark.sql import SparkSession
from pyspark.testing.connectutils import should_test_connect, connect_requirement_message
if should_test_connect:
    from pyspark.ml.tests.connect.test_legacy_mode_summarizer import SummarizerTestsMixin

@unittest.skipIf(not should_test_connect, connect_requirement_message)
class SummarizerTestsOnConnect(SummarizerTestsMixin, unittest.TestCase):

    def setUp(self) -> None:
        if False:
            return 10
        self.spark = SparkSession.builder.remote('local[2]').getOrCreate()

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.spark.stop()
if __name__ == '__main__':
    from pyspark.ml.tests.connect.test_connect_summarizer import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)