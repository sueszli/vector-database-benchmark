import unittest
from pyspark.pandas.tests.test_frame_spark import SparkFrameMethodsTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils, TestUtils

class SparkFrameMethodsParityTests(SparkFrameMethodsTestsMixin, TestUtils, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @unittest.skip('Test depends on checkpoint which is not supported from Spark Connect.')
    def test_checkpoint(self):
        if False:
            return 10
        super().test_checkpoint()

    @unittest.skip('Test depends on RDD which is not supported from Spark Connect.')
    def test_coalesce(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_coalesce()

    @unittest.skip('Test depends on localCheckpoint which is not supported from Spark Connect.')
    def test_local_checkpoint(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_local_checkpoint()

    @unittest.skip('Test depends on RDD which is not supported from Spark Connect.')
    def test_repartition(self):
        if False:
            print('Hello World!')
        super().test_repartition()
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.test_parity_frame_spark import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)