import unittest
from pyspark.pandas.tests.test_default_index import DefaultIndexTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class DefaultIndexParityTests(DefaultIndexTestsMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @unittest.skip('Test depends on SparkContext which is not supported from Spark Connect.')
    def test_index_distributed_sequence_cleanup(self):
        if False:
            print('Hello World!')
        super().test_index_distributed_sequence_cleanup()
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.test_parity_default_index import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)