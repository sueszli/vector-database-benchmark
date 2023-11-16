import unittest
from pyspark.sql.tests.test_readwriter import ReadwriterTestsMixin, ReadwriterV2TestsMixin
from pyspark.testing.connectutils import should_test_connect, ReusedConnectTestCase
if should_test_connect:
    from pyspark.sql.connect.readwriter import DataFrameWriterV2

class ReadwriterParityTests(ReadwriterTestsMixin, ReusedConnectTestCase):
    pass

class ReadwriterV2ParityTests(ReadwriterV2TestsMixin, ReusedConnectTestCase):

    def test_api(self):
        if False:
            return 10
        self.check_api(DataFrameWriterV2)

    def test_partitioning_functions(self):
        if False:
            return 10
        self.check_partitioning_functions(DataFrameWriterV2)
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_readwriter import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)