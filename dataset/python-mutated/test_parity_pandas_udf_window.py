import unittest
from pyspark.sql.tests.pandas.test_pandas_udf_window import WindowPandasUDFTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class PandasUDFWindowParityTests(WindowPandasUDFTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_invalid_args(self):
        if False:
            i = 10
            return i + 15
        self.check_invalid_args()
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_parity_pandas_udf_window import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)