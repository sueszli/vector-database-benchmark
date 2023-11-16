import unittest
from pyspark.sql.tests.pandas.test_pandas_udf_grouped_agg import GroupedAggPandasUDFTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class PandasUDFGroupedAggParityTests(GroupedAggPandasUDFTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_unsupported_types(self):
        if False:
            print('Hello World!')
        self.check_unsupported_types()

    def test_invalid_args(self):
        if False:
            print('Hello World!')
        self.check_invalid_args()

    @unittest.skip("Spark Connect doesn't support RDD but the test depends on it.")
    def test_grouped_with_empty_partition(self):
        if False:
            i = 10
            return i + 15
        super().test_grouped_with_empty_partition()

    @unittest.skip('Spark Connect does not support convert UNPARSED to catalyst types.')
    def test_manual(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_manual()
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_parity_pandas_udf_grouped_agg import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)