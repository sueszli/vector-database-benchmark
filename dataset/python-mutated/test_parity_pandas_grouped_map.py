import unittest
from pyspark.sql.tests.pandas.test_pandas_grouped_map import GroupedApplyInPandasTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class GroupedApplyInPandasTests(GroupedApplyInPandasTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_supported_types(self):
        if False:
            print('Hello World!')
        super().test_supported_types()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_wrong_return_type(self):
        if False:
            return 10
        self.check_wrong_return_type()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_wrong_args(self):
        if False:
            i = 10
            return i + 15
        self.check_wrong_args()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_unsupported_types(self):
        if False:
            return 10
        self.check_unsupported_types()

    def test_register_grouped_map_udf(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_register_grouped_map_udf()

    def test_column_order(self):
        if False:
            print('Hello World!')
        self.check_column_order()

    def test_apply_in_pandas_returning_wrong_column_names(self):
        if False:
            print('Hello World!')
        self.check_apply_in_pandas_returning_wrong_column_names()

    def test_apply_in_pandas_returning_no_column_names_and_wrong_amount(self):
        if False:
            i = 10
            return i + 15
        self.check_apply_in_pandas_returning_no_column_names_and_wrong_amount()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_apply_in_pandas_returning_incompatible_type(self):
        if False:
            return 10
        self.check_apply_in_pandas_returning_incompatible_type()

    def test_apply_in_pandas_not_returning_pandas_dataframe(self):
        if False:
            print('Hello World!')
        self.check_apply_in_pandas_not_returning_pandas_dataframe()

    @unittest.skip("Spark Connect doesn't support RDD but the test depends on it.")
    def test_grouped_with_empty_partition(self):
        if False:
            print('Hello World!')
        super().test_grouped_with_empty_partition()
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_parity_pandas_grouped_map import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)