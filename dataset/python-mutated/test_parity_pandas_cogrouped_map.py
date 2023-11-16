import unittest
from pyspark.sql.tests.pandas.test_pandas_cogrouped_map import CogroupedApplyInPandasTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class CogroupedApplyInPandasTests(CogroupedApplyInPandasTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_different_group_key_cardinality(self):
        if False:
            i = 10
            return i + 15
        self.check_different_group_key_cardinality()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_wrong_args(self):
        if False:
            return 10
        self.check_wrong_args()

    def test_apply_in_pandas_not_returning_pandas_dataframe(self):
        if False:
            i = 10
            return i + 15
        self.check_apply_in_pandas_not_returning_pandas_dataframe()

    def test_apply_in_pandas_returning_wrong_column_names(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_apply_in_pandas_returning_wrong_column_names()

    def test_apply_in_pandas_returning_no_column_names_and_wrong_amount(self):
        if False:
            i = 10
            return i + 15
        self.check_apply_in_pandas_returning_no_column_names_and_wrong_amount()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_apply_in_pandas_returning_incompatible_type(self):
        if False:
            while True:
                i = 10
        self.check_apply_in_pandas_returning_incompatible_type()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_wrong_return_type(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_wrong_return_type()
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_parity_pandas_cogrouped_map import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)