import unittest
from pyspark.sql.tests.pandas.test_pandas_map import MapInPandasTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class MapInPandasParityTests(MapInPandasTestsMixin, ReusedConnectTestCase):

    def test_other_than_dataframe_iter(self):
        if False:
            while True:
                i = 10
        self.check_other_than_dataframe_iter()

    def test_dataframes_with_other_column_names(self):
        if False:
            return 10
        self.check_dataframes_with_other_column_names()

    def test_dataframes_with_duplicate_column_names(self):
        if False:
            return 10
        self.check_dataframes_with_duplicate_column_names()

    def test_dataframes_with_less_columns(self):
        if False:
            while True:
                i = 10
        self.check_dataframes_with_less_columns()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_dataframes_with_incompatible_types(self):
        if False:
            print('Hello World!')
        self.check_dataframes_with_incompatible_types()

    def test_empty_dataframes_with_less_columns(self):
        if False:
            return 10
        self.check_empty_dataframes_with_less_columns()

    def test_empty_dataframes_with_other_columns(self):
        if False:
            return 10
        self.check_empty_dataframes_with_other_columns()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_pandas_map import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)