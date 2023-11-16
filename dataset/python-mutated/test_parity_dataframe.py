import unittest
from pyspark.sql.tests.test_dataframe import DataFrameTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class DataFrameParityTests(DataFrameTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_help_command(self):
        if False:
            print('Hello World!')
        super().test_help_command()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_observe_str(self):
        if False:
            while True:
                i = 10
        super().test_observe_str()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_pandas_api(self):
        if False:
            while True:
                i = 10
        super().test_pandas_api()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_repartitionByRange_dataframe(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_repartitionByRange_dataframe()

    @unittest.skip('Spark Connect does not SparkContext but the tests depend on them.')
    def test_same_semantics_error(self):
        if False:
            return 10
        super().test_same_semantics_error()

    def test_sample(self):
        if False:
            return 10
        super().test_sample()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_toDF_with_schema_string(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_toDF_with_schema_string()

    def test_to_local_iterator_not_fully_consumed(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_to_local_iterator_not_fully_consumed()

    def test_to_pandas_for_array_of_struct(self):
        if False:
            for i in range(10):
                print('nop')
        super().check_to_pandas_for_array_of_struct(True)

    def test_to_pandas_from_null_dataframe(self):
        if False:
            print('Hello World!')
        self.check_to_pandas_from_null_dataframe()

    def test_to_pandas_on_cross_join(self):
        if False:
            return 10
        self.check_to_pandas_on_cross_join()

    def test_to_pandas_from_empty_dataframe(self):
        if False:
            i = 10
            return i + 15
        self.check_to_pandas_from_empty_dataframe()

    def test_to_pandas_with_duplicated_column_names(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_to_pandas_with_duplicated_column_names()

    def test_to_pandas_from_mixed_dataframe(self):
        if False:
            return 10
        self.check_to_pandas_from_mixed_dataframe()

    def test_toDF_with_string(self):
        if False:
            return 10
        super().test_toDF_with_string()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_dataframe import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)