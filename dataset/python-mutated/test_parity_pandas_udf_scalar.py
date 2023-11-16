import unittest
from pyspark.sql.tests.pandas.test_pandas_udf_scalar import ScalarPandasUDFTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class PandasUDFScalarParityTests(ScalarPandasUDFTestsMixin, ReusedConnectTestCase):

    def test_nondeterministic_vectorized_udf_in_aggregate(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_nondeterministic_analysis_exception()

    @unittest.skip("Spark Connect doesn't support RDD but the test depends on it.")
    def test_vectorized_udf_empty_partition(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_vectorized_udf_empty_partition()

    @unittest.skip("Spark Connect doesn't support RDD but the test depends on it.")
    def test_vectorized_udf_struct_with_empty_partition(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_vectorized_udf_struct_with_empty_partition()

    def test_vectorized_udf_exception(self):
        if False:
            i = 10
            return i + 15
        self.check_vectorized_udf_exception()

    def test_vectorized_udf_nested_struct(self):
        if False:
            return 10
        self.check_vectorized_udf_nested_struct()

    def test_vectorized_udf_return_scalar(self):
        if False:
            while True:
                i = 10
        self.check_vectorized_udf_return_scalar()

    def test_scalar_iter_udf_close(self):
        if False:
            while True:
                i = 10
        self.check_scalar_iter_udf_close()

    @unittest.skip('Fails in Spark Connect, should enable.')
    def test_vectorized_udf_wrong_return_type(self):
        if False:
            i = 10
            return i + 15
        self.check_vectorized_udf_wrong_return_type()

    def test_vectorized_udf_invalid_length(self):
        if False:
            i = 10
            return i + 15
        self.check_vectorized_udf_invalid_length()
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_parity_pandas_udf_scalar import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)