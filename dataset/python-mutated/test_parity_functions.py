import unittest
from pyspark.sql.tests.test_functions import FunctionsTestsMixin
from pyspark.testing.connectutils import should_test_connect, ReusedConnectTestCase
if should_test_connect:
    from pyspark.errors.exceptions.connect import SparkConnectException
    from pyspark.sql.connect.column import Column

class FunctionsParityTests(FunctionsTestsMixin, ReusedConnectTestCase):

    def test_assert_true(self):
        if False:
            print('Hello World!')
        self.check_assert_true(SparkConnectException)

    @unittest.skip('Spark Connect does not support Spark Context but the test depends on that.')
    def test_basic_functions(self):
        if False:
            while True:
                i = 10
        super().test_basic_functions()

    @unittest.skip('Spark Connect does not support Spark Context but the test depends on that.')
    def test_function_parity(self):
        if False:
            while True:
                i = 10
        super().test_function_parity()

    @unittest.skip('Spark Connect does not support Spark Context but the test depends on that.')
    def test_input_file_name_reset_for_rdd(self):
        if False:
            print('Hello World!')
        super().test_input_file_name_reset_for_rdd()

    def test_raise_error(self):
        if False:
            i = 10
            return i + 15
        self.check_raise_error(SparkConnectException)

    def test_sorting_functions_with_column(self):
        if False:
            print('Hello World!')
        self.check_sorting_functions_with_column(Column)
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_functions import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)