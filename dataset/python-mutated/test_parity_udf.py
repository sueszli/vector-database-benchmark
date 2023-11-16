import unittest
from pyspark.testing.connectutils import should_test_connect
if should_test_connect:
    from pyspark import sql
    from pyspark.sql.connect.udf import UserDefinedFunction
    sql.udf.UserDefinedFunction = UserDefinedFunction
from pyspark.sql.tests.test_udf import BaseUDFTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class UDFParityTests(BaseUDFTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Spark Connect does not support mapPartitions() but the test depends on it.')
    def test_worker_original_stdin_closed(self):
        if False:
            i = 10
            return i + 15
        super().test_worker_original_stdin_closed()

    @unittest.skip('Spark Connect does not support reading from Hadoop RDD but the test depends on it.')
    def test_udf_with_input_file_name_for_hadooprdd(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_udf_with_input_file_name_for_hadooprdd()

    @unittest.skip('Spark Connect does not support accumulator but the test depends on it.')
    def test_same_accumulator_in_udfs(self):
        if False:
            i = 10
            return i + 15
        super().test_same_accumulator_in_udfs()

    @unittest.skip('Spark Connect does not support spark.conf but the test depends on it.')
    def test_udf_timestamp_ntz(self):
        if False:
            print('Hello World!')
        super().test_udf_timestamp_ntz()

    @unittest.skip('Spark Connect does not support broadcast but the test depends on it.')
    def test_broadcast_in_udf(self):
        if False:
            while True:
                i = 10
        super().test_broadcast_in_udf()

    @unittest.skip('Spark Connect does not support cache() but the test depends on it.')
    def test_udf_cache(self):
        if False:
            print('Hello World!')
        super().test_udf_cache()

    @unittest.skip('Requires JVM access.')
    def test_udf_defers_judf_initialization(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_udf_defers_judf_initialization()

    @unittest.skip('Requires JVM access.')
    def test_nondeterministic_udf3(self):
        if False:
            return 10
        super().test_nondeterministic_udf3()

    def test_nondeterministic_udf_in_aggregate(self):
        if False:
            while True:
                i = 10
        self.check_nondeterministic_udf_in_aggregate()

    def test_udf_registration_return_type_not_none(self):
        if False:
            while True:
                i = 10
        self.check_udf_registration_return_type_not_none()

    @unittest.skip("Spark Connect doesn't support RDD but the test depends on it.")
    def test_worker_original_stdin_closed(self):
        if False:
            while True:
                i = 10
        super().test_worker_original_stdin_closed()

    @unittest.skip('Spark Connect does not support SQLContext but the test depends on it.')
    def test_udf_on_sql_context(self):
        if False:
            i = 10
            return i + 15
        super().test_udf_on_sql_context()

    @unittest.skip('Spark Connect does not support SQLContext but the test depends on it.')
    def test_non_existed_udf_with_sql_context(self):
        if False:
            return 10
        super().test_non_existed_udf_with_sql_context()

    @unittest.skip('Spark Connect does not support SQLContext but the test depends on it.')
    def test_udf_registration_returns_udf_on_sql_context(self):
        if False:
            i = 10
            return i + 15
        super().test_udf_registration_returns_udf_on_sql_context()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_udf import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)