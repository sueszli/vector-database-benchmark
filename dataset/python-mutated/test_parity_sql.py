import unittest
from pyspark.pandas.tests.test_sql import SQLTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class SQLParityTests(SQLTestsMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @unittest.skip('Test depends on temp view issue on JVM side.')
    def test_sql_with_index_col(self):
        if False:
            return 10
        super().test_sql_with_index_col()

    @unittest.skip('Test depends on temp view issue on JVM side.')
    def test_sql_with_pandas_on_spark_objects(self):
        if False:
            return 10
        super().test_sql_with_pandas_on_spark_objects()
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.test_parity_sql import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)