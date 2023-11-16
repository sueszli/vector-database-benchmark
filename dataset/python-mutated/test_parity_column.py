import unittest
from pyspark.testing.connectutils import should_test_connect
if should_test_connect:
    from pyspark import sql
    from pyspark.sql.connect.column import Column
    sql.Column = Column
from pyspark.sql.tests.test_column import ColumnTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class ColumnParityTests(ColumnTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Requires JVM access.')
    def test_validate_column_types(self):
        if False:
            i = 10
            return i + 15
        super().test_validate_column_types()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_column import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)