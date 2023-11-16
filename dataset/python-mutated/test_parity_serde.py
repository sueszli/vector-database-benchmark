import unittest
from pyspark.sql.tests.test_serde import SerdeTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class SerdeParityTests(SerdeTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_int_array_serialization(self):
        if False:
            i = 10
            return i + 15
        super().test_int_array_serialization()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_serialize_nested_array_and_map(self):
        if False:
            while True:
                i = 10
        super().test_serialize_nested_array_and_map()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_serde import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)