import unittest
from pyspark.sql.tests.test_arrow_map import MapInArrowTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class ArrowMapParityTests(MapInArrowTestsMixin, ReusedConnectTestCase):

    def test_other_than_recordbatch_iter(self):
        if False:
            print('Hello World!')
        self.check_other_than_recordbatch_iter()
if __name__ == '__main__':
    from pyspark.sql.tests.connect.test_parity_arrow_map import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)