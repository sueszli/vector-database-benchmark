import unittest
from pyspark.sql.tests.test_datasources import DataSourcesTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase

class DataSourcesParityTests(DataSourcesTestsMixin, ReusedConnectTestCase):

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_csv_sampling_ratio(self):
        if False:
            return 10
        super().test_csv_sampling_ratio()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_json_sampling_ratio(self):
        if False:
            return 10
        super().test_json_sampling_ratio()

    @unittest.skip('Spark Connect does not support RDD but the tests depend on them.')
    def test_xml_sampling_ratio(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_xml_sampling_ratio()
if __name__ == '__main__':
    import unittest
    from pyspark.sql.tests.connect.test_parity_datasources import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)