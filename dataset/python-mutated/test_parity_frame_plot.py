import unittest
from pyspark.pandas.tests.plot.test_frame_plot import DataFramePlotTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class DataFramePlotParityTests(DataFramePlotTestsMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @unittest.skip('Test depends on Spark ML which is not supported from Spark Connect.')
    def test_compute_hist_multi_columns(self):
        if False:
            while True:
                i = 10
        super().test_compute_hist_multi_columns()

    @unittest.skip('Test depends on Spark ML which is not supported from Spark Connect.')
    def test_compute_hist_single_column(self):
        if False:
            while True:
                i = 10
        super().test_compute_hist_single_column()
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.plot.test_parity_frame_plot import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)