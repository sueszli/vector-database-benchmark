import unittest
from pyspark.pandas.tests.plot.test_series_plot_matplotlib import SeriesPlotMatplotlibTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils, TestUtils

class SeriesPlotMatplotlibParityTests(SeriesPlotMatplotlibTestsMixin, PandasOnSparkTestUtils, TestUtils, ReusedConnectTestCase):

    @unittest.skip('Test depends on Spark ML which is not supported from Spark Connect.')
    def test_hist(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_hist()

    @unittest.skip('Test depends on Spark ML which is not supported from Spark Connect.')
    def test_hist_plot(self):
        if False:
            i = 10
            return i + 15
        super().test_hist_plot()

    @unittest.skip('Test depends on Spark ML which is not supported from Spark Connect.')
    def test_kde_plot(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_kde_plot()

    @unittest.skip('Test depends on Spark ML which is not supported from Spark Connect.')
    def test_single_value_hist(self):
        if False:
            for i in range(10):
                print('nop')
        super().test_single_value_hist()
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.plot.test_parity_series_plot_matplotlib import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)