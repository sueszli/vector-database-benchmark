import unittest
from pyspark.pandas.tests.plot.test_series_plot_plotly import SeriesPlotPlotlyTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils, TestUtils

class SeriesPlotPlotlyParityTests(SeriesPlotPlotlyTestsMixin, PandasOnSparkTestUtils, TestUtils, ReusedConnectTestCase):

    @unittest.skip('Test depends on Spark ML which is not supported from Spark Connect.')
    def test_hist_plot(self):
        if False:
            i = 10
            return i + 15
        super().test_hist_plot()

    @unittest.skip('Test depends on Spark ML which is not supported from Spark Connect.')
    def test_kde_plot(self):
        if False:
            return 10
        super().test_kde_plot()
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.plot.test_parity_series_plot_plotly import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)