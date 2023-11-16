import unittest
from pyspark.pandas.tests.diff_frames_ops.test_setitem_series import DiffFramesSetItemSeriesMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class DiffFramesParitySetItemSeriesTests(DiffFramesSetItemSeriesMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @unittest.skip('TODO(SPARK-44826): Resolve testing timeout issue from Spark Connect.')
    def test_series_iloc_setitem(self):
        if False:
            while True:
                i = 10
        super().test_series_iloc_setitem()
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.diff_frames_ops.test_parity_setitem_series import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)