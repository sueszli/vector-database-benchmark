import unittest
from pyspark import pandas as ps
from pyspark.pandas.tests.io.test_io import FrameIOMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class FrameParityIOTests(FrameIOMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @property
    def psdf(self):
        if False:
            i = 10
            return i + 15
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.io.test_parity_io import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)