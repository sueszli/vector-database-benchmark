import unittest
from pyspark import pandas as ps
from pyspark.pandas.tests.computation.test_melt import FrameMeltMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class FrameParityMeltTests(FrameMeltMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @property
    def psdf(self):
        if False:
            return 10
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.computation.test_parity_melt import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)