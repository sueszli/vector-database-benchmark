import unittest
from pyspark import pandas as ps
from pyspark.pandas.tests.frame.test_take import FrameTakeMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class FrameParityTakeTests(FrameTakeMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @property
    def psdf(self):
        if False:
            for i in range(10):
                print('nop')
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.frame.test_parity_take import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)