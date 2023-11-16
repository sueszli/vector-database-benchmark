import unittest
import pandas as pd
import numpy as np
from pyspark import pandas as ps
from pyspark.pandas.tests.test_extension import ExtensionTestsMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class ExtensionParityTests(ExtensionTestsMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @property
    def pdf(self):
        if False:
            return 10
        return pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=np.random.rand(9))

    @property
    def psdf(self):
        if False:
            for i in range(10):
                print('nop')
        return ps.from_pandas(self.pdf)
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.test_parity_extension import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)