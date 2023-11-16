import unittest
from pyspark.pandas.tests.groupby.test_apply_func import GroupbyApplyFuncMixin
from pyspark.testing.connectutils import ReusedConnectTestCase
from pyspark.testing.pandasutils import PandasOnSparkTestUtils

class GroupbyParityApplyFuncTests(GroupbyApplyFuncMixin, PandasOnSparkTestUtils, ReusedConnectTestCase):

    @unittest.skip('Test depends on SparkContext which is not supported from Spark Connect.')
    def test_apply_with_side_effect(self):
        if False:
            print('Hello World!')
        super().test_apply_with_side_effect()
if __name__ == '__main__':
    from pyspark.pandas.tests.connect.groupby.test_parity_apply_func import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)