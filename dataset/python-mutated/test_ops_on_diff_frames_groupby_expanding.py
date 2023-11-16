import pandas as pd
from pyspark import pandas as ps
from pyspark.pandas.config import set_option, reset_option
from pyspark.testing.pandasutils import PandasOnSparkTestCase, TestUtils

class OpsOnDiffFramesGroupByExpandingTestsMixin:

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        super().setUpClass()
        set_option('compute.ops_on_diff_frames', True)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        reset_option('compute.ops_on_diff_frames')
        super().tearDownClass()

    def _test_groupby_expanding_func(self, f):
        if False:
            i = 10
            return i + 15
        pser = pd.Series([1, 2, 3])
        pkey = pd.Series([1, 2, 3], name='a')
        psser = ps.from_pandas(pser)
        kkey = ps.from_pandas(pkey)
        self.assert_eq(getattr(psser.groupby(kkey).expanding(2), f)().sort_index(), getattr(pser.groupby(pkey).expanding(2), f)().sort_index())
        pdf = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [4.0, 2.0, 3.0, 1.0]})
        pkey = pd.Series([1, 2, 3, 2], name='a')
        psdf = ps.from_pandas(pdf)
        kkey = ps.from_pandas(pkey)
        self.assert_eq(getattr(psdf.groupby(kkey).expanding(2), f)().sort_index(), getattr(pdf.groupby(pkey).expanding(2), f)().sort_index())
        self.assert_eq(getattr(psdf.groupby(kkey)['b'].expanding(2), f)().sort_index(), getattr(pdf.groupby(pkey)['b'].expanding(2), f)().sort_index())
        self.assert_eq(getattr(psdf.groupby(kkey)[['b']].expanding(2), f)().sort_index(), getattr(pdf.groupby(pkey)[['b']].expanding(2), f)().sort_index())

    def test_groupby_expanding_count(self):
        if False:
            i = 10
            return i + 15
        self._test_groupby_expanding_func('count')

    def test_groupby_expanding_min(self):
        if False:
            i = 10
            return i + 15
        self._test_groupby_expanding_func('min')

    def test_groupby_expanding_max(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_groupby_expanding_func('max')

    def test_groupby_expanding_mean(self):
        if False:
            print('Hello World!')
        self._test_groupby_expanding_func('mean')

    def test_groupby_expanding_sum(self):
        if False:
            return 10
        self._test_groupby_expanding_func('sum')

    def test_groupby_expanding_std(self):
        if False:
            return 10
        self._test_groupby_expanding_func('std')

    def test_groupby_expanding_var(self):
        if False:
            i = 10
            return i + 15
        self._test_groupby_expanding_func('var')

class OpsOnDiffFramesGroupByExpandingTests(OpsOnDiffFramesGroupByExpandingTestsMixin, PandasOnSparkTestCase, TestUtils):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.test_ops_on_diff_frames_groupby_expanding import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)