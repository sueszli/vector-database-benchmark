import pandas as pd
from pyspark import pandas as ps
from pyspark.pandas.config import set_option, reset_option
from pyspark.testing.pandasutils import PandasOnSparkTestCase
from pyspark.testing.sqlutils import SQLTestUtils

class DiffFramesDotSeriesMixin:

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        set_option('compute.ops_on_diff_frames', True)

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        reset_option('compute.ops_on_diff_frames')
        super().tearDownClass()

    def test_series_dot(self):
        if False:
            while True:
                i = 10
        pser = pd.Series([90, 91, 85], index=[2, 4, 1])
        psser = ps.from_pandas(pser)
        pser_other = pd.Series([90, 91, 85], index=[2, 4, 1])
        psser_other = ps.from_pandas(pser_other)
        self.assert_eq(psser.dot(psser_other), pser.dot(pser_other))
        psser_other = ps.Series([90, 91, 85], index=[1, 2, 4])
        pser_other = pd.Series([90, 91, 85], index=[1, 2, 4])
        self.assert_eq(psser.dot(psser_other), pser.dot(pser_other))
        psser_other = ps.Series([90, 91, 85, 100], index=[2, 4, 1, 0])
        with self.assertRaisesRegex(ValueError, 'matrices are not aligned'):
            psser.dot(psser_other)
        midx = pd.MultiIndex([['lama', 'cow', 'falcon'], ['speed', 'weight', 'length']], [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        psser = ps.from_pandas(pser)
        pser_other = pd.Series([-450, 20, 12, -30, -250, 15, -320, 100, 3], index=midx)
        psser_other = ps.from_pandas(pser_other)
        self.assert_eq(psser.dot(psser_other), pser.dot(pser_other))
        pser = pd.Series([0, 1, 2, 3])
        psser = ps.from_pandas(pser)
        pdf = pd.DataFrame([[0, 1], [-2, 3], [4, -5], [6, 7]])
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psser.dot(psdf), pser.dot(pdf))
        pdf.columns = pd.Index(['x', 'y'])
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psser.dot(psdf), pser.dot(pdf))
        pdf.columns = pd.Index(['x', 'y'], name='cols_name')
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psser.dot(psdf), pser.dot(pdf))
        pdf = pdf.reindex([1, 0, 2, 3])
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psser.dot(psdf), pser.dot(pdf))
        pdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')])
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psser.dot(psdf), pser.dot(pdf))
        pdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')], names=['cols_name1', 'cols_name2'])
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psser.dot(psdf), pser.dot(pdf))
        psser = ps.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}).b
        pser = psser._to_pandas()
        psdf = ps.DataFrame({'c': [7, 8, 9]})
        pdf = psdf._to_pandas()
        self.assert_eq(psser.dot(psdf), pser.dot(pdf))
        pser = pd.Series([90, 91, 85], index=[0, 1, 2])
        psser = ps.from_pandas(pser)
        pser_other = pd.Series([90, 91, 85], index=[0, 1, 3])
        psser_other = ps.from_pandas(pser_other)
        pser_other2 = pd.Series([90, 91, 85, 100], index=[0, 1, 3, 5])
        psser_other2 = ps.from_pandas(pser_other2)
        with self.assertRaisesRegex(ValueError, 'matrices are not aligned'):
            psser.dot(psser_other)
        with ps.option_context('compute.eager_check', False), self.assertRaisesRegex(ValueError, 'matrices are not aligned'):
            psser.dot(psser_other2)
        with ps.option_context('compute.eager_check', True), self.assertRaisesRegex(ValueError, 'matrices are not aligned'):
            psser.dot(psser_other)
        with ps.option_context('compute.eager_check', False):
            self.assert_eq(psser.dot(psser_other), 16381)

class DiffFramesDotSeriesTests(DiffFramesDotSeriesMixin, PandasOnSparkTestCase, SQLTestUtils):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.diff_frames_ops.test_dot_series import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)