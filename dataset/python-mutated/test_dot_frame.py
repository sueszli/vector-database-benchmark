import pandas as pd
from pyspark import pandas as ps
from pyspark.pandas.config import set_option, reset_option
from pyspark.testing.pandasutils import PandasOnSparkTestCase
from pyspark.testing.sqlutils import SQLTestUtils

class DiffFramesDotFrameMixin:

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        set_option('compute.ops_on_diff_frames', True)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        reset_option('compute.ops_on_diff_frames')
        super().tearDownClass()

    def test_frame_dot(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
        psdf = ps.from_pandas(pdf)
        pser = pd.Series([1, 1, 2, 1])
        psser = ps.from_pandas(pser)
        self.assert_eq(psdf.dot(psser), pdf.dot(pser))
        pser = pser.reindex([1, 0, 2, 3])
        psser = ps.from_pandas(pser)
        self.assert_eq(psdf.dot(psser), pdf.dot(pser))
        pser.name = 'ser'
        psser = ps.from_pandas(pser)
        self.assert_eq(psdf.dot(psser), pdf.dot(pser))
        arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        pidx = pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        pser = pd.Series([1, 1, 2, 1], index=pidx)
        pdf = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]], columns=pidx)
        psdf = ps.from_pandas(pdf)
        psser = ps.from_pandas(pser)
        self.assert_eq(psdf.dot(psser), pdf.dot(pser))
        pidx = pd.Index([1, 2, 3, 4], name='number')
        pser = pd.Series([1, 1, 2, 1], index=pidx)
        pdf = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]], columns=pidx)
        psdf = ps.from_pandas(pdf)
        psser = ps.from_pandas(pser)
        self.assert_eq(psdf.dot(psser), pdf.dot(pser))
        pdf.index = pd.Index(['x', 'y'], name='char')
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psdf.dot(psser), pdf.dot(pser))
        pdf.index = pd.MultiIndex.from_arrays([[1, 1], ['red', 'blue']], names=('number', 'color'))
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psdf.dot(psser), pdf.dot(pser))
        pdf = pd.DataFrame([[1, 2], [3, 4]])
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psdf.dot(psdf[0]), pdf.dot(pdf[0]))
        self.assert_eq(psdf.dot(psdf[0] * 10), pdf.dot(pdf[0] * 10))
        self.assert_eq((psdf + 1).dot(psdf[0] * 10), (pdf + 1).dot(pdf[0] * 10))

class DiffFramesDotFrameTests(DiffFramesDotFrameMixin, PandasOnSparkTestCase, SQLTestUtils):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.diff_frames_ops.test_dot_frame import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)