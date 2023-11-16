import unittest
import numpy as np
import pandas as pd
from pyspark import pandas as ps
from pyspark.testing.pandasutils import ComparisonTestBase
from pyspark.testing.sqlutils import SQLTestUtils

class FrameAnyAllMixin:

    @property
    def pdf(self):
        if False:
            for i in range(10):
                print('nop')
        return pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=np.random.rand(9))

    @property
    def df_pair(self):
        if False:
            return 10
        pdf = self.pdf
        psdf = ps.from_pandas(pdf)
        return (pdf, psdf)

    def test_all(self):
        if False:
            print('Hello World!')
        pdf = pd.DataFrame({'col1': [False, False, False], 'col2': [True, False, False], 'col3': [0, 0, 1], 'col4': [0, 1, 2], 'col5': [False, False, None], 'col6': [True, False, None]}, index=np.random.rand(3))
        pdf.name = 'x'
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psdf.all(), pdf.all())
        self.assert_eq(psdf.all(bool_only=True), pdf.all(bool_only=True))
        self.assert_eq(psdf.all(bool_only=False), pdf.all(bool_only=False))
        self.assert_eq(psdf[['col5']].all(bool_only=True), pdf[['col5']].all(bool_only=True))
        self.assert_eq(psdf[['col5']].all(bool_only=False), pdf[['col5']].all(bool_only=False))
        columns = pd.MultiIndex.from_tuples([('a', 'col1'), ('a', 'col2'), ('a', 'col3'), ('b', 'col4'), ('b', 'col5'), ('c', 'col6')])
        pdf.columns = columns
        psdf.columns = columns
        self.assert_eq(psdf.all(), pdf.all())
        self.assert_eq(psdf.all(bool_only=True), pdf.all(bool_only=True))
        self.assert_eq(psdf.all(bool_only=False), pdf.all(bool_only=False))
        columns.names = ['X', 'Y']
        pdf.columns = columns
        psdf.columns = columns
        self.assert_eq(psdf.all(), pdf.all())
        self.assert_eq(psdf.all(bool_only=True), pdf.all(bool_only=True))
        self.assert_eq(psdf.all(bool_only=False), pdf.all(bool_only=False))
        with self.assertRaisesRegex(NotImplementedError, 'axis should be either 0 or "index" currently.'):
            psdf.all(axis=1)
        pdf = pd.DataFrame({'A': [True, True], 'B': [1, np.nan], 'C': [True, None]})
        pdf.name = 'x'
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psdf[['A', 'B']].all(skipna=False), pdf[['A', 'B']].all(skipna=False))
        self.assert_eq(psdf[['A', 'C']].all(skipna=False), pdf[['A', 'C']].all(skipna=False))
        self.assert_eq(psdf[['B', 'C']].all(skipna=False), pdf[['B', 'C']].all(skipna=False))
        self.assert_eq(psdf.all(skipna=False), pdf.all(skipna=False))
        self.assert_eq(psdf.all(skipna=True), pdf.all(skipna=True))
        self.assert_eq(psdf.all(), pdf.all())
        self.assert_eq(ps.DataFrame([np.nan]).all(skipna=False), pd.DataFrame([np.nan]).all(skipna=False), almost=True)
        self.assert_eq(ps.DataFrame([None]).all(skipna=True), pd.DataFrame([None]).all(skipna=True), almost=True)

    def test_any(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = pd.DataFrame({'col1': [False, False, False], 'col2': [True, False, False], 'col3': [0, 0, 1], 'col4': [0, 1, 2], 'col5': [False, False, None], 'col6': [True, False, None]}, index=np.random.rand(3))
        pdf.name = 'x'
        psdf = ps.from_pandas(pdf)
        self.assert_eq(psdf.any(), pdf.any())
        self.assert_eq(psdf.any(bool_only=True), pdf.any(bool_only=True))
        self.assert_eq(psdf.any(bool_only=False), pdf.any(bool_only=False))
        self.assert_eq(psdf[['col5']].all(bool_only=True), pdf[['col5']].all(bool_only=True))
        self.assert_eq(psdf[['col5']].all(bool_only=False), pdf[['col5']].all(bool_only=False))
        columns = pd.MultiIndex.from_tuples([('a', 'col1'), ('a', 'col2'), ('a', 'col3'), ('b', 'col4'), ('b', 'col5'), ('c', 'col6')])
        pdf.columns = columns
        psdf.columns = columns
        self.assert_eq(psdf.any(), pdf.any())
        self.assert_eq(psdf.any(bool_only=True), pdf.any(bool_only=True))
        self.assert_eq(psdf.any(bool_only=False), pdf.any(bool_only=False))
        columns.names = ['X', 'Y']
        pdf.columns = columns
        psdf.columns = columns
        self.assert_eq(psdf.any(), pdf.any())
        self.assert_eq(psdf.any(bool_only=True), pdf.any(bool_only=True))
        self.assert_eq(psdf.any(bool_only=False), pdf.any(bool_only=False))
        with self.assertRaisesRegex(NotImplementedError, 'axis should be either 0 or "index" currently.'):
            psdf.any(axis=1)

class FrameAnyAllTests(FrameAnyAllMixin, ComparisonTestBase, SQLTestUtils):
    pass
if __name__ == '__main__':
    from pyspark.pandas.tests.computation.test_any_all import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)