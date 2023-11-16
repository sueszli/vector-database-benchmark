import unittest
import sys
import pandas as pd
from pyspark import pandas as ps
from pyspark.testing.pandasutils import PandasOnSparkTestCase
from pyspark.testing.sqlutils import SQLTestUtils

class SeriesConversionTestsMixin:

    @property
    def pser(self):
        if False:
            while True:
                i = 10
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def psser(self):
        if False:
            while True:
                i = 10
        return ps.from_pandas(self.pser)

    @unittest.skipIf(sys.platform == 'linux' or sys.platform == 'linux2', 'Pyperclip could not find a copy/paste mechanism for Linux.')
    def test_to_clipboard(self):
        if False:
            for i in range(10):
                print('nop')
        pser = self.pser
        psser = self.psser
        self.assert_eq(psser.to_clipboard(), pser.to_clipboard())
        self.assert_eq(psser.to_clipboard(excel=False), pser.to_clipboard(excel=False))
        self.assert_eq(psser.to_clipboard(sep=',', index=False), pser.to_clipboard(sep=',', index=False))

    def test_to_latex(self):
        if False:
            return 10
        pser = self.pser
        psser = self.psser
        self.assert_eq(psser.to_latex(), pser.to_latex())
        self.assert_eq(psser.to_latex(header=True), pser.to_latex(header=True))
        self.assert_eq(psser.to_latex(index=False), pser.to_latex(index=False))
        self.assert_eq(psser.to_latex(na_rep='-'), pser.to_latex(na_rep='-'))
        self.assert_eq(psser.to_latex(float_format='%.1f'), pser.to_latex(float_format='%.1f'))
        self.assert_eq(psser.to_latex(sparsify=False), pser.to_latex(sparsify=False))
        self.assert_eq(psser.to_latex(index_names=False), pser.to_latex(index_names=False))
        self.assert_eq(psser.to_latex(bold_rows=True), pser.to_latex(bold_rows=True))
        self.assert_eq(psser.to_latex(decimal=','), pser.to_latex(decimal=','))

class SeriesConversionTests(SeriesConversionTestsMixin, PandasOnSparkTestCase, SQLTestUtils):
    pass
if __name__ == '__main__':
    from pyspark.pandas.tests.test_series_conversion import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)