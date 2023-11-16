import pandas as pd
from pyspark import pandas as ps
from pyspark.pandas.config import set_option, reset_option
from pyspark.testing.pandasutils import PandasOnSparkTestCase
from pyspark.testing.sqlutils import SQLTestUtils

class DiffFramesSetItemFrameMixin:

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

    def test_frame_loc_setitem(self):
        if False:
            i = 10
            return i + 15
        pdf_orig = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'], columns=['max_speed', 'shield'])
        psdf_orig = ps.DataFrame(pdf_orig)
        pdf = pdf_orig.copy()
        psdf = psdf_orig.copy()
        pser1 = pdf.max_speed
        pser2 = pdf.shield
        psser1 = psdf.max_speed
        psser2 = psdf.shield
        another_psdf = ps.DataFrame(pdf_orig)
        psdf.loc[['viper', 'sidewinder'], ['shield']] = -another_psdf.max_speed
        pdf.loc[['viper', 'sidewinder'], ['shield']] = -pdf.max_speed
        self.assert_eq(psdf, pdf)
        self.assert_eq(psser1, pser1)
        self.assert_eq(psser2, pser2)
        pdf = pdf_orig.copy()
        psdf = psdf_orig.copy()
        pser1 = pdf.max_speed
        pser2 = pdf.shield
        psser1 = psdf.max_speed
        psser2 = psdf.shield
        psdf.loc[another_psdf.max_speed < 5, ['shield']] = -psdf.max_speed
        pdf.loc[pdf.max_speed < 5, ['shield']] = -pdf.max_speed
        self.assert_eq(psdf, pdf)
        self.assert_eq(psser1, pser1)
        self.assert_eq(psser2, pser2)
        pdf = pdf_orig.copy()
        psdf = psdf_orig.copy()
        pser1 = pdf.max_speed
        pser2 = pdf.shield
        psser1 = psdf.max_speed
        psser2 = psdf.shield
        psdf.loc[another_psdf.max_speed < 5, ['shield']] = -another_psdf.max_speed
        pdf.loc[pdf.max_speed < 5, ['shield']] = -pdf.max_speed
        self.assert_eq(psdf, pdf)
        self.assert_eq(psser1, pser1)
        self.assert_eq(psser2, pser2)

    def test_frame_iloc_setitem(self):
        if False:
            return 10
        pdf = pd.DataFrame([[1, 2], [4, 5], [7, 8]], index=['cobra', 'viper', 'sidewinder'], columns=['max_speed', 'shield'])
        psdf = ps.DataFrame(pdf)
        another_psdf = ps.DataFrame(pdf)
        psdf.iloc[[0, 1, 2], 1] = -another_psdf.max_speed
        pdf.iloc[[0, 1, 2], 1] = -pdf.max_speed
        self.assert_eq(psdf, pdf)
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            psdf.iloc[[1, 2], [1]] = -another_psdf.max_speed
        psdf.iloc[[0, 1, 2], 1] = 10 * another_psdf.max_speed
        pdf.iloc[[0, 1, 2], 1] = 10 * pdf.max_speed
        self.assert_eq(psdf, pdf)
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            psdf.iloc[[0], 1] = 10 * another_psdf.max_speed

class DiffFramesSetItemFrameTests(DiffFramesSetItemFrameMixin, PandasOnSparkTestCase, SQLTestUtils):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.diff_frames_ops.test_setitem_frame import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)