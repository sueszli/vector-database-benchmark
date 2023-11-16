import numpy as np
import pandas as pd
import pyspark.pandas as ps
from pyspark.testing.pandasutils import PandasOnSparkTestCase, TestUtils

class FrameInterpolateTestsMixin:

    def test_interpolate_error(self):
        if False:
            for i in range(10):
                print('nop')
        psdf = ps.range(10)
        with self.assertRaisesRegex(NotImplementedError, "interpolate currently works only for method='linear'"):
            psdf.interpolate(method='quadratic')
        with self.assertRaisesRegex(NotImplementedError, "interpolate currently works only for method='linear'"):
            psdf.id.interpolate(method='quadratic')
        with self.assertRaisesRegex(ValueError, 'limit must be > 0'):
            psdf.interpolate(limit=0)
        with self.assertRaisesRegex(ValueError, 'limit must be > 0'):
            psdf.id.interpolate(limit=0)
        with self.assertRaisesRegex(ValueError, 'invalid limit_direction'):
            psdf.interpolate(limit_direction='jump')
        with self.assertRaisesRegex(ValueError, 'invalid limit_direction'):
            psdf.id.interpolate(limit_direction='jump')
        with self.assertRaisesRegex(ValueError, 'invalid limit_area'):
            psdf.interpolate(limit_area='jump')
        with self.assertRaisesRegex(ValueError, 'invalid limit_area'):
            psdf.id.interpolate(limit_area='jump')
        with self.assertRaisesRegex(TypeError, 'Cannot interpolate with all object-dtype columns in the DataFrame.'):
            ps.DataFrame({'A': ['a', 'b', 'c'], 'B': ['a', 'b', 'c']}).interpolate()

    def _test_interpolate(self, pobj):
        if False:
            for i in range(10):
                print('nop')
        psobj = ps.from_pandas(pobj)
        self.assert_eq(psobj.interpolate(), pobj.interpolate())
        for limit in range(1, 5):
            for limit_direction in [None, 'forward', 'backward', 'both']:
                for limit_area in [None, 'inside', 'outside']:
                    self.assert_eq(psobj.interpolate(limit=limit, limit_direction=limit_direction, limit_area=limit_area), pobj.interpolate(limit=limit, limit_direction=limit_direction, limit_area=limit_area))

    def test_interpolate(self):
        if False:
            i = 10
            return i + 15
        pdf = pd.DataFrame([(1, 0.0, np.nan), (2, np.nan, 2.0), (3, 2.0, 3.0), (4, np.nan, 4.0), (5, np.nan, 1.0)], columns=list('abc'))
        self._test_interpolate(pdf)
        pdf = pd.DataFrame([(0.0, np.nan, -1.0, 1.0, np.nan), (np.nan, 2.0, np.nan, np.nan, np.nan), (2.0, 3.0, np.nan, 9.0, np.nan), (np.nan, 4.0, -4.0, 16.0, np.nan), (np.nan, 1.0, np.nan, 7.0, np.nan)], columns=list('abcde'))
        self._test_interpolate(pdf)
        pdf = pd.DataFrame([(0.0, np.nan, -1.0, False, np.nan), (np.nan, 2.0, np.nan, True, np.nan), (2.0, 3.0, np.nan, True, np.nan), (np.nan, 4.0, -4.0, False, np.nan), (np.nan, 1.0, np.nan, True, np.nan)], columns=list('abcde'))
        self._test_interpolate(pdf)

class FrameInterpolateTests(FrameInterpolateTestsMixin, PandasOnSparkTestCase, TestUtils):
    pass
if __name__ == '__main__':
    import unittest
    from pyspark.pandas.tests.test_frame_interpolate import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)