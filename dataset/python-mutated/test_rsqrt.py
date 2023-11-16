import unittest
import numpy
import cupy
from cupy import testing
import cupyx

class TestRsqrt(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    def test_rsqrt(self, dtype):
        if False:
            print('Hello World!')
        a = testing.shaped_arange((2, 3), numpy, dtype) + 1.0
        out = cupyx.rsqrt(cupy.array(a))
        testing.assert_allclose(out, 1.0 / numpy.sqrt(a))