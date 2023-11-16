"""Test cases for complex numbers division."""
import os
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import googletest
os.environ['XLA_FLAGS'] = '--xla_cpu_fast_math_honor_nans=true --xla_cpu_fast_math_honor_infs=true'

class ComplexNumbersDivisionTest(xla_test.XLATestCase):
    """Test cases for complex numbers division operators."""

    def _testBinary(self, op, a, b, expected, equality_test=None):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as session:
            with self.test_scope():
                pa = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name='a')
                pb = array_ops.placeholder(dtypes.as_dtype(b.dtype), b.shape, name='b')
                output = op(pa, pb)
            result = session.run(output, {pa: a, pb: b})
            if equality_test is None:
                equality_test = self.assertAllCloseAccordingToType
            equality_test(np.real(result), np.real(expected), rtol=0.001)
            equality_test(np.imag(result), np.imag(expected), rtol=0.001)

    def testComplexOps(self):
        if False:
            return 10
        for dtype in self.complex_types:
            self._testBinary(gen_math_ops.real_div, np.array([complex(1, 1), complex(1, np.inf), complex(1, np.nan), complex(np.inf, 1), complex(np.inf, np.inf), complex(np.inf, np.nan), complex(np.nan, 1), complex(np.nan, np.inf), complex(np.nan, np.nan), complex(-np.inf, np.nan)], dtype=dtype), np.array([0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0.0 + 0j], dtype=dtype), expected=np.array([complex(np.inf, np.inf), complex(np.inf, np.inf), complex(np.inf, np.nan), complex(np.inf, np.inf), complex(np.inf, np.inf), complex(np.inf, np.nan), complex(np.nan, np.inf), complex(np.nan, np.inf), complex(np.nan, np.nan), complex(-np.inf, np.nan)], dtype=dtype))
            self._testBinary(gen_math_ops.real_div, np.array([1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j], dtype=dtype), np.array([complex(1, np.inf), complex(1, np.nan), complex(np.inf, 1), complex(np.inf, np.inf), complex(np.inf, np.nan), complex(np.nan, 1), complex(np.nan, np.inf), complex(np.nan, -np.inf), complex(np.nan, np.nan)], dtype=dtype), expected=np.array([(1 + 1j) / complex(1, np.inf), (1 + 1j) / complex(1, np.nan), (1 + 1j) / complex(np.inf, 1), complex(0 + 0j), complex(0 + 0j), (1 + 1j) / complex(np.nan, 1), complex(0 + 0j), complex(0 - 0j), (1 + 1j) / complex(np.nan, np.nan)], dtype=dtype))
            self._testBinary(gen_math_ops.real_div, np.array([complex(1, np.inf), complex(1, np.nan), complex(np.inf, 1), complex(np.inf, np.inf), complex(np.inf, np.nan), complex(np.nan, 1), complex(np.nan, np.inf), complex(np.nan, np.nan), complex(np.nan, -np.inf)], dtype=dtype), np.array([1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, -1 - 1j], dtype=dtype), expected=np.array([complex(np.inf, np.inf), complex(1 / np.nan) / (1 + 1j), complex(np.inf / 1) / (1 + 1j), complex(np.inf, -np.nan), complex(np.inf, -np.inf), complex(np.nan / 1) / (1 + 1j), complex(np.inf, np.inf), complex(np.nan / np.nan) / (1 + 1j), complex(np.inf, np.inf)], dtype=dtype))
if __name__ == '__main__':
    googletest.main()