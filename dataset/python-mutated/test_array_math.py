import numpy
import pytest
import modin.numpy as np
from .utils import assert_scalar_or_array_equal

@pytest.mark.parametrize('data', [[3, 2, 1, 1], [-87.434, -90.908, -87.152, -84.903], [-87.434, -90.908, np.nan, -87.152, -84.903]], ids=['ints', 'floats', 'floats with nan'])
@pytest.mark.parametrize('op', ['argmin', 'argmax'])
def test_argmax_argmin(data, op):
    if False:
        i = 10
        return i + 15
    numpy_result = getattr(numpy, op)(numpy.array(data))
    modin_result = getattr(np, op)(np.array(data))
    assert_scalar_or_array_equal(modin_result, numpy_result)

def test_rem_mod():
    if False:
        for i in range(10):
            print('nop')
    'Tests remainder and mod, which, unlike the C/matlab equivalents, are identical in numpy.'
    a = numpy.array([[2, -1], [10, -3]])
    b = numpy.array(([-3, 3], [3, -7]))
    numpy_result = numpy.remainder(a, b)
    modin_result = np.remainder(np.array(a), np.array(b))
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_result = numpy.mod(a, b)
    modin_result = np.mod(np.array(a), np.array(b))
    assert_scalar_or_array_equal(modin_result, numpy_result)