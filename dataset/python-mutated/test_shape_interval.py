from deeplake.util.exceptions import InvalidShapeIntervalError
import pytest
from deeplake.util.shape_interval import ShapeInterval

def assert_fixed(raw_shape):
    if False:
        print('Hello World!')
    fixed_shape = ShapeInterval(raw_shape)
    assert fixed_shape.lower == raw_shape
    assert fixed_shape.upper == raw_shape
    assert not fixed_shape.is_dynamic

def assert_dynamic(raw_lower, raw_upper):
    if False:
        print('Hello World!')
    dynamic_shape = ShapeInterval(raw_lower, raw_upper)
    assert dynamic_shape.lower == raw_lower
    assert dynamic_shape.upper == raw_upper
    assert dynamic_shape.is_dynamic

def test_compatible_shapes():
    if False:
        for i in range(10):
            print('nop')
    assert_fixed((100, 100, 3))
    assert_fixed((1,))
    assert_dynamic((100, 100, 3), (100, 100, 4))
    assert_dynamic((1,), (50000,))
FAILURES = [[(100,), (100, 1)], [(1, 1, 1), (1,)], [(100,), (1,)], [(100, 100, 3), (100, 100, 1)], [(-1,), None], [(1,), (-1,)]]

@pytest.mark.xfail(raises=InvalidShapeIntervalError, strict=True)
@pytest.mark.parametrize('upper,lower', FAILURES)
def test_invalid_shapes(upper, lower):
    if False:
        i = 10
        return i + 15
    ShapeInterval(upper, lower)