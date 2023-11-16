from assertpy import assert_that
from polynomial import strip
from polynomial import Polynomial

def test_strip_empty():
    if False:
        i = 10
        return i + 15
    assert_that(strip([], 1)).is_equal_to([])

def test_strip_single():
    if False:
        return 10
    assert_that(strip([1, 2, 3, 1], 1)).is_equal_to([1, 2, 3])

def test_strip_many():
    if False:
        return 10
    assert_that(strip([1, 2, 3, 1, 1, 1, 1], 1)).is_equal_to([1, 2, 3])

def test_strip_string():
    if False:
        return 10
    assert_that(strip('123111', '1')).is_equal_to('123')

def test_polynomial_zero():
    if False:
        for i in range(10):
            print('nop')
    assert_that(Polynomial([0]).coefficients).is_equal_to([])
    assert_that(len(Polynomial([0]))).is_equal_to(0)

def test_polynomial_repr():
    if False:
        i = 10
        return i + 15
    f = Polynomial([1, 2, 3])
    assert_that(repr(f)).is_equal_to('1 + 2 x^1 + 3 x^2')

def test_polynomial_add():
    if False:
        while True:
            i = 10
    f = Polynomial([1, 2, 3])
    g = Polynomial([4, 5, 6])
    assert_that((f + g).coefficients).is_equal_to([5, 7, 9])

def test_polynomial_sub():
    if False:
        while True:
            i = 10
    f = Polynomial([1, 2, 3])
    g = Polynomial([4, 5, 6])
    assert_that((f - g).coefficients).is_equal_to([-3, -3, -3])

def test_polynomial_add_zero():
    if False:
        while True:
            i = 10
    f = Polynomial([1, 2, 3])
    g = Polynomial([0])
    assert_that((f + g).coefficients).is_equal_to([1, 2, 3])

def test_polynomial_negate():
    if False:
        print('Hello World!')
    f = Polynomial([1, 2, 3])
    assert_that((-f).coefficients).is_equal_to([-1, -2, -3])

def test_polynomial_multiply():
    if False:
        return 10
    f = Polynomial([1, 2, 3])
    g = Polynomial([4, 5, 6])
    assert_that((f * g).coefficients).is_equal_to([4, 13, 28, 27, 18])

def test_polynomial_evaluate_at():
    if False:
        return 10
    f = Polynomial([1, 2, 3])
    assert_that(f(2)).is_equal_to(1 + 4 + 12)