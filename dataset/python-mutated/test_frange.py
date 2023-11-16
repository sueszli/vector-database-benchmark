import unittest
from robot.utils.frange import frange, _digits
from robot.utils.asserts import assert_equal, assert_true, assert_raises

class TestFrange(unittest.TestCase):

    def test_basics(self):
        if False:
            for i in range(10):
                print('nop')
        for (input, expected) in [([6.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), ([6.01], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), ([-2.4, 2.1], [-2.4, -1.4, -0.4, 0.6, 1.6]), ([0, 0.5, 0.1], [0, 0.1, 0.2, 0.3, 0.4])]:
            assert_equal(frange(*input), expected)

    def test_numbers_with_e(self):
        if False:
            return 10
        for (input, expected) in [([1e+20, 1e+21, 2e+20], [1e+20, 3e+20, 5e+20, 7e+20, 9e+20]), ([1e-21, 1.1e-20, 3e-21], [1e-21, 4e-21, 7e-21, 1e-20]), ([1.1e-20, 1.1e-21, -5e-21], [1.1e-20, 6e-21])]:
            result = frange(*input)
            assert_equal(len(result), len(expected))
            diffs = [round(r - e, 30) for (r, e) in zip(result, expected)]
            assert_equal(sum(diffs), 0)

    def test_compatibility_with_range(self):
        if False:
            for i in range(10):
                print('nop')
        for input in [(10,), (-10,), (1, 10), (1, 10, 2), (10, -5, -2)]:
            assert_equal(frange(*input), list(range(*input)))
            assert_equal(frange(*(float(i) for i in input)), list(range(*input)))

    def test_preserve_type(self):
        if False:
            i = 10
            return i + 15
        for input in [(2,), (0, 2), (0, 2, 1)]:
            assert_true(all((isinstance(item, int) for item in frange(*input))))
        for input in [(2.0,), (0, 2.0), (0.0, 2), (0, 2, 1.0), (0, 2.0, 1)]:
            assert_true(all((isinstance(item, float) for item in frange(*input))))

    def test_invalid_args(self):
        if False:
            while True:
                i = 10
        assert_raises(TypeError, frange, ())
        assert_raises(TypeError, frange, (1, 2, 3, 4))

    def test_digits(self):
        if False:
            for i in range(10):
                print('nop')
        for (input, expected) in [(3, 0), (3.0, 0), ('3.1', 1), ('3.14', 2), ('3.141592653589793', len('141592653589793')), (1000.1, 1), ('-2.458', 3), (1e+50, 0), (1.23e+50, 0), (1e-50, 50), ('1.23e-50', 52), ('1.23e3', 0), ('1.23e2', 0), ('1.23e1', 1), ('1.23e0', 2), ('1.23e-1', 3), ('1.23e-2', 4)]:
            assert_equal(_digits(input), expected, input)
if __name__ == '__main__':
    unittest.main()