import decimal
import sys
import unittest
import warnings
from typing import Iterable, Optional, Union
from unittest.mock import patch
import pytest
from faker import Faker

@pytest.mark.parametrize('object_type', (None, bool, str, float, int, tuple, set, list, Iterable, dict))
def test_pyobject(object_type: Optional[Union[bool, str, float, int, tuple, set, list, Iterable, dict]]):
    if False:
        return 10
    random_object = Faker().pyobject(object_type=object_type)
    if object_type is None:
        assert random_object is None
    else:
        assert isinstance(random_object, object_type)

@pytest.mark.parametrize('object_type', (object, type, callable))
def test_pyobject_with_unknown_object_type(object_type):
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match=f'Object type `{object_type}` is not supported by `pyobject` function'):
        assert Faker().pyobject(object_type=object_type)

@pytest.mark.parametrize('mock_random_number_source, right_digits, expected_decimal_part', (('1234567', 5, '12345'), ('1234567', 0, '1'), ('1234567', 1, '1'), ('1234567', 2, '12'), ('0123', 1, '1')))
def test_pyfloat_right_and_left_digits_positive(mock_random_number_source, right_digits, expected_decimal_part):
    if False:
        print('Hello World!')

    def mock_random_number(self, digits=None, fix_len=False):
        if False:
            return 10
        return int(mock_random_number_source[:digits or 1])
    with patch('faker.providers.BaseProvider.random_number', mock_random_number):
        result = Faker().pyfloat(left_digits=1, right_digits=right_digits, positive=True)
        decimal_part = str(result).split('.')[1]
        assert decimal_part == expected_decimal_part

def test_pyfloat_right_or_left_digit_overflow():
    if False:
        print('Hello World!')
    max_float_digits = sys.float_info.dig
    faker = Faker()

    def mock_random_int(self, min=0, max=9999, step=1):
        if False:
            while True:
                i = 10
        return max

    def mock_random_number(self, digits=None, fix_len=False):
        if False:
            i = 10
            return i + 15
        return int('12345678901234567890'[:digits or 1])
    with patch('faker.providers.BaseProvider.random_int', mock_random_int):
        with patch('faker.providers.BaseProvider.random_number', mock_random_number):
            with pytest.raises(ValueError, match='Asking for too many digits'):
                faker.pyfloat(left_digits=max_float_digits // 2 + 1, right_digits=max_float_digits // 2 + 1)
            with pytest.raises(ValueError, match='Asking for too many digits'):
                faker.pyfloat(left_digits=max_float_digits)
            with pytest.raises(ValueError, match='Asking for too many digits'):
                faker.pyfloat(right_digits=max_float_digits)
            result = faker.pyfloat(left_digits=max_float_digits - 1)
            assert str(abs(result)) == '12345678901234.1'
            result = faker.pyfloat(right_digits=max_float_digits - 1)
            assert str(abs(result)) == '1.12345678901234'

@pytest.mark.skipif(sys.version_info < (3, 10), reason='Only relevant for Python 3.10 and later.')
@pytest.mark.parametrize(('min_value', 'max_value'), [(1.5, None), (-1.5, None), (None, -1.5), (None, 1.5), (-1.5, 1.5)])
@pytest.mark.parametrize('left_digits', [None, 5])
@pytest.mark.parametrize('right_digits', [None, 5])
@pytest.mark.filterwarnings('error:non-integer arguments to randrange\\(\\):DeprecationWarning')
def test_float_min_and_max_value_does_not_crash(left_digits: Optional[int], right_digits: Optional[int], min_value: Optional[float], max_value: Optional[float]):
    if False:
        for i in range(10):
            print('nop')
    '\n    Float arguments to randrange are deprecated from Python 3.10. This is a regression\n    test to check that `pydecimal` does not cause a crash on any code path.\n    '
    Faker().pydecimal(left_digits, right_digits, min_value=min_value, max_value=max_value)

class TestPyint(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fake = Faker()
        Faker.seed(0)

    def test_pyint(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(self.fake.pyint(), int)

    def test_pyint_bounds(self):
        if False:
            while True:
                i = 10
        self.assertTrue(0 <= self.fake.pyint() <= 9999)

    def test_pyint_step(self):
        if False:
            for i in range(10):
                print('nop')
        random_int = self.fake.pyint(step=2)
        self.assertEqual(0, random_int % 2)

    def test_pyint_bound_0(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, self.fake.pyint(min_value=0, max_value=0))

    def test_pyint_bound_positive(self):
        if False:
            print('Hello World!')
        self.assertEqual(5, self.fake.pyint(min_value=5, max_value=5))

    def test_pyint_bound_negative(self):
        if False:
            print('Hello World!')
        self.assertEqual(-5, self.fake.pyint(min_value=-5, max_value=-5))

    def test_pyint_range(self):
        if False:
            while True:
                i = 10
        self.assertTrue(0 <= self.fake.pyint(min_value=0, max_value=2) <= 2)

class TestPyfloat(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.fake = Faker()
        Faker.seed(0)

    def test_pyfloat(self):
        if False:
            return 10
        result = self.fake.pyfloat()
        self.assertIsInstance(result, float)

    def test_left_digits(self):
        if False:
            print('Hello World!')
        expected_left_digits = 10
        result = self.fake.pyfloat(left_digits=expected_left_digits)
        left_digits = len(str(abs(int(result))))
        self.assertGreaterEqual(expected_left_digits, left_digits)

    def test_right_digits(self):
        if False:
            while True:
                i = 10
        expected_right_digits = 10
        result = self.fake.pyfloat(right_digits=expected_right_digits)
        right_digits = len(('%r' % result).split('.')[1])
        self.assertGreaterEqual(expected_right_digits, right_digits)

    def test_positive(self):
        if False:
            i = 10
            return i + 15
        result = self.fake.pyfloat(positive=True)
        self.assertGreater(result, 0)
        self.assertEqual(result, abs(result))

    def test_min_value(self):
        if False:
            for i in range(10):
                print('nop')
        min_values = (0, 10, -1000, 1000, 999999)
        for min_value in min_values:
            result = self.fake.pyfloat(min_value=min_value)
            self.assertGreaterEqual(result, min_value)

    def test_min_value_and_left_digits(self):
        if False:
            return 10
        '\n        Combining the min_value and left_digits keyword arguments produces\n        numbers that obey both of those constraints.\n        '
        result = self.fake.pyfloat(left_digits=1, min_value=0)
        self.assertLess(result, 10)
        self.assertGreaterEqual(result, 0)

    def test_max_value(self):
        if False:
            return 10
        max_values = (0, 10, -1000, 1000, 999999)
        for max_value in max_values:
            result = self.fake.pyfloat(max_value=max_value)
            self.assertLessEqual(result, max_value)

    def test_max_value_zero_and_left_digits(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Combining the max_value and left_digits keyword arguments produces\n        numbers that obey both of those constraints.\n        '
        result = self.fake.pyfloat(left_digits=2, max_value=0)
        self.assertLessEqual(result, 0)
        self.assertGreater(result, -100)

    def test_max_value_should_be_greater_than_min_value(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An exception should be raised if min_value is greater than max_value\n        '
        expected_message = 'Min value cannot be greater than max value'
        with self.assertRaises(ValueError) as raises:
            self.fake.pyfloat(min_value=100, max_value=0)
        message = str(raises.exception)
        self.assertEqual(message, expected_message)

    def test_max_value_and_positive(self):
        if False:
            print('Hello World!')
        '\n        Combining the max_value and positive keyword arguments produces\n        numbers that obey both of those constraints.\n        '
        result = self.fake.pyfloat(positive=True, max_value=100)
        self.assertLessEqual(result, 100)
        self.assertGreater(result, 0)

    def test_max_and_min_value_positive_with_decimals(self):
        if False:
            return 10
        '\n        Combining the max_value and min_value keyword arguments with\n        positive values for each produces numbers that obey both of\n        those constraints.\n        '
        for _ in range(1000):
            result = self.fake.pyfloat(min_value=100.123, max_value=200.321)
            self.assertLessEqual(result, 200.321)
            self.assertGreaterEqual(result, 100.123)

    def test_max_and_min_value_negative(self):
        if False:
            return 10
        '\n        Combining the max_value and min_value keyword arguments with\n        negative values for each produces numbers that obey both of\n        those constraints.\n        '
        result = self.fake.pyfloat(max_value=-100, min_value=-200)
        self.assertLessEqual(result, -100)
        self.assertGreaterEqual(result, -200)

    def test_max_and_min_value_negative_with_decimals(self):
        if False:
            i = 10
            return i + 15
        '\n        Combining the max_value and min_value keyword arguments with\n        negative values for each produces numbers that obey both of\n        those constraints.\n        '
        for _ in range(1000):
            result = self.fake.pyfloat(max_value=-100.123, min_value=-200.321)
            self.assertLessEqual(result, -100.123)
            self.assertGreaterEqual(result, -200.321)

    def test_positive_and_min_value_incompatible(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An exception should be raised if positive=True is set, but\n        a negative min_value is provided.\n        '
        expected_message = 'Cannot combine positive=True with negative or zero min_value'
        with self.assertRaises(ValueError) as raises:
            self.fake.pyfloat(min_value=-100, positive=True)
        message = str(raises.exception)
        self.assertEqual(message, expected_message)

    def test_positive_doesnt_return_zero(self):
        if False:
            return 10
        "\n        Choose the right_digits and max_value so it's guaranteed to return zero,\n        then watch as it doesn't because positive=True\n        "
        result = self.fake.pyfloat(positive=True, right_digits=0, max_value=1)
        self.assertGreater(result, 0)

    @pytest.mark.skipif(sys.version_info < (3, 10), reason='Only relevant for Python 3.10 and later.')
    @pytest.mark.filterwarnings('error:non-integer arguments to randrange\\(\\):DeprecationWarning')
    def test_float_min_and_max_value_does_not_warn(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Float arguments to randrange are deprecated from Python 3.10. This is a regression\n        test to check that `pyfloat` does not cause a deprecation warning.\n        '
        self.fake.pyfloat(min_value=-1.0, max_value=1.0)

    def test_float_min_and_max_value_with_same_whole(self):
        if False:
            for i in range(10):
                print('nop')
        self.fake.pyfloat(min_value=2.3, max_value=2.5)

class TestPydecimal(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fake = Faker()
        Faker.seed(0)

    def test_pydecimal(self):
        if False:
            while True:
                i = 10
        result = self.fake.pydecimal()
        self.assertIsInstance(result, decimal.Decimal)

    def test_left_digits(self):
        if False:
            for i in range(10):
                print('nop')
        expected_left_digits = 10
        result = self.fake.pydecimal(left_digits=expected_left_digits)
        left_digits = len(str(abs(int(result))))
        self.assertGreaterEqual(expected_left_digits, left_digits)

    def test_left_digits_can_be_zero(self):
        if False:
            return 10
        expected_left_digits = 0
        result = self.fake.pydecimal(left_digits=expected_left_digits)
        left_digits = int(result)
        self.assertEqual(expected_left_digits, left_digits)

    def test_right_digits(self):
        if False:
            print('Hello World!')
        expected_right_digits = 10
        result = self.fake.pydecimal(right_digits=expected_right_digits)
        right_digits = len(str(result).split('.')[1])
        self.assertGreaterEqual(expected_right_digits, right_digits)

    def test_positive(self):
        if False:
            return 10
        result = self.fake.pydecimal(positive=True)
        self.assertGreater(result, 0)
        abs_result = -result if result < 0 else result
        self.assertEqual(result, abs_result)

    def test_min_value(self):
        if False:
            print('Hello World!')
        min_values = (0, 10, -1000, 1000, 999999)
        for min_value in min_values:
            result = self.fake.pydecimal(min_value=min_value)
            self.assertGreaterEqual(result, min_value)

    def test_min_value_always_returns_a_decimal(self):
        if False:
            while True:
                i = 10
        min_values = (0, 10, -1000, 1000, 999999)
        for min_value in min_values:
            result = self.fake.pydecimal(min_value=min_value)
            self.assertIsInstance(result, decimal.Decimal)

    def test_min_value_and_left_digits(self):
        if False:
            return 10
        '\n        Combining the min_value and left_digits keyword arguments produces\n        numbers that obey both of those constraints.\n        '
        result = self.fake.pydecimal(left_digits=1, min_value=0)
        self.assertLess(result, 10)
        self.assertGreaterEqual(result, 0)

    def test_max_value(self):
        if False:
            i = 10
            return i + 15
        max_values = (0, 10, -1000, 1000, 999999)
        for max_value in max_values:
            result = self.fake.pydecimal(max_value=max_value)
            self.assertLessEqual(result, max_value)

    def test_max_value_always_returns_a_decimal(self):
        if False:
            i = 10
            return i + 15
        max_values = (0, 10, -1000, 1000, 999999)
        for max_value in max_values:
            result = self.fake.pydecimal(max_value=max_value)
            self.assertIsInstance(result, decimal.Decimal)

    def test_max_value_zero_and_left_digits(self):
        if False:
            i = 10
            return i + 15
        '\n        Combining the max_value and left_digits keyword arguments produces\n        numbers that obey both of those constraints.\n        '
        result = self.fake.pydecimal(left_digits=2, max_value=0)
        self.assertLessEqual(result, 0)
        self.assertGreater(result, -100)

    def test_max_value_should_be_greater_than_min_value(self):
        if False:
            i = 10
            return i + 15
        '\n        An exception should be raised if min_value is greater than max_value\n        '
        expected_message = 'Min value cannot be greater than max value'
        with self.assertRaises(ValueError) as raises:
            self.fake.pydecimal(min_value=100, max_value=0)
        message = str(raises.exception)
        self.assertEqual(message, expected_message)

    def test_max_value_and_positive(self):
        if False:
            print('Hello World!')
        '\n        Combining the max_value and positive keyword arguments produces\n        numbers that obey both of those constraints.\n        '
        result = self.fake.pydecimal(positive=True, max_value=100)
        self.assertLessEqual(result, 100)
        self.assertGreater(result, 0)

    def test_max_and_min_value_negative(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Combining the max_value and min_value keyword arguments with\n        negative values for each produces numbers that obey both of\n        those constraints.\n        '
        result = self.fake.pydecimal(max_value=-100, min_value=-200)
        self.assertLessEqual(result, -100)
        self.assertGreaterEqual(result, -200)

    def test_positive_and_min_value_incompatible(self):
        if False:
            print('Hello World!')
        '\n        An exception should be raised if positive=True is set, but\n        a negative min_value is provided.\n        '
        expected_message = 'Cannot combine positive=True with negative or zero min_value'
        with self.assertRaises(ValueError) as raises:
            self.fake.pydecimal(min_value=-100, positive=True)
        message = str(raises.exception)
        self.assertEqual(message, expected_message)

    def test_positive_doesnt_return_zero(self):
        if False:
            i = 10
            return i + 15
        "\n        Choose the right_digits and max_value so it's guaranteed to return zero,\n        then watch as it doesn't because positive=True\n        "
        result = self.fake.pydecimal(positive=True, right_digits=0, max_value=1)
        self.assertGreater(result, 0)

    def test_min_value_zero_doesnt_return_negative(self):
        if False:
            return 10
        Faker.seed('1')
        result = self.fake.pydecimal(left_digits=3, right_digits=2, min_value=0, max_value=999)
        self.assertGreater(result, 0)

    def test_min_value_one_hundred_doesnt_return_negative(self):
        if False:
            while True:
                i = 10
        Faker.seed('1')
        result = self.fake.pydecimal(left_digits=3, right_digits=2, min_value=100, max_value=999)
        self.assertGreater(result, 100)

    def test_min_value_minus_one_doesnt_return_positive(self):
        if False:
            while True:
                i = 10
        Faker.seed('5')
        result = self.fake.pydecimal(left_digits=3, right_digits=2, min_value=-999, max_value=0)
        self.assertLess(result, 0)

    def test_min_value_minus_one_hundred_doesnt_return_positive(self):
        if False:
            print('Hello World!')
        Faker.seed('5')
        result = self.fake.pydecimal(left_digits=3, right_digits=2, min_value=-999, max_value=-100)
        self.assertLess(result, -100)

    def test_min_value_10_pow_1000_return_greater_number(self):
        if False:
            i = 10
            return i + 15
        Faker.seed('2')
        result = self.fake.pydecimal(min_value=10 ** 1000)
        self.assertGreater(result, 10 ** 1000)

class TestPystr(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fake = Faker(includes=['tests.mymodule.en_US'])
        Faker.seed(0)

    def test_no_parameters(self):
        if False:
            print('Hello World!')
        some_string = self.fake.pystr()
        assert isinstance(some_string, str)
        assert len(some_string) <= 20

    def test_lower_length_limit(self):
        if False:
            i = 10
            return i + 15
        some_string = self.fake.pystr(min_chars=3)
        assert isinstance(some_string, str)
        assert len(some_string) >= 3
        assert len(some_string) <= 20

    def test_upper_length_limit(self):
        if False:
            while True:
                i = 10
        some_string = self.fake.pystr(max_chars=5)
        assert isinstance(some_string, str)
        assert len(some_string) <= 5

    def test_invalid_length_limits(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(AssertionError):
            self.fake.pystr(min_chars=6, max_chars=5)

    def test_exact_length(self):
        if False:
            for i in range(10):
                print('nop')
        some_string = self.fake.pystr(min_chars=5, max_chars=5)
        assert isinstance(some_string, str)
        assert len(some_string) == 5

    def test_prefix(self):
        if False:
            i = 10
            return i + 15
        some_string = self.fake.pystr(prefix='START_')
        assert isinstance(some_string, str)
        assert some_string.startswith('START_')
        assert len(some_string) == 26

    def test_suffix(self):
        if False:
            while True:
                i = 10
        some_string = self.fake.pystr(suffix='_END')
        assert isinstance(some_string, str)
        assert some_string.endswith('_END')
        assert len(some_string) == 24

    def test_prefix_and_suffix(self):
        if False:
            for i in range(10):
                print('nop')
        some_string = self.fake.pystr(min_chars=9, max_chars=20, prefix='START_', suffix='_END')
        assert isinstance(some_string, str)
        assert some_string.startswith('START_')
        assert some_string.endswith('_END')
        assert len(some_string) >= 19

class TestPystrFormat(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fake = Faker(includes=['tests.mymodule.en_US'])
        Faker.seed(0)

    def test_formatter_invocation(self):
        if False:
            return 10
        with patch.object(self.fake['en_US'], 'foo') as mock_foo:
            with patch('faker.providers.BaseProvider.bothify', wraps=self.fake.bothify) as mock_bothify:
                mock_foo.return_value = 'barbar'
                value = self.fake.pystr_format('{{foo}}?#?{{foo}}?#?{{foo}}', letters='abcde')
                assert value.count('barbar') == 3
                assert mock_foo.call_count == 3
                mock_bothify.assert_called_once_with('barbar?#?barbar?#?barbar', letters='abcde')

class TestPython(unittest.TestCase):
    """Tests python generators"""

    def setUp(self):
        if False:
            print('Hello World!')
        self.factory = Faker()

    def test_pybool_return_type(self):
        if False:
            i = 10
            return i + 15
        some_bool = self.factory.pybool()
        assert isinstance(some_bool, bool)

    def __test_pybool_truth_probability(self, truth_probability: int, deviation_threshold: int=5, iterations: int=999):
        if False:
            while True:
                i = 10
        truth_count_expected = iterations * truth_probability / 100
        truth_count_actual = 0
        for iteration in range(iterations):
            boolean = self.factory.pybool(truth_probability=truth_probability)
            assert isinstance(boolean, bool)
            if boolean is True:
                truth_count_actual += 1
        deviation_absolute = abs(truth_count_expected - truth_count_actual)
        deviation_percentage = deviation_absolute / iterations * 100
        assert deviation_percentage <= deviation_threshold

    def test_pybool_truth_probability_zero(self):
        if False:
            for i in range(10):
                print('nop')
        self.__test_pybool_truth_probability(0, deviation_threshold=0)

    def test_pybool_truth_probability_twenty_five(self):
        if False:
            while True:
                i = 10
        self.__test_pybool_truth_probability(25)

    def test_pybool_truth_probability_fifty(self):
        if False:
            return 10
        self.__test_pybool_truth_probability(50)

    def test_pybool_truth_probability_seventy_five(self):
        if False:
            while True:
                i = 10
        self.__test_pybool_truth_probability(75)

    def test_pybool_truth_probability_hundred(self):
        if False:
            i = 10
            return i + 15
        self.__test_pybool_truth_probability(100, deviation_threshold=0)

    def __test_pybool_invalid_truth_probability(self, truth_probability: int):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError) as exception:
            self.factory.pybool(truth_probability=truth_probability)
        message_expected = 'Invalid `truth_probability` value: must be between `0` and `100` inclusive'
        message_actual = str(exception.value)
        assert message_expected == message_actual

    def test_pybool_truth_probability_less_than_zero(self):
        if False:
            return 10
        self.__test_pybool_invalid_truth_probability(-1)

    def test_pybool_truth_probability_more_than_hundred(self):
        if False:
            while True:
                i = 10
        self.__test_pybool_invalid_truth_probability(101)

    def test_pytuple(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as w:
            some_tuple = Faker().pytuple()
            assert len(w) == 0
        assert some_tuple
        assert isinstance(some_tuple, tuple)

    def test_pytuple_size(self):
        if False:
            while True:
                i = 10

        def mock_pyint(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return 1
        with patch('faker.providers.python.Provider.pyint', mock_pyint):
            some_tuple = Faker().pytuple(nb_elements=3, variable_nb_elements=False, value_types=[int])
            assert some_tuple == (1, 1, 1)

    def test_pylist(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as w:
            some_list = self.factory.pylist()
            assert len(w) == 0
        assert some_list
        assert isinstance(some_list, list)

    def test_pylist_types(self):
        if False:
            print('Hello World!')
        with warnings.catch_warnings(record=True) as w:
            some_list = self.factory.pylist(10, True, [int])
            assert len(w) == 0
        assert some_list
        for item in some_list:
            assert isinstance(item, int)
        with warnings.catch_warnings(record=True) as w:
            some_list = self.factory.pylist(10, True, value_types=[int])
            assert len(w) == 0
        assert some_list
        for item in some_list:
            assert isinstance(item, int)
        with warnings.catch_warnings(record=True) as w:
            some_list = self.factory.pylist(10, True, int)
            assert len(w) == 1
        assert some_list
        for item in some_list:
            assert isinstance(item, int)
        with warnings.catch_warnings(record=True) as w:
            some_list = self.factory.pylist(10, True, int, float)
            assert len(w) == 2
        assert some_list
        for item in some_list:
            assert isinstance(item, (int, float))