"""Testing utilities for tfdbg command-line interface."""
import re
import numpy as np

def assert_lines_equal_ignoring_whitespace(test, expected_lines, actual_lines):
    if False:
        return 10
    'Assert equality in lines, ignoring all whitespace.\n\n  Args:\n    test: An instance of unittest.TestCase or its subtypes (e.g.,\n      TensorFlowTestCase).\n    expected_lines: Expected lines as an iterable of strings.\n    actual_lines: Actual lines as an iterable of strings.\n  '
    test.assertEqual(len(expected_lines), len(actual_lines), 'Mismatch in the number of lines: %d vs %d' % (len(expected_lines), len(actual_lines)))
    for (expected_line, actual_line) in zip(expected_lines, actual_lines):
        test.assertEqual(''.join(expected_line.split()), ''.join(actual_line.split()))
_ARRAY_VALUE_SEPARATOR_REGEX = re.compile('(array|\\(|\\[|\\]|\\)|\\||,)')

def assert_array_lines_close(test, expected_array, array_lines):
    if False:
        print('Hello World!')
    'Assert that the array value represented by lines is close to expected.\n\n  Note that the shape of the array represented by the `array_lines` is ignored.\n\n  Args:\n    test: An instance of TensorFlowTestCase.\n    expected_array: Expected value of the array.\n    array_lines: A list of strings representing the array.\n      E.g., "array([[ 1.0, 2.0 ], [ 3.0, 4.0 ]])"\n      Assumes that values are separated by commas, parentheses, brackets, "|"\n      characters and whitespace.\n  '
    elements = []
    for line in array_lines:
        line = re.sub(_ARRAY_VALUE_SEPARATOR_REGEX, ' ', line)
        elements.extend((float(s) for s in line.split()))
    test.assertAllClose(np.array(expected_array).flatten(), elements)