import unittest
import warnings
from ...utility import xl_col_to_name

class TestUtility(unittest.TestCase):
    """
    Test xl_col_to_name() utility function.

    """

    def test_xl_col_to_name(self):
        if False:
            print('Hello World!')
        'Test xl_col_to_name()'
        tests = [(0, 'A'), (1, 'B'), (2, 'C'), (9, 'J'), (24, 'Y'), (25, 'Z'), (26, 'AA'), (254, 'IU'), (255, 'IV'), (256, 'IW'), (16383, 'XFD'), (16384, 'XFE'), (-1, None)]
        for (col, string) in tests:
            exp = string
            got = xl_col_to_name(col)
            warnings.filterwarnings('ignore')
            self.assertEqual(got, exp)

    def test_xl_col_to_name_abs(self):
        if False:
            i = 10
            return i + 15
        'Test xl_col_to_name() with absolute references'
        tests = [(0, True, '$A'), (-1, True, None)]
        for (col, col_abs, string) in tests:
            exp = string
            got = xl_col_to_name(col, col_abs)
            warnings.filterwarnings('ignore')
            self.assertEqual(got, exp)