import unittest
from collections import namedtuple
from io import StringIO
from ...worksheet import Worksheet

class TestWriteCell(unittest.TestCase):
    """
    Test the Worksheet _write_cell() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_cell_number(self):
        if False:
            print('Hello World!')
        'Test the _write_cell() method for numbers.'
        cell_tuple = namedtuple('Number', 'number, format')
        cell = cell_tuple(1, None)
        self.worksheet._write_cell(0, 0, cell)
        exp = '<c r="A1"><v>1</v></c>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_cell_string(self):
        if False:
            return 10
        'Test the _write_cell() method for strings.'
        cell_tuple = namedtuple('String', 'string, format')
        cell = cell_tuple(0, None)
        self.worksheet._write_cell(3, 1, cell)
        exp = '<c r="B4" t="s"><v>0</v></c>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_cell_formula01(self):
        if False:
            return 10
        'Test the _write_cell() method for formulas.'
        cell_tuple = namedtuple('Formula', 'formula, format, value')
        cell = cell_tuple('A3+A5', None, 0)
        self.worksheet._write_cell(1, 2, cell)
        exp = '<c r="C2"><f>A3+A5</f><v>0</v></c>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_cell_formula02(self):
        if False:
            print('Hello World!')
        'Test the _write_cell() method for formulas.'
        cell_tuple = namedtuple('Formula', 'formula, format, value')
        cell = cell_tuple('A3+A5', None, 7)
        self.worksheet._write_cell(1, 2, cell)
        exp = '<c r="C2"><f>A3+A5</f><v>7</v></c>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)