import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWritePrintOptions(unittest.TestCase):
    """
    Test the Worksheet _write_print_options() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_print_options_default(self):
        if False:
            return 10
        'Test the _write_print_options() method without options'
        self.worksheet._write_print_options()
        exp = ''
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_print_options_hcenter(self):
        if False:
            print('Hello World!')
        'Test the _write_print_options() method with horizontal center'
        self.worksheet.center_horizontally()
        self.worksheet._write_print_options()
        exp = '<printOptions horizontalCentered="1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_print_options_vcenter(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_print_options() method with vertical center'
        self.worksheet.center_vertically()
        self.worksheet._write_print_options()
        exp = '<printOptions verticalCentered="1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_print_options_center(self):
        if False:
            return 10
        'Test the _write_print_options() method with horiz + vert center'
        self.worksheet.center_horizontally()
        self.worksheet.center_vertically()
        self.worksheet._write_print_options()
        exp = '<printOptions horizontalCentered="1" verticalCentered="1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_print_options_gridlines_default(self):
        if False:
            return 10
        'Test the _write_print_options() method with default value'
        self.worksheet.hide_gridlines()
        self.worksheet._write_print_options()
        exp = ''
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_print_options_gridlines_0(self):
        if False:
            print('Hello World!')
        'Test the _write_print_options() method with 0 value'
        self.worksheet.hide_gridlines(0)
        self.worksheet._write_print_options()
        exp = '<printOptions gridLines="1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)