import unittest
from io import StringIO
from ...worksheet import Worksheet
from ...format import Format
from ...sharedstrings import SharedStringTable

class TestWriteMergeCells(unittest.TestCase):
    """
    Test the Worksheet _write_merge_cells() method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)
        self.worksheet.str_table = SharedStringTable()

    def test_write_merge_cells_1(self):
        if False:
            while True:
                i = 10
        'Test the _write_merge_cells() method'
        cell_format = Format()
        self.worksheet.merge_range(2, 1, 2, 2, 'Foo', cell_format)
        self.worksheet._write_merge_cells()
        exp = '<mergeCells count="1"><mergeCell ref="B3:C3"/></mergeCells>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_merge_cells_2(self):
        if False:
            while True:
                i = 10
        'Test the _write_merge_cells() method'
        cell_format = Format()
        self.worksheet.merge_range('B3:C3', 'Foo', cell_format)
        self.worksheet._write_merge_cells()
        exp = '<mergeCells count="1"><mergeCell ref="B3:C3"/></mergeCells>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_merge_cells_3(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_merge_cells() method'
        cell_format = Format()
        self.worksheet.merge_range('B3:C3', 'Foo', cell_format)
        self.worksheet.merge_range('A2:D2', 'Foo', cell_format)
        self.worksheet._write_merge_cells()
        exp = '<mergeCells count="2"><mergeCell ref="B3:C3"/><mergeCell ref="A2:D2"/></mergeCells>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)