import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteTabColor(unittest.TestCase):
    """
    Test the Worksheet _write_tab_color() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_tab_color(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_tab_color() method'
        self.worksheet.set_tab_color('red')
        self.worksheet._write_tab_color()
        exp = '<tabColor rgb="FFFF0000"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)