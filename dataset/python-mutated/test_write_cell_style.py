import unittest
from io import StringIO
from ...styles import Styles

class TestWriteCellStyle(unittest.TestCase):
    """
    Test the Styles _write_cell_style() method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.styles = Styles()
        self.styles._set_filehandle(self.fh)

    def test_write_cell_style(self):
        if False:
            return 10
        'Test the _write_cell_style() method'
        self.styles._write_cell_style()
        exp = '<cellStyle name="Normal" xfId="0" builtinId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)