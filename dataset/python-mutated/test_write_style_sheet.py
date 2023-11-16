import unittest
from io import StringIO
from ...styles import Styles

class TestWriteStyleSheet(unittest.TestCase):
    """
    Test the Styles _write_style_sheet() method.

    """

    def setUp(self):
        if False:
            return 10
        self.fh = StringIO()
        self.styles = Styles()
        self.styles._set_filehandle(self.fh)

    def test_write_style_sheet(self):
        if False:
            while True:
                i = 10
        'Test the _write_style_sheet() method'
        self.styles._write_style_sheet()
        exp = '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)