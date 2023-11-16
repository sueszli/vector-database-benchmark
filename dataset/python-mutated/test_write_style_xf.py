import unittest
from io import StringIO
from ...styles import Styles

class TestWriteStyleXf(unittest.TestCase):
    """
    Test the Styles _write_style_xf() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.styles = Styles()
        self.styles._set_filehandle(self.fh)

    def test_write_style_xf(self):
        if False:
            while True:
                i = 10
        'Test the _write_style_xf() method'
        self.styles._write_style_xf()
        exp = '<xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)