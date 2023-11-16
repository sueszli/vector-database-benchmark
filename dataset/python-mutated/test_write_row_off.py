import unittest
from io import StringIO
from ...drawing import Drawing

class TestWriteXdrrowOff(unittest.TestCase):
    """
    Test the Drawing _write_row_off() method.

    """

    def setUp(self):
        if False:
            return 10
        self.fh = StringIO()
        self.drawing = Drawing()
        self.drawing._set_filehandle(self.fh)

    def test_write_row_off(self):
        if False:
            while True:
                i = 10
        'Test the _write_row_off() method'
        self.drawing._write_row_off(104775)
        exp = '<xdr:rowOff>104775</xdr:rowOff>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)