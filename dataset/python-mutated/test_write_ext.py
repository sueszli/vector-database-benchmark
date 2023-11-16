import unittest
from io import StringIO
from ...drawing import Drawing

class TestWriteXdrext(unittest.TestCase):
    """
    Test the Drawing _write_ext() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.drawing = Drawing()
        self.drawing._set_filehandle(self.fh)

    def test_write_xdr_ext(self):
        if False:
            while True:
                i = 10
        'Test the _write_ext() method'
        self.drawing._write_xdr_ext(9308969, 6078325)
        exp = '<xdr:ext cx="9308969" cy="6078325"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)