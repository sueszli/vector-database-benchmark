import unittest
from io import StringIO
from ...drawing import Drawing

class TestWriteXdrpos(unittest.TestCase):
    """
    Test the Drawing _write_pos() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.drawing = Drawing()
        self.drawing._set_filehandle(self.fh)

    def test_write_pos(self):
        if False:
            while True:
                i = 10
        'Test the _write_pos() method'
        self.drawing._write_pos(0, 0)
        exp = '<xdr:pos x="0" y="0"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)