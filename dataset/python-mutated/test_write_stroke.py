import unittest
from io import StringIO
from ...vml import Vml

class TestWriteVstroke(unittest.TestCase):
    """
    Test the Vml _write_stroke() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_write_stroke(self):
        if False:
            while True:
                i = 10
        'Test the _write_stroke() method'
        self.vml._write_stroke()
        exp = '<v:stroke joinstyle="miter"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)