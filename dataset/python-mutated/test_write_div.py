import unittest
from io import StringIO
from ...vml import Vml

class TestWriteDiv(unittest.TestCase):
    """
    Test the Vml _write_div() method.

    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_write_div(self):
        if False:
            return 10
        'Test the _write_div() method'
        self.vml._write_div('left')
        exp = '<div style="text-align:left"></div>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)