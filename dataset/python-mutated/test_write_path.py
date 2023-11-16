import unittest
from io import StringIO
from ...vml import Vml

class TestWriteVpath(unittest.TestCase):
    """
    Test the Vml _write_path() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_write_comment_path_1(self):
        if False:
            while True:
                i = 10
        'Test the _write_comment_path() method'
        self.vml._write_comment_path('t', 'rect')
        exp = '<v:path gradientshapeok="t" o:connecttype="rect"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_comment_path_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_comment_path() method'
        self.vml._write_comment_path(None, 'none')
        exp = '<v:path o:connecttype="none"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_button_path(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_button_path() method'
        self.vml._write_button_path()
        exp = '<v:path shadowok="f" o:extrusionok="f" strokeok="f" fillok="f" o:connecttype="rect"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)