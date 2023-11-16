import unittest
from io import StringIO
from ...vml import Vml

class TestWriteVtextbox(unittest.TestCase):
    """
    Test the Vml _write_textbox() method.

    """

    def setUp(self):
        if False:
            return 10
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_write_comment_textbox(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_comment_textbox() method'
        self.vml._write_comment_textbox()
        exp = '<v:textbox style="mso-direction-alt:auto"><div style="text-align:left"></div></v:textbox>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)