import unittest
from io import StringIO
from ...styles import Styles

class TestWriteMruColors(unittest.TestCase):
    """
    Test the Styles _write_mru_colors() method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.styles = Styles()
        self.styles._set_filehandle(self.fh)

    def test_write_mru_colors(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_mru_colors() method'
        self.styles._write_mru_colors(['FF26DA55', 'FF792DC8', 'FF646462'])
        exp = '<mruColors><color rgb="FF646462"/><color rgb="FF792DC8"/><color rgb="FF26DA55"/></mruColors>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)