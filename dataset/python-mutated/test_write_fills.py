import unittest
from io import StringIO
from ...styles import Styles

class TestWriteFills(unittest.TestCase):
    """
    Test the Styles _write_fills() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.styles = Styles()
        self.styles._set_filehandle(self.fh)

    def test_write_fills(self):
        if False:
            print('Hello World!')
        'Test the _write_fills() method'
        self.styles.fill_count = 2
        self.styles._write_fills()
        exp = '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)