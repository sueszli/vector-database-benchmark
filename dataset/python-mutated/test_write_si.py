import unittest
from io import StringIO
from ...sharedstrings import SharedStrings

class TestWriteSi(unittest.TestCase):
    """
    Test the SharedStrings _write_si() method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.sharedstrings = SharedStrings()
        self.sharedstrings._set_filehandle(self.fh)

    def test_write_si(self):
        if False:
            return 10
        'Test the _write_si() method'
        self.sharedstrings._write_si('neptune')
        exp = '<si><t>neptune</t></si>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)