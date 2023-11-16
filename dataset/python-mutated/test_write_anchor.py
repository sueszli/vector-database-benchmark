import unittest
from io import StringIO
from ...vml import Vml

class TestWriteXAnchor(unittest.TestCase):
    """
    Test the Vml _write_anchor() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_write_anchor(self):
        if False:
            print('Hello World!')
        'Test the _write_anchor() method'
        self.vml._write_anchor([2, 0, 15, 10, 4, 4, 15, 4])
        exp = '<x:Anchor>2, 15, 0, 10, 4, 15, 4, 4</x:Anchor>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)