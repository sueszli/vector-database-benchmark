import unittest
from io import StringIO
from ...vml import Vml

class TestWriteXSizeWithCells(unittest.TestCase):
    """
    Test the Vml _write_size_with_cells() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_write_size_with_cells(self):
        if False:
            return 10
        'Test the _write_size_with_cells() method'
        self.vml._write_size_with_cells()
        exp = '<x:SizeWithCells/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)