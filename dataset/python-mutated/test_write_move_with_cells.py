import unittest
from io import StringIO
from ...vml import Vml

class TestWriteXMoveWithCells(unittest.TestCase):
    """
    Test the Vml _write_move_with_cells() method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.vml = Vml()
        self.vml._set_filehandle(self.fh)

    def test_write_move_with_cells(self):
        if False:
            print('Hello World!')
        'Test the _write_move_with_cells() method'
        self.vml._write_move_with_cells()
        exp = '<x:MoveWithCells/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)