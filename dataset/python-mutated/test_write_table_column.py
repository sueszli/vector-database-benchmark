import unittest
from io import StringIO
from ...table import Table

class TestWriteTableColumn(unittest.TestCase):
    """
    Test the Table _write_table_column() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.table = Table()
        self.table._set_filehandle(self.fh)

    def test_write_table_column(self):
        if False:
            return 10
        'Test the _write_table_column() method'
        self.table._write_table_column({'name': 'Column1', 'id': 1})
        exp = '<tableColumn id="1" name="Column1"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)