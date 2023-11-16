import unittest
from io import StringIO
from ...workbook import Workbook
from ...exceptions import DuplicateTableName

class TestAddTable(unittest.TestCase):
    """
    Test exceptions with add_table().

    """

    def test_duplicate_table_name(self):
        if False:
            print('Hello World!')
        'Test that adding 2 tables with the same name raises an exception.'
        fh = StringIO()
        workbook = Workbook()
        workbook._set_filehandle(fh)
        worksheet = workbook.add_worksheet()
        worksheet.add_table('B1:F3', {'name': 'SalesData'})
        worksheet.add_table('B4:F7', {'name': 'SalesData'})
        self.assertRaises(DuplicateTableName, workbook._prepare_tables)
        workbook.fileclosed = True