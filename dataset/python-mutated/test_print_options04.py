from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('print_options04.xlsx')
        self.ignore_files = ['xl/printerSettings/printerSettings1.bin', 'xl/worksheets/_rels/sheet1.xml.rels']
        self.ignore_elements = {'[Content_Types].xml': ['<Default Extension="bin"'], 'xl/worksheets/sheet1.xml': ['<pageMargins', '<pageSetup']}

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file with print options.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.print_row_col_headers()
        worksheet.write('A1', 'Foo')
        workbook.close()
        self.assertExcelEqual()