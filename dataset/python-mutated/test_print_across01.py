from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_filename('print_across01.xlsx')
        self.ignore_files = ['xl/printerSettings/printerSettings1.bin', 'xl/worksheets/_rels/sheet1.xml.rels']
        self.ignore_elements = {'[Content_Types].xml': ['<Default Extension="bin"'], 'xl/worksheets/sheet1.xml': ['<pageMargins']}

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with fit across.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.print_across()
        worksheet.set_paper(9)
        worksheet.write('A1', 'Foo')
        workbook.close()
        self.assertExcelEqual()