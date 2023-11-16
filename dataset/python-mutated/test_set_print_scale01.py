from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('set_print_scale01.xlsx')
        self.ignore_files = ['xl/printerSettings/printerSettings1.bin', 'xl/worksheets/_rels/sheet1.xml.rels']
        self.ignore_elements = {'[Content_Types].xml': ['<Default Extension="bin"'], 'xl/worksheets/sheet1.xml': ['<pageMargins']}

    def test_create_file(self):
        if False:
            i = 10
            return i + 15
        'Test the creation of a simple XlsxWriter file with printer settings.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_print_scale(110)
        worksheet.set_paper(9)
        worksheet.write('A1', 'Foo')
        workbook.close()
        self.assertExcelEqual()