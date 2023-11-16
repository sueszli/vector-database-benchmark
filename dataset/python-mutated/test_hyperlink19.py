from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook
from ...sharedstrings import SharedStringTable

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_filename('hyperlink19.xlsx')
        self.ignore_files = ['xl/calcChain.xml', '[Content_Types].xml', 'xl/_rels/workbook.xml.rels']

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_url('A1', 'http://www.perl.com/')
        worksheet.write_formula('A1', '=1+1', None, 2)
        workbook.str_table = SharedStringTable()
        workbook.close()
        self.assertExcelEqual()