from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_filename('defined_name04.xlsx')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the creation of a simple XlsxWriter file with defined names.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        workbook.define_name('\\__', '=Sheet1!$A$1')
        workbook.define_name('a3f6', '=Sheet1!$A$2')
        workbook.define_name('afoo.bar', '=Sheet1!$A$3')
        workbook.define_name('étude', '=Sheet1!$A$4')
        workbook.define_name('eésumé', '=Sheet1!$A$5')
        workbook.define_name('b', '=Sheet1!$A$6')
        import warnings
        warnings.filterwarnings('ignore')
        workbook.define_name('.abc', '=Sheet1!$B$1')
        workbook.define_name('GFG$', '=Sheet1!$B$1')
        workbook.define_name('A1', '=Sheet1!$B$1')
        workbook.define_name('XFD1048576', '=Sheet1!$B$1')
        workbook.define_name('1A', '=Sheet1!$B$1')
        workbook.define_name('A A', '=Sheet1!$B$1')
        workbook.define_name('c', '=Sheet1!$B$1')
        workbook.define_name('r', '=Sheet1!$B$1')
        workbook.define_name('C', '=Sheet1!$B$1')
        workbook.define_name('R', '=Sheet1!$B$1')
        workbook.define_name('R1', '=Sheet1!$B$1')
        workbook.define_name('C1', '=Sheet1!$B$1')
        workbook.define_name('R1C1', '=Sheet1!$B$1')
        workbook.define_name('R13C99', '=Sheet1!$B$1')
        warnings.resetwarnings()
        workbook.close()
        self.assertExcelEqual()