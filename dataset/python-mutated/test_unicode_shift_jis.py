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
        self.set_filename('unicode_shift_jis.xlsx')
        self.set_text_file('unicode_shift_jis.txt')

    def test_create_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test example file converting Unicode text.'
        textfile = open(self.txt_filename, mode='r', encoding='shift_jis')
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('A:A', 50)
        row = 0
        col = 0
        for line in textfile:
            if line.startswith('#'):
                continue
            worksheet.write(row, col, line.rstrip('\n'))
            row += 1
        workbook.close()
        textfile.close()
        self.assertExcelEqual()