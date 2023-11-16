import warnings
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
        self.set_filename('simple01.xlsx')

    def test_close_file_twice(self):
        if False:
            while True:
                i = 10
        'Test warning when closing workbook more than once.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.write_string(0, 0, 'Hello')
        worksheet.write_number(1, 0, 123)
        workbook.close()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            workbook.close()
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
        self.assertExcelEqual()