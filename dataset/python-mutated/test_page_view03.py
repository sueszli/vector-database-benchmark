from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('page_view03.xlsx')

    def test_create_file(self):
        if False:
            print('Hello World!')
        'Test the creation of a simple XlsxWriter file with print options.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_pagebreak_view()
        worksheet.set_zoom(75)
        worksheet.set_paper(9)
        worksheet.vertical_dpi = 200
        worksheet.write('A1', 'Foo')
        workbook.close()
        self.assertExcelEqual()