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
        self.set_filename('table01.xlsx')
        self.ignore_files = ['xl/sharedStrings.xml']

    def test_create_file(self):
        if False:
            while True:
                i = 10
        'Test the creation of a simple XlsxWriter file with tables.'
        workbook = Workbook(self.got_filename)
        worksheet = workbook.add_worksheet()
        worksheet.set_column('C:F', 10.288)
        worksheet.add_table('C3:F13')
        import warnings
        warnings.filterwarnings('ignore')
        worksheet.add_table('G3:H3', {'columns': [{'header': 'Column1'}, {'header': 'Column1'}]})
        workbook.close()
        self.assertExcelEqual()