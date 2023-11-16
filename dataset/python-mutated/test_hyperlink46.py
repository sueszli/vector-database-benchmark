from ..excel_comparison_test import ExcelComparisonTest
from ...workbook import Workbook

class TestCompareXLSXFiles(ExcelComparisonTest):
    """
    Test file created by XlsxWriter against a file created by Excel.

    """

    def setUp(self):
        if False:
            return 10
        self.set_filename('hyperlink46.xlsx')

    def test_create_file(self):
        if False:
            return 10
        'Test the creation of a simple XlsxWriter file with hyperlinks.'
        workbook = Workbook(self.got_filename, {'max_url_length': 255})
        worksheet = workbook.add_worksheet()
        worksheet.write('A1', 'Foo')
        worksheet.write('A3', 'Bar')
        import warnings
        warnings.filterwarnings('ignore')
        worksheet.write_url('A2', 'http://foo.com/this_is_a_long_hyperlink_that_exceeds_a_limit_of_255_characters_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        workbook.close()
        self.assertExcelEqual()