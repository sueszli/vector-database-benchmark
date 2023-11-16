import unittest
import os
from .helperfunctions import _compare_xlsx_files

class ExcelComparisonTest(unittest.TestCase):
    """
    Test class for comparing a file created by XlsxWriter against a file
    created by Excel.

    """

    def set_filename(self, filename):
        if False:
            return 10
        self.maxDiff = None
        self.got_filename = ''
        self.exp_filename = ''
        self.ignore_files = []
        self.ignore_elements = {}
        self.test_dir = 'xlsxwriter/test/comparison/'
        self.vba_dir = self.test_dir + 'xlsx_files/'
        self.image_dir = self.test_dir + 'images/'
        self.exp_filename = self.test_dir + 'xlsx_files/' + filename
        self.got_filename = self.test_dir + '_test_' + filename

    def set_text_file(self, filename):
        if False:
            while True:
                i = 10
        self.txt_filename = self.test_dir + 'xlsx_files/' + filename

    def assertExcelEqual(self):
        if False:
            i = 10
            return i + 15
        (got, exp) = _compare_xlsx_files(self.got_filename, self.exp_filename, self.ignore_files, self.ignore_elements)
        self.assertEqual(exp, got)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if os.path.exists(self.got_filename):
            os.remove(self.got_filename)