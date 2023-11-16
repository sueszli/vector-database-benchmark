import os
import tempfile
import unittest
import warnings
from ...workbook import Workbook
from ...exceptions import FileCreateError

class TestCloseWithException(unittest.TestCase):
    """
    Test the Workbook close() exception.

    """

    def test_non_existent_dir(self):
        if False:
            i = 10
            return i + 15
        'Test the _check_sheetname() method'
        self.workbook = Workbook('non_existent_path/test.xlsx')
        self.workbook.add_worksheet()
        with self.assertRaises(FileCreateError):
            self.workbook.close()

    def test_workbook_closes_all_handles(self):
        if False:
            i = 10
            return i + 15
        'Test that close() closes all file handles'
        filepath = tempfile.mktemp()
        warnings.simplefilter('always')
        with warnings.catch_warnings(record=True) as warnings_emitted:
            workbook = Workbook(filepath, dict(constant_memory=True))
            workbook.close()
            del workbook
        os.unlink(filepath)
        self.assertFalse(warnings_emitted)