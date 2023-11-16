import unittest
from io import StringIO
from ...workbook import Workbook

class TestWriteWorkbookPr(unittest.TestCase):
    """
    Test the Workbook _write_workbook_pr() method.

    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.fh = StringIO()
        self.workbook = Workbook()
        self.workbook._set_filehandle(self.fh)

    def test_write_workbook_pr(self):
        if False:
            print('Hello World!')
        'Test the _write_workbook_pr() method'
        self.workbook._write_workbook_pr()
        exp = '<workbookPr defaultThemeVersion="124226"/>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.workbook.fileclosed = 1