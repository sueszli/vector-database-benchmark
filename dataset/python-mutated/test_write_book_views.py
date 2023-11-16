import unittest
from io import StringIO
from ...workbook import Workbook

class TestWriteBookViews(unittest.TestCase):
    """
    Test the Workbook _write_book_views() method.

    """

    def setUp(self):
        if False:
            return 10
        self.fh = StringIO()
        self.workbook = Workbook()
        self.workbook._set_filehandle(self.fh)

    def test_write_book_views(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the _write_book_views() method'
        self.workbook._write_book_views()
        exp = '<bookViews><workbookView xWindow="240" yWindow="15" windowWidth="16095" windowHeight="9660"/></bookViews>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.workbook.fileclosed = 1