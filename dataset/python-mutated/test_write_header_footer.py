import unittest
from io import StringIO
from ...worksheet import Worksheet

class TestWriteHeaderFooter(unittest.TestCase):
    """
    Test the Worksheet _write_header_footer() method.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.fh = StringIO()
        self.worksheet = Worksheet()
        self.worksheet._set_filehandle(self.fh)

    def test_write_header_footer_header_only(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_header_footer() method header only'
        self.worksheet.set_header('Page &P of &N')
        self.worksheet._write_header_footer()
        exp = '<headerFooter><oddHeader>Page &amp;P of &amp;N</oddHeader></headerFooter>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_header_footer_footer_only(self):
        if False:
            print('Hello World!')
        'Test the _write_header_footer() method footer only'
        self.worksheet.set_footer('&F')
        self.worksheet._write_header_footer()
        exp = '<headerFooter><oddFooter>&amp;F</oddFooter></headerFooter>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_header_footer_both(self):
        if False:
            return 10
        'Test the _write_header_footer() method header and footer'
        self.worksheet.set_header('Page &P of &N')
        self.worksheet.set_footer('&F')
        self.worksheet._write_header_footer()
        exp = '<headerFooter><oddHeader>Page &amp;P of &amp;N</oddHeader><oddFooter>&amp;F</oddFooter></headerFooter>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)