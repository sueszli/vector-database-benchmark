import unittest
from io import StringIO
from ...comments import Comments

class TestWriteText(unittest.TestCase):
    """
    Test the Comments _write_text_t() method.

    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.fh = StringIO()
        self.comments = Comments()
        self.comments._set_filehandle(self.fh)

    def test_write_text_t_1(self):
        if False:
            while True:
                i = 10
        'Test the _write_text_t() method'
        self.comments._write_text_t('Some text')
        exp = '<t>Some text</t>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_text_t_2(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_text_t() method'
        self.comments._write_text_t(' Some text')
        exp = '<t xml:space="preserve"> Some text</t>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_text_t_3(self):
        if False:
            i = 10
            return i + 15
        'Test the _write_text_t() method'
        self.comments._write_text_t('Some text ')
        exp = '<t xml:space="preserve">Some text </t>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_text_t_4(self):
        if False:
            print('Hello World!')
        'Test the _write_text_t() method'
        self.comments._write_text_t(' Some text ')
        exp = '<t xml:space="preserve"> Some text </t>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)

    def test_write_text_t_5(self):
        if False:
            while True:
                i = 10
        'Test the _write_text_t() method'
        self.comments._write_text_t('Some text\n')
        exp = '<t xml:space="preserve">Some text\n</t>'
        got = self.fh.getvalue()
        self.assertEqual(got, exp)