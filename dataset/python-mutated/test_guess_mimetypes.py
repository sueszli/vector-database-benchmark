import os.path
import unittest
from odoo.tools.mimetypes import guess_mimetype

def contents(extension):
    if False:
        for i in range(10):
            print('nop')
    with open(os.path.join(os.path.dirname(__file__), 'testfiles', 'case.{}'.format(extension)), 'rb') as f:
        return f.read()

class TestMimeGuessing(unittest.TestCase):

    def test_doc(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(guess_mimetype(contents('doc')), 'application/msword')

    def test_xls(self):
        if False:
            return 10
        self.assertEqual(guess_mimetype(contents('xls')), 'application/vnd.ms-excel')

    def test_docx(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(guess_mimetype(contents('docx')), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')

    def test_xlsx(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(guess_mimetype(contents('xlsx')), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    def test_odt(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(guess_mimetype(contents('odt')), 'application/vnd.oasis.opendocument.text')

    def test_ods(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(guess_mimetype(contents('ods')), 'application/vnd.oasis.opendocument.spreadsheet')

    def test_zip(self):
        if False:
            return 10
        self.assertEqual(guess_mimetype(contents('zip')), 'application/zip')

    def test_gif(self):
        if False:
            return 10
        self.assertEqual(guess_mimetype(contents('gif')), 'image/gif')

    def test_jpeg(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(guess_mimetype(contents('jpg')), 'image/jpeg')

    def test_unknown(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(guess_mimetype(contents('csv')), 'application/octet-stream')