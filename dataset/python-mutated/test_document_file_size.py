import typing
import unittest
from decimal import Decimal
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestDocumentFileSize(TestCase):

    def test_write_hello_world(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(Paragraph('Hello World!'))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_check_file_size_001(self):
        if False:
            while True:
                i = 10
        with open(self.get_first_output_file(), 'rb') as pdf_file_handle:
            document = PDF.loads(pdf_file_handle)
        s: typing.Optional[Decimal] = document.get_document_info().get_file_size()
        assert s is not None
        assert 1000 <= s <= 1200

    def test_check_file_size_002(self):
        if False:
            print('Hello World!')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(Paragraph('Hello World!'))
        s: typing.Optional[Decimal] = pdf.get_document_info().get_file_size()
        assert s is None