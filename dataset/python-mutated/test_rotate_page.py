import random
import typing
import unittest
from borb.pdf import Lipsum
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestRemovePage(TestCase):

    def test_create_dummy_pdf_document(self):
        if False:
            i = 10
            return i + 15
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test creates a simple dummy PDF.'))
        random.seed(2048)
        for s in [Lipsum.generate_arthur_conan_doyle_text(random.choice([5, 6, 7])) for _ in range(0, 20)]:
            layout.add(Paragraph(s))
        pdf.pop_page(0)
        with open(self.get_first_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_rotate_page_right(self):
        if False:
            print('Hello World!')
        doc: typing.Optional[Document] = None
        with open(self.get_first_output_file(), 'rb') as in_file_handle:
            doc = PDF.loads(in_file_handle)
        doc.get_page(0).rotate_right()
        with open(self.get_second_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, doc)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_rotate_page_left(self):
        if False:
            i = 10
            return i + 15
        doc: typing.Optional[Document] = None
        with open(self.get_first_output_file(), 'rb') as in_file_handle:
            doc = PDF.loads(in_file_handle)
        doc.get_page(0).rotate_left()
        with open(self.get_third_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, doc)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())