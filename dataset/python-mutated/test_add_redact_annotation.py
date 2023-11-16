import typing
import unittest
from decimal import Decimal
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.redact_annotation import RedactAnnotation
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.page_layout.page_layout import PageLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestAddRedactAnnotation(TestCase):

    def test_create_pdf_to_redact(self):
        if False:
            while True:
                i = 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout: PageLayout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a SquigglyAnnotation to a PDF.'))
        layout.add(Paragraph('\n            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \n            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. \n            Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \n            Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.\n            '))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_redact_annotation(self):
        if False:
            while True:
                i = 10
        doc: typing.Optional[Document] = None
        with open(self.get_first_output_file(), 'rb') as fh:
            doc = PDF.loads(fh)
        assert doc is not None
        doc.get_page(0).add_annotation(RedactAnnotation(bounding_box=Rectangle(Decimal(250), Decimal(625), Decimal(52), Decimal(17)), fill_color=HexColor('000000')))
        with open(self.get_second_output_file(), 'wb') as fh:
            PDF.dumps(fh, doc)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_apply_redact_annotation(self):
        if False:
            i = 10
            return i + 15
        doc: typing.Optional[Document] = None
        with open(self.get_second_output_file(), 'rb') as fh:
            doc = PDF.loads(fh)
        assert doc is not None
        doc.get_page(0).apply_redact_annotations()
        with open(self.get_third_output_file(), 'wb') as fh:
            PDF.dumps(fh, doc)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())