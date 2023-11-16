import unittest
from decimal import Decimal
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.free_text_annotation import FreeTextAnnotation
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestAddFreeTextAnnotation(TestCase):

    def test_add_free_text_annotation(self):
        if False:
            while True:
                i = 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a FreeTextAnnotation to a PDF.'))
        w: Decimal = pdf.get_page(0).get_page_info().get_width()
        h: Decimal = pdf.get_page(0).get_page_info().get_height()
        pdf.get_page(0).add_annotation(FreeTextAnnotation(bounding_box=Rectangle(w / Decimal(2) - Decimal(32), h / Decimal(2) - Decimal(32), Decimal(64), Decimal(64)), contents='Lorem Ipsum Dolor Sit Amet'))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())