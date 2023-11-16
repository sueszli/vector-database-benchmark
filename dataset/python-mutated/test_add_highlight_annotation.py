from decimal import Decimal
from borb.pdf import PageLayout
from borb.pdf.canvas.geometry.rectangle import Rectangle
from borb.pdf.canvas.layout.annotation.highlight_annotation import HighlightAnnotation
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase

class TestAddHighlightAnnotation(TestCase):

    def test_add_highlight_annotation(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout: PageLayout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a HighlightAnnotation to a PDF.'))
        layout.add(Paragraph('\n            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \n            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. \n            Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \n            Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.\n            '))
        page.add_annotation(HighlightAnnotation(bounding_box=Rectangle(Decimal(250), Decimal(625), Decimal(52), Decimal(17))))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())