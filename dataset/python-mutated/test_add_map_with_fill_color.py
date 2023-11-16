from borb.pdf import Document
from borb.pdf import HexColor
from borb.pdf import MapOfTheWorld
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import SingleColumnLayout
from tests.test_case import TestCase

class TestAddMapWithFillColor(TestCase):

    def test_add_map_with_fill_color(self):
        if False:
            print('Hello World!')
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header('This tests creates a PDF with a MapOfTheWorld in it.'))
        layout.add(MapOfTheWorld().set_fill_color(HexColor('#f0f0f0')).set_fill_color(HexColor('#f1cd2e'), key='United States of America'))
        with open(self.get_first_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())