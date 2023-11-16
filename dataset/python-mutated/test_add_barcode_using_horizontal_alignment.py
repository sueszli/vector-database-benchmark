from _decimal import Decimal
from borb.pdf import Alignment
from borb.pdf import Barcode
from borb.pdf import BarcodeType
from borb.pdf import Document
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import PageLayout
from borb.pdf import SingleColumnLayout
from tests.test_case import TestCase

class TestAddBarcodeUsingHorizontalAlignment(TestCase):

    def test_add_barcode_using_horizontal_alignment_left(self):
        if False:
            for i in range(10):
                print('nop')
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header(test_description=f'This test creates a PDF with a Barcode in it.'))
        page_layout.add(Barcode(data='https://www.borbpdf.com', type=BarcodeType.QR, width=Decimal(100), height=Decimal(100), horizontal_alignment=Alignment.LEFT))
        with open(self.get_first_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_barcode_using_horizontal_alignment_centered(self):
        if False:
            i = 10
            return i + 15
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header(test_description=f'This test creates a PDF with a Barcode in it.'))
        page_layout.add(Barcode(data='https://www.borbpdf.com', type=BarcodeType.QR, width=Decimal(100), height=Decimal(100), horizontal_alignment=Alignment.CENTERED))
        with open(self.get_second_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_add_barcode_using_horizontal_alignment_right(self):
        if False:
            return 10
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header(test_description=f'This test creates a PDF with a Barcode in it.'))
        page_layout.add(Barcode(data='https://www.borbpdf.com', type=BarcodeType.QR, width=Decimal(100), height=Decimal(100), horizontal_alignment=Alignment.RIGHT))
        with open(self.get_third_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())