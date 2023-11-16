import unittest
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase

class TestCreateEmptyPDF(TestCase):
    """
    This test attempts to extract the text of each PDF in the corpus
    """

    def test_create_empty_pdf(self):
        if False:
            while True:
                i = 10
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        with open(self.get_first_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())
if __name__ == '__main__':
    unittest.main()