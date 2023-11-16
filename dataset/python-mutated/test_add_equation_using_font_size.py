from decimal import Decimal
from borb.pdf import PageLayout
from borb.pdf import SingleColumnLayout
from borb.pdf.canvas.layout.equation.equation import Equation
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase

class TestAddEquationUsingFontSize(TestCase):
    """
    This test attempts to extract the text of each PDF in the corpus
    """

    def test_add_equation_using_font_size(self):
        if False:
            i = 10
            return i + 15
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header('This test creates a PDF with an equation in it with a non-default font_size.'))
        page_layout.add(Equation('sin(x)/cos(x)=tan(x)', font_size=Decimal(20)))
        with open(self.get_first_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())