import unittest
from borb.pdf import PageLayout
from borb.pdf import SingleColumnLayout
from borb.pdf.canvas.layout.equation.equation import Equation
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase

class TestAddEquation(TestCase):
    """
    This test attempts to extract the text of each PDF in the corpus
    """

    def test_add_equation_addition(self):
        if False:
            i = 10
            return i + 15
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header('This test creates a PDF with an equation in it.'))
        page_layout.add(Equation('1+2'))
        with open(self.get_first_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_equation_divide_by(self):
        if False:
            for i in range(10):
                print('nop')
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header('This test creates a PDF with an equation in it.'))
        page_layout.add(Equation('sin(x)/cos(x)'))
        with open(self.get_second_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_add_equation_multiplication(self):
        if False:
            i = 10
            return i + 15
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header('This test creates a PDF with an equation in it.'))
        page_layout.add(Equation('sin(x)*sin(x)'))
        with open(self.get_third_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())

    def test_add_equation_equality(self):
        if False:
            i = 10
            return i + 15
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header('This test creates a PDF with an equation in it.'))
        page_layout.add(Equation('sin(x)/cos(x)=tan(x)'))
        with open(self.get_fourth_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_fourth_output_file())
        self.check_pdf_using_validator(self.get_fourth_output_file())
if __name__ == '__main__':
    unittest.main()