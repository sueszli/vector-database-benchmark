from borb.pdf import Document
from borb.pdf import HexColor
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import SingleColumnLayout
from borb.pdf.canvas.layout.text.line_of_text import LineOfText
from tests.test_case import TestCase

class TestAddLineOfTextUsingFontColor(TestCase):

    def test_add_lineoftext_using_font_color_001(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a LineOfText to the PDF with font_color 023047'))
        layout.add(LineOfText('Lorem Ipsum Dolor', font_color=HexColor('023047')))
        with open(self.get_first_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_lineoftext_using_font_color_002(self):
        if False:
            i = 10
            return i + 15
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a LineOfText to the PDF with font_color FFB703'))
        layout.add(LineOfText('Lorem Ipsum Dolor', font_color=HexColor('FFB703')))
        with open(self.get_second_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_add_lineoftext_using_font_color_003(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a LineOfText to the PDF with font_color FB8500'))
        layout.add(LineOfText('Lorem Ipsum Dolor', font_color=HexColor('FB8500')))
        with open(self.get_third_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())

    def test_add_lineoftext_using_font_color_004(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a LineOfText to the PDF with font_color 219EBC'))
        layout.add(LineOfText('Lorem Ipsum Dolor', font_color=HexColor('219EBC')))
        with open(self.get_fourth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_fourth_output_file())
        self.check_pdf_using_validator(self.get_fourth_output_file())

    def test_add_lineoftext_using_font_color_005(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a LineOfText to the PDF with font_color 8ECAE6'))
        layout.add(LineOfText('Lorem Ipsum Dolor', font_color=HexColor('8ECAE6')))
        with open(self.get_fifth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_fifth_output_file())
        self.check_pdf_using_validator(self.get_fifth_output_file())