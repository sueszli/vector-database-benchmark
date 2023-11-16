import unittest
from decimal import Decimal
from borb.pdf.canvas.color.color import HexColor
from borb.pdf.canvas.layout.hyphenation.hyphenation import Hyphenation
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.page_layout.page_layout import PageLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase

class TestAddParagraphUsingHyphenation(TestCase):

    def test_add_paragraph_using_hyphenation_001(self):
        if False:
            return 10
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header(test_description='This test creates a PDF with a Paragraph of text in it, with and without hyphenation.'))
        hyph: Hyphenation = Hyphenation('en-gb')
        page_layout.add(Paragraph('Without hyphenation', font_size=Decimal(20), font_color=HexColor('f1cd2e')))
        page_layout.add(Paragraph("\n        Lorem Ipsum is simply dummy text of the printing and typesetting industry. \n        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, \n        when an unknown printer took a galley of type and scrambled it to make a type specimen book. \n        It has survived not only five centuries, but also the leap into electronic typesetting, \n        remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, \n        and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n        "))
        page_layout.add(Paragraph('With hyphenation', font_size=Decimal(20), font_color=HexColor('f1cd2e')))
        page_layout.add(Paragraph("\n        Lorem Ipsum is simply dummy text of the printing and typesetting industry. \n        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, \n        when an unknown printer took a galley of type and scrambled it to make a type specimen book. \n        It has survived not only five centuries, but also the leap into electronic typesetting, \n        remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, \n        and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n        ", hyphenation=hyph))
        with open(self.get_first_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_paragraph_using_hyphenation_002(self):
        if False:
            return 10
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header(test_description='This test creates a PDF with a Paragraph of text in it, with and without hyphenation.'))
        hyph: Hyphenation = Hyphenation('en-gb')
        page_layout.add(Paragraph('Without hyphenation', font_size=Decimal(20), font_color=HexColor('f1cd2e')))
        page_layout.add(Paragraph('\n                Still others clutched their children closely to their breasts. One girl stood alone, slightly apart from the rest.\n                She was quite young, not more than eighteen.\n                '))
        page_layout.add(Paragraph('With hyphenation', font_size=Decimal(20), font_color=HexColor('f1cd2e')))
        page_layout.add(Paragraph('\n                Still others clutched their children closely to their breasts. One girl stood alone, slightly apart from the rest.\n                She was quite young, not more than eighteen.\n                ', hyphenation=hyph))
        with open(self.get_second_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_add_paragraph_using_hyphenation_003(self):
        if False:
            i = 10
            return i + 15
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header(test_description='This test creates a PDF with a Paragraph of text in it, with hyphenation.'))
        page_layout.add(Paragraph('alignment', font_size=Decimal(120), hyphenation=Hyphenation('en-gb')))
        with open(self.get_third_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())
if __name__ == '__main__':
    unittest.main()