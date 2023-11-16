from borb.pdf import ChunkOfText
from borb.pdf import Document
from borb.pdf import HeterogeneousParagraph
from borb.pdf import HexColor
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import SingleColumnLayout
from borb.pdf.canvas.layout.emoji.emoji import Emojis
from tests.test_case import TestCase

class TestAddHeterogeneousParagraphUsingFontColor(TestCase):

    def test_add_heterogeneousparagraph_using_font_color_001(self):
        if False:
            while True:
                i = 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a HeterogeneousParagraph to the PDF with font_color 12'))
        layout.add(HeterogeneousParagraph(chunks_of_text=[ChunkOfText('Lorem', font_color=HexColor('023047')), Emojis.OCTOCAT.value, ChunkOfText('Ipsum', font_color=HexColor('023047')), Emojis.SMILE.value, ChunkOfText('Dolor', font_color=HexColor('023047'))]))
        with open(self.get_first_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_heterogeneousparagraph_using_font_color_002(self):
        if False:
            i = 10
            return i + 15
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a HeterogeneousParagraph to the PDF with font_color 14'))
        layout.add(HeterogeneousParagraph(chunks_of_text=[ChunkOfText('Lorem', font_color=HexColor('FFB703')), Emojis.OCTOCAT.value, ChunkOfText('Ipsum', font_color=HexColor('FFB703')), Emojis.SMILE.value, ChunkOfText('Dolor', font_color=HexColor('FFB703'))]))
        with open(self.get_second_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_add_heterogeneousparagraph_using_font_color_003(self):
        if False:
            print('Hello World!')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a HeterogeneousParagraph to the PDF with font_color 16'))
        layout.add(HeterogeneousParagraph(chunks_of_text=[ChunkOfText('Lorem', font_color=HexColor('FB8500')), Emojis.OCTOCAT.value, ChunkOfText('Ipsum', font_color=HexColor('FB8500')), Emojis.SMILE.value, ChunkOfText('Dolor', font_color=HexColor('FB8500'))]))
        with open(self.get_third_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())

    def test_add_heterogeneousparagraph_using_font_color_004(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a HeterogeneousParagraph to the PDF with font_color 18'))
        layout.add(HeterogeneousParagraph(chunks_of_text=[ChunkOfText('Lorem', font_color=HexColor('219EBC')), Emojis.OCTOCAT.value, ChunkOfText('Ipsum', font_color=HexColor('219EBC')), Emojis.SMILE.value, ChunkOfText('Dolor', font_color=HexColor('219EBC'))]))
        with open(self.get_fourth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_fourth_output_file())
        self.check_pdf_using_validator(self.get_fourth_output_file())

    def test_add_heterogeneousparagraph_using_font_color_005(self):
        if False:
            while True:
                i = 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a HeterogeneousParagraph to the PDF with font_color 20'))
        layout.add(HeterogeneousParagraph(chunks_of_text=[ChunkOfText('Lorem', font_color=HexColor('8ECAE6')), Emojis.OCTOCAT.value, ChunkOfText('Ipsum', font_color=HexColor('8ECAE6')), Emojis.SMILE.value, ChunkOfText('Dolor', font_color=HexColor('8ECAE6'))]))
        with open(self.get_fifth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_fifth_output_file())
        self.check_pdf_using_validator(self.get_fifth_output_file())