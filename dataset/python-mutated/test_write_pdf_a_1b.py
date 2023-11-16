from borb.io.read.types import Dictionary
from borb.io.read.types import Name
from borb.io.read.types import String
from borb.io.write.conformance_level import ConformanceLevel
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.page_layout.page_layout import PageLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase

class TestWritePDFA1B(TestCase):
    """
    This test creates a PDF with a few PDF graphics in it
    """

    def test_write_pdf_a_1b(self):
        if False:
            i = 10
            return i + 15
        pdf = Document(ConformanceLevel.PDFA_1A)
        page = Page()
        pdf.add_page(page)
        layout: PageLayout = SingleColumnLayout(page)
        layout.add(Paragraph('Hello World!'))
        layout.add(Paragraph('Hello World!'))
        info_dictionary: Dictionary = Dictionary()
        info_dictionary[Name('Title')] = String('Title Value')
        info_dictionary[Name('Subject')] = String('Subject Value')
        info_dictionary[Name('Creator')] = String('Creator Value')
        info_dictionary[Name('Author')] = String('Author Value')
        info_dictionary[Name('Keywords')] = String('Keyword1 Keyword2 Keyword3')
        pdf['XRef']['Trailer'][Name('Info')] = info_dictionary
        with open(self.get_first_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_re_open_pdfa_1_b(self):
        if False:
            for i in range(10):
                print('nop')
        with open(self.get_first_output_file(), 'rb') as in_file_handle:
            pdf = PDF.loads(in_file_handle)
        xmp = pdf.get_xmp_document_info()
        assert xmp.get_title() == 'Title Value'
        assert xmp.get_creator() == 'Creator Value'
        assert xmp.get_author() == 'Author Value'
        assert xmp.get_subject() == 'Subject Value'
        assert xmp.get_keywords() == 'Keyword1 Keyword2 Keyword3'

    def test_re_save_pdf_a_1_b(self):
        if False:
            print('Hello World!')
        with open(self.get_first_output_file(), 'rb') as in_file_handle:
            pdf = PDF.loads(in_file_handle)
        with open(self.get_second_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        with open(self.get_first_output_file(), 'rb') as in_file_handle:
            pdf = PDF.loads(in_file_handle)
        xmp = pdf.get_xmp_document_info()
        assert xmp.get_title() == 'Title Value'
        assert xmp.get_creator() == 'Creator Value'
        assert xmp.get_author() == 'Author Value'
        assert xmp.get_subject() == 'Subject Value'
        assert xmp.get_keywords() == 'Keyword1 Keyword2 Keyword3'
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())