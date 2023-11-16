import re
import typing
import unittest
from decimal import Decimal
from borb.pdf.canvas.layout.layout_element import Alignment
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from borb.toolkit.text.simple_text_extraction import SimpleTextExtraction
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestExtractText(TestCase):
    """
    This test attempts to extract the text of each PDF in the corpus
    """

    def test_create_dummy_pdf(self):
        if False:
            print('Hello World!')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test creates a PDF with an empty Page, and a Paragraph of text. A subsequent test will attempt to extract all the text from this PDF.'))
        layout.add(Paragraph('\n            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \n            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. \n            Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. \n            Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.\n            ', font_size=Decimal(10), vertical_alignment=Alignment.TOP, horizontal_alignment=Alignment.LEFT, padding_top=Decimal(5), padding_right=Decimal(5), padding_bottom=Decimal(5), padding_left=Decimal(5)))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_extract_text(self):
        if False:
            return 10
        doc: typing.Optional[None] = None
        l = SimpleTextExtraction()
        with open(self.get_first_output_file(), 'rb') as file_handle:
            doc = PDF.loads(file_handle, [l])
        assert doc is not None
        page_content: str = l.get_text()[0]
        ground_truth: str = '\n        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et\n        dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex\n        ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu\n        fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt\n        mollit anim id est laborum.         \n        '
        for w in re.split('[^a-zA-Z]+', ground_truth):
            assert w in page_content, "Word '%s' not found in extracted text" % w
if __name__ == '__main__':
    unittest.main()