import unittest
from borb.pdf import Document
from borb.pdf import HexColor
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import Paragraph
from borb.pdf import SingleColumnLayout
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestWriteHelloWorldWithEasierImports(TestCase):

    def test_write_hello_world_with_easier_imports(self):
        if False:
            i = 10
            return i + 15
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(Paragraph('Hello World!', font_color=HexColor('56cbf9')))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.check_pdf_using_validator(self.get_first_output_file())