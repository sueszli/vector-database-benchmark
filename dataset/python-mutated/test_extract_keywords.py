import unittest
from borb.pdf.canvas.layout.list.unordered_list import UnorderedList
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from borb.toolkit.text.stop_words import ENGLISH_STOP_WORDS
from borb.toolkit.text.text_rank_keyword_extraction import TextRankKeywordExtraction
from borb.toolkit.text.tf_idf_keyword_extraction import TFIDFKeywordExtraction
from tests.test_case import TestCase
unittest.TestLoader.sortTestMethodsUsing = None

class TestExtractKeywords(TestCase):
    """
    This test attempts to extract the keywords (TF-IDF)
    from each PDF in the corpus
    """

    def test_create_dummy_pdf(self):
        if False:
            i = 10
            return i + 15
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test creates a PDF with an empty Page, and a Paragraph of text. A subsequent test will attempt to extract the keywords from this text.'))
        layout.add(Paragraph("\n            Lorem Ipsum is simply dummy text of the printing and typesetting industry. \n            Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, \n            when an unknown printer took a galley of type and scrambled it to make a type specimen book. \n            It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. \n            It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, \n            and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\n            "))
        layout.add(Paragraph("\n            It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. \n            The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, \n            as opposed to using 'Content here, content here', making it look like readable English. \n            Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, \n            and a search for 'lorem ipsum' will uncover many web sites still in their infancy. \n            Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).\n            "))
        with open(self.get_first_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_extract_keywords_using_tf_idf_from_document(self):
        if False:
            return 10
        with open(self.get_first_output_file(), 'rb') as pdf_file_handle:
            l = TFIDFKeywordExtraction(ENGLISH_STOP_WORDS)
            doc = PDF.loads(pdf_file_handle, [l])
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test creates a PDF with an empty Page, and adds the keywords it found in the previously made PDF.'))
        layout.add(Paragraph('Following keywords were found:'))
        ul: UnorderedList = UnorderedList()
        for k in l.get_keywords()[0][:5]:
            ul.add(Paragraph(k[0]))
        layout.add(ul)
        with open(self.get_second_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_extract_keywords_using_textrank_from_document(self):
        if False:
            print('Hello World!')
        l = TextRankKeywordExtraction()
        with open(self.get_first_output_file(), 'rb') as pdf_file_handle:
            doc = PDF.loads(pdf_file_handle, [l])
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test creates a PDF with an empty Page, and adds the keywords it foundin the previously made PDF.'))
        layout.add(Paragraph('Following keywords were found:'))
        ul: UnorderedList = UnorderedList()
        for k in l.get_keywords()[0][:5]:
            ul.add(Paragraph(k[0]))
        layout.add(ul)
        with open(self.get_third_output_file(), 'wb') as out_file_handle:
            PDF.dumps(out_file_handle, pdf)
        self.check_pdf_using_validator(self.get_third_output_file())
if __name__ == '__main__':
    unittest.main()