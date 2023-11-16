import unittest
from borb.pdf import Alignment
from borb.pdf import FlexibleColumnWidthTable
from borb.pdf import RomanNumeralOrderedList
from borb.pdf import UnorderedList
from borb.pdf.canvas.layout.list.ordered_list import OrderedList
from borb.pdf.canvas.layout.page_layout.multi_column_layout import SingleColumnLayout
from borb.pdf.canvas.layout.text.paragraph import Paragraph
from borb.pdf.document.document import Document
from borb.pdf.page.page import Page
from borb.pdf.pdf import PDF
from tests.test_case import TestCase

class TestAddRomanNumeralOrderedList(TestCase):
    """
    This test creates a PDF with an ordered list in it.
    """

    def test_add_romannumeralorderedlist(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds an RomanNumeralOrderedList to a PDF.'))
        layout.add(RomanNumeralOrderedList().add(Paragraph(text='Lorem Ipsum Dolor Sit Amet Consectetur Nunc')).add(Paragraph(text='Ipsum')).add(Paragraph(text='Dolor')).add(Paragraph(text='Sit')).add(Paragraph(text='Amet')))
        with open(self.get_first_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_orderedlist_of_romannumeralorderedlists(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds an OrderedList of RomanNumeralOrderedList to a PDF.'))
        layout.add(OrderedList().add(RomanNumeralOrderedList().add(Paragraph(text='Lorem Ipsum Dolor Sit Amet Consectetur Nunc')).add(Paragraph(text='Ipsum'))).add(RomanNumeralOrderedList().add(Paragraph(text='Dolor')).add(Paragraph(text='Sit')).add(Paragraph(text='Amet'))))
        with open(self.get_second_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_add_unorderedlist_of_romannumeralorderedlists(self):
        if False:
            while True:
                i = 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds an UnorderedList of RomanNumeralOrderedLists to a PDF.'))
        layout.add(UnorderedList().add(RomanNumeralOrderedList().add(Paragraph(text='Lorem Ipsum Dolor Sit Amet Consectetur Nunc')).add(Paragraph(text='Ipsum'))).add(RomanNumeralOrderedList().add(Paragraph(text='Dolor')).add(Paragraph(text='Sit')).add(Paragraph(text='Amet'))))
        with open(self.get_third_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())

    def test_add_table_of_romannumeralorderedlists(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Table of RomanNumeralOrderedList to a PDF.'))
        layout.add(FlexibleColumnWidthTable(number_of_columns=2, number_of_rows=2).add(RomanNumeralOrderedList().add(Paragraph(text='Lorem Ipsum')).add(Paragraph(text='Ipsum'))).add(RomanNumeralOrderedList().add(Paragraph(text='Dolor')).add(Paragraph(text='Sit')).add(Paragraph(text='Amet'))).add(RomanNumeralOrderedList().add(Paragraph(text='Lorem Ipsum')).add(Paragraph(text='Ipsum'))).add(RomanNumeralOrderedList().add(Paragraph(text='Dolor')).add(Paragraph(text='Sit')).add(Paragraph(text='Amet'))))
        with open(self.get_fourth_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_fourth_output_file())
        self.check_pdf_using_validator(self.get_fourth_output_file())

    def test_add_romannumeralorderedlist_using_horizontal_alignment_left(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds an RomanNumeralOrderedList to a PDF, the list has horizontal alignment LEFT.'))
        layout.add(RomanNumeralOrderedList(horizontal_alignment=Alignment.LEFT).add(Paragraph(text='Lorem Ipsum Dolor Sit Amet Consectetur Nunc')).add(Paragraph(text='Ipsum')).add(Paragraph(text='Dolor')).add(Paragraph(text='Sit')).add(Paragraph(text='Amet')))
        with open(self.get_fifth_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_fifth_output_file())
        self.check_pdf_using_validator(self.get_fifth_output_file())

    def test_add_romannumeralorderedlist_using_horizontal_alignment_centered(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds an RomanNumeralOrderedList to a PDF, the list has horizontal alignment CENTERED.'))
        layout.add(RomanNumeralOrderedList(horizontal_alignment=Alignment.CENTERED).add(Paragraph(text='Lorem Ipsum Dolor Sit Amet Consectetur Nunc')).add(Paragraph(text='Ipsum')).add(Paragraph(text='Dolor')).add(Paragraph(text='Sit')).add(Paragraph(text='Amet')))
        with open(self.get_sixth_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_sixth_output_file())
        self.check_pdf_using_validator(self.get_sixth_output_file())

    def test_add_romannumeralorderedlist_using_horizontal_alignment_right(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds an RomanNumeralOrderedList to a PDF, the list has horizontal alignment RIGHT.'))
        layout.add(RomanNumeralOrderedList(horizontal_alignment=Alignment.RIGHT).add(Paragraph(text='Lorem Ipsum Dolor Sit Amet Consectetur Nunc')).add(Paragraph(text='Ipsum')).add(Paragraph(text='Dolor')).add(Paragraph(text='Sit')).add(Paragraph(text='Amet')))
        with open(self.get_seventh_output_file(), 'wb') as in_file_handle:
            PDF.dumps(in_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_seventh_output_file())
        self.check_pdf_using_validator(self.get_seventh_output_file())
if __name__ == '__main__':
    unittest.main()