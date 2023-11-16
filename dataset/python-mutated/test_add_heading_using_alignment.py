from borb.pdf import Alignment
from borb.pdf import Document
from borb.pdf import Heading
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import SingleColumnLayout
from tests.test_case import TestCase

class TestAddHeadingUsingAlignment(TestCase):

    def test_add_heading(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF.'))
        layout.add(Heading('Lorem Ipsum Dolor'))
        with open(self.get_first_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_heading_align_left_top(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment LEFT and vertical alignment TOP.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.LEFT, vertical_alignment=Alignment.TOP))
        with open(self.get_second_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())

    def test_add_heading_align_left_middle(self):
        if False:
            i = 10
            return i + 15
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment LEFT and vertical alignment MIDDLE.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.LEFT, vertical_alignment=Alignment.MIDDLE))
        with open(self.get_third_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_third_output_file())
        self.check_pdf_using_validator(self.get_third_output_file())

    def test_add_heading_align_left_bottom(self):
        if False:
            print('Hello World!')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment LEFT and vertical alignment BOTTOM.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.LEFT, vertical_alignment=Alignment.BOTTOM))
        with open(self.get_fourth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_fourth_output_file())
        self.check_pdf_using_validator(self.get_fourth_output_file())

    def test_add_heading_align_centered_top(self):
        if False:
            return 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment CENTERED and vertical alignment TOP.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.CENTERED, vertical_alignment=Alignment.TOP))
        with open(self.get_fifth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_fifth_output_file())
        self.check_pdf_using_validator(self.get_fifth_output_file())

    def test_add_heading_align_centered_middle(self):
        if False:
            print('Hello World!')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment CENTERED and vertical alignment MIDDLE.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.CENTERED, vertical_alignment=Alignment.MIDDLE))
        with open(self.get_sixth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_sixth_output_file())
        self.check_pdf_using_validator(self.get_sixth_output_file())

    def test_add_heading_align_centered_bottom(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment CENTERED and vertical alignment BOTTOM.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.CENTERED, vertical_alignment=Alignment.BOTTOM))
        with open(self.get_seventh_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_seventh_output_file())
        self.check_pdf_using_validator(self.get_seventh_output_file())

    def test_add_heading_align_right_top(self):
        if False:
            while True:
                i = 10
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment RIGHT and vertical alignment TOP.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.RIGHT, vertical_alignment=Alignment.TOP))
        with open(self.get_eight_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_eight_output_file())
        self.check_pdf_using_validator(self.get_eight_output_file())

    def test_add_heading_align_right_middle(self):
        if False:
            for i in range(10):
                print('nop')
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment RIGHT and vertical alignment MIDDLE.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.RIGHT, vertical_alignment=Alignment.MIDDLE))
        with open(self.get_nineth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_nineth_output_file())
        self.check_pdf_using_validator(self.get_nineth_output_file())

    def test_add_heading_align_right_bottom(self):
        if False:
            i = 10
            return i + 15
        pdf = Document()
        page = Page()
        pdf.add_page(page)
        layout = SingleColumnLayout(page)
        layout.add(self.get_test_header(test_description='This test adds a Heading to the PDF with horizontal alignment RIGHT and vertical alignment BOTTOM.'))
        layout.add(Heading('Lorem Ipsum Dolor', horizontal_alignment=Alignment.RIGHT, vertical_alignment=Alignment.BOTTOM))
        with open(self.get_tenth_output_file(), 'wb') as fh:
            PDF.dumps(fh, pdf)
        self.compare_visually_to_ground_truth(self.get_tenth_output_file())
        self.check_pdf_using_validator(self.get_tenth_output_file())