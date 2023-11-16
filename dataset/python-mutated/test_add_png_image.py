import io
from _decimal import Decimal
from pathlib import Path
import PIL.Image
from borb.pdf import Paragraph, Alignment
from borb.pdf import Document
from borb.pdf import Image
from borb.pdf import PDF
from borb.pdf import Page
from borb.pdf import PageLayout
from borb.pdf import SingleColumnLayout
from borb.pdf.canvas.geometry.rectangle import Rectangle
from tests.test_case import TestCase

class TestAddImage(TestCase):

    def test_add_image_001(self):
        if False:
            for i in range(10):
                print('nop')

        def always_no(a, b) -> bool:
            if False:
                i = 10
                return i + 15
            return False
        from borb.io.write.image.rgba_image_transformer import RGBAImageTransformer
        prev_can_be_transformed = RGBAImageTransformer.can_be_transformed
        RGBAImageTransformer.can_be_transformed = always_no
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header('This test creates a PDF with an RGBA Image in it.This test first disables RGBAImageTransformer.'))
        N: int = 128
        raw_img: PIL.Image = PIL.Image.new(mode='RGBA', size=(N, N))
        for i in range(0, N):
            for j in range(0, N):
                d: float = ((i - N / 2) ** 2 + (j - N / 2) ** 2) ** 0.5
                if d < 12:
                    raw_img.putpixel((i, j), (255, 0, 0, 255))
                else:
                    raw_img.putpixel((i, j), (0, 0, 0, 0))
        obj_img: Image = Image(raw_img, border_top=True, border_right=True, border_bottom=True, border_left=True)
        page_layout.add(obj_img)
        text_under: str = ''.join(['1 ' for _ in range(0, 256)])[:-1]
        Paragraph(text_under, font_size=Decimal(7.925), horizontal_alignment=Alignment.CENTERED).paint(page, obj_img.get_previous_paint_box())
        with open(self.get_first_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        RGBAImageTransformer.can_be_transformed = prev_can_be_transformed
        self.compare_visually_to_ground_truth(self.get_first_output_file())
        self.check_pdf_using_validator(self.get_first_output_file())

    def test_add_image_002(self):
        if False:
            return 10
        pdf: Document = Document()
        page: Page = Page()
        pdf.add_page(page)
        page_layout: PageLayout = SingleColumnLayout(page)
        page_layout.add(self.get_test_header('This test creates a PDF with an RGBA Image in it.'))
        N: int = 128
        raw_img: PIL.Image = PIL.Image.new(mode='RGBA', size=(N, N))
        for i in range(0, N):
            for j in range(0, N):
                d: float = ((i - N / 2) ** 2 + (j - N / 2) ** 2) ** 0.5
                if d < 12:
                    raw_img.putpixel((i, j), (255, 0, 0, 255))
                else:
                    raw_img.putpixel((i, j), (0, 0, 0, 0))
        obj_img: Image = Image(raw_img, border_top=True, border_right=True, border_bottom=True, border_left=True)
        page_layout.add(obj_img)
        text_under: str = ''.join(['1 ' for _ in range(0, 256)])[:-1]
        Paragraph(text_under, font_size=Decimal(7.925), horizontal_alignment=Alignment.CENTERED).paint(page, obj_img.get_previous_paint_box())
        with open(self.get_second_output_file(), 'wb') as pdf_file_handle:
            PDF.dumps(pdf_file_handle, pdf)
        self.compare_visually_to_ground_truth(self.get_second_output_file())
        self.check_pdf_using_validator(self.get_second_output_file())