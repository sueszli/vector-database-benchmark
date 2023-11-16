"""
    This implementation of EventListener renders a PDF to a PIL Image
"""
import io
import platform
import typing
from decimal import Decimal
from pathlib import Path
from PIL import Image as PILImage
from PIL import ImageDraw
from PIL import ImageFont
from borb.pdf.canvas.canvas import Canvas
from borb.pdf.canvas.canvas_stream_processor import CanvasStreamProcessor
from borb.pdf.canvas.color.color import Color
from borb.pdf.canvas.event.begin_page_event import BeginPageEvent
from borb.pdf.canvas.event.end_page_event import EndPageEvent
from borb.pdf.page.page import Page
from borb.pdf.page.page_size import PageSize
from borb.toolkit.export.pdf_to_svg import PDFToSVG

class PDFToJPG(PDFToSVG):
    """
    This implementation of EventListener renders a PDF to a PIL Image
    """

    def __init__(self, default_page_width: Decimal=Decimal(PageSize.A4_PORTRAIT.value[0]), default_page_height: Decimal=Decimal(PageSize.A4_PORTRAIT.value[1])):
        if False:
            i = 10
            return i + 15
        super(PDFToJPG, self).__init__(default_page_width=default_page_width, default_page_height=default_page_height)
        self._jpg_image_per_page: typing.Dict[int, PILImage] = {}
        self._regular_font: typing.Optional[Path] = None
        self._bold_font: typing.Optional[Path] = None
        self._italic_font: typing.Optional[Path] = None
        self._bold_italic_font: typing.Optional[Path] = None
        self._find_font_families()

    def _begin_page(self, page_nr: Decimal, page_width: Decimal, page_height: Decimal) -> None:
        if False:
            return 10
        self._jpg_image_per_page[int(page_nr)] = PILImage.new('RGB', (int(page_width), int(page_height)), color=(255, 255, 255))

    def _find_font_families(self):
        if False:
            return 10
        system: str = platform.system()
        assert system in ['Darwin', 'Linux', 'Windows']
        root_font_dir: typing.Optional[Path] = None
        if system == 'Linux':
            root_font_dir = Path('/usr/share/fonts')
        if system == 'Darwin':
            root_font_dir = Path('/Library/Fonts/')
        if system == 'Windows':
            root_font_dir = Path('C:/Windows/Fonts')
        ttf_font_files = []
        file_stk: typing.List[Path] = [root_font_dir]
        while len(file_stk) > 0:
            f = file_stk[0]
            file_stk.pop(0)
            if f.is_dir():
                for subdir in f.iterdir():
                    file_stk.append(subdir)
            elif f.name.endswith('.ttf'):
                ttf_font_files.append(f)
        for c in ['LiberationSans', 'LiberationMono', 'arial']:
            suffixes: typing.List[str] = ['-Regular', '-Italic', '-Bold', '-BoldItalic']
            all_fonts_present = all([y in [x.name for x in ttf_font_files] for y in [c + x + '.ttf' for x in suffixes]])
            if all_fonts_present:
                self._regular_font = [x for x in ttf_font_files if x.name.endswith(c + '-Regular.ttf')][0]
                self._bold_font = [x for x in ttf_font_files if x.name.endswith(c + '-Bold.ttf')][0]
                self._italic_font = [x for x in ttf_font_files if x.name.endswith(c + '-Italic.ttf')][0]
                self._bold_italic_font = [x for x in ttf_font_files if x.name.endswith(c + '-BoldItalic.ttf')][0]
            suffixes = ['', 'i', 'bd', 'bi']
            all_fonts_present = all([y in [x.name for x in ttf_font_files] for y in [c + x + '.ttf' for x in suffixes]])
            if all_fonts_present:
                self._regular_font = [x for x in ttf_font_files if x.name.endswith(c + '.ttf')][0]
                self._bold_font = [x for x in ttf_font_files if x.name.endswith(c + 'bd.ttf')][0]
                self._italic_font = [x for x in ttf_font_files if x.name.endswith(c + 'i.ttf')][0]
                self._bold_italic_font = [x for x in ttf_font_files if x.name.endswith(c + 'bi.ttf')][0]

    def _render_image(self, page_nr: Decimal, page_width: Decimal, page_height: Decimal, x: Decimal, y: Decimal, image_width: Decimal, image_height: Decimal, image: PILImage):
        if False:
            for i in range(10):
                print('nop')
        page_image = self._jpg_image_per_page.get(int(page_nr))
        assert page_image is not None
        image = image.resize((int(image_width), int(image_height)))
        page_image.paste(image, (int(x), int(page_height - y - image_height)))

    def _render_text(self, page_nr: Decimal, page_width: Decimal, page_height: Decimal, x: Decimal, y: Decimal, font_color: Color, font_size: Decimal, font_name: str, bold: bool, italic: bool, text: str):
        if False:
            i = 10
            return i + 15
        if len(text.strip()) == 0:
            return
        assert self._bold_font
        assert self._bold_italic_font
        assert self._italic_font
        assert self._regular_font
        font_path = self._regular_font
        if bold and italic:
            font_path = self._bold_italic_font
        elif bold:
            font_path = self._bold_font
        elif italic:
            font_path = self._italic_font
        font = ImageFont.truetype(str(font_path), int(font_size))
        assert self._jpg_image_per_page.get(int(page_nr)) is not None
        draw = ImageDraw.Draw(self._jpg_image_per_page[int(page_nr)])
        draw.text((float(x), float(page_height - y)), text, font=font, fill=(int(font_color.to_rgb().red), int(font_color.to_rgb().green), int(font_color.to_rgb().blue)))

    @staticmethod
    def convert_pdf_to_jpg(pdf: 'Document') -> typing.Dict[int, PILImage.Image]:
        if False:
            while True:
                i = 10
        '\n        This function converts a PDF to an PIL.Image.Image\n        '
        image_of_each_page: typing.Dict[int, PILImage.Image] = {}
        number_of_pages: int = int(pdf.get_document_info().get_number_of_pages() or 0)
        for page_nr in range(0, number_of_pages):
            page: Page = pdf.get_page(page_nr)
            page_source: io.BytesIO = io.BytesIO(page['Contents']['DecodedBytes'])
            cse: 'PDFToJPG' = PDFToJPG()
            cse._event_occurred(BeginPageEvent(page))
            CanvasStreamProcessor(page, Canvas(), []).read(page_source, [cse])
            cse._event_occurred(EndPageEvent(page))
            image_of_each_page[page_nr] = cse.convert_to_jpg()[0]
        return image_of_each_page

    def convert_to_jpg(self) -> typing.Dict[int, PILImage.Image]:
        if False:
            while True:
                i = 10
        '\n        This function returns the PIL.Image for a given page_nr\n        '
        return self._jpg_image_per_page