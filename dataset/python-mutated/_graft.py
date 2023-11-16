"""For grafting text-only PDF pages onto freeform PDF pages."""
from __future__ import annotations
import logging
from contextlib import suppress
from enum import Enum
from pathlib import Path
from pikepdf import Dictionary, Name, Operator, Page, Pdf, PdfError, PdfMatrix, Stream, parse_content_stream, unparse_content_stream
from ocrmypdf._jobcontext import PdfContext

class RenderMode(Enum):
    ON_TOP = 0
    UNDERNEATH = 1
log = logging.getLogger(__name__)
MAX_REPLACE_PAGES = 100

def _ensure_dictionary(obj: Dictionary | Stream, name: Name):
    if False:
        return 10
    if name not in obj:
        obj[name] = Dictionary({})
    return obj[name]

def _update_resources(*, obj: Dictionary | Stream, font: Dictionary | None, font_key: Name | None):
    if False:
        print('Hello World!')
    "Update this obj's fonts with a reference to the Glyphless font.\n\n    obj can be a page or Form XObject.\n    "
    resources = _ensure_dictionary(obj, Name.Resources)
    fonts = _ensure_dictionary(resources, Name.Font)
    if font_key is not None and font_key not in fonts:
        fonts[font_key] = font

def strip_invisible_text(pdf: Pdf, page: Page):
    if False:
        return 10
    stream = []
    in_text_obj = False
    render_mode = 0
    text_objects = []
    for (operands, operator) in parse_content_stream(page, ''):
        if not in_text_obj:
            if operator == Operator('BT'):
                in_text_obj = True
                render_mode = 0
                text_objects.append((operands, operator))
            else:
                stream.append((operands, operator))
        else:
            if operator == Operator('Tr'):
                render_mode = operands[0]
            text_objects.append((operands, operator))
            if operator == Operator('ET'):
                in_text_obj = False
                if render_mode != 3:
                    stream.extend(text_objects)
                text_objects.clear()
    content_stream = unparse_content_stream(stream)
    page.Contents = Stream(pdf, content_stream)

class OcrGrafter:
    """Manages grafting text-only PDFs onto regular PDFs."""

    def __init__(self, context: PdfContext):
        if False:
            return 10
        self.context = context
        self.path_base = context.origin
        self.pdf_base = Pdf.open(self.path_base)
        self.font: Dictionary | None = None
        self.font_key: Name | None = None
        self.pdfinfo = context.pdfinfo
        self.output_file = context.get_path('graft_layers.pdf')
        self.emplacements = 1
        self.interim_count = 0
        self.render_mode = RenderMode.UNDERNEATH

    def graft_page(self, *, pageno: int, image: Path | None, textpdf: Path | None, autorotate_correction: int):
        if False:
            print('Hello World!')
        if textpdf and (not self.font):
            (self.font, self.font_key) = self._find_font(textpdf)
        emplaced_page = False
        content_rotation = self.pdfinfo[pageno].rotation
        path_image = Path(image).resolve() if image else None
        if path_image is not None and path_image != self.path_base:
            log.debug('Emplacement update')
            with Pdf.open(path_image) as pdf_image:
                self.emplacements += 1
                foreign_image_page = pdf_image.pages[0]
                self.pdf_base.pages.append(foreign_image_page)
                local_image_page = self.pdf_base.pages[-1]
                self.pdf_base.pages[pageno].emplace(local_image_page, retain=(Name.Parent,))
                del self.pdf_base.pages[-1]
            emplaced_page = True
        if emplaced_page:
            content_rotation = autorotate_correction
        text_rotation = autorotate_correction
        text_misaligned = (text_rotation - content_rotation) % 360
        log.debug(f'Text rotation: (text, autorotate, content) -> text misalignment = ({text_rotation}, {autorotate_correction}, {content_rotation}) -> {text_misaligned}')
        if textpdf and self.font:
            if self.font_key is None:
                raise ValueError('Font key is not set')
            strip_old = self.context.options.redo_ocr
            self._graft_text_layer(page_num=pageno + 1, textpdf=textpdf, font=self.font, font_key=self.font_key, text_rotation=text_misaligned, strip_old_text=strip_old)
        page_rotation = (content_rotation - autorotate_correction) % 360
        self.pdf_base.pages[pageno].Rotate = page_rotation
        log.debug(f'Page rotation: (content, auto) -> page = ({content_rotation}, {autorotate_correction}) -> {page_rotation}')
        if self.emplacements % MAX_REPLACE_PAGES == 0:
            self.save_and_reload()

    def save_and_reload(self) -> None:
        if False:
            return 10
        "Save and reload the Pdf.\n\n        This will keep a lid on our memory usage for very large files. Attach\n        the font to page 1 even if page 1 doesn't use it, so we have a way to get it\n        back.\n        "
        page0 = self.pdf_base.pages[0]
        _update_resources(obj=page0.obj, font=self.font, font_key=self.font_key)
        old_file = self.output_file.with_suffix(f'.working{self.interim_count - 1}.pdf')
        if not self.context.options.keep_temporary_files:
            with suppress(FileNotFoundError):
                old_file.unlink()
        next_file = self.output_file.with_suffix(f'.working{self.interim_count + 1}.pdf')
        self.pdf_base.save(next_file)
        self.pdf_base.close()
        self.pdf_base = Pdf.open(next_file)
        (self.font, self.font_key) = (None, None)
        self.interim_count += 1

    def finalize(self):
        if False:
            print('Hello World!')
        self.pdf_base.save(self.output_file)
        self.pdf_base.close()
        return self.output_file

    def _find_font(self, text: Path) -> tuple[Dictionary | None, Name | None]:
        if False:
            for i in range(10):
                print('nop')
        'Copy a font from the filename text into pdf_base.'
        (font, font_key) = (None, None)
        possible_font_names = ('/f-0-0', '/F1')
        try:
            with Pdf.open(text) as pdf_text:
                try:
                    pdf_text_fonts = pdf_text.pages[0].Resources.get(Name.Font, Dictionary())
                except (AttributeError, IndexError, KeyError):
                    return (None, None)
                if not isinstance(pdf_text_fonts, Dictionary):
                    log.warning('Page fonts are not stored in a dictionary')
                    return (None, None)
                pdf_text_font = None
                for f in possible_font_names:
                    pdf_text_font = pdf_text_fonts.get(f, None)
                    if pdf_text_font is not None:
                        font_key = Name(f)
                        break
                if pdf_text_font:
                    font = self.pdf_base.copy_foreign(pdf_text_font)
                if not isinstance(font, Dictionary):
                    log.warning('Font is not a dictionary')
                    (font, font_key) = (None, None)
                return (font, font_key)
        except (FileNotFoundError, PdfError):
            return (None, None)

    def _graft_text_layer(self, *, page_num: int, textpdf: Path, font: Dictionary, font_key: Name, text_rotation: int, strip_old_text: bool):
        if False:
            while True:
                i = 10
        'Insert the text layer from text page 0 on to pdf_base at page_num.'
        log.debug('Grafting')
        if Path(textpdf).stat().st_size == 0:
            return
        with Pdf.open(textpdf) as pdf_text:
            pdf_text_contents = pdf_text.pages[0].Contents.read_bytes()
            base_page = self.pdf_base.pages.p(page_num)
            mediabox = pdf_text.pages[0].mediabox
            (wt, ht) = (mediabox[2] - mediabox[0], mediabox[3] - mediabox[1])
            mediabox = base_page.mediabox
            (wp, hp) = (mediabox[2] - mediabox[0], mediabox[3] - mediabox[1])
            translate = PdfMatrix().translated(-wt / 2, -ht / 2)
            untranslate = PdfMatrix().translated(wp / 2, hp / 2)
            corner = PdfMatrix().translated(mediabox[0], mediabox[1])
            text_rotation = -text_rotation % 360
            rotate = PdfMatrix().rotated(text_rotation)
            if text_rotation in (90, 270):
                (wt, ht) = (ht, wt)
            scale_x = wp / wt
            scale_y = hp / ht
            scale = PdfMatrix().scaled(scale_x, scale_y)
            ctm = translate @ rotate @ scale @ untranslate @ corner
            base_resources = _ensure_dictionary(base_page.obj, Name.Resources)
            base_xobjs = _ensure_dictionary(base_resources, Name.XObject)
            text_xobj_name = Name.random(prefix='OCR-')
            xobj = self.pdf_base.make_stream(pdf_text_contents)
            base_xobjs[text_xobj_name] = xobj
            xobj.Type = Name.XObject
            xobj.Subtype = Name.Form
            xobj.FormType = 1
            xobj.BBox = mediabox
            _update_resources(obj=xobj, font=font, font_key=font_key)
            pdf_draw_xobj = b'q %s cm\n' % ctm.encode() + b'%s Do\n' % text_xobj_name + b'\nQ\n'
            new_text_layer = Stream(self.pdf_base, pdf_draw_xobj)
            if strip_old_text:
                strip_invisible_text(self.pdf_base, base_page)
            base_page.contents_add(new_text_layer, prepend=self.render_mode == RenderMode.UNDERNEATH)
            _update_resources(obj=base_page.obj, font=font, font_key=font_key)