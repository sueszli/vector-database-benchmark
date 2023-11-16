"""Tesseract no-op plugin.

To quickly run tests where getting OCR output is not necessary.

In 'hocr' mode, create a .hocr file that specifies no text found.

In 'pdf' mode, convert the image to PDF using another program.

In orientation check mode, report the orientation is upright.
"""
from __future__ import annotations
import pikepdf
from PIL import Image
from ocrmypdf import OcrEngine, OrientationConfidence, hookimpl
HOCR_TEMPLATE = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">\n <head>\n  <title></title>\n  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />\n  <meta name=\'ocr-system\' content=\'tesseract 4.1.1\' />\n  <meta name=\'ocr-capabilities\'\n    content=\'ocr_page ocr_carea ocr_par ocr_line ocrx_word\'/>\n </head>\n <body>\n  <div class=\'ocr_page\' id=\'page_1\' title=\'image "x.tif"; bbox 0 0 {0} {1}; ppageno 0\'>\n   <div class=\'ocr_carea\' id=\'block_1_1\' title="bbox 0 1 {0} {1}">\n    <p class=\'ocr_par\' dir=\'ltr\' id=\'par_1\' title="bbox 0 1 {0} {1}">\n     <span class=\'ocr_line\' id=\'line_1\' title="bbox 0 1 {0} {1}">\n       <span class=\'ocrx_word\' id=\'word_1\' title="bbox 0 1 {0} {1}"> </span>\n     </span>\n    </p>\n   </div>\n  </div>\n </body>\n</html>'

class NoopOcrEngine(OcrEngine):

    @staticmethod
    def version():
        if False:
            for i in range(10):
                print('nop')
        return '4.1.1'

    @staticmethod
    def creator_tag(options):
        if False:
            i = 10
            return i + 15
        tag = '-PDF' if options.pdf_renderer == 'sandwich' else ''
        return f'NO-OP {tag} {NoopOcrEngine.version()}'

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f'NO-OP {NoopOcrEngine.version()}'

    @staticmethod
    def languages(options):
        if False:
            i = 10
            return i + 15
        return {'eng'}

    @staticmethod
    def get_orientation(input_file, options):
        if False:
            while True:
                i = 10
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file, options):
        if False:
            i = 10
            return i + 15
        return 0.0

    @staticmethod
    def generate_hocr(input_file, output_hocr, output_text, options):
        if False:
            print('Hello World!')
        with Image.open(input_file) as im, open(output_hocr, 'w', encoding='utf-8') as f:
            (w, h) = im.size
            f.write(HOCR_TEMPLATE.format(str(w), str(h)))
        with open(output_text, 'w') as f:
            f.write('')

    @staticmethod
    def generate_pdf(input_file, output_pdf, output_text, options):
        if False:
            return 10
        with Image.open(input_file) as im:
            dpi = im.info['dpi']
            pagesize = (im.size[0] / dpi[0], im.size[1] / dpi[1])
        ptsize = (pagesize[0] * 72, pagesize[1] * 72)
        pdf = pikepdf.new()
        pdf.add_blank_page(page_size=ptsize)
        pdf.save(output_pdf, static_id=True)
        output_text.write_text('')

@hookimpl
def get_ocr_engine():
    if False:
        i = 10
        return i + 15
    return NoopOcrEngine()