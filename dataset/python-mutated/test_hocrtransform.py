from __future__ import annotations
import re
from io import StringIO
import pytest
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from PIL import Image
from ocrmypdf import hocrtransform
from ocrmypdf._exec.tesseract import generate_hocr
from ocrmypdf.helpers import check_pdf
from .conftest import check_ocrmypdf

def text_from_pdf(filename):
    if False:
        print('Hello World!')
    output_string = StringIO()
    with open(filename, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    return output_string.getvalue()

@pytest.fixture
def blank_hocr(tmp_path):
    if False:
        print('Hello World!')
    im = Image.new('1', (8, 8), 0)
    im.save(tmp_path / 'blank.tif', format='TIFF')
    generate_hocr(input_file=tmp_path / 'blank.tif', output_hocr=tmp_path / 'blank.hocr', output_text=tmp_path / 'blank.txt', languages=['eng'], engine_mode=1, tessconfig=[], pagesegmode=3, thresholding=0, user_words=None, user_patterns=None, timeout=None)
    return tmp_path / 'blank.hocr'

def test_mono_image(blank_hocr, outdir):
    if False:
        print('Hello World!')
    im = Image.new('1', (8, 8), 0)
    for n in range(8):
        im.putpixel((n, n), 1)
    im.save(outdir / 'mono.tif', format='TIFF')
    hocr = hocrtransform.HocrTransform(hocr_filename=str(blank_hocr), dpi=300)
    hocr.to_pdf(out_filename=str(outdir / 'mono.pdf'), image_filename=str(outdir / 'mono.tif'))
    check_pdf(str(outdir / 'mono.pdf'))

@pytest.mark.slow
def test_hocrtransform_matches_sandwich(resources, outdir):
    if False:
        while True:
            i = 10
    check_ocrmypdf(resources / 'ccitt.pdf', outdir / 'hocr.pdf', '--pdf-renderer=hocr')
    check_ocrmypdf(resources / 'ccitt.pdf', outdir / 'tess.pdf', '--pdf-renderer=sandwich')

    def clean(s):
        if False:
            print('Hello World!')
        s = re.sub('\\s+', ' ', s)
        words = s.split(' ')
        return '\n'.join(sorted(words))
    hocr_txt = clean(text_from_pdf(outdir / 'hocr.pdf'))
    tess_txt = clean(text_from_pdf(outdir / 'tess.pdf'))
    assert hocr_txt == tess_txt