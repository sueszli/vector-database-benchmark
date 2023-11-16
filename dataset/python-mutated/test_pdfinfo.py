from __future__ import annotations
import pickle
from io import BytesIO
from math import isclose
import img2pdf
import pikepdf
import pytest
from PIL import Image
from reportlab.lib.units import inch
from reportlab.pdfgen.canvas import Canvas
from ocrmypdf import pdfinfo
from ocrmypdf.exceptions import InputFileError
from ocrmypdf.helpers import IMG2PDF_KWARGS, Resolution
from ocrmypdf.pdfinfo import Colorspace, Encoding
from ocrmypdf.pdfinfo.layout import PDFPage

@pytest.fixture
def single_page_text(outdir):
    if False:
        while True:
            i = 10
    filename = outdir / 'text.pdf'
    pdf = Canvas(str(filename), pagesize=(8 * inch, 6 * inch))
    text = pdf.beginText()
    text.setFont('Helvetica', 12)
    text.setTextOrigin(1 * inch, 3 * inch)
    text.textLine("Methink'st thou art a general offence and every man should beat thee.")
    pdf.drawText(text)
    pdf.showPage()
    pdf.save()
    return filename

def test_single_page_text(single_page_text):
    if False:
        print('Hello World!')
    info = pdfinfo.PdfInfo(single_page_text)
    assert len(info) == 1
    page = info[0]
    assert page.has_text
    assert len(page.images) == 0

@pytest.fixture(scope='session')
def eight_by_eight():
    if False:
        for i in range(10):
            print('nop')
    im = Image.new('1', (8, 8), 0)
    for n in range(8):
        im.putpixel((n, n), 1)
    return im

@pytest.fixture
def eight_by_eight_regular_image(eight_by_eight, outpdf):
    if False:
        print('Hello World!')
    im = eight_by_eight
    bio = BytesIO()
    im.save(bio, format='PNG')
    bio.seek(0)
    imgsize = ((img2pdf.ImgSize.dpi, 8), (img2pdf.ImgSize.dpi, 8))
    layout_fun = img2pdf.get_layout_fun(None, imgsize, None, None, None)
    with outpdf.open('wb') as f:
        img2pdf.convert(bio, producer='img2pdf', layout_fun=layout_fun, outputstream=f, **IMG2PDF_KWARGS)
    return outpdf

def test_single_page_image(eight_by_eight_regular_image):
    if False:
        print('Hello World!')
    info = pdfinfo.PdfInfo(eight_by_eight_regular_image)
    assert len(info) == 1
    page = info[0]
    assert not page.has_text
    assert len(page.images) == 1
    pdfimage = page.images[0]
    assert pdfimage.width == 8
    assert pdfimage.color == Colorspace.gray
    assert isclose(pdfimage.dpi.x, 8)
    assert isclose(pdfimage.dpi.y, 8)

@pytest.fixture
def eight_by_eight_inline_image(eight_by_eight, outpdf):
    if False:
        return 10
    pdf = Canvas(str(outpdf), pagesize=(8 * 72, 6 * 72))
    pdf.drawInlineImage(eight_by_eight, 0, 0, width=72, height=72)
    pdf.showPage()
    pdf.save()
    return outpdf

def test_single_page_inline_image(eight_by_eight_inline_image):
    if False:
        for i in range(10):
            print('nop')
    info = pdfinfo.PdfInfo(eight_by_eight_inline_image)
    print(info)
    pdfimage = info[0].images[0]
    assert isclose(pdfimage.dpi.x, 8)
    assert pdfimage.color == Colorspace.gray
    assert pdfimage.width == 8

def test_jpeg(resources):
    if False:
        while True:
            i = 10
    filename = resources / 'c02-22.pdf'
    pdf = pdfinfo.PdfInfo(filename)
    pdfimage = pdf[0].images[0]
    assert pdfimage.enc == Encoding.jpeg
    assert isclose(pdfimage.dpi.x, 150)

def test_form_xobject(resources):
    if False:
        while True:
            i = 10
    filename = resources / 'formxobject.pdf'
    pdf = pdfinfo.PdfInfo(filename)
    pdfimage = pdf[0].images[0]
    assert pdfimage.width == 50

def test_no_contents(resources):
    if False:
        print('Hello World!')
    filename = resources / 'no_contents.pdf'
    pdf = pdfinfo.PdfInfo(filename)
    assert len(pdf[0].images) == 0
    assert not pdf[0].has_text

def test_oversized_page(resources):
    if False:
        i = 10
        return i + 15
    pdf = pdfinfo.PdfInfo(resources / 'poster.pdf')
    image = pdf[0].images[0]
    assert image.width * image.dpi.x > 200, 'this is supposed to be oversized'

def test_pickle(resources):
    if False:
        i = 10
        return i + 15
    filename = resources / 'graph_ocred.pdf'
    pdf = pdfinfo.PdfInfo(filename)
    pickle.dumps(pdf)

def test_vector(resources):
    if False:
        print('Hello World!')
    filename = resources / 'vector.pdf'
    pdf = pdfinfo.PdfInfo(filename)
    assert pdf[0].has_vector
    assert not pdf[0].has_text

def test_ocr_detection(resources):
    if False:
        return 10
    filename = resources / 'graph_ocred.pdf'
    pdf = pdfinfo.PdfInfo(filename)
    assert not pdf[0].has_vector
    assert pdf[0].has_text

@pytest.mark.parametrize('testfile', ('truetype_font_nomapping.pdf', 'type3_font_nomapping.pdf'))
def test_corrupt_font_detection(resources, testfile):
    if False:
        print('Hello World!')
    filename = resources / testfile
    pdf = pdfinfo.PdfInfo(filename, detailed_analysis=True)
    assert pdf[0].has_corrupt_text

def test_stack_abuse():
    if False:
        i = 10
        return i + 15
    p = pikepdf.Pdf.new()
    stream = pikepdf.Stream(p, b'q ' * 35)
    with pytest.warns(UserWarning, match='overflowed'):
        pdfinfo.info._interpret_contents(stream)
    stream = pikepdf.Stream(p, b'q Q Q Q Q')
    with pytest.warns(UserWarning, match='underflowed'):
        pdfinfo.info._interpret_contents(stream)
    stream = pikepdf.Stream(p, b'q ' * 135)
    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError):
            pdfinfo.info._interpret_contents(stream)

def test_pages_issue700(monkeypatch, resources):
    if False:
        return 10

    def get_no_pages(*args, **kwargs):
        if False:
            return 10
        return iter([])
    monkeypatch.setattr(PDFPage, 'get_pages', get_no_pages)
    with pytest.raises(InputFileError, match='pdfminer'):
        pdfinfo.PdfInfo(resources / 'cardinal.pdf', detailed_analysis=True, progbar=False, max_workers=1)

@pytest.fixture
def image_scale0(resources, outpdf):
    if False:
        while True:
            i = 10
    with pikepdf.open(resources / 'cmyk.pdf') as cmyk:
        xobj = cmyk.pages[0].as_form_xobject()
        p = pikepdf.Pdf.new()
        p.add_blank_page(page_size=(72, 72))
        objname = p.pages[0].add_resource(p.copy_foreign(xobj), pikepdf.Name.XObject, pikepdf.Name.Im0)
        print(objname)
        p.pages[0].Contents = pikepdf.Stream(p, b'q 0 0 0 0 0 0 cm %s Do Q' % bytes(objname))
        p.save(outpdf)
    return outpdf

def test_image_scale0(image_scale0):
    if False:
        i = 10
        return i + 15
    pi = pdfinfo.PdfInfo(image_scale0, detailed_analysis=True, progbar=False, max_workers=1)
    assert not pi.pages[0]._images[0].dpi.is_finite
    assert pi.pages[0].dpi == Resolution(0, 0)