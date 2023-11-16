"""
Pixmap tests
* make pixmap of a page and assert bbox size
* make pixmap from a PDF xref and compare with extracted image
* pixmap from file and from binary image and compare
"""
import os
import tempfile

import fitz

scriptdir = os.path.abspath(os.path.dirname(__file__))
epub = os.path.join(scriptdir, "resources", "Bezier.epub")
pdf = os.path.join(scriptdir, "resources", "001003ED.pdf")
imgfile = os.path.join(scriptdir, "resources", "nur-ruhig.jpg")


def test_pagepixmap():
    # pixmap from an EPUB page
    doc = fitz.open(epub)
    page = doc[0]
    pix = page.get_pixmap()
    assert pix.irect == page.rect.irect
    pix = page.get_pixmap(alpha=True)
    assert pix.alpha
    assert pix.n == pix.colorspace.n + pix.alpha


def test_pdfpixmap():
    # pixmap from xref in a PDF
    doc = fitz.open(pdf)
    # take first image item of first page
    img = doc.get_page_images(0)[0]
    # make pixmap of it
    pix = fitz.Pixmap(doc, img[0])
    # assert pixmap properties
    assert pix.width == img[2]
    assert pix.height == img[3]
    # extract image and compare metadata
    extractimg = doc.extract_image(img[0])
    assert extractimg["width"] == pix.width
    assert extractimg["height"] == pix.height


def test_filepixmap():
    # pixmaps from file and from stream
    # should lead to same result
    pix1 = fitz.Pixmap(imgfile)
    stream = open(imgfile, "rb").read()
    pix2 = fitz.Pixmap(stream)
    assert repr(pix1) == repr(pix2)
    assert pix1.digest == pix2.digest


def test_pilsave():
    # pixmaps from file then save to pillow image
    # make pixmap from this and confirm equality
    pix1 = fitz.Pixmap(imgfile)
    try:
        stream = pix1.pil_tobytes("JPEG")
        pix2 = fitz.Pixmap(stream)
        assert repr(pix1) == repr(pix2)
    except:
        pass


def test_save(tmpdir):
    # pixmaps from file then save to image
    # make pixmap from this and confirm equality
    pix1 = fitz.Pixmap(imgfile)
    outfile = os.path.join(tmpdir, "foo.png")
    pix1.save(outfile, output="png")
    # read it back
    pix2 = fitz.Pixmap(outfile)
    assert repr(pix1) == repr(pix2)


def test_setalpha():
    # pixmap from JPEG file, then add an alpha channel
    # with 30% transparency
    pix1 = fitz.Pixmap(imgfile)
    opa = int(255 * 0.3)  # corresponding to 30% transparency
    alphas = [opa] * (pix1.width * pix1.height)
    alphas = bytearray(alphas)
    pix2 = fitz.Pixmap(pix1, 1)  # add alpha channel
    pix2.set_alpha(alphas)  # make image 30% transparent
    samples = pix2.samples  # copy of samples
    # confirm correct the alpha bytes
    t = bytearray([samples[i] for i in range(3, len(samples), 4)])
    assert t == alphas

def test_color_count():
    pm = fitz.Pixmap(imgfile)
    assert pm.color_count() == 40624

def test_memoryview():
    pm = fitz.Pixmap(imgfile)
    samples = pm.samples_mv
    assert isinstance( samples, memoryview)
    print( f'samples={samples} samples.itemsize={samples.itemsize} samples.nbytes={samples.nbytes} samples.ndim={samples.ndim} samples.shape={samples.shape} samples.strides={samples.strides}')
    assert samples.itemsize == 1
    assert samples.nbytes == 659817
    assert samples.ndim == 1
    assert samples.shape == (659817,)
    assert samples.strides == (1,)
    
    color = pm.pixel( 100, 100)
    print( f'color={color}')
    assert color == (83, 66, 40)

def test_samples_ptr():
    pm = fitz.Pixmap(imgfile)
    samples = pm.samples_ptr
    print( f'samples={samples}')
    assert isinstance( samples, int)

def test_2369():

    width, height = 13, 37
    image = fitz.Pixmap(fitz.csGRAY, width, height, b"\x00" * (width * height), False)

    with fitz.Document(stream=image.tobytes(output="pam"), filetype="pam") as doc:
        test_pdf_bytes = doc.convert_to_pdf()
    
    with fitz.Document(stream=test_pdf_bytes) as doc:
        page = doc[0]
        img_xref = page.get_images()[0][0]
        img = doc.extract_image(img_xref)
        img_bytes = img["image"]
        fitz.Pixmap(img_bytes)

