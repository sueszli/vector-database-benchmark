"""
Tests:
    * Convert some image to a PDF
    * Insert it rotated in some rectangle of a PDF page
    * Assert PDF Form XObject has been created
    * Assert that image contained in inserted PDF is inside given retangle
"""
import os
import fitz
scriptdir = os.path.abspath(os.path.dirname(__file__))
imgfile = os.path.join(scriptdir, 'resources', 'nur-ruhig.jpg')

def test_insert():
    if False:
        print('Hello World!')
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 100, 100)
    img = fitz.open(imgfile)
    tobytes = img.convert_to_pdf()
    src = fitz.open('pdf', tobytes)
    xref = page.show_pdf_page(rect, src, 0, rotate=-23)
    img = page.get_images(True)[0]
    assert img[-1] == xref
    img = page.get_image_info()[0]
    assert img['bbox'] in rect + (-1, -1, 1, 1)