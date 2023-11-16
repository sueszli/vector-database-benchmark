"""
Demo of Story class in PyMuPDF
-------------------------------

This script demonstrates how to the results of a fitz.Story output can be
placed in a rectangle of an existing (!) PDF page.

"""
import io
import os
import fitz

def make_pdf(fileptr, text, rect, font='sans-serif', archive=None):
    if False:
        for i in range(10):
            print('nop')
    'Make a memory DocumentWriter from HTML text and a rect.\n\n    Args:\n        fileptr: a Python file object. For example an io.BytesIO().\n        text: the text to output (HTML format)\n        rect: the target rectangle. Will use its width / height as mediabox\n        font: (str) font family name, default sans-serif\n        archive: fitz.Archive parameter. To be used if e.g. images or special\n                fonts should be used.\n    Returns:\n        The matrix to convert page rectangles of the created PDF back\n        to rectangle coordinates in the parameter "rect".\n        Normal use will expect to fit all the text in the given rect.\n        However, if an overflow occurs, this function will output multiple\n        pages, and the caller may decide to either accept or retry with\n        changed parameters.\n    '
    mediabox = fitz.Rect(0, 0, rect.width, rect.height)
    matrix = mediabox.torect(rect)
    story = fitz.Story(text, archive=archive)
    body = story.body
    body.set_properties(font=font)
    writer = fitz.DocumentWriter(fileptr)
    while True:
        device = writer.begin_page(mediabox)
        (more, _) = story.place(mediabox)
        story.draw(device)
        writer.end_page()
        if not more:
            break
    writer.close()
    return matrix
HTML = '\n<p>PyMuPDF is a great package! And it still improves significantly from one version to the next one!</p>\n<p>It is a Python binding for <b>MuPDF</b>, a lightweight PDF, XPS, and E-book viewer, renderer, and toolkit.<br> Both are maintained and developed by Artifex Software, Inc.</p>\n<p>Via MuPDF it can access files in PDF, XPS, OpenXPS, CBZ, EPUB, MOBI and FB2 (e-books) formats,<br> and it is known for its top\n<b><i>performance</i></b> and <b><i>rendering quality.</p>'
root = os.path.abspath(f'{__file__}/..')
doc = fitz.open(f'{root}/mupdf-title.pdf')
page = doc[0]
WHERE = fitz.Rect(50, 100, 250, 500)
fileptr = io.BytesIO()
matrix = make_pdf(fileptr, HTML, WHERE)
src = fitz.open('pdf', fileptr)
if src.page_count > 1:
    raise ValueError('target WHERE too small')
page.show_pdf_page(WHERE, src, 0)
doc.ez_save(f'{root}/mupdf-title-after.pdf')