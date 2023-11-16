"""
This is a demo script using PyMuPDF's Story class to output text as a PDF with
a two-column page layout.

The script demonstrates the following features:
* Layout text around images of an existing ("target") PDF.
* Based on a few global parameters, areas on each page are identified, that
  can be used to receive text layouted by a Story.
* These global parameters are not stored anywhere in the target PDF and
  must therefore be provided in some way.
  - The width of the border(s) on each page.
  - The fontsize to use for text. This value determines whether the provided
    text will fit in the empty spaces of the (fixed) pages of target PDF. It
    cannot be predicted in any way. The script ends with an exception if
    target PDF has not enough pages, and prints a warning message if not all
    pages receive at least some text. In both cases, the FONTSIZE value
    can be changed (a float value).
  - Use of a 2-column page layout for the text.
* The layout creates a temporary (memory) PDF. Its produced page content
  (the text) is used to overlay the corresponding target page. If text
  requires more pages than are available in target PDF, an exception is raised.
  If not all target pages receive at least some text, a warning is printed.
* The script reads "image-no-go.pdf" in its own folder. This is the "target" PDF.
  It contains 2 pages with each 2 images (from the original article), which are
  positioned at places that create a broad overall test coverage. Otherwise the
  pages are empty.
* The script produces "quickfox-image-no-go.pdf" which contains the original pages
  and image positions, but with the original article text laid out around them.

Note:
--------------
This script version uses just image positions to derive "No-Go areas" for
layouting the text. Other PDF objects types are detectable by PyMuPDF and may
be taken instead or in addition, without influencing the layouting.
The following are candidates for other such "No-Go areas". Each can be detected
and located by PyMuPDF:
* Annotations
* Drawings
* Existing text

--------------
The text and images are taken from the somewhat modified Wikipedia article
https://en.wikipedia.org/wiki/The_quick_brown_fox_jumps_over_the_lazy_dog.
--------------
"""
import io
import os
import zipfile
import fitz
thisdir = os.path.dirname(os.path.abspath(__file__))
myzip = zipfile.ZipFile(os.path.join(thisdir, 'quickfox.zip'))
docname = os.path.join(thisdir, 'image-no-go.pdf')
outname = os.path.join(thisdir, 'quickfox-image-no-go.pdf')
BORDER = 36
FONTSIZE = 12.5
COLS = 2

def analyze_page(page):
    if False:
        return 10
    'Compute MediaBox and rectangles on page that are free to receive text.\n\n    Notes:\n        Assume a BORDER around the page, make 2 columns of the resulting\n        sub-rectangle and extract the rectangles of all images on page.\n        For demo purposes, the image rectangles are taken as "NO-GO areas"\n        on the page when writing text with the Story.\n        The function returns free areas for each of the columns.\n\n    Returns:\n        (page.number, mediabox, CELLS), where CELLS is a list of free cells.\n    '
    prect = page.rect
    where = prect + (BORDER, BORDER, -BORDER, -BORDER)
    TABLE = fitz.make_table(where, rows=1, cols=COLS)
    IMG_RECTS = sorted([fitz.Rect(item['bbox']) for item in page.get_image_info()], key=lambda b: (b.y1, b.x0))

    def free_cells(column):
        if False:
            for i in range(10):
                print('nop')
        'Return free areas in this column.'
        free_stripes = []
        col_imgs = [(b.y0, b.y1) for b in IMG_RECTS if abs(b & column) > 0]
        s_y0 = column.y0
        for (y0, y1) in col_imgs:
            if y0 > s_y0 + FONTSIZE:
                free_stripes.append((s_y0, y0))
            s_y0 = y1
        if s_y0 + FONTSIZE < column.y1:
            free_stripes.append((s_y0, column.y1))
        if free_stripes == []:
            free_stripes.append((column.y0, column.y1))
        CELLS = [fitz.Rect(column.x0, y0, column.x1, y1) for (y0, y1) in free_stripes]
        return CELLS
    CELLS = []
    for i in range(COLS):
        CELLS.extend(free_cells(TABLE[0][i]))
    return (page.number, prect, CELLS)
HTML = myzip.read('quickfox.html').decode()
story = fitz.Story(HTML)
body = story.body
body.set_properties(font='sans-serif')
para = body.find('p', None, None)
while para != None:
    para.set_properties(indent=15, fontsize=FONTSIZE)
    para = para.find_next('p', None, None)
img = body.find('img', None, None)
while img != None:
    next_img = img.find_next('img', None, None)
    img.remove()
    img = next_img
page_info = {}
doc = fitz.open(docname)
for page in doc:
    (pno, mediabox, cells) = analyze_page(page)
    page_info[pno] = (mediabox, cells)
doc.close()
fileobject = io.BytesIO()
writer = fitz.DocumentWriter(fileobject)
more = 1
pno = 0
while more:
    try:
        (MEDIABOX, CELLS) = page_info[pno]
    except KeyError:
        raise ValueError('text does not fit on target PDF')
    dev = writer.begin_page(MEDIABOX)
    for cell in CELLS:
        if not more:
            continue
        (more, _) = story.place(cell)
        story.draw(dev)
    writer.end_page()
    pno += 1
writer.close()
src = fitz.open('pdf', fileobject)
doc = fitz.open(doc.name)
for page in doc:
    if page.number >= src.page_count:
        print(f'Text only uses {src.page_count} target pages!')
        continue
    page.show_pdf_page(page.rect, src, page.number)
doc.ez_save(outname)