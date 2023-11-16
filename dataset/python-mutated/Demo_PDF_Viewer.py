"""
@created: 2018-08-19 18:00:00

@author: (c) 2018 Jorj X. McKie

Display a PyMuPDF Document using Tkinter
-------------------------------------------------------------------------------

Dependencies:
-------------
PyMuPDF, PySimpleGUI > v2.9.0, Tkinter with Tk v8.6+, Python 3


License:
--------
GNU GPL V3+

Description
------------
Read filename from command line and start display with page 1.
Pages can be directly jumped to, or buttons for paging can be used.
For experimental / demonstration purposes, we have included options to zoom
into the four page quadrants (top-left, bottom-right, etc.).

We also interpret keyboard events to support paging by PageDown / PageUp
keys as if the resp. buttons were clicked. Similarly, we do not include
a 'Quit' button. Instead, the ESCAPE key can be used, or cancelling the window.

To improve paging performance, we are not directly creating pixmaps from
pages, but instead from the fitz.DisplayList of the page. A display list
will be stored in a list and looked up by page number. This way, zooming
pixmaps and page re-visits will re-use a once-created display list.

"""
import sys
import fitz
import PySimpleGUI as sg
from sys import exit
sg.theme('GreenTan')
if len(sys.argv) == 1:
    fname = sg.popup_get_file('PDF Browser', 'PDF file to open', file_types=(('PDF Files', '*.pdf'),))
    if fname is None:
        sg.popup_cancel('Cancelling')
        exit(0)
else:
    fname = sys.argv[1]
doc = fitz.open(fname)
page_count = len(doc)
dlist_tab = [None] * page_count
title = "PyMuPDF display of '%s', pages: %i" % (fname, page_count)

def get_page(pno, zoom=0):
    if False:
        return 10
    'Return a PNG image for a document page number. If zoom is other than 0, one of the 4 page quadrants are zoomed-in instead and the corresponding clip returned.\n\n    '
    dlist = dlist_tab[pno]
    if not dlist:
        dlist_tab[pno] = doc[pno].getDisplayList()
        dlist = dlist_tab[pno]
    r = dlist.rect
    mp = r.tl + (r.br - r.tl) * 0.5
    mt = r.tl + (r.tr - r.tl) * 0.5
    ml = r.tl + (r.bl - r.tl) * 0.5
    mr = r.tr + (r.br - r.tr) * 0.5
    mb = r.bl + (r.br - r.bl) * 0.5
    mat = fitz.Matrix(2, 2)
    if zoom == 1:
        clip = fitz.Rect(r.tl, mp)
    elif zoom == 4:
        clip = fitz.Rect(mp, r.br)
    elif zoom == 2:
        clip = fitz.Rect(mt, mr)
    elif zoom == 3:
        clip = fitz.Rect(ml, mb)
    if zoom == 0:
        pix = dlist.getPixmap(alpha=False)
    else:
        pix = dlist.getPixmap(alpha=False, matrix=mat, clip=clip)
    return pix.getPNGData()
cur_page = 0
data = get_page(cur_page)
image_elem = sg.Image(data=data)
goto = sg.InputText(str(cur_page + 1), size=(5, 1))
layout = [[sg.Button('Prev'), sg.Button('Next'), sg.Text('Page:'), goto], [sg.Text('Zoom:'), sg.Button('Top-L'), sg.Button('Top-R'), sg.Button('Bot-L'), sg.Button('Bot-R')], [image_elem]]
my_keys = ('Next', 'Next:34', 'Prev', 'Prior:33', 'Top-L', 'Top-R', 'Bot-L', 'Bot-R', 'MouseWheel:Down', 'MouseWheel:Up')
zoom_buttons = ('Top-L', 'Top-R', 'Bot-L', 'Bot-R')
window = sg.Window(title, layout, return_keyboard_events=True, use_default_focus=False)
old_page = 0
old_zoom = 0
while True:
    (event, values) = window.read(timeout=100)
    zoom = 0
    force_page = False
    if event == sg.WIN_CLOSED:
        break
    if event in ('Escape:27',):
        break
    if event[0] == chr(13):
        try:
            cur_page = int(values[0]) - 1
            while cur_page < 0:
                cur_page += page_count
        except:
            cur_page = 0
        goto.update(str(cur_page + 1))
    elif event in ('Next', 'Next:34', 'MouseWheel:Down'):
        cur_page += 1
    elif event in ('Prev', 'Prior:33', 'MouseWheel:Up'):
        cur_page -= 1
    elif event == 'Top-L':
        zoom = 1
    elif event == 'Top-R':
        zoom = 2
    elif event == 'Bot-L':
        zoom = 3
    elif event == 'Bot-R':
        zoom = 4
    if cur_page >= page_count:
        cur_page = 0
    while cur_page < 0:
        cur_page += page_count
    if cur_page != old_page:
        zoom = old_zoom = 0
        force_page = True
    if event in zoom_buttons:
        if 0 < zoom == old_zoom:
            zoom = 0
            force_page = True
        if zoom != old_zoom:
            force_page = True
    if force_page:
        data = get_page(cur_page, zoom)
        image_elem.update(data=data)
        old_page = cur_page
    old_zoom = zoom
    if event in my_keys or not values[0]:
        goto.update(str(cur_page + 1))