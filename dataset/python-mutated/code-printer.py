"""
Demo script PyMuPDF Story class
-------------------------------

Read the Python sources in the script directory and create a PDF of all their
source codes.

The following features are included as a specialty:
1. HTML source for fitz.Story created via Python API exclusively
2. Separate Story objects for page headers and footers
3. Use of HTML "id" elements for identifying source start pages
4. Generate a Table of Contents pointing to source file starts. This
   - uses the new Stoy callback feature
   - uses Story also for making the TOC page(s)

"""
import io
import os
import time
import fitz
THISDIR = os.path.dirname(os.path.abspath(__file__))
TOC = []
CURRENT_ID = ''
MEDIABOX = fitz.paper_rect('a4-l')
WHERE = MEDIABOX + (36, 50, -36, -36)
HDR_WHERE = (36, 5, MEDIABOX.width - 36, 40)
FTR_WHERE = (36, MEDIABOX.height - 36, MEDIABOX.width - 36, MEDIABOX.height)

def recorder(elpos):
    if False:
        while True:
            i = 10
    'Callback function invoked during story.place().\n    This function generates / collects all TOC items and updates the value of\n    CURRENT_ID - which is used to update the footer line of each page.\n    '
    global TOC, CURRENT_ID
    if not elpos.open_close & 1:
        return
    level = elpos.heading
    y0 = elpos.rect[1]
    if level > 0:
        pno = elpos.page + 1
        TOC.append((level, elpos.text, elpos.page + 1, y0))
        return
    CURRENT_ID = elpos.id if elpos.id else ''
    return

def header_story(text):
    if False:
        while True:
            i = 10
    'Make the page header'
    header = fitz.Story()
    hdr_body = header.body
    hdr_body.add_paragraph().set_properties(align=fitz.fitz.TEXT_ALIGN_CENTER, bgcolor='#eee', font='sans-serif', bold=True, fontsize=12, color='green').add_text(text)
    return header

def footer_story(text):
    if False:
        print('Hello World!')
    'Make the page footer'
    footer = fitz.Story()
    ftr_body = footer.body
    ftr_body.add_paragraph().set_properties(bgcolor='#eee', align=fitz.TEXT_ALIGN_CENTER, color='blue', fontsize=10, font='sans-serif').add_text(text)
    return footer

def code_printer(outfile):
    if False:
        while True:
            i = 10
    'Output the generated PDF to outfile.'
    global MAX_TITLE_LEN
    where = +WHERE
    writer = fitz.DocumentWriter(outfile, '')
    print_time = time.strftime('%Y-%m-%d %H:%M:%S (%z)')
    thispath = os.path.abspath(os.curdir)
    basename = os.path.basename(thispath)
    story = fitz.Story()
    body = story.body
    body.set_properties(font='sans-serif')
    text = f"Python sources in folder '{THISDIR}'"
    body.add_header(1).add_text(text)
    files = os.listdir(THISDIR)
    i = 1
    for code_file in files:
        if not code_file.endswith('.py'):
            continue
        fileinput = open(os.path.join(THISDIR, code_file), 'rb')
        text = fileinput.read().decode()
        fileinput.close()
        hdr = body.add_header(2)
        if i > 1:
            hdr.set_pagebreak_before()
        hdr.add_text(f"{i}. Listing of file '{code_file}'")
        body.add_codeblock().set_bgcolor((240, 255, 210)).set_color('blue').set_id(code_file).set_fontsize(10).add_text(text)
        body.add_paragraph().set_align(fitz.TEXT_ALIGN_CENTER).add_text(f"---------- End of File '{code_file}' ----------")
        i += 1
    i = 0
    while True:
        i += 1
        device = writer.begin_page(MEDIABOX)
        header = header_story(f"Python Files in '{THISDIR}'")
        (hdr_ok, _) = header.place(HDR_WHERE)
        if hdr_ok != 0:
            raise ValueError('header does not fit')
        header.draw(device, None)
        (more, filled) = story.place(where)
        story.element_positions(recorder, {'page': i - 1})
        story.draw(device, None)
        if CURRENT_ID:
            text = f"File '{CURRENT_ID}' printed at {print_time}{chr(160) * 5}{'-' * 10}{chr(160) * 5}Page {i}"
        else:
            text = f"Printed at {print_time}{chr(160) * 5}{'-' * 10}{chr(160) * 5}Page {i}"
        footer = footer_story(text)
        (ftr_ok, _) = footer.place(FTR_WHERE)
        if ftr_ok != 0:
            raise ValueError('footer does not fit')
        footer.draw(device, None)
        writer.end_page()
        if more == 0:
            break
    writer.close()
if __name__ == '__main__' or os.environ.get('PYTEST_CURRENT_TEST'):
    fileptr1 = io.BytesIO()
    t0 = time.perf_counter()
    code_printer(fileptr1)
    t1 = time.perf_counter()
    doc = fitz.open('pdf', fileptr1)
    old_count = doc.page_count
    story = fitz.Story()
    body = story.body
    body.add_header(1).set_font('sans-serif').add_text('Table of Contents')
    TOC.insert(0, [1, 'Table of Contents', old_count + 1, 36])
    for item in TOC[1:]:
        body.add_paragraph().set_font('sans-serif').add_text(item[1] + f' - ({item[2]})')
    fileptr2 = io.BytesIO()
    writer = fitz.DocumentWriter(fileptr2)
    i = 1
    more = 1
    while more:
        device = writer.begin_page(MEDIABOX)
        header = header_story(f"Python Files in '{THISDIR}'")
        (hdr_ok, _) = header.place(HDR_WHERE)
        header.draw(device, None)
        (more, filled) = story.place(WHERE)
        story.draw(device, None)
        footer = footer_story(f'TOC-{i}')
        (ftr_ok, _) = footer.place(FTR_WHERE)
        footer.draw(device, None)
        writer.end_page()
        i += 1
    writer.close()
    doc2 = fitz.open('pdf', fileptr2)
    doc.insert_pdf(doc2)
    new_range = range(old_count, doc.page_count)
    pages = [doc[i] for i in new_range]
    for item in TOC:
        for page in pages:
            rl = page.search_for(item[1], flags=~(fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_SPANS))
            if rl != []:
                break
        rect = rl[0]
        link = {'kind': fitz.LINK_GOTO, 'from': rect, 'to': fitz.Point(0, item[3]), 'page': item[2] - 1}
        page.insert_link(link)
    doc.set_toc(TOC)
    for i in new_range:
        doc.move_page(doc.page_count - 1, 0)
    doc.ez_save(__file__.replace('.py', '.pdf'))