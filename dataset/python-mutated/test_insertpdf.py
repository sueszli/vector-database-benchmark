"""
* Join multiple PDFs into a new one.
* Compare with stored earlier result:
    - must have identical object definitions
    - must have different trailers
* Try inserting files in a loop.
"""
import os
import re
import fitz
scriptdir = os.path.abspath(os.path.dirname(__file__))
resources = os.path.join(scriptdir, 'resources')

def approx_parse(text):
    if False:
        for i in range(10):
            print('nop')
    "\n    Splits <text> into sequence of (text, number) pairs. Where sequence of\n    [0-9.] is not convertible to a number (e.g. '4.5.6'), <number> will be\n    None.\n    "
    ret = []
    for m in re.finditer('([^0-9]+)([0-9.]*)', text):
        text = m.group(1)
        try:
            number = float(m.group(2))
        except Exception:
            text += m.group(2)
            number = None
        ret.append((text, number))
    return ret

def approx_compare(a, b, max_delta):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compares <a> and <b>, allowing numbers to differ by up to <delta>.\n    '
    aa = approx_parse(a)
    bb = approx_parse(b)
    if len(aa) != len(bb):
        return 1
    ret = 1
    for ((at, an), (bt, bn)) in zip(aa, bb):
        if at != bt:
            break
        if an is not None and bn is not None:
            if abs(an - bn) >= max_delta:
                print(f'diff={an - bn}: an={an} bn={bn}')
                break
        elif (an is None) != (bn is None):
            break
    else:
        ret = 0
    if ret:
        print(f'Differ:\n    a={a!r}\n    b={b!r}')
    return ret

def test_insert():
    if False:
        return 10
    all_text_original = []
    all_text_combined = []
    doc1 = fitz.open()
    for i in range(5):
        text = f'doc 1, page {i}'
        page = doc1.new_page()
        page.insert_text((100, 72), text)
        all_text_original.append(text)
    doc2 = fitz.open()
    for i in range(4):
        text = f'doc 2, page {i}'
        page = doc2.new_page()
        page.insert_text((100, 72), text)
        all_text_original.append(text)
    doc3 = fitz.open()
    for i in range(3):
        text = f'doc 3, page {i}'
        page = doc3.new_page()
        page.insert_text((100, 72), text)
        all_text_original.append(text)
    doc4 = fitz.open()
    for i in range(6):
        text = f'doc 4, page {i}'
        page = doc4.new_page()
        page.insert_text((100, 72), text)
        all_text_original.append(text)
    new_doc = fitz.open()
    new_doc.insert_pdf(doc1)
    new_doc.insert_pdf(doc2)
    new_doc.insert_pdf(doc3)
    new_doc.insert_pdf(doc4)
    for page in new_doc:
        all_text_combined.append(page.get_text().replace('\n', ''))
    assert all_text_combined == all_text_original

def test_issue1417_insertpdf_in_loop():
    if False:
        print('Hello World!')
    'Using a context manager instead of explicitly closing files'
    f = os.path.join(resources, '1.pdf')
    big_doc = fitz.open()
    fd1 = os.open(f, os.O_RDONLY)
    os.close(fd1)
    for n in range(0, 1025):
        with fitz.open(f) as pdf:
            big_doc.insert_pdf(pdf)
        fd2 = os.open(f, os.O_RDONLY)
        assert fd2 == fd1
        os.close(fd2)
    big_doc.close()

def _test_insert_adobe():
    if False:
        for i in range(10):
            print('nop')
    path = os.path.abspath(f'{__file__}/../../../PyMuPDF-performance/adobe.pdf')
    if not os.path.exists(path):
        print(f'Not running test_insert_adobe() because does not exist: {os.path.relpath(path)}')
        return
    a = fitz.Document()
    b = fitz.Document(path)
    a.insert_pdf(b)