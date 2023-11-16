"""
----------------------------------------------------
This tests correct functioning of multi-page delete
----------------------------------------------------
Create a PDF in memory with 100 pages with a unique text each.
Also create a TOC with a bookmark per page.
On every page after the first to-be-deleted page, also insert a link, which
points to this page.
The bookmark text equals the text on the page for easy verification.

Then delete some pages and verify:
- the new TOC has empty items exactly for every deleted page
- the remaining TOC items still point to the correct page
- the document has no more links at all
"""
import fitz
page_count = 100
r = range(5, 35, 5)
link = {'from': fitz.Rect(100, 100, 120, 120), 'kind': fitz.LINK_GOTO, 'page': r[0], 'to': fitz.Point(100, 100)}

def test_deletion():
    if False:
        print('Hello World!')
    doc = fitz.open()
    toc = []
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((100, 100), '%i' % i)
        if i > r[0]:
            page.insert_link(link)
        toc.append([1, '%i' % i, i + 1])
    doc.set_toc(toc)
    assert doc.has_links()
    del doc[r]
    assert not doc.has_links()
    assert doc.page_count == page_count - len(r)
    toc_new = doc.get_toc()
    assert len([item for item in toc_new if item[-1] == -1]) == len(r)
    for i in r:
        assert toc_new[i][-1] == -1
    for item in toc_new:
        pno = item[-1]
        if pno == -1:
            continue
        pno -= 1
        text = doc[pno].get_text().replace('\n', '')
        assert text == item[1]
    doc.delete_page(0)
    del doc[5:10]
    doc.select(range(doc.page_count))
    doc.copy_page(0)
    doc.move_page(0)
    doc.fullcopy_page(0)