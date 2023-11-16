import fitz
import os
import textwrap

def test_story():
    if False:
        for i in range(10):
            print('nop')
    otf = os.path.abspath(f'{__file__}/../resources/PragmaticaC.otf')
    CSS = f'\n        @font-face {{font-family: test; src: url({otf});}}\n    '
    HTML = '\n    <p style="font-family: test;color: blue">We shall meet again at a place where there is no darkness.</p>\n    '
    MEDIABOX = fitz.paper_rect('letter')
    WHERE = MEDIABOX + (36, 36, -36, -36)
    arch = fitz.Archive('.')
    story = fitz.Story(HTML, user_css=CSS, archive=arch)
    writer = fitz.DocumentWriter('output.pdf')
    more = 1
    while more:
        device = writer.begin_page(MEDIABOX)
        (more, _) = story.place(WHERE)
        story.draw(device)
        writer.end_page()
    writer.close()

def test_2753():
    if False:
        i = 10
        return i + 15

    def rectfn(rect_num, filled):
        if False:
            return 10
        return (fitz.Rect(0, 0, 200, 200), fitz.Rect(50, 50, 100, 100), None)

    def make_pdf(html, path_out):
        if False:
            while True:
                i = 10
        story = fitz.Story(html=html)
        document = story.write_with_links(rectfn)
        print(f'Writing to: path_out={path_out!r}.')
        document.save(path_out)
        return document
    doc_before = make_pdf(textwrap.dedent('\n                <p>Before</p>\n                <p style="page-break-before: always;"></p>\n                <p>After</p>\n                '), os.path.abspath(f'{__file__}/../../tests/test_2753-out-before.pdf'))
    doc_after = make_pdf(textwrap.dedent('\n                <p>Before</p>\n                <p style="page-break-after: always;"></p>\n                <p>After</p>\n                '), os.path.abspath(f'{__file__}/../../tests/test_2753-out-after.pdf'))
    assert len(doc_before) == 2
    if fitz.mupdf_version_tuple > (1, 23, 5) and fitz.mupdf_version_tuple < (1, 24, 0):
        assert len(doc_after) == 2
    else:
        assert len(doc_after) == 1