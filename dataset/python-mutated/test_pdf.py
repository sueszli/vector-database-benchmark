"""Test PDF-related code, including metadata, bookmarks and hyperlinks."""
import hashlib
import io
import os
import re
from codecs import BOM_UTF16_BE
import pytest
from weasyprint import Attachment
from weasyprint.document import Document, DocumentMetadata
from weasyprint.text.fonts import FontConfiguration
from weasyprint.urls import path2url
from .testing_utils import FakeHTML, assert_no_logs, capture_logs, resource_filename
TOP = round(297 * 72 / 25.4, 6)
RIGHT = round(210 * 72 / 25.4, 6)

@assert_no_logs
@pytest.mark.parametrize('zoom', (1, 1.5, 0.5))
def test_page_size_zoom(zoom):
    if False:
        return 10
    pdf = FakeHTML(string='<style>@page{size:3in 4in').write_pdf(zoom=zoom)
    (width, height) = (int(216 * zoom), int(288 * zoom))
    assert f'/MediaBox [0 0 {width} {height}]'.encode() in pdf

@assert_no_logs
def test_bookmarks_1():
    if False:
        print('Hello World!')
    pdf = FakeHTML(string='\n      <h1>a</h1>  #\n      <h4>b</h4>  ####\n      <h3>c</h3>  ###\n      <h2>d</h2>  ##\n      <h1>e</h1>  #\n    ').write_pdf()
    assert re.findall(b'/Count ([0-9-]*)', pdf)[-1] == b'5'
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a', b'b', b'c', b'd', b'e']

@assert_no_logs
def test_bookmarks_2():
    if False:
        return 10
    pdf = FakeHTML(string='<body>').write_pdf()
    assert b'Outlines' not in pdf

@assert_no_logs
def test_bookmarks_3():
    if False:
        print('Hello World!')
    pdf = FakeHTML(string='<h1>a\xa0nbsp…</h1>').write_pdf()
    assert re.findall(b'/Title <(\\w*)>', pdf) == [b'feff006100a0006e0062007300702026']

@assert_no_logs
def test_bookmarks_4():
    if False:
        return 10
    pdf = FakeHTML(string='\n      <style>\n        * { height: 90pt; margin: 0 0 10pt 0 }\n      </style>\n      <h1>1</h1>\n      <h1>2</h1>\n      <h2 style="position: relative; left: 20pt">3</h2>\n      <h2>4</h2>\n      <h3>5</h3>\n      <span style="display: block; page-break-before: always"></span>\n      <h2>6</h2>\n      <h1>7</h1>\n      <h2>8</h2>\n      <h3>9</h3>\n      <h1>10</h1>\n      <h2>11</h2>\n    ').write_pdf()
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [str(i).encode() for i in range(1, 12)]
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    counts.pop(0)
    outlines = counts.pop()
    assert outlines == b'11'
    assert counts == [b'0', b'4', b'0', b'1', b'0', b'0', b'2', b'1', b'0', b'1', b'0']

@assert_no_logs
def test_bookmarks_5():
    if False:
        return 10
    pdf = FakeHTML(string='\n      <h2>1</h2> level 1\n      <h4>2</h4> level 2\n      <h2>3</h2> level 1\n      <h3>4</h3> level 2\n      <h4>5</h4> level 3\n    ').write_pdf()
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [str(i).encode() for i in range(1, 6)]
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    counts.pop(0)
    outlines = counts.pop()
    assert outlines == b'5'
    assert counts == [b'1', b'0', b'2', b'1', b'0']

@assert_no_logs
def test_bookmarks_6():
    if False:
        return 10
    pdf = FakeHTML(string='\n      <h2>1</h2> h2 level 1\n      <h4>2</h4> h4 level 2\n      <h3>3</h3> h3 level 2\n      <h5>4</h5> h5 level 3\n      <h1>5</h1> h1 level 1\n      <h2>6</h2> h2 level 2\n      <h2>7</h2> h2 level 2\n      <h4>8</h4> h4 level 3\n      <h1>9</h1> h1 level 1\n    ').write_pdf()
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [str(i).encode() for i in range(1, 10)]
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    counts.pop(0)
    outlines = counts.pop()
    assert outlines == b'9'
    assert counts == [b'3', b'0', b'1', b'0', b'3', b'0', b'1', b'0', b'0']

@assert_no_logs
def test_bookmarks_7():
    if False:
        return 10
    pdf = FakeHTML(string='<h2>a</h2>').write_pdf()
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a']
    (dest,) = re.findall(b'/Dest \\[(.*)\\]', pdf)
    y = round(float(dest.strip().split()[-2]))
    pdf = FakeHTML(string='<h2>a</h2>').write_pdf(zoom=1.5)
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a']
    (dest,) = re.findall(b'/Dest \\[(.*)\\]', pdf)
    assert round(float(dest.strip().split()[-2])) == 1.5 * y

@assert_no_logs
def test_bookmarks_8():
    if False:
        return 10
    pdf = FakeHTML(string='\n      <h1>a</h1>\n      <h2>b</h2>\n      <h3>c</h3>\n      <h2 style="bookmark-state: closed">d</h2>\n      <h3>e</h3>\n      <h4>f</h4>\n      <h1>g</h1>\n    ').write_pdf()
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a', b'b', b'c', b'd', b'e', b'f', b'g']
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    counts.pop(0)
    outlines = counts.pop()
    assert outlines == b'5'
    assert counts == [b'3', b'1', b'0', b'-2', b'1', b'0', b'0']

@assert_no_logs
def test_bookmarks_9():
    if False:
        print('Hello World!')
    pdf = FakeHTML(string='\n      <h1 style="bookmark-label: \'h1 on page \' counter(page)">a</h1>\n    ').write_pdf()
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    outlines = counts.pop()
    assert outlines == b'1'
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'h1 on page 1']

@assert_no_logs
def test_bookmarks_10():
    if False:
        for i in range(10):
            print('nop')
    pdf = FakeHTML(string="\n      <style>\n      div:before, div:after {\n         content: '';\n         bookmark-level: 1;\n         bookmark-label: 'x';\n      }\n      </style>\n      <div>a</div>\n    ").write_pdf()
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    outlines = counts.pop()
    assert outlines == b'2'
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'x', b'x']

@assert_no_logs
def test_bookmarks_11():
    if False:
        for i in range(10):
            print('nop')
    pdf = FakeHTML(string='\n      <div style="display:inline; white-space:pre;\n       bookmark-level:1; bookmark-label:\'a\'">\n      a\n      a\n      a\n      </div>\n      <div style="bookmark-level:1; bookmark-label:\'b\'">\n        <div>b</div>\n        <div style="break-before:always">c</div>\n      </div>\n    ').write_pdf()
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    outlines = counts.pop()
    assert outlines == b'2'
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a', b'b']

@assert_no_logs
def test_bookmarks_12():
    if False:
        print('Hello World!')
    pdf = FakeHTML(string='\n      <div style="bookmark-level:1; bookmark-label:contents">a</div>\n    ').write_pdf()
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    outlines = counts.pop()
    assert outlines == b'1'
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a']

@assert_no_logs
def test_bookmarks_13():
    if False:
        return 10
    pdf = FakeHTML(string='\n      <div style="bookmark-level:1; bookmark-label:contents;\n                  text-transform:uppercase">a</div>\n    ').write_pdf()
    counts = re.findall(b'/Count ([0-9-]*)', pdf)
    outlines = counts.pop()
    assert outlines == b'1'
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a']

@assert_no_logs
def test_bookmarks_14():
    if False:
        i = 10
        return i + 15
    pdf = FakeHTML(string='\n      <h1>a</h1>\n      <h1> b c d </h1>\n      <h1> e\n             f </h1>\n      <h1> g <span> h </span> i </h1>\n    ').write_pdf()
    assert re.findall(b'/Count ([0-9-]*)', pdf)[-1] == b'4'
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a', b'b c d', b'e f', b'g h i']

@assert_no_logs
def test_bookmarks_15():
    if False:
        print('Hello World!')
    pdf = FakeHTML(string='\n      <style>@page { size: 10pt 10pt }</style>\n      <h1>a</h1>\n    ').write_pdf()
    assert re.findall(b'/Count ([0-9-]*)', pdf)[-1] == b'1'
    assert re.findall(b'/Title \\((.*)\\)', pdf) == [b'a']
    assert b'/XYZ 0 10 0' in pdf

@assert_no_logs
def test_links_none():
    if False:
        while True:
            i = 10
    pdf = FakeHTML(string='<body>').write_pdf()
    assert b'Annots' not in pdf

@assert_no_logs
def test_links():
    if False:
        while True:
            i = 10
    pdf = FakeHTML(string='\n      <style>\n        body { margin: 0; font-size: 10pt; line-height: 2 }\n        p { display: block; height: 90pt; margin: 0 0 10pt 0 }\n        img { width: 30pt; vertical-align: top }\n      </style>\n      <p><a href="https://weasyprint.org"><img src=pattern.png></a></p>\n      <p style="padding: 0 10pt"><a\n         href="#lipsum"><img style="border: solid 1pt"\n                             src=pattern.png></a></p>\n      <p id=hello>Hello, World</p>\n      <p id=lipsum>\n        <a style="display: block; page-break-before: always; height: 30pt"\n           href="#hel%6Co"></a>a\n      </p>\n    ', base_url=resource_filename('<inline HTML>')).write_pdf()
    uris = re.findall(b'/URI \\((.*)\\)', pdf)
    types = re.findall(b'/S (/\\w*)', pdf)
    subtypes = re.findall(b'/Subtype (/\\w*)', pdf)
    rects = [[float(number) for number in match.split()] for match in re.findall(b'/Rect \\[([\\d\\.]+ [\\d\\.]+ [\\d\\.]+ [\\d\\.]+)\\]', pdf)]
    assert uris.pop(0) == b'https://weasyprint.org'
    assert subtypes.pop(0) == b'/Link'
    assert types.pop(0) == b'/URI'
    assert rects.pop(0) == [0, TOP, 30, TOP - 20]
    assert uris.pop(0) == b'https://weasyprint.org'
    assert subtypes.pop(0) == b'/Link'
    assert types.pop(0) == b'/URI'
    assert rects.pop(0) == [0, TOP, 30, TOP - 30]
    assert subtypes.pop(0) == b'/Link'
    assert b'/Dest (lipsum)' in pdf
    link = re.search(b'\\(lipsum\\) \\[\\d+ 0 R /XYZ ([\\d\\.]+ [\\d\\.]+ [\\d\\.]+)]', pdf).group(1)
    assert [float(number) for number in link.split()] == [0, TOP, 0]
    assert rects.pop(0) == [10, TOP - 100, 10 + 32, TOP - 100 - 20]
    assert subtypes.pop(0) == b'/Link'
    assert rects.pop(0) == [10, TOP - 100, 10 + 32, TOP - 100 - 32]
    assert subtypes.pop(0) == b'/Link'
    assert b'/Dest (hello)' in pdf
    link = re.search(b'\\(hello\\) \\[\\d+ 0 R /XYZ ([\\d\\.]+ [\\d\\.]+ [\\d\\.]+)]', pdf).group(1)
    assert [float(number) for number in link.split()] == [0, TOP - 200, 0]
    assert rects.pop(0) == [0, TOP, RIGHT, TOP - 30]

@assert_no_logs
def test_sorted_links():
    if False:
        print('Hello World!')
    pdf = FakeHTML(string='\n      <p id="zzz">zzz</p>\n      <p id="aaa">aaa</p>\n      <a href="#zzz">z</a>\n      <a href="#aaa">a</a>\n    ', base_url=resource_filename('<inline HTML>')).write_pdf()
    assert b'(zzz) [' in pdf.split(b'(aaa) [')[-1]

@assert_no_logs
def test_relative_links_no_height():
    if False:
        i = 10
        return i + 15
    pdf = FakeHTML(string='<a href="../lipsum" style="display: block"></a>a', base_url='https://weasyprint.org/foo/bar/').write_pdf()
    assert b'/S /URI\n/URI (https://weasyprint.org/foo/lipsum)'
    assert f'/Rect [0 {TOP} {RIGHT} {TOP}]'.encode() in pdf

@assert_no_logs
def test_relative_links_missing_base():
    if False:
        i = 10
        return i + 15
    pdf = FakeHTML(string='<a href="../lipsum" style="display: block"></a>a', base_url=None).write_pdf()
    assert b'/S /URI\n/URI (../lipsum)'
    assert f'/Rect [0 {TOP} {RIGHT} {TOP}]'.encode() in pdf

@assert_no_logs
def test_relative_links_missing_base_link():
    if False:
        i = 10
        return i + 15
    with capture_logs() as logs:
        pdf = FakeHTML(string='<div style="-weasy-link: url(../lipsum)">', base_url=None).write_pdf()
    assert b'/Annots' not in pdf
    assert len(logs) == 1
    assert 'WARNING: Ignored `-weasy-link: url(../lipsum)`' in logs[0]
    assert 'Relative URI reference without a base URI' in logs[0]

@assert_no_logs
def test_relative_links_internal():
    if False:
        return 10
    pdf = FakeHTML(string='<a href="#lipsum" id="lipsum" style="display: block"></a>a', base_url=None).write_pdf()
    assert b'/Dest (lipsum)' in pdf
    link = re.search(b'\\(lipsum\\) \\[\\d+ 0 R /XYZ ([\\d\\.]+ [\\d\\.]+ [\\d\\.]+)]', pdf).group(1)
    assert [float(number) for number in link.split()] == [0, TOP, 0]
    rect = re.search(b'/Rect \\[([\\d\\.]+ [\\d\\.]+ [\\d\\.]+ [\\d\\.]+)\\]', pdf).group(1)
    assert [float(number) for number in rect.split()] == [0, TOP, RIGHT, TOP]

@assert_no_logs
def test_relative_links_anchors():
    if False:
        i = 10
        return i + 15
    pdf = FakeHTML(string='<div style="-weasy-link: url(#lipsum)" id="lipsum"></div>a', base_url=None).write_pdf()
    assert b'/Dest (lipsum)' in pdf
    link = re.search(b'\\(lipsum\\) \\[\\d+ 0 R /XYZ ([\\d\\.]+ [\\d\\.]+ [\\d\\.]+)]', pdf).group(1)
    assert [float(number) for number in link.split()] == [0, TOP, 0]
    rect = re.search(b'/Rect \\[([\\d\\.]+ [\\d\\.]+ [\\d\\.]+ [\\d\\.]+)\\]', pdf).group(1)
    assert [float(number) for number in rect.split()] == [0, TOP, RIGHT, TOP]

@assert_no_logs
def test_relative_links_different_base():
    if False:
        for i in range(10):
            print('nop')
    pdf = FakeHTML(string='<a href="/test/lipsum"></a>a', base_url='https://weasyprint.org/foo/bar/').write_pdf()
    assert b'https://weasyprint.org/test/lipsum' in pdf

@assert_no_logs
def test_relative_links_same_base():
    if False:
        return 10
    pdf = FakeHTML(string='<a id="test" href="/foo/bar/#test"></a>a', base_url='https://weasyprint.org/foo/bar/').write_pdf()
    assert b'/Dest (test)' in pdf

@assert_no_logs
def test_missing_links():
    if False:
        return 10
    with capture_logs() as logs:
        pdf = FakeHTML(string='\n          <style> a { display: block; height: 15pt } </style>\n          <a href="#lipsum"></a>\n          <a href="#missing" id="lipsum"></a>\n          <a href=""></a>a\n        ', base_url=None).write_pdf()
    assert b'/Dest (lipsum)' in pdf
    assert len(logs) == 1
    link = re.search(b'\\(lipsum\\) \\[\\d+ 0 R /XYZ ([\\d\\.]+ [\\d\\.]+ [\\d\\.]+)]', pdf).group(1)
    assert [float(number) for number in link.split()] == [0, TOP - 15, 0]
    rect = re.search(b'/Rect \\[([\\d\\.]+ [\\d\\.]+ [\\d\\.]+ [\\d\\.]+)\\]', pdf).group(1)
    assert [float(number) for number in rect.split()] == [0, TOP, RIGHT, TOP - 15]
    assert 'ERROR: No anchor #missing for internal URI reference' in logs[0]

@assert_no_logs
def test_anchor_multiple_pages():
    if False:
        i = 10
        return i + 15
    pdf = FakeHTML(string='\n      <style> a { display: block; break-after: page } </style>\n      <div id="lipsum">\n        <a href="#lipsum"></a>\n        <a href="#lipsum"></a>\n        <a href="#lipsum"></a>\n      </div>\n    ', base_url=None).write_pdf()
    (first_page,) = re.findall(b'/Kids \\[(\\d+) 0 R', pdf)
    assert b'/Names [(lipsum) [' + first_page in pdf

@assert_no_logs
def test_embed_gif():
    if False:
        i = 10
        return i + 15
    assert b'/Filter /DCTDecode' not in FakeHTML(base_url=resource_filename('dummy.html'), string='<img src="pattern.gif">').write_pdf()

@assert_no_logs
def test_embed_jpeg():
    if False:
        while True:
            i = 10
    assert b'/Filter /DCTDecode' in FakeHTML(base_url=resource_filename('dummy.html'), string='<img src="blue.jpg">').write_pdf()

@assert_no_logs
def test_embed_image_once():
    if False:
        print('Hello World!')
    assert FakeHTML(base_url=resource_filename('dummy.html'), string='\n          <img src="blue.jpg">\n          <div style="background: url(blue.jpg)"></div>\n          <img src="blue.jpg">\n          <div style="background: url(blue.jpg) no-repeat"></div>\n        ').write_pdf().count(b'/Filter /DCTDecode') == 1

@assert_no_logs
def test_embed_images_from_pages():
    if False:
        return 10
    (page1,) = FakeHTML(base_url=resource_filename('dummy.html'), string='<img src="blue.jpg">').render().pages
    (page2,) = FakeHTML(base_url=resource_filename('dummy.html'), string='<img src="not-optimized.jpg">').render().pages
    document = Document((page1, page2), metadata=DocumentMetadata(), font_config=FontConfiguration(), url_fetcher=None).write_pdf()
    assert document.count(b'/Filter /DCTDecode') == 2

@assert_no_logs
def test_document_info():
    if False:
        return 10
    pdf = FakeHTML(string='\n      <meta name=author content="I Me &amp; Myself">\n      <title>Test document</title>\n      <h1>Another title</h1>\n      <meta name=generator content="Human\xa0after\xa0all">\n      <meta name=keywords content="html ,\tcss,\n                                   pdf,css">\n      <meta name=description content="Blah… ">\n      <meta name=dcterms.created content=2011-04-21T23:00:00Z>\n      <meta name=dcterms.modified content=2013-07-21T23:46+01:00>\n    ').write_pdf()
    assert b'/Author (I Me & Myself)' in pdf
    assert b'/Title (Test document)' in pdf
    assert b'/Creator <feff00480075006d0061006e00a00061006600740065007200a00061006c006c>' in pdf
    assert b'/Keywords (html, css, pdf)' in pdf
    assert b'/Subject <feff0042006c0061006820260020>' in pdf
    assert b'/CreationDate (D:20110421230000Z)' in pdf
    assert b"/ModDate (D:20130721234600+01'00)" in pdf

@assert_no_logs
def test_embedded_files_attachments(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    absolute_tmp_file = tmpdir.join('some_file.txt').strpath
    adata = b'12345678'
    with open(absolute_tmp_file, 'wb') as afile:
        afile.write(adata)
    absolute_url = path2url(absolute_tmp_file)
    assert absolute_url.startswith('file://')
    relative_tmp_file = tmpdir.join('äöü.txt').strpath
    rdata = b'abcdefgh'
    with open(relative_tmp_file, 'wb') as rfile:
        rfile.write(rdata)
    pdf = FakeHTML(string='\n          <title>Test document</title>\n          <meta charset="utf-8">\n          <link\n            rel="attachment"\n            title="some file attachment äöü"\n            href="data:,hi%20there">\n          <link rel="attachment" href="{0}">\n          <link rel="attachment" href="{1}">\n          <h1>Heading 1</h1>\n          <h2>Heading 2</h2>\n        '.format(absolute_url, os.path.basename(relative_tmp_file)), base_url=tmpdir.strpath).write_pdf(attachments=[Attachment('data:,oob attachment', description='Hello'), 'data:,raw URL', io.BytesIO(b'file like obj')])
    assert '<{}>'.format(hashlib.md5(b'hi there').hexdigest()).encode() in pdf
    assert b'/F ()' in pdf
    assert b'/UF (attachment.bin)' in pdf
    name = BOM_UTF16_BE + 'some file attachment äöü'.encode('utf-16-be')
    assert b'/Desc <' + name.hex().encode() + b'>' in pdf
    assert hashlib.md5(adata).hexdigest().encode() in pdf
    assert os.path.basename(absolute_tmp_file).encode() in pdf
    assert hashlib.md5(rdata).hexdigest().encode() in pdf
    name = BOM_UTF16_BE + 'some file attachment äöü'.encode('utf-16-be')
    assert b'/Desc <' + name.hex().encode() + b'>' in pdf
    assert hashlib.md5(b'oob attachment').hexdigest().encode() in pdf
    assert b'/Desc (Hello)' in pdf
    assert hashlib.md5(b'raw URL').hexdigest().encode() in pdf
    assert hashlib.md5(b'file like obj').hexdigest().encode() in pdf
    assert b'/EmbeddedFiles' in pdf
    assert b'/Outlines' in pdf

@assert_no_logs
def test_attachments_data():
    if False:
        return 10
    pdf = FakeHTML(string='\n      <title>Test document 2</title>\n      <meta charset="utf-8">\n      <link rel="attachment" href="data:,some data">\n    ').write_pdf()
    md5 = '<{}>'.format(hashlib.md5(b'some data').hexdigest()).encode()
    assert md5 in pdf
    assert b'EmbeddedFiles' in pdf

@assert_no_logs
def test_attachments_data_with_anchor():
    if False:
        for i in range(10):
            print('nop')
    pdf = FakeHTML(string='\n      <title>Test document 2</title>\n      <meta charset="utf-8">\n      <link rel="attachment" href="data:,some data">\n      <h1 id="title">Title</h1>\n      <a href="#title">example</a>\n    ').write_pdf()
    md5 = '<{}>'.format(hashlib.md5(b'some data').hexdigest()).encode()
    assert md5 in pdf
    assert b'EmbeddedFiles' in pdf

@assert_no_logs
def test_attachments_no_href():
    if False:
        print('Hello World!')
    with capture_logs() as logs:
        pdf = FakeHTML(string='\n          <title>Test document 2</title>\n          <meta charset="utf-8">\n          <link rel="attachment">\n        ').write_pdf()
    assert b'Names' not in pdf
    assert b'Outlines' not in pdf
    assert len(logs) == 1
    assert 'Missing href' in logs[0]

@assert_no_logs
def test_attachments_none():
    if False:
        i = 10
        return i + 15
    pdf = FakeHTML(string='\n      <title>Test document 3</title>\n      <meta charset="utf-8">\n      <h1>Heading</h1>\n    ').write_pdf()
    assert b'Names' not in pdf
    assert b'Outlines' in pdf

@assert_no_logs
def test_attachments_none_empty():
    if False:
        while True:
            i = 10
    pdf = FakeHTML(string='\n      <title>Test document 3</title>\n      <meta charset="utf-8">\n    ').write_pdf()
    assert b'Names' not in pdf
    assert b'Outlines' not in pdf

@assert_no_logs
def test_annotations():
    if False:
        while True:
            i = 10
    pdf = FakeHTML(string='\n      <title>Test document</title>\n      <meta charset="utf-8">\n      <a\n        rel="attachment"\n        href="data:,some data"\n        download>A link that lets you download an attachment</a>\n    ').write_pdf()
    assert hashlib.md5(b'some data').hexdigest().encode() in pdf
    assert b'/FileAttachment' in pdf
    assert b'/EmbeddedFiles' not in pdf

@pytest.mark.parametrize('style, media, bleed, trim', (('bleed: 30pt; size: 10pt', [-30, -30, 40, 40], [-10, -10, 20, 20], [0, 0, 10, 10]), ('bleed: 15pt 3pt 6pt 18pt; size: 12pt 15pt', [-18, -15, 15, 21], [-10, -10, 15, 21], [0, 0, 12, 15])))
@assert_no_logs
def test_bleed(style, media, bleed, trim):
    if False:
        print('Hello World!')
    pdf = FakeHTML(string='\n      <title>Test document</title>\n      <style>@page { %s }</style>\n      <body>test\n    ' % style).write_pdf()
    assert '/MediaBox [{} {} {} {}]'.format(*media).encode() in pdf
    assert '/BleedBox [{} {} {} {}]'.format(*bleed).encode() in pdf
    assert '/TrimBox [{} {} {} {}]'.format(*trim).encode() in pdf