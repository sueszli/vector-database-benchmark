import io
import textwrap
import re
import uuid
import pytest
mhtml = pytest.importorskip('qutebrowser.browser.webkit.mhtml')

@pytest.fixture(autouse=True)
def patch_uuid(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setattr(uuid, 'uuid4', lambda : 'UUID')

class Checker:
    """A helper to check mhtml output.

    Attributes:
        fp: A BytesIO object for passing to MHTMLWriter.write_to.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.fp = io.BytesIO()

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        return self.fp.getvalue()

    def expect(self, expected):
        if False:
            print('Hello World!')
        actual = self.value.decode('ascii')
        assert re.search('\\r[^\\n]', actual) is None
        assert re.search('[^\\r]\\n', actual) is None
        actual = actual.replace('\r\n', '\n')
        expected = textwrap.dedent(expected).lstrip('\n')
        assert expected == actual

@pytest.fixture
def checker():
    if False:
        return 10
    return Checker()

def test_quoted_printable_umlauts(checker):
    if False:
        for i in range(10):
            print('nop')
    content = 'Die süße Hündin läuft in die Höhle des Bären'
    content = content.encode('iso-8859-1')
    writer = mhtml.MHTMLWriter(root_content=content, content_location='localhost', content_type='text/plain')
    writer.write_to(checker.fp)
    checker.expect('\n        Content-Type: multipart/related; boundary="---=_qute-UUID"\n        MIME-Version: 1.0\n\n        -----=_qute-UUID\n        Content-Location: localhost\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        Die s=FC=DFe H=FCndin l=E4uft in die H=F6hle des B=E4ren\n        -----=_qute-UUID--\n        ')

@pytest.mark.parametrize('header, value', [('content_location', 'http://brötli.com'), ('content_type', 'text/pläin')])
def test_refuses_non_ascii_header_value(checker, header, value):
    if False:
        i = 10
        return i + 15
    defaults = {'root_content': b'', 'content_location': 'http://example.com', 'content_type': 'text/plain', header: value}
    writer = mhtml.MHTMLWriter(**defaults)
    with pytest.raises(UnicodeEncodeError, match="'ascii' codec can't encode"):
        writer.write_to(checker.fp)

def test_file_encoded_as_base64(checker):
    if False:
        while True:
            i = 10
    content = b'Image file attached'
    writer = mhtml.MHTMLWriter(root_content=content, content_type='text/plain', content_location='http://example.com')
    writer.add_file(location='http://a.example.com/image.png', content='😁 image data'.encode('utf-8'), content_type='image/png', transfer_encoding=mhtml.E_BASE64)
    writer.write_to(checker.fp)
    checker.expect('\n        Content-Type: multipart/related; boundary="---=_qute-UUID"\n        MIME-Version: 1.0\n\n        -----=_qute-UUID\n        Content-Location: http://example.com\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        Image file attached\n        -----=_qute-UUID\n        Content-Location: http://a.example.com/image.png\n        MIME-Version: 1.0\n        Content-Type: image/png\n        Content-Transfer-Encoding: base64\n\n        8J+YgSBpbWFnZSBkYXRh\n\n        -----=_qute-UUID--\n        ')

@pytest.mark.parametrize('transfer_encoding', [pytest.param(mhtml.E_BASE64, id='base64'), pytest.param(mhtml.E_QUOPRI, id='quoted-printable')])
def test_payload_lines_wrap(checker, transfer_encoding):
    if False:
        return 10
    payload = b'1234567890' * 10
    writer = mhtml.MHTMLWriter(root_content=b'', content_type='text/plain', content_location='http://example.com')
    writer.add_file(location='http://example.com/payload', content=payload, content_type='text/plain', transfer_encoding=transfer_encoding)
    writer.write_to(checker.fp)
    for line in checker.value.split(b'\r\n'):
        assert len(line) < 77

def test_files_appear_sorted(checker):
    if False:
        print('Hello World!')
    writer = mhtml.MHTMLWriter(root_content=b'root file', content_type='text/plain', content_location='http://www.example.com/')
    for subdomain in 'ahgbizt':
        writer.add_file(location='http://{}.example.com/'.format(subdomain), content='file {}'.format(subdomain).encode('utf-8'), content_type='text/plain', transfer_encoding=mhtml.E_QUOPRI)
    writer.write_to(checker.fp)
    checker.expect('\n        Content-Type: multipart/related; boundary="---=_qute-UUID"\n        MIME-Version: 1.0\n\n        -----=_qute-UUID\n        Content-Location: http://www.example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        root file\n        -----=_qute-UUID\n        Content-Location: http://a.example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        file a\n        -----=_qute-UUID\n        Content-Location: http://b.example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        file b\n        -----=_qute-UUID\n        Content-Location: http://g.example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        file g\n        -----=_qute-UUID\n        Content-Location: http://h.example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        file h\n        -----=_qute-UUID\n        Content-Location: http://i.example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        file i\n        -----=_qute-UUID\n        Content-Location: http://t.example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        file t\n        -----=_qute-UUID\n        Content-Location: http://z.example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        file z\n        -----=_qute-UUID--\n        ')

def test_empty_content_type(checker):
    if False:
        while True:
            i = 10
    writer = mhtml.MHTMLWriter(root_content=b'', content_location='http://example.com/', content_type='text/plain')
    writer.add_file('http://example.com/file', b'file content')
    writer.write_to(checker.fp)
    checker.expect('\n        Content-Type: multipart/related; boundary="---=_qute-UUID"\n        MIME-Version: 1.0\n\n        -----=_qute-UUID\n        Content-Location: http://example.com/\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n\n        -----=_qute-UUID\n        MIME-Version: 1.0\n        Content-Location: http://example.com/file\n        Content-Transfer-Encoding: quoted-printable\n\n        file content\n        -----=_qute-UUID--\n        ')

@pytest.mark.parametrize('style, expected_urls', [pytest.param("@import 'default.css'", ['default.css'], id='import with apostrophe'), pytest.param('@import "default.css"', ['default.css'], id='import with quote'), pytest.param("@import \t 'tabbed.css'", ['tabbed.css'], id='import with tab'), pytest.param("@import url('default.css')", ['default.css'], id='import with url()'), pytest.param('body {\n    background: url("/bg-img.png")\n    }', ['/bg-img.png'], id='background with body'), pytest.param('background: url(folder/file.png) no-repeat', ['folder/file.png'], id='background'), pytest.param('content: url()', [], id='content')])
def test_css_url_scanner(monkeypatch, style, expected_urls):
    if False:
        return 10
    expected_urls.sort()
    urls = mhtml._get_css_imports(style)
    urls.sort()
    assert urls == expected_urls

def test_quoted_printable_spaces(checker):
    if False:
        print('Hello World!')
    content = b' ' * 100
    writer = mhtml.MHTMLWriter(root_content=content, content_location='localhost', content_type='text/plain')
    writer.write_to(checker.fp)
    checker.expect('\n        Content-Type: multipart/related; boundary="---=_qute-UUID"\n        MIME-Version: 1.0\n\n        -----=_qute-UUID\n        Content-Location: localhost\n        MIME-Version: 1.0\n        Content-Type: text/plain\n        Content-Transfer-Encoding: quoted-printable\n\n        {}=\n        {}=20\n        -----=_qute-UUID--\n        '.format(' ' * 75, ' ' * 24))

class TestNoCloseBytesIO:

    def test_fake_close(self):
        if False:
            for i in range(10):
                print('nop')
        fp = mhtml._NoCloseBytesIO()
        fp.write(b'Value')
        fp.close()
        assert fp.getvalue() == b'Value'
        fp.write(b'Eulav')
        assert fp.getvalue() == b'ValueEulav'

    def test_actual_close(self):
        if False:
            print('Hello World!')
        fp = mhtml._NoCloseBytesIO()
        fp.write(b'Value')
        fp.actual_close()
        with pytest.raises(ValueError, match='I/O operation on closed file.'):
            fp.getvalue()
        with pytest.raises(ValueError, match='I/O operation on closed file.'):
            fp.getvalue()
            fp.write(b'Closed')