__license__ = 'GPL v3'
__copyright__ = '2014, Kovid Goyal <kovid at kovidgoyal.net>'
from tinycss.decoding import decode
from tinycss.tests import BaseTest

def params(css, encoding, use_bom=False, expect_error=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Nicer syntax to make a tuple.'
    return (css, encoding, use_bom, expect_error, kwargs)

class TestDecoding(BaseTest):

    def test_decoding(self):
        if False:
            return 10
        for (css, encoding, use_bom, expect_error, kwargs) in [params('', 'utf8'), params('𐂃', 'utf8'), params('é', 'latin1'), params('£', 'ShiftJIS', expect_error=True), params('£', 'ShiftJIS', protocol_encoding='Shift-JIS'), params('£', 'ShiftJIS', linking_encoding='Shift-JIS'), params('£', 'ShiftJIS', document_encoding='Shift-JIS'), params('£', 'ShiftJIS', protocol_encoding='utf8', document_encoding='ShiftJIS'), params('@charset "utf8"; £', 'ShiftJIS', expect_error=True), params('@charset "utf£8"; £', 'ShiftJIS', expect_error=True), params('@charset "unknown-encoding"; £', 'ShiftJIS', expect_error=True), params('@charset "utf8"; £', 'ShiftJIS', document_encoding='ShiftJIS'), params('£', 'ShiftJIS', linking_encoding='utf8', document_encoding='ShiftJIS'), params('@charset "utf-32"; 𐂃', 'utf-32-be'), params('@charset "Shift-JIS"; £', 'ShiftJIS'), params('@charset "ISO-8859-8"; £', 'ShiftJIS', expect_error=True), params('𐂃', 'utf-16-le', expect_error=True), params('𐂃', 'utf-16-le', use_bom=True), params('𐂃', 'utf-32-be', expect_error=True), params('𐂃', 'utf-32-be', use_bom=True), params('𐂃', 'utf-32-be', document_encoding='utf-32-be'), params('𐂃', 'utf-32-be', linking_encoding='utf-32-be'), params('@charset "utf-32-le"; 𐂃', 'utf-32-be', use_bom=True, expect_error=True), params('@charset "ISO-8859-8"; £', 'ShiftJIS', protocol_encoding='Shift-JIS'), params('@charset "unknown-encoding"; £', 'ShiftJIS', protocol_encoding='Shift-JIS'), params('@charset "Shift-JIS"; £', 'ShiftJIS', protocol_encoding='utf8'), params('@charset "Shift-JIS"; £', 'ShiftJIS', document_encoding='ISO-8859-8'), params('@charset "Shift-JIS"; £', 'ShiftJIS', linking_encoding='ISO-8859-8'), params('£', 'ShiftJIS', linking_encoding='Shift-JIS', document_encoding='ISO-8859-8')]:
            if use_bom:
                source = '\ufeff' + css
            else:
                source = css
            css_bytes = source.encode(encoding)
            (result, result_encoding) = decode(css_bytes, **kwargs)
            if expect_error:
                self.assertNotEqual(result, css)
            else:
                self.ae(result, css)