import _multibytecodec
import codecs
import io
import sys
import textwrap
import unittest
from test import support
from test.support import os_helper
from test.support.os_helper import TESTFN
ALL_CJKENCODINGS = ['gb2312', 'gbk', 'gb18030', 'hz', 'big5hkscs', 'cp932', 'shift_jis', 'euc_jp', 'euc_jisx0213', 'shift_jisx0213', 'euc_jis_2004', 'shift_jis_2004', 'cp949', 'euc_kr', 'johab', 'big5', 'cp950', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2', 'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr']

class Test_MultibyteCodec(unittest.TestCase):

    def test_nullcoding(self):
        if False:
            while True:
                i = 10
        for enc in ALL_CJKENCODINGS:
            self.assertEqual(b''.decode(enc), '')
            self.assertEqual(str(b'', enc), '')
            self.assertEqual(''.encode(enc), b'')

    def test_str_decode(self):
        if False:
            i = 10
            return i + 15
        for enc in ALL_CJKENCODINGS:
            self.assertEqual('abcd'.encode(enc), b'abcd')

    def test_errorcallback_longindex(self):
        if False:
            print('Hello World!')
        dec = codecs.getdecoder('euc-kr')
        myreplace = lambda exc: ('', sys.maxsize + 1)
        codecs.register_error('test.cjktest', myreplace)
        self.assertRaises(IndexError, dec, b'apple\x92ham\x93spam', 'test.cjktest')

    def test_errorcallback_custom_ignore(self):
        if False:
            while True:
                i = 10
        data = 100 * '\udc00'
        codecs.register_error('test.ignore', codecs.ignore_errors)
        for enc in ALL_CJKENCODINGS:
            self.assertEqual(data.encode(enc, 'test.ignore'), b'')

    def test_codingspec(self):
        if False:
            print('Hello World!')
        try:
            for enc in ALL_CJKENCODINGS:
                code = '# coding: {}\n'.format(enc)
                exec(code)
        finally:
            os_helper.unlink(TESTFN)

    def test_init_segfault(self):
        if False:
            while True:
                i = 10
        self.assertRaises(AttributeError, _multibytecodec.MultibyteStreamReader, None)
        self.assertRaises(AttributeError, _multibytecodec.MultibyteStreamWriter, None)

    def test_decode_unicode(self):
        if False:
            print('Hello World!')
        for enc in ALL_CJKENCODINGS:
            self.assertRaises(TypeError, codecs.getdecoder(enc), '')

class Test_IncrementalEncoder(unittest.TestCase):

    def test_stateless(self):
        if False:
            i = 10
            return i + 15
        encoder = codecs.getincrementalencoder('cp949')()
        self.assertEqual(encoder.encode('파이썬 마을'), b'\xc6\xc4\xc0\xcc\xbd\xe3 \xb8\xb6\xc0\xbb')
        self.assertEqual(encoder.reset(), None)
        self.assertEqual(encoder.encode('☆∼☆', True), b'\xa1\xd9\xa1\xad\xa1\xd9')
        self.assertEqual(encoder.reset(), None)
        self.assertEqual(encoder.encode('', True), b'')
        self.assertEqual(encoder.encode('', False), b'')
        self.assertEqual(encoder.reset(), None)

    def test_stateful(self):
        if False:
            print('Hello World!')
        encoder = codecs.getincrementalencoder('jisx0213')()
        self.assertEqual(encoder.encode('æ̀'), b'\xab\xc4')
        self.assertEqual(encoder.encode('æ'), b'')
        self.assertEqual(encoder.encode('̀'), b'\xab\xc4')
        self.assertEqual(encoder.encode('æ', True), b'\xa9\xdc')
        self.assertEqual(encoder.reset(), None)
        self.assertEqual(encoder.encode('̀'), b'\xab\xdc')
        self.assertEqual(encoder.encode('æ'), b'')
        self.assertEqual(encoder.encode('', True), b'\xa9\xdc')
        self.assertEqual(encoder.encode('', True), b'')

    def test_stateful_keep_buffer(self):
        if False:
            while True:
                i = 10
        encoder = codecs.getincrementalencoder('jisx0213')()
        self.assertEqual(encoder.encode('æ'), b'')
        self.assertRaises(UnicodeEncodeError, encoder.encode, 'ģ')
        self.assertEqual(encoder.encode('̀æ'), b'\xab\xc4')
        self.assertRaises(UnicodeEncodeError, encoder.encode, 'ģ')
        self.assertEqual(encoder.reset(), None)
        self.assertEqual(encoder.encode('̀'), b'\xab\xdc')
        self.assertEqual(encoder.encode('æ'), b'')
        self.assertRaises(UnicodeEncodeError, encoder.encode, 'ģ')
        self.assertEqual(encoder.encode('', True), b'\xa9\xdc')

    def test_state_methods_with_buffer_state(self):
        if False:
            while True:
                i = 10
        encoder = codecs.getincrementalencoder('euc_jis_2004')()
        initial_state = encoder.getstate()
        self.assertEqual(encoder.encode('æ̀'), b'\xab\xc4')
        encoder.setstate(initial_state)
        self.assertEqual(encoder.encode('æ̀'), b'\xab\xc4')
        self.assertEqual(encoder.encode('æ'), b'')
        partial_state = encoder.getstate()
        self.assertEqual(encoder.encode('̀'), b'\xab\xc4')
        encoder.setstate(partial_state)
        self.assertEqual(encoder.encode('̀'), b'\xab\xc4')

    def test_state_methods_with_non_buffer_state(self):
        if False:
            print('Hello World!')
        encoder = codecs.getincrementalencoder('iso2022_jp')()
        self.assertEqual(encoder.encode('z'), b'z')
        en_state = encoder.getstate()
        self.assertEqual(encoder.encode('あ'), b'\x1b$B$"')
        jp_state = encoder.getstate()
        self.assertEqual(encoder.encode('z'), b'\x1b(Bz')
        encoder.setstate(jp_state)
        self.assertEqual(encoder.encode('あ'), b'$"')
        encoder.setstate(en_state)
        self.assertEqual(encoder.encode('z'), b'z')

    def test_getstate_returns_expected_value(self):
        if False:
            print('Hello World!')
        buffer_state_encoder = codecs.getincrementalencoder('euc_jis_2004')()
        self.assertEqual(buffer_state_encoder.getstate(), 0)
        buffer_state_encoder.encode('æ')
        self.assertEqual(buffer_state_encoder.getstate(), int.from_bytes(b'\x02\xc3\xa6\x00\x00\x00\x00\x00\x00\x00\x00', 'little'))
        buffer_state_encoder.encode('̀')
        self.assertEqual(buffer_state_encoder.getstate(), 0)
        non_buffer_state_encoder = codecs.getincrementalencoder('iso2022_jp')()
        self.assertEqual(non_buffer_state_encoder.getstate(), int.from_bytes(b'\x00BB\x00\x00\x00\x00\x00\x00', 'little'))
        non_buffer_state_encoder.encode('あ')
        self.assertEqual(non_buffer_state_encoder.getstate(), int.from_bytes(b'\x00\xc2B\x00\x00\x00\x00\x00\x00', 'little'))

    def test_setstate_validates_input_size(self):
        if False:
            for i in range(10):
                print('nop')
        encoder = codecs.getincrementalencoder('euc_jp')()
        pending_size_nine = int.from_bytes(b'\t\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', 'little')
        self.assertRaises(UnicodeError, encoder.setstate, pending_size_nine)

    def test_setstate_validates_input_bytes(self):
        if False:
            while True:
                i = 10
        encoder = codecs.getincrementalencoder('euc_jp')()
        invalid_utf8 = int.from_bytes(b'\x01\xff\x00\x00\x00\x00\x00\x00\x00\x00', 'little')
        self.assertRaises(UnicodeDecodeError, encoder.setstate, invalid_utf8)

    def test_issue5640(self):
        if False:
            return 10
        encoder = codecs.getincrementalencoder('shift-jis')('backslashreplace')
        self.assertEqual(encoder.encode('ÿ'), b'\\xff')
        self.assertEqual(encoder.encode('\n'), b'\n')

    @support.cpython_only
    def test_subinterp(self):
        if False:
            print('Hello World!')
        import _testcapi
        encoding = 'cp932'
        text = 'Python の開発は、1990 年ごろから開始されています。'
        code = textwrap.dedent('\n            import codecs\n            encoding = %r\n            text = %r\n            encoder = codecs.getincrementalencoder(encoding)()\n            text2 = encoder.encode(text).decode(encoding)\n            if text2 != text:\n                raise ValueError(f"encoding issue: {text2!a} != {text!a}")\n        ') % (encoding, text)
        res = _testcapi.run_in_subinterp(code)
        self.assertEqual(res, 0)

class Test_IncrementalDecoder(unittest.TestCase):

    def test_dbcs(self):
        if False:
            for i in range(10):
                print('nop')
        decoder = codecs.getincrementaldecoder('cp949')()
        self.assertEqual(decoder.decode(b'\xc6\xc4\xc0\xcc\xbd'), '파이')
        self.assertEqual(decoder.decode(b'\xe3 \xb8\xb6\xc0\xbb'), '썬 마을')
        self.assertEqual(decoder.decode(b''), '')

    def test_dbcs_keep_buffer(self):
        if False:
            while True:
                i = 10
        decoder = codecs.getincrementaldecoder('cp949')()
        self.assertEqual(decoder.decode(b'\xc6\xc4\xc0'), '파')
        self.assertRaises(UnicodeDecodeError, decoder.decode, b'', True)
        self.assertEqual(decoder.decode(b'\xcc'), '이')
        self.assertEqual(decoder.decode(b'\xc6\xc4\xc0'), '파')
        self.assertRaises(UnicodeDecodeError, decoder.decode, b'\xcc\xbd', True)
        self.assertEqual(decoder.decode(b'\xcc'), '이')

    def test_iso2022(self):
        if False:
            i = 10
            return i + 15
        decoder = codecs.getincrementaldecoder('iso2022-jp')()
        ESC = b'\x1b'
        self.assertEqual(decoder.decode(ESC + b'('), '')
        self.assertEqual(decoder.decode(b'B', True), '')
        self.assertEqual(decoder.decode(ESC + b'$'), '')
        self.assertEqual(decoder.decode(b'B@$'), '世')
        self.assertEqual(decoder.decode(b'@$@'), '世')
        self.assertEqual(decoder.decode(b'$', True), '世')
        self.assertEqual(decoder.reset(), None)
        self.assertEqual(decoder.decode(b'@$'), '@$')
        self.assertEqual(decoder.decode(ESC + b'$'), '')
        self.assertRaises(UnicodeDecodeError, decoder.decode, b'', True)
        self.assertEqual(decoder.decode(b'B@$'), '世')

    def test_decode_unicode(self):
        if False:
            return 10
        for enc in ALL_CJKENCODINGS:
            decoder = codecs.getincrementaldecoder(enc)()
            self.assertRaises(TypeError, decoder.decode, '')

    def test_state_methods(self):
        if False:
            while True:
                i = 10
        decoder = codecs.getincrementaldecoder('euc_jp')()
        self.assertEqual(decoder.decode(b'\xa4\xa6'), 'う')
        (pending1, _) = decoder.getstate()
        self.assertEqual(pending1, b'')
        self.assertEqual(decoder.decode(b'\xa4'), '')
        (pending2, flags2) = decoder.getstate()
        self.assertEqual(pending2, b'\xa4')
        self.assertEqual(decoder.decode(b'\xa6'), 'う')
        (pending3, _) = decoder.getstate()
        self.assertEqual(pending3, b'')
        decoder.setstate((pending2, flags2))
        self.assertEqual(decoder.decode(b'\xa6'), 'う')
        (pending4, _) = decoder.getstate()
        self.assertEqual(pending4, b'')
        decoder.setstate((b'abc', 123456789))
        self.assertEqual(decoder.getstate(), (b'abc', 123456789))

    def test_setstate_validates_input(self):
        if False:
            while True:
                i = 10
        decoder = codecs.getincrementaldecoder('euc_jp')()
        self.assertRaises(TypeError, decoder.setstate, 123)
        self.assertRaises(TypeError, decoder.setstate, ('invalid', 0))
        self.assertRaises(TypeError, decoder.setstate, (b'1234', 'invalid'))
        self.assertRaises(UnicodeError, decoder.setstate, (b'123456789', 0))

class Test_StreamReader(unittest.TestCase):

    def test_bug1728403(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            f = open(TESTFN, 'wb')
            try:
                f.write(b'\xa1')
            finally:
                f.close()
            f = codecs.open(TESTFN, encoding='cp949')
            try:
                self.assertRaises(UnicodeDecodeError, f.read, 2)
            finally:
                f.close()
        finally:
            os_helper.unlink(TESTFN)

class Test_StreamWriter(unittest.TestCase):

    def test_gb18030(self):
        if False:
            return 10
        s = io.BytesIO()
        c = codecs.getwriter('gb18030')(s)
        c.write('123')
        self.assertEqual(s.getvalue(), b'123')
        c.write('𒍅')
        self.assertEqual(s.getvalue(), b'123\x907\x959')
        c.write('가¬')
        self.assertEqual(s.getvalue(), b'123\x907\x959\x827\xcf5\x810\x851')

    def test_utf_8(self):
        if False:
            for i in range(10):
                print('nop')
        s = io.BytesIO()
        c = codecs.getwriter('utf-8')(s)
        c.write('123')
        self.assertEqual(s.getvalue(), b'123')
        c.write('𒍅')
        self.assertEqual(s.getvalue(), b'123\xf0\x92\x8d\x85')
        c.write('가¬')
        self.assertEqual(s.getvalue(), b'123\xf0\x92\x8d\x85\xea\xb0\x80\xc2\xac')

    def test_streamwriter_strwrite(self):
        if False:
            for i in range(10):
                print('nop')
        s = io.BytesIO()
        wr = codecs.getwriter('gb18030')(s)
        wr.write('abcd')
        self.assertEqual(s.getvalue(), b'abcd')

class Test_ISO2022(unittest.TestCase):

    def test_g2(self):
        if False:
            while True:
                i = 10
        iso2022jp2 = b'\x1b(B:hu4:unit\x1b.A\x1bNi de famille'
        uni = ':hu4:unité de famille'
        self.assertEqual(iso2022jp2.decode('iso2022-jp-2'), uni)

    def test_iso2022_jp_g0(self):
        if False:
            return 10
        self.assertNotIn(b'\x0e', '\xad'.encode('iso-2022-jp-2'))
        for encoding in ('iso-2022-jp-2004', 'iso-2022-jp-3'):
            e = '㐆'.encode(encoding)
            self.assertFalse(any((x > 128 for x in e)))

    def test_bug1572832(self):
        if False:
            i = 10
            return i + 15
        for x in range(65536, 1114112):
            chr(x).encode('iso_2022_jp', 'ignore')

class TestStateful(unittest.TestCase):
    text = '世世'
    encoding = 'iso-2022-jp'
    expected = b'\x1b$B@$@$'
    reset = b'\x1b(B'
    expected_reset = expected + reset

    def test_encode(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.text.encode(self.encoding), self.expected_reset)

    def test_incrementalencoder(self):
        if False:
            i = 10
            return i + 15
        encoder = codecs.getincrementalencoder(self.encoding)()
        output = b''.join((encoder.encode(char) for char in self.text))
        self.assertEqual(output, self.expected)
        self.assertEqual(encoder.encode('', final=True), self.reset)
        self.assertEqual(encoder.encode('', final=True), b'')

    def test_incrementalencoder_final(self):
        if False:
            print('Hello World!')
        encoder = codecs.getincrementalencoder(self.encoding)()
        last_index = len(self.text) - 1
        output = b''.join((encoder.encode(char, index == last_index) for (index, char) in enumerate(self.text)))
        self.assertEqual(output, self.expected_reset)
        self.assertEqual(encoder.encode('', final=True), b'')

class TestHZStateful(TestStateful):
    text = '聊聊'
    encoding = 'hz'
    expected = b'~{ADAD'
    reset = b'~}'
    expected_reset = expected + reset
if __name__ == '__main__':
    unittest.main()