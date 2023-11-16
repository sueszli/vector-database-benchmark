from tornado.httputil import url_concat, parse_multipart_form_data, HTTPHeaders, format_timestamp, HTTPServerRequest, parse_request_start_line, parse_cookie, qs_to_qsl, HTTPInputError, HTTPFile
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List

def form_data_args() -> Tuple[Dict[str, List[bytes]], Dict[str, List[HTTPFile]]]:
    if False:
        return 10
    'Return two empty dicts suitable for use with parse_multipart_form_data.\n\n    mypy insists on type annotations for dict literals, so this lets us avoid\n    the verbose types throughout this test.\n    '
    return ({}, {})

class TestUrlConcat(unittest.TestCase):

    def test_url_concat_no_query_params(self):
        if False:
            for i in range(10):
                print('nop')
        url = url_concat('https://localhost/path', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?y=y&z=z')

    def test_url_concat_encode_args(self):
        if False:
            while True:
                i = 10
        url = url_concat('https://localhost/path', [('y', '/y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?y=%2Fy&z=z')

    def test_url_concat_trailing_q(self):
        if False:
            i = 10
            return i + 15
        url = url_concat('https://localhost/path?', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?y=y&z=z')

    def test_url_concat_q_with_no_trailing_amp(self):
        if False:
            for i in range(10):
                print('nop')
        url = url_concat('https://localhost/path?x', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?x=&y=y&z=z')

    def test_url_concat_trailing_amp(self):
        if False:
            i = 10
            return i + 15
        url = url_concat('https://localhost/path?x&', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?x=&y=y&z=z')

    def test_url_concat_mult_params(self):
        if False:
            for i in range(10):
                print('nop')
        url = url_concat('https://localhost/path?a=1&b=2', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?a=1&b=2&y=y&z=z')

    def test_url_concat_no_params(self):
        if False:
            i = 10
            return i + 15
        url = url_concat('https://localhost/path?r=1&t=2', [])
        self.assertEqual(url, 'https://localhost/path?r=1&t=2')

    def test_url_concat_none_params(self):
        if False:
            while True:
                i = 10
        url = url_concat('https://localhost/path?r=1&t=2', None)
        self.assertEqual(url, 'https://localhost/path?r=1&t=2')

    def test_url_concat_with_frag(self):
        if False:
            for i in range(10):
                print('nop')
        url = url_concat('https://localhost/path#tab', [('y', 'y')])
        self.assertEqual(url, 'https://localhost/path?y=y#tab')

    def test_url_concat_multi_same_params(self):
        if False:
            for i in range(10):
                print('nop')
        url = url_concat('https://localhost/path', [('y', 'y1'), ('y', 'y2')])
        self.assertEqual(url, 'https://localhost/path?y=y1&y=y2')

    def test_url_concat_multi_same_query_params(self):
        if False:
            print('Hello World!')
        url = url_concat('https://localhost/path?r=1&r=2', [('y', 'y')])
        self.assertEqual(url, 'https://localhost/path?r=1&r=2&y=y')

    def test_url_concat_dict_params(self):
        if False:
            for i in range(10):
                print('nop')
        url = url_concat('https://localhost/path', dict(y='y'))
        self.assertEqual(url, 'https://localhost/path?y=y')

class QsParseTest(unittest.TestCase):

    def test_parsing(self):
        if False:
            while True:
                i = 10
        qsstring = 'a=1&b=2&a=3'
        qs = urllib.parse.parse_qs(qsstring)
        qsl = list(qs_to_qsl(qs))
        self.assertIn(('a', '1'), qsl)
        self.assertIn(('a', '3'), qsl)
        self.assertIn(('b', '2'), qsl)

class MultipartFormDataTest(unittest.TestCase):

    def test_file_upload(self):
        if False:
            while True:
                i = 10
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'ab.txt')
        self.assertEqual(file['body'], b'Foo')

    def test_unquoted_names(self):
        if False:
            for i in range(10):
                print('nop')
        data = b'--1234\nContent-Disposition: form-data; name=files; filename=ab.txt\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'ab.txt')
        self.assertEqual(file['body'], b'Foo')

    def test_special_filenames(self):
        if False:
            for i in range(10):
                print('nop')
        filenames = ['a;b.txt', 'a"b.txt', 'a";b.txt', 'a;"b.txt', 'a";";.txt', 'a\\"b.txt', 'a\\b.txt']
        for filename in filenames:
            logging.debug('trying filename %r', filename)
            str_data = '--1234\nContent-Disposition: form-data; name="files"; filename="%s"\n\nFoo\n--1234--' % filename.replace('\\', '\\\\').replace('"', '\\"')
            data = utf8(str_data.replace('\n', '\r\n'))
            (args, files) = form_data_args()
            parse_multipart_form_data(b'1234', data, args, files)
            file = files['files'][0]
            self.assertEqual(file['filename'], filename)
            self.assertEqual(file['body'], b'Foo')

    def test_non_ascii_filename(self):
        if False:
            while True:
                i = 10
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"; filename*=UTF-8\'\'%C3%A1b.txt\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'áb.txt')
        self.assertEqual(file['body'], b'Foo')

    def test_boundary_starts_and_ends_with_quotes(self):
        if False:
            print('Hello World!')
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        parse_multipart_form_data(b'"1234"', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'ab.txt')
        self.assertEqual(file['body'], b'Foo')

    def test_missing_headers(self):
        if False:
            print('Hello World!')
        data = b'--1234\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        with ExpectLog(gen_log, 'multipart/form-data missing headers'):
            parse_multipart_form_data(b'1234', data, args, files)
        self.assertEqual(files, {})

    def test_invalid_content_disposition(self):
        if False:
            while True:
                i = 10
        data = b'--1234\nContent-Disposition: invalid; name="files"; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        with ExpectLog(gen_log, 'Invalid multipart/form-data'):
            parse_multipart_form_data(b'1234', data, args, files)
        self.assertEqual(files, {})

    def test_line_does_not_end_with_correct_line_break(self):
        if False:
            print('Hello World!')
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"\n\nFoo--1234--'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        with ExpectLog(gen_log, 'Invalid multipart/form-data'):
            parse_multipart_form_data(b'1234', data, args, files)
        self.assertEqual(files, {})

    def test_content_disposition_header_without_name_parameter(self):
        if False:
            i = 10
            return i + 15
        data = b'--1234\nContent-Disposition: form-data; filename="ab.txt"\n\nFoo\n--1234--'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        with ExpectLog(gen_log, 'multipart/form-data value missing name'):
            parse_multipart_form_data(b'1234', data, args, files)
        self.assertEqual(files, {})

    def test_data_after_final_boundary(self):
        if False:
            print('Hello World!')
        data = b'--1234\nContent-Disposition: form-data; name="files"; filename="ab.txt"\n\nFoo\n--1234--\n'.replace(b'\n', b'\r\n')
        (args, files) = form_data_args()
        parse_multipart_form_data(b'1234', data, args, files)
        file = files['files'][0]
        self.assertEqual(file['filename'], 'ab.txt')
        self.assertEqual(file['body'], b'Foo')

class HTTPHeadersTest(unittest.TestCase):

    def test_multi_line(self):
        if False:
            while True:
                i = 10
        data = 'Foo: bar\n baz\nAsdf: qwer\n\tzxcv\nFoo: even\n     more\n     lines\n'.replace('\n', '\r\n')
        headers = HTTPHeaders.parse(data)
        self.assertEqual(headers['asdf'], 'qwer zxcv')
        self.assertEqual(headers.get_list('asdf'), ['qwer zxcv'])
        self.assertEqual(headers['Foo'], 'bar baz,even more lines')
        self.assertEqual(headers.get_list('foo'), ['bar baz', 'even more lines'])
        self.assertEqual(sorted(list(headers.get_all())), [('Asdf', 'qwer zxcv'), ('Foo', 'bar baz'), ('Foo', 'even more lines')])

    def test_malformed_continuation(self):
        if False:
            print('Hello World!')
        data = ' Foo: bar'
        self.assertRaises(HTTPInputError, HTTPHeaders.parse, data)

    def test_unicode_newlines(self):
        if False:
            for i in range(10):
                print('nop')
        newlines = ['\x1b', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029']
        for newline in newlines:
            for encoding in ['utf8', 'latin1']:
                try:
                    try:
                        encoded = newline.encode(encoding)
                    except UnicodeEncodeError:
                        continue
                    data = b'Cookie: foo=' + encoded + b'bar'
                    headers = HTTPHeaders.parse(native_str(data.decode('latin1')))
                    expected = [('Cookie', 'foo=' + native_str(encoded.decode('latin1')) + 'bar')]
                    self.assertEqual(expected, list(headers.get_all()))
                except Exception:
                    gen_log.warning('failed while trying %r in %s', newline, encoding)
                    raise

    def test_optional_cr(self):
        if False:
            return 10
        headers = HTTPHeaders.parse('CRLF: crlf\r\nLF: lf\nCR: cr\rMore: more\r\n')
        self.assertEqual(sorted(headers.get_all()), [('Cr', 'cr\rMore: more'), ('Crlf', 'crlf'), ('Lf', 'lf')])

    def test_copy(self):
        if False:
            return 10
        all_pairs = [('A', '1'), ('A', '2'), ('B', 'c')]
        h1 = HTTPHeaders()
        for (k, v) in all_pairs:
            h1.add(k, v)
        h2 = h1.copy()
        h3 = copy.copy(h1)
        h4 = copy.deepcopy(h1)
        for headers in [h1, h2, h3, h4]:
            self.assertEqual(list(sorted(headers.get_all())), all_pairs)
        for headers in [h2, h3, h4]:
            self.assertIsNot(headers, h1)
            self.assertIsNot(headers.get_list('A'), h1.get_list('A'))

    def test_pickle_roundtrip(self):
        if False:
            i = 10
            return i + 15
        headers = HTTPHeaders()
        headers.add('Set-Cookie', 'a=b')
        headers.add('Set-Cookie', 'c=d')
        headers.add('Content-Type', 'text/html')
        pickled = pickle.dumps(headers)
        unpickled = pickle.loads(pickled)
        self.assertEqual(sorted(headers.get_all()), sorted(unpickled.get_all()))
        self.assertEqual(sorted(headers.items()), sorted(unpickled.items()))

    def test_setdefault(self):
        if False:
            print('Hello World!')
        headers = HTTPHeaders()
        headers['foo'] = 'bar'
        self.assertEqual(headers.setdefault('foo', 'baz'), 'bar')
        self.assertEqual(headers['foo'], 'bar')
        self.assertEqual(headers.setdefault('quux', 'xyzzy'), 'xyzzy')
        self.assertEqual(headers['quux'], 'xyzzy')
        self.assertEqual(sorted(headers.get_all()), [('Foo', 'bar'), ('Quux', 'xyzzy')])

    def test_string(self):
        if False:
            i = 10
            return i + 15
        headers = HTTPHeaders()
        headers.add('Foo', '1')
        headers.add('Foo', '2')
        headers.add('Foo', '3')
        headers2 = HTTPHeaders.parse(str(headers))
        self.assertEqual(headers, headers2)

class FormatTimestampTest(unittest.TestCase):
    TIMESTAMP = 1359312200.503611
    EXPECTED = 'Sun, 27 Jan 2013 18:43:20 GMT'

    def check(self, value):
        if False:
            return 10
        self.assertEqual(format_timestamp(value), self.EXPECTED)

    def test_unix_time_float(self):
        if False:
            for i in range(10):
                print('nop')
        self.check(self.TIMESTAMP)

    def test_unix_time_int(self):
        if False:
            print('Hello World!')
        self.check(int(self.TIMESTAMP))

    def test_struct_time(self):
        if False:
            i = 10
            return i + 15
        self.check(time.gmtime(self.TIMESTAMP))

    def test_time_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        tup = tuple(time.gmtime(self.TIMESTAMP))
        self.assertEqual(9, len(tup))
        self.check(tup)

    def test_utc_naive_datetime(self):
        if False:
            while True:
                i = 10
        self.check(datetime.datetime.fromtimestamp(self.TIMESTAMP, datetime.timezone.utc).replace(tzinfo=None))

    def test_utc_naive_datetime_deprecated(self):
        if False:
            print('Hello World!')
        with ignore_deprecation():
            self.check(datetime.datetime.utcfromtimestamp(self.TIMESTAMP))

    def test_utc_aware_datetime(self):
        if False:
            while True:
                i = 10
        self.check(datetime.datetime.fromtimestamp(self.TIMESTAMP, datetime.timezone.utc))

    def test_other_aware_datetime(self):
        if False:
            while True:
                i = 10
        self.check(datetime.datetime.fromtimestamp(self.TIMESTAMP, datetime.timezone(datetime.timedelta(hours=-4))))

class HTTPServerRequestTest(unittest.TestCase):

    def test_default_constructor(self):
        if False:
            print('Hello World!')
        HTTPServerRequest(uri='/')

    def test_body_is_a_byte_string(self):
        if False:
            print('Hello World!')
        requets = HTTPServerRequest(uri='/')
        self.assertIsInstance(requets.body, bytes)

    def test_repr_does_not_contain_headers(self):
        if False:
            for i in range(10):
                print('nop')
        request = HTTPServerRequest(uri='/', headers=HTTPHeaders({'Canary': ['Coal Mine']}))
        self.assertTrue('Canary' not in repr(request))

class ParseRequestStartLineTest(unittest.TestCase):
    METHOD = 'GET'
    PATH = '/foo'
    VERSION = 'HTTP/1.1'

    def test_parse_request_start_line(self):
        if False:
            print('Hello World!')
        start_line = ' '.join([self.METHOD, self.PATH, self.VERSION])
        parsed_start_line = parse_request_start_line(start_line)
        self.assertEqual(parsed_start_line.method, self.METHOD)
        self.assertEqual(parsed_start_line.path, self.PATH)
        self.assertEqual(parsed_start_line.version, self.VERSION)

class ParseCookieTest(unittest.TestCase):

    def test_python_cookies(self):
        if False:
            print('Hello World!')
        "\n        Test cases copied from Python's Lib/test/test_http_cookies.py\n        "
        self.assertEqual(parse_cookie('chips=ahoy; vienna=finger'), {'chips': 'ahoy', 'vienna': 'finger'})
        self.assertEqual(parse_cookie('keebler="E=mc2; L=\\"Loves\\"; fudge=\\012;"'), {'keebler': '"E=mc2', 'L': '\\"Loves\\"', 'fudge': '\\012', '': '"'})
        self.assertEqual(parse_cookie('keebler=E=mc2'), {'keebler': 'E=mc2'})
        self.assertEqual(parse_cookie('key:term=value:term'), {'key:term': 'value:term'})
        self.assertEqual(parse_cookie('a=b; c=[; d=r; f=h'), {'a': 'b', 'c': '[', 'd': 'r', 'f': 'h'})

    def test_cookie_edgecases(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(parse_cookie('a=b; Domain=example.com'), {'a': 'b', 'Domain': 'example.com'})
        self.assertEqual(parse_cookie('a=b; h=i; a=c'), {'a': 'c', 'h': 'i'})

    def test_invalid_cookies(self):
        if False:
            print('Hello World!')
        '\n        Cookie strings that go against RFC6265 but browsers will send if set\n        via document.cookie.\n        '
        self.assertIn('django_language', parse_cookie('abc=def; unnamed; django_language=en').keys())
        self.assertEqual(parse_cookie('a=b; "; c=d'), {'a': 'b', '': '"', 'c': 'd'})
        self.assertEqual(parse_cookie('a b c=d e = f; gh=i'), {'a b c': 'd e = f', 'gh': 'i'})
        self.assertEqual(parse_cookie('a   b,c<>@:/[]?{}=d  "  =e,f g'), {'a   b,c<>@:/[]?{}': 'd  "  =e,f g'})
        self.assertEqual(parse_cookie('saint=André Bessette'), {'saint': native_str('André Bessette')})
        self.assertEqual(parse_cookie('  =  b  ;  ;  =  ;   c  =  ;  '), {'': 'b', 'c': ''})